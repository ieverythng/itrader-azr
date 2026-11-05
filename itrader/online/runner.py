"""The paper loop: one command, one DecisionRecord per cycle.

    cycle:  sync data -> frame (candidates + WAIT) -> gate -> portfolio -> risk -> execute
            -> mark equity -> DecisionRecord (JSONL) -> alerts

Sleeve bookkeeping. Strategies emit weights in "strategy space" (one weight per symbol). The
runner expands them into book legs: perp strategies trade `SYM`, spot strategies `SYM.SPOT`,
and spot+perp carry both `+w SYM.SPOT` and `-w SYM`. A sleeve that the gate refused only on
the cost check keeps its current position (it is not worth re-trading); a sleeve refused for
any other reason, or absent, goes to zero capital.

Replay mode runs the same cycle over past timestamps with the data store truncated at each
one, so the replay and the live loop share every line of decision code.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from itrader.rl_lib.tasks.decision_record import DecisionRecord

from .alerts import Alerts
from .bench.engine import funding_per_bar
from .contracts import candidate_id
from .data.store import DataStore
from .data.universe import SPOT_SUFFIX, leg_symbol
from .execution import Executor, PaperBroker
from .framer import MarketFramer
from .gate import DecisionGate
from .portfolio import Portfolio, load_bench_returns, load_profiles
from .risk import RiskEngine
from .settings import Settings, settings as default_settings
from .strategies import Strategy, all_strategies


class TruncatedStore(DataStore):
    """A DataStore view that hides everything after `cutoff` (for replay)."""

    def __init__(self, base: DataStore, cutoff: datetime) -> None:
        self.root, self.feed, self.cutoff = base.root, base.feed, pd.Timestamp(cutoff)

    def load(self, symbol: str, bar: str) -> pd.DataFrame:
        df = super().load(symbol, bar)
        return df[df.ts <= self.cutoff].reset_index(drop=True)

    def load_funding(self, symbol: str) -> pd.DataFrame:
        df = super().load_funding(symbol)
        return df[df.ts <= self.cutoff].reset_index(drop=True)


def expand_legs(name: str, instrument: str, w: pd.Series) -> pd.Series:
    if instrument == "spot":
        return pd.Series({f"{k}{SPOT_SUFFIX}": max(0.0, float(v)) for k, v in w.items()}, dtype=float)
    if instrument == "spot+perp":
        out: Dict[str, float] = {}
        for k, v in w.items():
            v = max(0.0, float(v))
            out[f"{k}{SPOT_SUFFIX}"] = v
            out[k] = -v
        return pd.Series(out, dtype=float)
    return w.astype(float)


class PaperRunner:
    def __init__(self, s: Optional[Settings] = None, store: Optional[DataStore] = None, strategies: Optional[List[Strategy]] = None,
                 bench_dir: Optional[Path] = None, out_dir: Optional[Path] = None, log: Callable[[str], None] = print,
                 sync: bool = True, broker: Optional[PaperBroker] = None, alerts_enabled: bool = True) -> None:
        self.s = s or default_settings()
        self.log = log
        self.root = self.s.resolve(self.s.general.data_root)
        self.store = store or DataStore(self.root)
        self.out = Path(out_dir) if out_dir else self.s.resolve(self.s.general.out_root)
        self.out.mkdir(parents=True, exist_ok=True)
        bench_dir = Path(bench_dir) if bench_dir else self.s.resolve(self.s.general.out_root) / "bench"
        self.profiles = load_profiles(bench_dir)
        self.bench_returns = load_bench_returns(bench_dir)
        self.strategies = strategies or all_strategies()
        self.framer = MarketFramer(self.strategies, self.store, self.s, self.profiles)
        self.gate = DecisionGate(self.s)
        self.portfolio = Portfolio(self.s)
        self.risk = RiskEngine(self.s, state_path=self.out / "risk_state.json")
        self.broker = broker or PaperBroker(self.s, feed=self.store.feed, state_path=self.out / "paper_broker.json")
        self.executor = Executor(self.broker, self.s)
        self.alerts = Alerts(self.s, log_path=self.out / "alerts.jsonl", enabled=alerts_enabled)
        self.decisions = self.out / "decisions.jsonl"
        self.state_path = self.out / "state.json"
        self.state: Dict[str, Any] = {"sleeve_weights": {}, "cycles": 0, "last_equity": None, "last_funding_ts": None, "paper_returns": []}
        if self.state_path.exists():
            try:
                self.state.update(json.loads(self.state_path.read_text()))
            except Exception:  # noqa: BLE001
                pass
        self.sync = sync
        self.by_name = {x.name: x for x in self.strategies}

    # ------------------------------------------------------------------ helpers
    def _kill_present(self) -> bool:
        return self.s.resolve(self.s.risk.kill_file).exists()

    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=1, default=str))

    def _marks(self, frame) -> Dict[str, float]:
        return dict(frame.prices)

    def _paper_returns(self) -> pd.DataFrame:
        """Daily paper returns per sleeve so far (from the state), to extend the bench history."""
        rows = self.state.get("paper_returns", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["t"] = pd.to_datetime(df["t"], utc=True)
        return df.set_index("t")

    def paper_stats(self) -> Dict[str, Dict[str, Any]]:
        """Days, Sharpe and PSR of the portfolio paper record, for the ladder and the allocator."""
        from .bench.metrics import _moments, probabilistic_sharpe

        rows = self.state.get("paper_returns", [])
        if len(rows) < 3:
            return {}
        s = pd.Series([r["portfolio"] for r in rows], index=pd.to_datetime([r["t"] for r in rows], utc=True))
        daily = (1 + s).resample("D").prod() - 1
        n, mu, sd, sk, ku = _moments(daily)
        sr = mu / sd if sd > 0 else 0.0
        out = {"days": int(n), "sharpe": float(sr * (365 ** 0.5)), "psr": probabilistic_sharpe(sr, n, sk, ku)}
        return {name: out for name in self.by_name}

    # ------------------------------------------------------------------ one cycle
    def cycle(self, now: Optional[datetime] = None, store: Optional[DataStore] = None) -> Dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        t0 = time.time()
        if self.sync and store is None:
            try:
                self.store.sync(list(self.s.data.universe), list(self.s.data.bars), years=0.1, log=lambda *_: None)
            except Exception as exc:  # noqa: BLE001
                self.alerts.send("error", f"data sync failed: {type(exc).__name__}: {exc}")
        framer = self.framer if store is None else MarketFramer(self.strategies, store, self.s, self.profiles)
        sleeve_w = {k: pd.Series(v, dtype=float) for k, v in self.state["sleeve_weights"].items()}
        current_raw = sum((w for w in sleeve_w.values()), pd.Series(dtype=float)) if sleeve_w else pd.Series(dtype=float)
        frame = framer.frame(now, current=current_raw if len(current_raw) else None)
        # marks and spreads from the live book where reachable, so fills and equity use the same prices;
        # otherwise the freshest close from the store
        marks = self._marks(frame)
        spreads: Dict[str, Any] = {}
        if store is None:
            for sym in list(marks):
                try:
                    b = self.broker.book(sym, marks.get(sym))
                    spreads[sym] = b.get("spread_bps")
                    if b.get("source") == "live" and b.get("mid"):
                        marks[sym] = float(b["mid"])
                except Exception:  # noqa: BLE001
                    spreads[sym] = None
        marks.update({f"{k}{SPOT_SUFFIX}": v for k, v in list(marks.items())})
        equity = self.broker.equity(marks)
        self.risk.mark(equity, now)

        locked = bool(self.risk.state.locked_until)
        gd = self.gate.decide(frame, current_raw, risk_locked=locked, lock_reason=self.risk.state.lock_reason)

        # sleeve targets in strategy space: approved -> new target; failed only on cost -> hold; else zero
        targets: Dict[str, pd.Series] = {}
        held: List[str] = []
        for p in frame.proposals:
            checks = gd.checks.get(p.candidate_id, [])
            failed = [c["check"] for c in checks if not c["ok"]]
            if not failed:
                targets[p.strategy] = pd.Series(p.weights, dtype=float)
            elif failed == ["utility_vs_cost"] and p.strategy in sleeve_w:
                targets[p.strategy] = sleeve_w[p.strategy]
                held.append(p.strategy)
        legs = {n: expand_legs(n, self.by_name[n].instrument, w) for n, w in targets.items()}
        hist = self.bench_returns
        pr = self._paper_returns()
        if len(pr):
            hist = pd.concat([hist, pr.reindex(columns=hist.columns)]).sort_index() if len(hist) else pr
        current_book = self.executor.current_weights(marks)
        pdec = self.portfolio.build(legs, hist, current=current_book, profiles=self.profiles, paper_stats=self.paper_stats(),
                                    regime=frame.regime) if legs else None
        book_target = pdec.asset_weights if pdec is not None else pd.Series(dtype=float)

        verdict = self.risk.check(book_target, current_book, now, quality=frame.quality, prices={k: v for k, v in marks.items() if not k.endswith(SPOT_SUFFIX)},
                                  spreads_bps={k: v for k, v in spreads.items()} | {f"{k}{SPOT_SUFFIX}": v for k, v in spreads.items()})
        cancelled = self.executor.cancel_all(now) if verdict.fast_cancel else 0
        # anything the risk engine forced (flatten, locks, reductions only) goes as a taker; the rest follows [execution] style
        report = self.executor.rebalance(verdict.approved, marks, now, reason="cycle", ranges=frame.ranges,
                                         urgent=bool(verdict.flatten or verdict.blocked))
        report.cancelled += cancelled
        # closed trades feed the cooldown rules (loss streaks); a flatten by the risk engine counts as a stop-out
        seen = int(self.state.get("closed_trades_seen", 0))
        new_trades = self.broker.closed_trades[seen:] if hasattr(self.broker, "closed_trades") else []
        for tr in new_trades:
            self.risk.on_trade_closed(float(tr.get("pnl", 0.0)), now, stopped=bool(verdict.flatten and not self._kill_present()))
        self.state["closed_trades_seen"] = seen + len(new_trades)

        # funding on perps at settlement bars (8h) in paper
        fund_paid = 0.0
        funding = framer.store.funding_panel(list(self.s.data.universe))
        if len(funding):
            last_ts = funding.index[-1]
            if str(last_ts) != self.state.get("last_funding_ts"):
                if self.state.get("last_funding_ts") is not None:        # never charge the first cycle's history
                    fund_paid = self.broker.apply_funding(funding.iloc[-1].dropna().to_dict(), marks, now, key=str(last_ts))
                self.state["last_funding_ts"] = str(last_ts)

        eq_after = self.broker.equity(marks)
        ret = eq_after / equity - 1.0 if equity else 0.0
        # sleeve-level paper returns are attributed by capital share (the paper book is one account)
        shares = pdec.sleeve_capital if pdec is not None else {}
        tot = sum(abs(v) for v in shares.values()) or 1.0
        self.state["paper_returns"].append({"t": now.isoformat(timespec="seconds"), "portfolio": ret, **{k: ret * abs(v) / tot for k, v in shares.items()}})
        self.state["paper_returns"] = self.state["paper_returns"][-20000:]
        if verdict.flatten:
            self.state["sleeve_weights"] = {}
        else:
            self.state["sleeve_weights"] = {n: {k: float(v) for k, v in w.items()} for n, w in targets.items()}
        self.state["cycles"] += 1
        self.state["last_equity"] = eq_after
        self._save()

        # the record
        rec = DecisionRecord(
            market_summary_ref=frame.market_summary_ref,
            candidate_tasks=frame.candidates,
            selected_candidate_id=gd.selected_candidate_id,
            wait_selected=gd.wait or verdict.flatten or (pdec is None),
            solver_scores={"gate": gd.to_dict(), "candidate_ids": [candidate_id(c) for c in frame.candidates]},
            risk_decisions=verdict.to_dict(),
            executor_action={**report.to_dict(), "funding_paid": fund_paid, "held_on_cost": held},
            realized_outcome={"equity_before": equity, "equity_after": eq_after, "return": ret, "previous_cycle_return": self.state["paper_returns"][-2]["portfolio"] if len(self.state["paper_returns"]) > 1 else None},
            created_at=now.isoformat(timespec="seconds"),
            metadata={"market_summary": frame.market_summary, "portfolio": pdec.to_dict() if pdec is not None else None, "quality": {k: v.to_dict() for k, v in frame.quality.items() if not v.ok},
                      "settings_hash": self.s.hash, "mode": self.s.general.mode, "seconds": round(time.time() - t0, 2), "notes": frame.notes},
        )
        with self.decisions.open("a") as fh:
            fh.write(json.dumps(rec.to_dict(), default=str) + "\n")
        # alerts
        for ev in self.risk.state.events[-5:]:
            if ev.get("t", "") >= now.isoformat(timespec="seconds")[:16]:
                self.alerts.send("risk_lock" if ev["kind"] in ("lock", "kill") else ev["kind"], f"{ev['kind']}: {ev.get('reason', ev)}", ev)
        bad = [k for k, v in frame.quality.items() if not v.ok]
        if bad:
            self.alerts.send("stale_data", f"data not ok for {', '.join(bad)}", {k: frame.quality[k].reasons for k in bad})
        # one summary per UTC day, at the first cycle of the day
        today = now.date().isoformat()
        if self.state.get("summary_day") != today:
            if self.state.get("summary_day") is not None:
                prev = float(self.state.get("summary_equity") or eq_after)
                recent = self.state["paper_returns"][-24:]
                self.alerts.send("daily_summary", f"equity {eq_after:,.2f} ({eq_after / prev - 1.0:+.2%} since the last summary), gross {verdict.approved.abs().sum():.2f}, "
                                                  f"regime {frame.regime}, {self.state['cycles']} cycles, "
                                                  f"{'LOCKED: ' + self.risk.state.lock_reason if self.risk.state.locked_until else 'risk ok'}",
                                 {"equity": eq_after, "previous": prev, "cycles_last_day": len(recent), "sleeves": pdec.sleeve_capital if pdec is not None else {}})
            self.state["summary_day"], self.state["summary_equity"] = today, eq_after
            self._save()
        d = rec.to_dict()
        self.log(f"[{now:%Y-%m-%d %H:%M}] eq {eq_after:,.2f} ({ret:+.3%}) | {gd.reason} | gross {verdict.approved.abs().sum():.2f} | "
                 f"orders {len(report.orders)} fees {report.fees:.2f} | {'; '.join(verdict.reasons) or 'risk ok'}")
        return d

    # ------------------------------------------------------------------ loops
    def run(self, cycles: Optional[int] = None, interval_s: Optional[int] = None) -> None:
        bar = str(self.s.runner.cycle_bar)
        step = {"1h": 3600, "1d": 86400, "15m": 900, "5m": 300}.get(bar, 3600)
        n = 0
        self.alerts.send("daily_summary", f"paper loop started (mode {self.s.general.mode}, settings {self.s.hash})")
        while cycles is None or n < cycles:
            try:
                self.cycle()
            except Exception as exc:  # noqa: BLE001
                self.alerts.send("error", f"cycle failed: {type(exc).__name__}: {exc}")
                self.log(f"[!] cycle failed: {type(exc).__name__}: {exc}")
            n += 1
            if cycles is not None and n >= cycles:
                break
            now = time.time()
            wait = interval_s if interval_s is not None else (step - (now % step) + 20)
            target = now + max(5, wait)
            # Sleep in short slices against the wall clock. time.sleep does not advance while the machine is
            # asleep, so one long sleep overshoots the bar close by however long the lid was shut.
            while time.time() < target:
                time.sleep(min(30.0, max(0.5, target - time.time())))

    def replay(self, start: datetime, end: datetime, step: timedelta = timedelta(hours=1), max_cycles: Optional[int] = None) -> List[Dict[str, Any]]:
        out = []
        t = start
        n = 0
        while t <= end and (max_cycles is None or n < max_cycles):
            out.append(self.cycle(now=t, store=TruncatedStore(self.store, t)))
            t += step
            n += 1
        return out
