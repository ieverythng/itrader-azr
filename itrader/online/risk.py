"""RiskEngine: the last word before any order, in every mode (spec 12).

Checks, in the order they run on a proposed book (asset -> target weight):

 1. kill file            -> flatten everything, block, alert
 2. drawdown lock         -> flatten, lock until an operator unlocks
 3. daily loss lock       -> flatten, lock until the next UTC day
 4. fast cancel           -> a move beyond `fast_cancel_move_pct` since the last cycle: no new risk, cancel resting orders
 5. stale or bad data     -> that symbol may only be reduced, never increased
 6. no-trade windows      -> explicit UTC windows and a keyword calendar: reductions only
 7. cooldown              -> after a stop-out or a loss streak: reductions only
 8. spread                -> a symbol whose live spread exceeds the cap is not traded this cycle
 9. caps                  -> per asset, per base name (spot + perp), gross
10. gap-to-zero sizing    -> gross scaled so a `gap_move` against everything costs at most `max_gap_loss_pct`

Nothing upstream can override a verdict. The engine keeps its own state file (equity marks,
locks, streaks) so a restart does not forget a lock. Every verdict lists each adjustment with
the rule and the numbers, for the DecisionRecord.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data.store import Quality
from .data.universe import base_asset
from .settings import Settings, settings as default_settings


@dataclass
class RiskState:
    equity: float = 0.0
    peak_equity: float = 0.0
    day: str = ""
    day_start_equity: float = 0.0
    realised_today: float = 0.0
    locked_until: Optional[str] = None      # ISO time, or "manual"
    lock_reason: str = ""
    cooldown_until: Optional[str] = None
    cooldown_reason: str = ""
    loss_streak: int = 0
    last_prices: Dict[str, float] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    approved: pd.Series                 # asset -> weight after every rule
    blocked: bool                        # True when nothing new may be opened this cycle
    flatten: bool                        # True when the book must go to zero
    fast_cancel: List[str]               # symbols whose resting orders must be cancelled now
    adjustments: List[Dict[str, Any]]
    reasons: List[str]
    state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"approved": {k: float(v) for k, v in self.approved.items() if abs(v) > 1e-12}, "blocked": self.blocked, "flatten": self.flatten,
                "fast_cancel": self.fast_cancel, "adjustments": self.adjustments, "reasons": self.reasons,
                "state": {k: v for k, v in self.state.items() if k != "events"}}


def _parse_windows(items: List[str]) -> List[tuple]:
    out = []
    for it in items:
        try:
            a, b = it.split("/")
            out.append((pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"), it))
        except Exception:  # noqa: BLE001
            continue
    return out


class RiskEngine:
    def __init__(self, s: Optional[Settings] = None, state_path: Optional[Path] = None, root: Optional[Path] = None) -> None:
        self.s = s or default_settings()
        self.r = self.s.risk
        self.root = root
        self.state_path = state_path
        self.state = RiskState()
        if state_path and Path(state_path).exists():
            try:
                self.state = RiskState(**json.loads(Path(state_path).read_text()))
            except Exception:  # noqa: BLE001
                self.state = RiskState()
        self.windows = _parse_windows(list(self.r.no_trade_windows_utc))
        self.keywords = [k.lower() for k in self.r.no_trade_keywords]

    # ------------------------------------------------------------------ persistence and marks
    def save(self) -> None:
        if self.state_path:
            Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.state_path).write_text(json.dumps(self.state.to_dict(), indent=1, default=str))

    def _event(self, kind: str, now: datetime, **kw) -> None:
        self.state.events.append({"t": now.isoformat(timespec="seconds"), "kind": kind, **kw})
        self.state.events = self.state.events[-200:]

    def mark(self, equity: float, now: datetime) -> None:
        """Call once per cycle with the marked equity. Rolls the day, tracks the peak."""
        st = self.state
        today = now.date().isoformat()
        if st.day != today:
            st.day, st.day_start_equity, st.realised_today = today, equity, 0.0
            if st.locked_until and st.locked_until != "manual" and pd.Timestamp(st.locked_until) <= pd.Timestamp(now):
                st.locked_until, st.lock_reason = None, ""
        st.equity = equity
        st.peak_equity = max(st.peak_equity, equity) if st.peak_equity else equity
        st.realised_today = equity / st.day_start_equity - 1.0 if st.day_start_equity else 0.0

    def on_trade_closed(self, pnl: float, now: datetime, stopped: bool = False) -> None:
        st = self.state
        st.loss_streak = st.loss_streak + 1 if pnl < 0 else 0
        if stopped:
            st.cooldown_until = (now + timedelta(hours=int(self.r.cooldown_bars_after_stop))).isoformat(timespec="seconds")
            st.cooldown_reason = "stop-out"
            self._event("cooldown", now, reason="stop-out", until=st.cooldown_until)
        elif st.loss_streak >= int(self.r.loss_streak_cooldown):
            st.cooldown_until = (now + timedelta(hours=int(self.r.cooldown_bars_after_streak))).isoformat(timespec="seconds")
            st.cooldown_reason = f"{st.loss_streak} losses in a row"
            self._event("cooldown", now, reason=st.cooldown_reason, until=st.cooldown_until)

    def unlock(self, now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        self._event("unlock", now, previous=self.state.lock_reason)
        self.state.locked_until, self.state.lock_reason = None, ""
        self.save()

    # ------------------------------------------------------------------ helpers
    def _kill(self) -> bool:
        p = self.s.resolve(self.r.kill_file, self.root)
        return p.exists()

    def _calendar_hits(self, now: datetime) -> List[str]:
        hits = []
        for a, b, label in self.windows:
            if a <= pd.Timestamp(now) <= b:
                hits.append(f"window {label}")
        p = self.s.resolve(self.r.calendar_file, self.root)
        if p.exists():
            try:
                for ev in json.loads(p.read_text()):
                    title = str(ev.get("title", "")).lower()
                    if not any(k in title for k in self.keywords):
                        continue
                    a = pd.Timestamp(ev["start"], tz="UTC") if pd.Timestamp(ev["start"]).tzinfo is None else pd.Timestamp(ev["start"])
                    b = pd.Timestamp(ev.get("end", ev["start"]))
                    b = b.tz_localize("UTC") if b.tzinfo is None else b
                    pre = pd.Timedelta(minutes=int(ev.get("pre_minutes", 30)))
                    post = pd.Timedelta(minutes=int(ev.get("post_minutes", 60)))
                    if a - pre <= pd.Timestamp(now) <= b + post:
                        hits.append(f"calendar '{ev.get('title')}'")
            except Exception as exc:  # noqa: BLE001
                hits.append(f"calendar unreadable ({type(exc).__name__}): treating as no-trade")
        return hits

    def fast_check(self, prices: Dict[str, float]) -> List[str]:
        """Symbols that moved more than `fast_cancel_move_pct` since the last cycle."""
        hit = []
        last = self.state.last_prices
        for sym, px in prices.items():
            p0 = last.get(sym)
            if p0 and p0 > 0 and abs(px / p0 - 1.0) > float(self.r.fast_cancel_move_pct):
                hit.append(sym)
        return hit

    # ------------------------------------------------------------------ the verdict
    def check(self, target: pd.Series, current: pd.Series, now: datetime, quality: Optional[Dict[str, Quality]] = None,
              prices: Optional[Dict[str, float]] = None, spreads_bps: Optional[Dict[str, float]] = None) -> Verdict:
        r = self.r
        tgt = target.astype(float).copy()
        cur = current.reindex(tgt.index.union(current.index)).fillna(0.0).astype(float)
        tgt = tgt.reindex(cur.index).fillna(0.0)
        adj: List[Dict[str, Any]] = []
        reasons: List[str] = []
        blocked = False
        flatten = False
        fast: List[str] = []
        st = self.state

        def reductions_only(why: str) -> None:
            nonlocal tgt, blocked
            blocked = True
            allowed = tgt.where(tgt.abs() < cur.abs(), cur)        # may shrink toward zero, never grow or flip
            allowed = allowed.where((allowed * cur >= 0) | (cur == 0), 0.0)
            changed = (allowed != tgt).sum()
            if changed:
                adj.append({"rule": "reductions_only", "why": why, "symbols_held": int(changed)})
            tgt = allowed
            reasons.append(why)

        # 1. kill file
        if self._kill():
            flatten, blocked = True, True
            tgt[:] = 0.0
            reasons.append(f"kill file present: {r.kill_file}")
            adj.append({"rule": "kill", "action": "flatten"})
            self._event("kill", now)
        # 2. drawdown lock
        dd = (st.equity / st.peak_equity - 1.0) if st.peak_equity else 0.0
        if not flatten and dd <= -float(r.drawdown_lock_pct):
            if st.locked_until != "manual":
                st.locked_until, st.lock_reason = "manual", f"drawdown {dd:.1%} beyond {float(r.drawdown_lock_pct):.0%}"
                self._event("lock", now, reason=st.lock_reason)
        if st.locked_until == "manual":
            flatten, blocked = True, True
            tgt[:] = 0.0
            reasons.append(f"locked: {st.lock_reason} (operator unlock required)")
            adj.append({"rule": "drawdown_lock", "drawdown": dd, "limit": -float(r.drawdown_lock_pct), "action": "flatten"})
        # 3. daily loss lock
        if not flatten and st.realised_today <= -float(r.daily_loss_limit_pct):
            nxt = (datetime.combine(now.date(), datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)).isoformat(timespec="seconds")
            if st.locked_until != nxt:
                st.locked_until, st.lock_reason = nxt, f"daily loss {st.realised_today:.2%} beyond {float(r.daily_loss_limit_pct):.1%}"
                self._event("lock", now, reason=st.lock_reason, until=nxt)
        if not flatten and st.locked_until and st.locked_until != "manual" and pd.Timestamp(st.locked_until) > pd.Timestamp(now):
            flatten, blocked = True, True
            tgt[:] = 0.0
            reasons.append(f"locked until {st.locked_until}: {st.lock_reason}")
            adj.append({"rule": "daily_loss_lock", "realised_today": st.realised_today, "action": "flatten"})
        # 4. fast cancel
        if prices:
            fast = self.fast_check(prices)
            if fast and not flatten:
                reductions_only(f"fast cancel: {', '.join(fast)} moved more than {float(r.fast_cancel_move_pct):.0%} inside one cycle")
                self._event("fast_cancel", now, symbols=fast)
            st.last_prices = dict(prices)
        # 5. stale or bad data
        if quality and bool(r.stale_block) and not flatten:
            for sym, q in quality.items():
                if not q.ok and sym in tgt.index and abs(tgt[sym]) > abs(cur[sym]):
                    adj.append({"rule": "stale_data", "symbol": sym, "reasons": q.reasons, "from": float(tgt[sym]), "to": float(cur[sym])})
                    tgt[sym] = cur[sym]
                    reasons.append(f"{sym}: data not ok ({'; '.join(q.reasons)}), no new risk")
        # 6. calendar
        if not flatten:
            hits = self._calendar_hits(now)
            if hits:
                reductions_only("no-trade: " + "; ".join(hits))
        # 7. cooldown
        if not flatten and st.cooldown_until and pd.Timestamp(st.cooldown_until) > pd.Timestamp(now):
            reductions_only(f"cooldown until {st.cooldown_until} ({st.cooldown_reason})")
        # 8. spread
        if spreads_bps and not flatten:
            for sym, bps in spreads_bps.items():
                if sym in tgt.index and bps is not None and bps > float(r.max_spread_bps) and abs(tgt[sym] - cur[sym]) > 1e-9:
                    adj.append({"rule": "spread", "symbol": sym, "spread_bps": float(bps), "limit": float(r.max_spread_bps)})
                    tgt[sym] = cur[sym]
                    reasons.append(f"{sym}: spread {bps:.1f} bps above {float(r.max_spread_bps):.0f}, not traded")
        # 9. caps
        if not flatten:
            over = tgt[tgt.abs() > float(r.max_asset_pct)]
            for sym, v in over.items():
                adj.append({"rule": "max_asset_pct", "symbol": sym, "from": float(v), "to": float(r.max_asset_pct) * (1 if v > 0 else -1)})
                tgt[sym] = float(r.max_asset_pct) * (1 if v > 0 else -1)
            names = tgt.groupby([base_asset(s) for s in tgt.index]).sum()
            for nm, v in names.items():
                if abs(v) > float(r.max_name_pct):
                    scale = float(r.max_name_pct) / abs(v)
                    syms = [s for s in tgt.index if base_asset(s) == nm]
                    for s_ in syms:
                        tgt[s_] *= scale
                    adj.append({"rule": "max_name_pct", "name": nm, "net": float(v), "scale": scale})
            gross = float(tgt.abs().sum())
            if gross > float(r.max_gross_pct):
                tgt *= float(r.max_gross_pct) / gross
                adj.append({"rule": "max_gross_pct", "from": gross, "to": float(r.max_gross_pct)})
            # 10. gap sizing
            gross = float(tgt.abs().sum())
            gap_loss = gross * float(r.gap_move)
            if gap_loss > float(r.max_gap_loss_pct) and gross > 0:
                scale = float(r.max_gap_loss_pct) / gap_loss
                tgt *= scale
                adj.append({"rule": "gap_to_zero", "gap_move": float(r.gap_move), "gross_from": gross, "gross_to": gross * scale,
                            "max_gap_loss_pct": float(r.max_gap_loss_pct)})
        self.save()
        return Verdict(tgt, blocked, flatten, fast, adj, reasons, st.to_dict())
