"""RiskEngine: the last word before any order, in every mode (spec 12).

Checks, in the order they run on a proposed book (asset -> target weight):

 1. kill file            -> flatten everything, block, alert
 2. drawdown lock         -> flatten, lock until an operator unlocks (unlock resets the peak)
 3. daily loss lock       -> flatten, lock until the next UTC day
 4. fast cancel           -> a move beyond `fast_cancel_move_pct` since the last cycle, or an unreadable price:
                             no new risk, cancel resting orders
 5. stale or bad data     -> that symbol may only be reduced, never increased or flipped (missing quality = bad)
 6. no-trade windows      -> explicit UTC windows and a keyword calendar: reductions only
 7. cooldown              -> after a stop-out or a loss streak: reductions only
 8. spread                -> a symbol whose live spread exceeds the cap may be reduced but not increased
 9. caps                  -> spot legs long only; per asset, per base name (spot + perp), gross
10. gap-to-zero sizing    -> exposure scaled so a `gap_move` against everything costs at most `max_gap_loss_pct`
                             (a spot long hedged by a perp short counts at 20%: basis risk)

Fail closed: an unreadable state file, a non-finite equity mark or a non-finite price all lock or
block rather than pass. Nothing upstream can override a verdict. State is written atomically so a
restart never forgets a lock. Every verdict lists each adjustment with the rule and the numbers.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data.store import Quality
from .data.universe import base_asset, is_crypto, is_spot, leg_symbol
from .settings import Settings, settings as default_settings


@dataclass
class RiskState:
    equity: float = 0.0
    peak_equity: float = 0.0
    day: str = ""
    day_start_equity: float = 0.0
    realised_today: float = 0.0            # mark-to-market since the day start (includes unrealised)
    locked_until: Optional[str] = None     # ISO time, or "manual"
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
    approved: pd.Series
    blocked: bool
    flatten: bool
    fast_cancel: List[str]
    adjustments: List[Dict[str, Any]]
    reasons: List[str]
    state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"approved": {k: float(v) for k, v in self.approved.items() if abs(v) > 1e-12}, "blocked": self.blocked, "flatten": self.flatten,
                "fast_cancel": self.fast_cancel, "adjustments": self.adjustments, "reasons": self.reasons,
                "state": {k: v for k, v in self.state.items() if k != "events"}}


def _aware(t: datetime) -> datetime:
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t


def _parse_windows(items: List[str]) -> tuple:
    out, bad = [], []
    for it in items:
        try:
            a, b = it.split("/")
            out.append((_ts(a), _ts(b), it))
        except Exception:  # noqa: BLE001
            bad.append(it)
    return out, bad


class RiskEngine:
    def __init__(self, s: Optional[Settings] = None, state_path: Optional[Path] = None, root: Optional[Path] = None) -> None:
        self.s = s or default_settings()
        self.r = self.s.risk
        self.root = root
        self.state_path = Path(state_path) if state_path else None
        self.state = RiskState()
        if self.state_path and self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text())
                known = {k: raw[k] for k in RiskState.__dataclass_fields__ if k in raw}
                self.state = RiskState(**known)
            except Exception as exc:  # noqa: BLE001
                # fail closed: an unreadable state could be hiding a lock
                self.state = RiskState(locked_until="manual", lock_reason=f"risk state file unreadable ({type(exc).__name__}); operator unlock required")
        self.windows, self.bad_windows = _parse_windows(list(self.r.no_trade_windows_utc))
        self.keywords = [k.lower() for k in self.r.no_trade_keywords]

    # ------------------------------------------------------------------ persistence and marks
    def save(self) -> None:
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state.to_dict(), indent=1, default=str))
            os.replace(tmp, self.state_path)

    def _event(self, kind: str, now: datetime, **kw) -> None:
        self.state.events.append({"t": _aware(now).isoformat(timespec="seconds"), "kind": kind, **kw})
        self.state.events = self.state.events[-200:]

    def mark(self, equity: float, now: datetime) -> None:
        """Call once per cycle with the marked equity. Rolls the day, tracks the peak. A non-finite
        equity locks the book: it means the broker or the marks are broken."""
        now = _aware(now)
        st = self.state
        if equity is None or not math.isfinite(float(equity)) or float(equity) <= 0:
            if st.locked_until != "manual":
                st.locked_until, st.lock_reason = "manual", f"equity mark not usable ({equity}); operator unlock required"
                self._event("lock", now, reason=st.lock_reason)
            self.save()
            return
        equity = float(equity)
        today = now.date().isoformat()
        if st.day != today:
            st.day, st.day_start_equity, st.realised_today = today, equity, 0.0
            if st.locked_until and st.locked_until != "manual" and _ts(st.locked_until) <= _ts(now):
                st.locked_until, st.lock_reason = None, ""
        st.equity = equity
        st.peak_equity = max(st.peak_equity, equity) if st.peak_equity else equity
        st.realised_today = equity / st.day_start_equity - 1.0 if st.day_start_equity else 0.0

    def on_trade_closed(self, pnl: float, now: datetime, stopped: bool = False) -> None:
        now = _aware(now)
        st = self.state
        st.loss_streak = st.loss_streak + 1 if pnl < 0 else 0
        until, why = None, ""
        if stopped:
            until, why = now + timedelta(hours=int(self.r.cooldown_bars_after_stop)), "stop-out"
        elif st.loss_streak >= int(self.r.loss_streak_cooldown):
            until, why = now + timedelta(hours=int(self.r.cooldown_bars_after_streak)), f"{st.loss_streak} losses in a row"
        if until is not None:
            if not st.cooldown_until or _ts(st.cooldown_until) < _ts(until):
                st.cooldown_until, st.cooldown_reason = until.isoformat(timespec="seconds"), why
                self._event("cooldown", now, reason=why, until=st.cooldown_until)
        self.save()

    def unlock(self, now: Optional[datetime] = None) -> None:
        """Operator action. Resets the peak to the current equity, otherwise the drawdown rule relocks at once."""
        now = _aware(now or datetime.now(timezone.utc))
        st = self.state
        self._event("unlock", now, previous=st.lock_reason, peak_from=st.peak_equity, peak_to=st.equity)
        st.locked_until, st.lock_reason = None, ""
        st.peak_equity = st.equity
        st.day_start_equity, st.realised_today = st.equity, 0.0
        self.save()

    # ------------------------------------------------------------------ helpers
    def _kill(self) -> bool:
        return self.s.resolve(self.r.kill_file, self.root).exists()

    def _calendar_hits(self, now: datetime) -> List[str]:
        hits = [f"unparseable no_trade window '{w}'" for w in self.bad_windows]     # fail closed
        for a, b, label in self.windows:
            if a <= _ts(now) <= b:
                hits.append(f"window {label}")
        p = self.s.resolve(self.r.calendar_file, self.root)
        if p.exists():
            try:
                for ev in json.loads(p.read_text()):
                    title = str(ev.get("title", "")).lower()
                    if not any(k in title for k in self.keywords):
                        continue
                    a = _ts(ev["start"])
                    b = _ts(ev.get("end") or ev["start"])
                    pre = pd.Timedelta(minutes=int(ev.get("pre_minutes", 30)))
                    post = pd.Timedelta(minutes=int(ev.get("post_minutes", 60)))
                    if a - pre <= _ts(now) <= b + post:
                        hits.append(f"calendar '{ev.get('title')}'")
            except Exception as exc:  # noqa: BLE001
                hits.append(f"calendar unreadable ({type(exc).__name__}): treating as no-trade")
        return hits

    def fast_check(self, prices: Dict[str, float]) -> List[str]:
        """Symbols that moved more than `fast_cancel_move_pct` since the last cycle, or whose price is unreadable."""
        hit = []
        last = self.state.last_prices
        for sym, px in prices.items():
            if px is None or not math.isfinite(float(px)) or float(px) <= 0:
                hit.append(sym)
                continue
            p0 = last.get(sym)
            if p0 and p0 > 0 and abs(float(px) / p0 - 1.0) > float(self.r.fast_cancel_move_pct):
                hit.append(sym)
        return hit

    # ------------------------------------------------------------------ the verdict
    def check(self, target: pd.Series, current: pd.Series, now: datetime, quality: Optional[Dict[str, Quality]] = None,
              prices: Optional[Dict[str, float]] = None, spreads_bps: Optional[Dict[str, float]] = None) -> Verdict:
        now = _aware(now)
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

        def hold(sym: str, rule: str, why: str, **kw) -> None:
            """No increase and no flip for one symbol; reductions toward zero are allowed."""
            nonlocal tgt
            t, c = float(tgt[sym]), float(cur[sym])
            if abs(t) > abs(c) or t * c < 0:
                new = c if abs(t) > abs(c) and t * c >= 0 else (0.0 if t * c < 0 else t)
                adj.append({"rule": rule, "symbol": sym, "from": t, "to": new, **kw})
                tgt[sym] = new
                reasons.append(f"{sym}: {why}")

        def reductions_only(why: str) -> None:
            nonlocal tgt, blocked
            blocked = True
            allowed = tgt.where(tgt.abs() < cur.abs(), cur)
            allowed = allowed.where((allowed * cur >= 0) | (cur == 0), 0.0)
            changed = int((allowed != tgt).sum())
            if changed:
                adj.append({"rule": "reductions_only", "why": why, "symbols_held": changed})
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
        if not flatten and dd <= -float(r.drawdown_lock_pct) and st.locked_until != "manual":
            st.locked_until, st.lock_reason = "manual", f"drawdown {dd:.1%} beyond {float(r.drawdown_lock_pct):.0%}"
            self._event("lock", now, reason=st.lock_reason)
        if st.locked_until == "manual":
            flatten, blocked = True, True
            tgt[:] = 0.0
            reasons.append(f"locked: {st.lock_reason} (operator unlock required)")
            adj.append({"rule": "manual_lock", "drawdown": dd, "action": "flatten"})
        # 3. daily loss lock
        if not flatten and st.realised_today <= -float(r.daily_loss_limit_pct):
            nxt = (datetime.combine(now.date(), datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)).isoformat(timespec="seconds")
            if st.locked_until != nxt:
                st.locked_until, st.lock_reason = nxt, f"daily loss {st.realised_today:.2%} beyond {float(r.daily_loss_limit_pct):.1%}"
                self._event("lock", now, reason=st.lock_reason, until=nxt)
        if not flatten and st.locked_until and st.locked_until != "manual" and _ts(st.locked_until) > _ts(now):
            flatten, blocked = True, True
            tgt[:] = 0.0
            reasons.append(f"locked until {st.locked_until}: {st.lock_reason}")
            adj.append({"rule": "daily_loss_lock", "realised_today": st.realised_today, "action": "flatten"})
        # 4. fast cancel
        if prices:
            fast = self.fast_check(prices)
            if fast and not flatten:
                reductions_only(f"fast cancel: {', '.join(fast)} moved more than {float(r.fast_cancel_move_pct):.0%} inside one cycle or has no usable price")
                self._event("fast_cancel", now, symbols=fast)
            st.last_prices.update({k: float(v) for k, v in prices.items() if v is not None and math.isfinite(float(v)) and float(v) > 0})
        # 5. stale or bad data (missing quality counts as bad)
        if bool(r.stale_block) and not flatten and quality is not None:
            for sym in tgt.index:
                q = quality.get(sym) or quality.get(leg_symbol(sym))
                if q is None:
                    hold(sym, "stale_data", "no data quality record, no new risk", reasons=["no quality record"])
                elif not q.ok:
                    hold(sym, "stale_data", f"data not ok ({'; '.join(q.reasons)}), no new risk", reasons=q.reasons)
        # 6. calendar
        if not flatten:
            hits = self._calendar_hits(now)
            if hits:
                reductions_only("no-trade: " + "; ".join(hits))
        # 7. cooldown
        if not flatten and st.cooldown_until and _ts(st.cooldown_until) > _ts(now):
            reductions_only(f"cooldown until {st.cooldown_until} ({st.cooldown_reason})")
        # 8. spread: increases and flips only; de-risking through a wide spread is allowed
        if spreads_bps and not flatten:
            for sym in tgt.index:
                bps = spreads_bps.get(sym, spreads_bps.get(leg_symbol(sym)))
                if bps is not None and math.isfinite(float(bps)) and float(bps) > float(r.max_spread_bps):
                    hold(sym, "spread", f"spread {float(bps):.1f} bps above {float(r.max_spread_bps):.0f}, no new risk", spread_bps=float(bps), limit=float(r.max_spread_bps))
        # 9. caps
        if not flatten:
            for sym in tgt.index:
                if is_spot(sym) and tgt[sym] < 0:
                    adj.append({"rule": "spot_long_only", "symbol": sym, "from": float(tgt[sym]), "to": 0.0})
                    tgt[sym] = 0.0
            over = tgt[tgt.abs() > float(r.max_asset_pct)]
            for sym, v in over.items():
                adj.append({"rule": "max_asset_pct", "symbol": sym, "from": float(v), "to": float(r.max_asset_pct) * (1 if v > 0 else -1)})
                tgt[sym] = float(r.max_asset_pct) * (1 if v > 0 else -1)
            names = tgt.groupby([base_asset(s) for s in tgt.index]).sum()
            for nm, v in names.items():
                if abs(v) > float(r.max_name_pct):
                    scale = float(r.max_name_pct) / abs(v)
                    for s_ in [s for s in tgt.index if base_asset(s) == nm]:
                        tgt[s_] *= scale
                    adj.append({"rule": "max_name_pct", "name": nm, "net": float(v), "scale": scale})
            gross = float(tgt.abs().sum())
            if gross > float(r.max_gross_pct):
                tgt *= float(r.max_gross_pct) / gross
                adj.append({"rule": "max_gross_pct", "from": gross, "to": float(r.max_gross_pct)})
            # 10. gap sizing
            gross = float(tgt.abs().sum())
            exposure = 0.0
            listed = float(r.get("gap_move_listed", r.gap_move)) / float(r.gap_move)   # diversified listed markets gap less than crypto
            for nm in {base_asset(s_) for s_ in tgt.index}:
                legs_ = tgt[[s_ for s_ in tgt.index if base_asset(s_) == nm]]
                if not any(is_crypto(s_) for s_ in legs_.index):
                    legs_ = legs_ * listed
                long_, short_ = float(legs_[legs_ > 0].sum()), float(-legs_[legs_ < 0].sum())
                exposure += abs(long_ - short_) + float(r.get("hedged_gap_factor", 0.2)) * min(long_, short_)
            gap_loss = exposure * float(r.gap_move)
            if gap_loss > float(r.max_gap_loss_pct) and exposure > 0:
                scale = float(r.max_gap_loss_pct) / gap_loss
                tgt *= scale
                adj.append({"rule": "gap_to_zero", "gap_move": float(r.gap_move), "exposure": exposure, "gross_from": gross, "gross_to": gross * scale,
                            "max_gap_loss_pct": float(r.max_gap_loss_pct)})
        self.save()
        return Verdict(tgt, blocked, flatten, fast, adj, reasons, st.to_dict())
