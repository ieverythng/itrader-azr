"""Executor: approved book -> orders -> fills -> report. Deterministic normalisation only.

Given the RiskEngine's approved weights, the current positions and marks, it computes the
notional to trade per leg, submits orders, and reports fills, fees and slippage. It never
changes a weight the risk engine approved, except to round to a lot size.

Two styles (`[execution] style`):

* taker: market orders, filled after the latency at mid plus half spread plus slippage.
* maker: a change in position is posted as a resting limit order at the touch (buy at the bid,
  sell at the ask) and left until the next cycle. At the next cycle the broker fills the ones
  the market traded through; whatever is still resting is cancelled and, if the book still wants
  the change, crossed as a taker ("chased"). So the worst case is one cycle's delay plus the taker
  cost, and the usual case pays the maker fee and no spread.

Anything the risk engine marks urgent (flatten, locks, reductions-only) is always taker: de-risking
does not wait in a queue. A spot leg and its perp hedge are also traded as takers, together.

Rules that keep "flatten" meaning flatten: a reducing order is clamped to the position and
floor-rounded (never rounds past zero), and a reduction is sent even below the minimum notional.
Order ids are derived from symbol, time, reason and type, so a replay reproduces the same fills.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from ..data.universe import base_asset, is_spot, leg_symbol
from ..settings import Settings, settings as default_settings
from .broker import Broker, Order
from .paper import mark_for


@dataclass
class ExecutionReport:
    orders: List[Order]
    equity_before: float
    equity_after: float
    traded_notional: float
    fees: float
    slippage_bps_weighted: float
    skipped: List[Dict[str, Any]]
    cancelled: int = 0
    notes: List[str] = field(default_factory=list)
    maker: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"orders": [o.to_dict() for o in self.orders], "equity_before": self.equity_before, "equity_after": self.equity_after,
                "traded_notional": self.traded_notional, "fees": self.fees, "slippage_bps_weighted": self.slippage_bps_weighted,
                "skipped": self.skipped, "cancelled": self.cancelled, "notes": self.notes, "maker": self.maker,
                "n_filled": sum(1 for o in self.orders if o.state.value == "filled"),
                "n_rejected": sum(1 for o in self.orders if o.state.value == "rejected"),
                "n_resting": sum(1 for o in self.orders if o.open)}


class Executor:
    def __init__(self, broker: Broker, s: Optional[Settings] = None, lot_steps: Optional[Dict[str, float]] = None) -> None:
        self.broker = broker
        self.s = s or default_settings()
        self.lot_steps = lot_steps or {}

    def current_weights(self, marks: Dict[str, float]) -> pd.Series:
        eq = self.broker.equity(marks)
        pos = self.broker.positions()
        if not math.isfinite(eq) or eq <= 0 or not pos:
            return pd.Series(dtype=float)
        return pd.Series({sym: q * (mark_for(marks, sym) or float("nan")) / eq for sym, q in pos.items()}, dtype=float)

    def _floor(self, symbol: str, qty: float) -> float:
        step = self.lot_steps.get(leg_symbol(symbol)) or 1e-6
        return math.floor(qty / step + 1e-9) * step

    def cancel_all(self, now: datetime) -> int:
        return self.broker.cancel_all(now)

    def rebalance(self, target: pd.Series, marks: Dict[str, float], now: datetime, reason: str = "rebalance",
                  wait: Optional[timedelta] = None, style: Optional[str] = None, ranges: Optional[Dict[str, Any]] = None,
                  urgent: bool = False) -> ExecutionReport:
        eq0 = self.broker.equity(marks)
        skipped: List[Dict[str, Any]] = []
        if not math.isfinite(eq0) or eq0 <= 0:
            return ExecutionReport([], eq0, eq0, 0.0, 0.0, 0.0, [{"symbol": "*", "why": f"equity not usable ({eq0}); no orders"}])
        can_rest = bool(getattr(self.broker, "supports_limit", False))
        style = str(style or self.s.execution.get("style", "taker"))
        if urgent or not can_rest:
            style = "taker"

        # 1. last cycle's resting orders: take the fills the market gave, cancel the rest and remember to chase them
        earlier: List[Order] = []
        chase: set = set()
        cancelled = 0
        if can_rest:
            earlier = [o for o in self.broker.poll(now, marks, ranges) if o.type == "limit" and o.fills]
            for o in [o for o in self.broker.open_orders() if o.type == "limit"]:
                chase.add(o.symbol)
                if self.broker.cancel(o.id, now):
                    cancelled += 1
                    o.error = o.error or "not filled within one cycle; cancelled"

        # 2. what the book still needs
        eq = self.broker.equity(marks)
        pos = self.broker.positions()
        cur = self.current_weights(marks)
        tgt = target.astype(float)
        idx = tgt.index.union(cur.index).union(pd.Index(list(pos)))
        hedged = {base_asset(x) for x in idx if is_spot(x)}             # names with a spot leg: both legs go together, as takers
        min_notional = float(self.s.execution.min_order_notional)
        orders: List[Order] = []
        posted = chased = 0
        for sym in idx:
            t = float(tgt.get(sym, 0.0))
            c = float(cur.get(sym, 0.0))
            mark = mark_for(marks, sym)
            if mark is None:
                if sym in pos or abs(t) > 0:
                    skipped.append({"symbol": sym, "why": "no usable mark", "target_w": t})
                continue
            if not math.isfinite(t) or not math.isfinite(c):
                skipped.append({"symbol": sym, "why": "non-finite weight", "target_w": t, "current_w": c})
                continue
            dw = t - c
            notional = abs(dw) * eq
            if notional < 1e-9:
                continue
            held = pos.get(sym, 0.0)
            reducing = held != 0 and (dw * held) < 0
            closing = reducing and abs(t) < 1e-9          # a full close is always sent, however small: dust must be closable
            if notional < min_notional and not closing:
                skipped.append({"symbol": sym, "why": f"below min notional {min_notional}", "notional": notional})
                continue
            qty = notional / mark
            if reducing:
                qty = min(qty, abs(held))
                qty = abs(held) if abs(t) < 1e-9 or abs(qty - abs(held)) <= abs(held) * 1e-6 else self._floor(sym, qty)
            else:
                qty = self._floor(sym, qty)
            if qty <= 0:
                skipped.append({"symbol": sym, "why": "rounds to zero", "notional": notional})
                continue
            rest = style == "maker" and sym not in chase and base_asset(sym) not in hedged
            side = "buy" if dw > 0 else "sell"
            kind = "limit" if rest else "market"
            oid = hashlib.sha1(f"{sym}|{now.isoformat()}|{reason}|{kind}".encode()).hexdigest()[:12]
            meta = {"target_w": t, "current_w": c, "reducing": reducing, "chased": sym in chase and style == "maker"}
            if rest:
                px = mark
                try:
                    b = self.broker.book(sym, mark)
                    cand = float(b["bid"] if side == "buy" else b["ask"])
                    px = cand if math.isfinite(cand) and cand > 0 else mark
                except Exception:  # noqa: BLE001
                    px = mark
                o = Order(symbol=sym, side=side, qty=qty, notional=qty * mark, reason=reason, type="limit", limit_price=px, id=oid, meta=meta)
                posted += 1
            else:
                o = Order(symbol=sym, side=side, qty=qty, notional=qty * mark, reason=reason, id=oid, meta=meta)
                chased += 1 if meta["chased"] else 0
            orders.append(self.broker.submit(o, mark, now))

        # 3. market orders are worked to completion now; limit orders are left resting for the next cycle
        market = [o for o in orders if o.type == "market"]
        deadline = now + (wait or timedelta(seconds=int(self.s.execution.order_ttl_s)))
        t_ = now
        step = timedelta(milliseconds=max(50, int(self.s.execution.latency_ms)))
        while any(o.open for o in market) and t_ <= deadline:
            t_ = t_ + step
            if getattr(self.broker, "live", False):
                time.sleep(step.total_seconds())
            self.broker.poll(t_, marks)
        for o in market:
            if o.open and self.broker.cancel(o.id, t_):
                cancelled += 1
                o.error = o.error or "ttl expired, cancelled"
        eq1 = self.broker.equity(marks)
        done = earlier + orders
        fills = [f for o in done for f in o.fills]
        traded = sum(f.qty * f.price for f in fills)
        slip = sum(abs(f.slippage_bps) * f.qty * f.price for f in fills) / traded if traded > 0 else 0.0
        maker_notional = sum(f.qty * f.price for o in done if o.meta.get("filled_as") == "maker" for f in o.fills)
        return ExecutionReport(done, eq0, eq1, traded, sum(f.fee for f in fills), slip, skipped, cancelled,
                               maker={"style": style, "posted": posted, "filled_since_last_cycle": len(earlier), "unfilled_cancelled": len(chase),
                                      "chased_as_taker": chased, "maker_notional": maker_notional, "taker_notional": traded - maker_notional})
