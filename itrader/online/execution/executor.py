"""Executor: approved book -> orders -> fills -> report. Deterministic normalisation only.

Given the RiskEngine's approved weights, the current positions and marks, it computes the
notional to trade per leg, submits market orders, polls until they are done or expire, and
reports fills, fees and slippage against the mid at submission. It never changes a weight the
risk engine approved, except to round to a lot size. Rules that keep "flatten" meaning flatten:
a reducing order is clamped to the position and floor-rounded (never rounds past zero), and a
reduction is sent even below the minimum notional (dust must be closable). Order ids are derived
from symbol, time and reason, so a replay reproduces the same ids and therefore the same fills.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from ..data.universe import leg_symbol
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

    def to_dict(self) -> Dict[str, Any]:
        return {"orders": [o.to_dict() for o in self.orders], "equity_before": self.equity_before, "equity_after": self.equity_after,
                "traded_notional": self.traded_notional, "fees": self.fees, "slippage_bps_weighted": self.slippage_bps_weighted,
                "skipped": self.skipped, "cancelled": self.cancelled, "notes": self.notes,
                "n_filled": sum(1 for o in self.orders if o.state.value == "filled"),
                "n_rejected": sum(1 for o in self.orders if o.state.value == "rejected")}


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
                  wait: Optional[timedelta] = None) -> ExecutionReport:
        eq0 = self.broker.equity(marks)
        skipped: List[Dict[str, Any]] = []
        orders: List[Order] = []
        if not math.isfinite(eq0) or eq0 <= 0:
            return ExecutionReport([], eq0, eq0, 0.0, 0.0, 0.0, [{"symbol": "*", "why": f"equity not usable ({eq0}); no orders"}])
        pos = self.broker.positions()
        cur = self.current_weights(marks)
        tgt = target.astype(float)
        idx = tgt.index.union(cur.index).union(pd.Index(pos))
        min_notional = float(self.s.execution.min_order_notional)
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
            notional = abs(dw) * eq0
            if notional < 1e-9:
                continue
            held = pos.get(sym, 0.0)
            reducing = held != 0 and (dw * held) < 0
            if notional < min_notional and not reducing:
                skipped.append({"symbol": sym, "why": f"below min notional {min_notional}", "notional": notional})
                continue
            qty = notional / mark
            if reducing:
                qty = min(qty, abs(held))
                qty = abs(held) if abs(t) < 1e-9 or abs(qty - abs(held)) <= self._floor(sym, abs(held)) * 1e-6 else self._floor(sym, qty)
            else:
                qty = self._floor(sym, qty)
            if qty <= 0:
                skipped.append({"symbol": sym, "why": "rounds to zero", "notional": notional})
                continue
            oid = hashlib.sha1(f"{sym}|{now.isoformat()}|{reason}".encode()).hexdigest()[:12]
            o = Order(symbol=sym, side="buy" if dw > 0 else "sell", qty=qty, notional=qty * mark, reason=reason, id=oid,
                      meta={"target_w": t, "current_w": c, "reducing": reducing})
            orders.append(self.broker.submit(o, mark, now))
        deadline = now + (wait or timedelta(seconds=int(self.s.execution.order_ttl_s)))
        t = now
        step = timedelta(milliseconds=max(50, int(self.s.execution.latency_ms)))
        while any(o.open for o in orders) and t <= deadline:
            t = t + step
            if getattr(self.broker, "live", False):
                time.sleep(step.total_seconds())
            self.broker.poll(t, marks)
        cancelled = 0
        for o in orders:
            if o.open:
                if self.broker.cancel(o.id, t):
                    cancelled += 1
                    o.error = o.error or "ttl expired, cancelled"
        eq1 = self.broker.equity(marks)
        fills = [f for o in orders for f in o.fills]
        traded = sum(f.qty * f.price for f in fills)
        slip = sum(abs(f.slippage_bps) * f.qty * f.price for f in fills) / traded if traded > 0 else 0.0
        return ExecutionReport(orders, eq0, eq1, traded, sum(f.fee for f in fills), slip, skipped, cancelled)
