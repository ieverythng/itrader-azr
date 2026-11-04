"""Executor: approved book -> orders -> fills -> report. Deterministic normalisation only.

Given the RiskEngine's approved weights, the current positions and marks, it computes the
notional to trade per leg, drops anything under the minimum order size, submits market orders,
polls until they are done or expire, and reports fills, fees and slippage against the mid at
submission. It never changes a weight the risk engine approved, except to round to the
venue's lot size when one is known.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from ..data.universe import leg_symbol
from ..settings import Settings, settings as default_settings
from .broker import Broker, Order


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
        if eq <= 0 or not pos:
            return pd.Series(dtype=float)
        return pd.Series({sym: q * (marks.get(leg_symbol(sym)) or 0.0) / eq for sym, q in pos.items()}, dtype=float)

    def _round(self, symbol: str, qty: float) -> float:
        step = self.lot_steps.get(leg_symbol(symbol))
        if step and step > 0:
            return (qty // step) * step
        return round(qty, 6)

    def cancel_all(self, now: datetime) -> int:
        return self.broker.cancel_all(now)

    def rebalance(self, target: pd.Series, marks: Dict[str, float], now: datetime, reason: str = "rebalance",
                  wait: Optional[timedelta] = None) -> ExecutionReport:
        eq0 = self.broker.equity(marks)
        cur = self.current_weights(marks)
        tgt = target.astype(float)
        idx = tgt.index.union(cur.index)
        delta = tgt.reindex(idx).fillna(0.0) - cur.reindex(idx).fillna(0.0)
        min_notional = float(self.s.execution.min_order_notional)
        orders: List[Order] = []
        skipped: List[Dict[str, Any]] = []
        for sym, dw in delta.items():
            notional = abs(dw) * eq0
            mark = marks.get(leg_symbol(sym))
            if notional < 1e-9:
                continue
            if not mark:
                skipped.append({"symbol": sym, "why": "no mark", "notional": notional})
                continue
            if notional < min_notional:
                skipped.append({"symbol": sym, "why": f"below min notional {min_notional}", "notional": notional})
                continue
            qty = self._round(sym, notional / mark)
            if qty <= 0:
                skipped.append({"symbol": sym, "why": "rounds to zero", "notional": notional})
                continue
            o = Order(symbol=sym, side="buy" if dw > 0 else "sell", qty=qty, notional=qty * mark, reason=reason,
                      meta={"target_w": float(tgt.get(sym, 0.0)), "current_w": float(cur.get(sym, 0.0))})
            orders.append(self.broker.submit(o, mark, now))
        # poll until done, expired, or the wait is over (paper fills are instantaneous after latency)
        deadline = now + (wait or timedelta(seconds=int(self.s.execution.order_ttl_s)))
        t = now
        step = timedelta(milliseconds=max(50, int(self.s.execution.latency_ms)))
        while any(o.open for o in orders) and t <= deadline:
            t = t + step
            self.broker.poll(t, marks)
        for o in orders:
            if o.open:
                self.broker.cancel(o.id, t)
                o.error = o.error or "ttl expired, cancelled"
        eq1 = self.broker.equity(marks)
        fills = [f for o in orders for f in o.fills]
        traded = sum(f.qty * f.price for f in fills)
        slip = sum(abs(f.slippage_bps) * f.qty * f.price for f in fills) / traded if traded > 0 else 0.0
        return ExecutionReport(orders, eq0, eq1, traded, sum(f.fee for f in fills), slip, skipped)
