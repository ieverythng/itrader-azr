"""Broker interface, order state machine, and the live guard.

Order states: NEW -> ACKED -> (PARTIAL ->)* FILLED | CANCELLED | REJECTED | EXPIRED.
Quantities are in base units (BTC, not USDT); `notional` is the USDT value at submission.
Book legs: "BTCUSDT" is the perpetual (signed), "BTCUSDT.SPOT" the spot holding (long only).
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class OrderState(str, Enum):
    NEW = "new"
    ACKED = "acked"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


OPEN_STATES = {OrderState.NEW, OrderState.ACKED, OrderState.PARTIAL}


@dataclass
class Fill:
    ts: str
    qty: float
    price: float
    fee: float
    mark: float                         # mid at the time, for slippage accounting

    @property
    def slippage_bps(self) -> float:
        return 1e4 * (self.price / self.mark - 1.0) if self.mark else 0.0


@dataclass
class Order:
    symbol: str
    side: str                           # "buy" | "sell"
    qty: float
    notional: float
    reason: str = ""
    type: str = "market"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: OrderState = OrderState.NEW
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
    acked_at: Optional[str] = None
    closed_at: Optional[str] = None
    fills: List[Fill] = field(default_factory=list)
    error: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def filled_qty(self) -> float:
        return float(sum(f.qty for f in self.fills))

    @property
    def remaining(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    @property
    def avg_price(self) -> float:
        q = self.filled_qty
        return float(sum(f.qty * f.price for f in self.fills) / q) if q > 0 else 0.0

    @property
    def fees(self) -> float:
        return float(sum(f.fee for f in self.fills))

    @property
    def open(self) -> bool:
        return self.state in OPEN_STATES

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "symbol": self.symbol, "side": self.side, "qty": self.qty, "notional": self.notional, "type": self.type,
                "state": self.state.value, "created": self.created, "acked_at": self.acked_at, "closed_at": self.closed_at, "reason": self.reason,
                "filled_qty": self.filled_qty, "avg_price": self.avg_price, "fees": self.fees,
                "fills": [{"ts": f.ts, "qty": f.qty, "price": f.price, "fee": f.fee, "mark": f.mark, "slippage_bps": f.slippage_bps} for f in self.fills],
                "error": self.error, "meta": self.meta}


class LiveDisabled(RuntimeError):
    pass


def live_guard(s, root: Optional[Path] = None, venue: str = "binance") -> Dict[str, Any]:
    """The four locks. All must be open for a live adapter to send an order. Returns the state of each."""
    lv = s.live
    key_env, sec_env = (lv.binance_key_env, lv.binance_secret_env) if venue == "binance" else (lv.alpaca_key_env, lv.alpaca_secret_env)
    conf = s.resolve(lv.confirmation_file, root)
    locks = {
        "settings.live.enabled": bool(lv.enabled),
        f"env {lv.env_flag}=1": os.environ.get(lv.env_flag, "") == "1",
        f"env {key_env} and {sec_env} set": bool(os.environ.get(key_env)) and bool(os.environ.get(sec_env)),
        f"confirmation file {lv.confirmation_file}": conf.exists(),
    }
    return {"open": all(locks.values()), "locks": locks}


class Broker:
    name: str = "broker"
    live: bool = False

    def book(self, symbol: str, mark: Optional[float] = None) -> Dict[str, float]:
        raise NotImplementedError

    def submit(self, order: Order, mark: float, now: datetime) -> Order:
        raise NotImplementedError

    def poll(self, now: datetime, marks: Dict[str, float]) -> List[Order]:
        raise NotImplementedError

    def cancel(self, order_id: str, now: datetime) -> bool:
        raise NotImplementedError

    def cancel_all(self, now: datetime) -> int:
        raise NotImplementedError

    def positions(self) -> Dict[str, float]:
        raise NotImplementedError

    def equity(self, marks: Dict[str, float]) -> float:
        raise NotImplementedError

    def open_orders(self) -> List[Order]:
        raise NotImplementedError
