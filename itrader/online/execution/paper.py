"""PaperBroker: fills against the live top of book, with fees, slippage, partial fills and latency.

Realistic in the ways that matter for a paper record to mean something:
* price = mid +/- (half spread + slippage), where the spread is the venue's live book ticker
  spread when reachable and the settings value otherwise;
* taker fee per side (spot and perp rates differ);
* a fill arrives `latency_ms` after the order is acknowledged, and with probability
  `partial_fill_prob` only a fraction arrives first;
* perp positions pay or receive funding when `apply_funding` is called at settlement;
* spot legs cannot go short; an order that would do so is rejected.
Everything is deterministic given the order ids (the RNG is seeded per order), so a replay
reproduces the same fills.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..bench.engine import CostModel
from ..data.feeds import Feed
from ..data.universe import is_spot, leg_symbol
from ..settings import Settings, settings as default_settings
from .broker import OPEN_STATES, Broker, Fill, Order, OrderState


class PaperBroker(Broker):
    name = "paper"
    live = False

    def __init__(self, s: Optional[Settings] = None, feed: Optional[Feed] = None, state_path: Optional[Path] = None,
                 equity_start: Optional[float] = None) -> None:
        self.s = s or default_settings()
        self.x = self.s.execution
        self.costs = CostModel.from_settings(self.s)
        self.feed = feed
        self.state_path = Path(state_path) if state_path else None
        self.cash: float = float(equity_start if equity_start is not None else self.s.general.equity_start)
        self.pos: Dict[str, float] = {}
        self.avg_cost: Dict[str, float] = {}
        self.realised: Dict[str, float] = {}
        self.fees_paid = 0.0
        self.funding_paid = 0.0
        self.orders: Dict[str, Order] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self._book_cache: Dict[str, Dict[str, float]] = {}
        if self.state_path and self.state_path.exists():
            self.load()

    # ------------------------------------------------------------------ persistence
    def snapshot(self) -> Dict[str, Any]:
        return {"cash": self.cash, "pos": self.pos, "avg_cost": self.avg_cost, "realised": self.realised, "fees_paid": self.fees_paid,
                "funding_paid": self.funding_paid, "orders": [o.to_dict() for o in self.orders.values() if o.open],
                "closed_trades": self.closed_trades[-500:]}

    def save(self) -> None:
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.snapshot(), indent=1))

    def load(self) -> None:
        d = json.loads(self.state_path.read_text())
        self.cash = float(d.get("cash", self.cash))
        self.pos = {k: float(v) for k, v in d.get("pos", {}).items()}
        self.avg_cost = {k: float(v) for k, v in d.get("avg_cost", {}).items()}
        self.realised = {k: float(v) for k, v in d.get("realised", {}).items()}
        self.fees_paid = float(d.get("fees_paid", 0.0))
        self.funding_paid = float(d.get("funding_paid", 0.0))
        self.closed_trades = list(d.get("closed_trades", []))

    # ------------------------------------------------------------------ market
    def book(self, symbol: str, mark: Optional[float] = None) -> Dict[str, float]:
        data_sym = leg_symbol(symbol)
        if self.feed is not None and bool(self.x.use_live_book):
            try:
                b = self.feed.book_ticker(data_sym)
                out = {"bid": b["bid"], "ask": b["ask"], "mid": b["mid"], "spread_bps": b["spread_bps"], "source": "live"}
                self._book_cache[data_sym] = out
                return out
            except Exception:  # noqa: BLE001
                pass
        if data_sym in self._book_cache and mark is None:
            return self._book_cache[data_sym]
        m = float(mark or 0.0)
        hs = self.costs.half_spread_bps / 1e4
        return {"bid": m * (1 - hs), "ask": m * (1 + hs), "mid": m, "spread_bps": 2 * self.costs.half_spread_bps, "source": "settings"}

    # ------------------------------------------------------------------ orders
    def submit(self, order: Order, mark: float, now: datetime) -> Order:
        if order.qty <= 0 or mark <= 0:
            order.state, order.error = OrderState.REJECTED, "non-positive qty or mark"
        elif is_spot(order.symbol) and order.side == "sell" and order.qty > self.pos.get(order.symbol, 0.0) + 1e-12:
            order.state, order.error = OrderState.REJECTED, "spot leg cannot go short"
        elif order.notional < float(self.x.min_order_notional):
            order.state, order.error = OrderState.REJECTED, f"below min notional {self.x.min_order_notional}"
        else:
            order.state, order.acked_at = OrderState.ACKED, now.isoformat(timespec="milliseconds")
            order.meta["mark_at_submit"] = mark
        self.orders[order.id] = order
        if order.state == OrderState.REJECTED:
            order.closed_at = now.isoformat(timespec="milliseconds")
        return order

    def _fill(self, o: Order, qty: float, mark: float, now: datetime) -> None:
        b = self.book(o.symbol, mark)
        half = max(0.0, b["spread_bps"] / 2.0) / 1e4
        slip = self.costs.slippage_bps / 1e4
        px = b["mid"] * (1 + half + slip) if o.side == "buy" else b["mid"] * (1 - half - slip)
        fee_bps = self.costs.spot_fee_bps if is_spot(o.symbol) else self.costs.fee_bps
        fee = qty * px * fee_bps / 1e4
        signed = qty if o.side == "buy" else -qty
        prev = self.pos.get(o.symbol, 0.0)
        new = prev + signed
        # realised pnl when reducing or flipping
        if prev != 0 and (signed * prev) < 0:
            closed = min(abs(signed), abs(prev))
            pnl = closed * (px - self.avg_cost.get(o.symbol, px)) * (1 if prev > 0 else -1)
            self.realised[o.symbol] = self.realised.get(o.symbol, 0.0) + pnl
            self.closed_trades.append({"ts": now.isoformat(timespec="seconds"), "symbol": o.symbol, "qty": closed, "pnl": pnl, "reason": o.reason})
            if abs(new) < 1e-12 or (new * prev) < 0:
                self.avg_cost[o.symbol] = px if abs(new) > 1e-12 else 0.0
        else:
            tot = abs(prev) + qty
            self.avg_cost[o.symbol] = (abs(prev) * self.avg_cost.get(o.symbol, px) + qty * px) / tot if tot > 0 else px
        self.pos[o.symbol] = new
        if abs(new) < 1e-12:
            self.pos.pop(o.symbol, None)
        self.cash -= signed * px + fee
        self.fees_paid += fee
        o.fills.append(Fill(now.isoformat(timespec="milliseconds"), qty, px, fee, b["mid"]))
        if o.remaining <= 1e-12:
            o.state, o.closed_at = OrderState.FILLED, now.isoformat(timespec="milliseconds")
        else:
            o.state = OrderState.PARTIAL

    def poll(self, now: datetime, marks: Dict[str, float]) -> List[Order]:
        changed: List[Order] = []
        lat = timedelta(milliseconds=int(self.x.latency_ms))
        ttl = timedelta(seconds=int(self.x.order_ttl_s))
        for o in list(self.orders.values()):
            if not o.open:
                continue
            acked = datetime.fromisoformat(o.acked_at) if o.acked_at else now
            if now < acked + lat:
                continue
            mark = marks.get(leg_symbol(o.symbol)) or marks.get(o.symbol)
            if not mark:
                if now > acked + ttl:
                    o.state, o.closed_at, o.error = OrderState.EXPIRED, now.isoformat(timespec="milliseconds"), "no mark"
                    changed.append(o)
                continue
            rng = random.Random(f"{o.id}:{len(o.fills)}")
            partial = o.state == OrderState.ACKED and rng.random() < float(self.x.partial_fill_prob)
            qty = o.remaining * (rng.uniform(float(self.x.partial_fill_min), 0.95) if partial else 1.0)
            self._fill(o, qty, float(mark), now)
            changed.append(o)
        self.save()
        return changed

    def cancel(self, order_id: str, now: datetime) -> bool:
        o = self.orders.get(order_id)
        if o and o.open:
            o.state, o.closed_at = OrderState.CANCELLED, now.isoformat(timespec="milliseconds")
            return True
        return False

    def cancel_all(self, now: datetime) -> int:
        return sum(1 for o in list(self.orders.values()) if self.cancel(o.id, now))

    def open_orders(self) -> List[Order]:
        return [o for o in self.orders.values() if o.state in OPEN_STATES]

    # ------------------------------------------------------------------ account
    def positions(self) -> Dict[str, float]:
        return dict(self.pos)

    def equity(self, marks: Dict[str, float]) -> float:
        val = self.cash
        for sym, q in self.pos.items():
            m = marks.get(leg_symbol(sym)) or marks.get(sym) or self.avg_cost.get(sym, 0.0)
            val += q * m
        return float(val)

    def weights(self, marks: Dict[str, float]) -> Dict[str, float]:
        eq = self.equity(marks)
        if eq <= 0:
            return {}
        return {sym: q * (marks.get(leg_symbol(sym)) or marks.get(sym) or 0.0) / eq for sym, q in self.pos.items()}

    def apply_funding(self, rates: Dict[str, float], marks: Dict[str, float], now: datetime) -> float:
        """Perp legs: long pays when the rate is positive. Returns the amount paid (negative = received)."""
        paid = 0.0
        for sym, q in self.pos.items():
            if is_spot(sym):
                continue
            rate = rates.get(sym)
            m = marks.get(sym)
            if rate is None or not m:
                continue
            paid += q * m * rate
        self.cash -= paid
        self.funding_paid += paid
        return paid
