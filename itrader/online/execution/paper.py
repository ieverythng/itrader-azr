"""PaperBroker: fills against the live top of book, with fees, slippage, partial fills and latency.

Realistic in the ways that matter for a paper record to mean something:
* price = mid +/- (half spread + slippage), where the spread is the venue's live book ticker
  spread when reachable and sane, and the settings value otherwise;
* taker fee per side (spot and perp rates differ); realised pnl is net of the closing fee;
* a fill arrives `latency_ms` after the order is acknowledged, and with probability
  `partial_fill_prob` only a fraction arrives first;
* perp positions pay or receive funding when `apply_funding` is called at settlement, once per
  settlement key;
* spot legs cannot go short: checked at submit and again at every fill;
* non-finite marks, quantities or rates are refused rather than booked.
Fills are deterministic given the order ids (the RNG is seeded per order and fill number); the
Executor derives ids from symbol, time and reason, so a replay reproduces the same fills.
"""
from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..bench.engine import CostModel
from ..data.feeds import Feed
from ..data.universe import is_crypto, is_spot, leg_symbol
from ..settings import Settings, settings as default_settings
from .broker import OPEN_STATES, Broker, Fill, Order, OrderState


def mark_for(marks: Dict[str, float], symbol: str) -> Optional[float]:
    """The usable mark for a book leg: its own key, else the data symbol's. None when absent or not finite."""
    for key in (symbol, leg_symbol(symbol)):
        v = marks.get(key)
        if v is not None and math.isfinite(float(v)) and float(v) > 0:
            return float(v)
    return None


class PaperBroker(Broker):
    name = "paper"
    live = False
    supports_limit = True               # resting post-only orders, filled against the traded range since they were placed

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
        self.last_marks: Dict[str, float] = {}
        self.trip_pnl: Dict[str, float] = {}          # realised pnl of each open position since it was opened
        self.fees_paid = 0.0
        self.funding_paid = 0.0
        self.funding_keys: List[str] = []
        self.orders: Dict[str, Order] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self._book_cache: Dict[str, Dict[str, float]] = {}
        if self.state_path and self.state_path.exists():
            self.load()

    # ------------------------------------------------------------------ persistence
    def snapshot(self) -> Dict[str, Any]:
        return {"cash": self.cash, "pos": self.pos, "avg_cost": self.avg_cost, "realised": self.realised, "last_marks": self.last_marks, "trip_pnl": self.trip_pnl,
                "fees_paid": self.fees_paid, "funding_paid": self.funding_paid, "funding_keys": self.funding_keys[-50:],
                "orders": [o.to_dict() for o in self.orders.values() if o.open], "closed_trades": self.closed_trades[-500:]}

    def save(self) -> None:
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.snapshot(), indent=1))
            os.replace(tmp, self.state_path)

    def load(self) -> None:
        d = json.loads(self.state_path.read_text())
        self.cash = float(d.get("cash", self.cash))
        self.pos = {k: float(v) for k, v in d.get("pos", {}).items()}
        self.avg_cost = {k: float(v) for k, v in d.get("avg_cost", {}).items()}
        self.realised = {k: float(v) for k, v in d.get("realised", {}).items()}
        self.last_marks = {k: float(v) for k, v in d.get("last_marks", {}).items()}
        self.trip_pnl = {k: float(v) for k, v in d.get("trip_pnl", {}).items()}
        self.fees_paid = float(d.get("fees_paid", 0.0))
        self.funding_paid = float(d.get("funding_paid", 0.0))
        self.funding_keys = list(d.get("funding_keys", []))
        self.closed_trades = list(d.get("closed_trades", []))
        # open orders are not restored: a restart cancels them, which is the safe reading

    # ------------------------------------------------------------------ market
    def book(self, symbol: str, mark: Optional[float] = None) -> Dict[str, float]:
        data_sym = leg_symbol(symbol)
        if self.feed is not None and bool(self.x.use_live_book) and is_crypto(data_sym):   # listed markets have no book here
            try:
                b = self.feed.book_ticker(data_sym)
                bid, ask = float(b["bid"]), float(b["ask"])
                if math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask:
                    out = {"bid": bid, "ask": ask, "mid": (bid + ask) / 2, "spread_bps": 1e4 * (ask - bid) / ((ask + bid) / 2), "source": "live"}
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
    def _reduces(self, order: Order) -> bool:
        p = self.pos.get(order.symbol, 0.0)
        return (p > 0 and order.side == "sell") or (p < 0 and order.side == "buy")

    def submit(self, order: Order, mark: float, now: datetime) -> Order:
        ok_mark = mark is not None and math.isfinite(float(mark)) and float(mark) > 0
        if not (order.qty > 0 and math.isfinite(order.qty) and ok_mark and math.isfinite(order.notional)):
            order.state, order.error = OrderState.REJECTED, "non-finite or non-positive qty, notional or mark"
        elif order.type == "limit" and not (order.limit_price is not None and math.isfinite(float(order.limit_price)) and float(order.limit_price) > 0):
            order.state, order.error = OrderState.REJECTED, "limit order without a usable price"
        elif is_spot(order.symbol) and order.side == "sell" and order.qty > self.pos.get(order.symbol, 0.0) + 1e-12:
            order.state, order.error = OrderState.REJECTED, "spot leg cannot go short"
        elif order.notional < float(self.x.min_order_notional) and not self._reduces(order):
            order.state, order.error = OrderState.REJECTED, f"below min notional {self.x.min_order_notional}"
        else:
            order.state, order.acked_at = OrderState.ACKED, now.isoformat(timespec="milliseconds")
            order.meta["mark_at_submit"] = float(mark)
        self.orders[order.id] = order
        if order.state == OrderState.REJECTED:
            order.closed_at = now.isoformat(timespec="milliseconds")
        return order

    def _fill(self, o: Order, qty: float, mark: float, now: datetime, limit: bool = False) -> None:
        if is_spot(o.symbol) and o.side == "sell":
            qty = min(qty, max(0.0, self.pos.get(o.symbol, 0.0)))
            if qty <= 1e-12:
                o.state, o.closed_at, o.error = OrderState.REJECTED, now.isoformat(timespec="milliseconds"), "spot leg cannot go short (at fill)"
                return
        if limit:                                   # a resting order fills at its own price and pays the maker fee: no spread, no slippage
            px = float(o.limit_price)
            mid = float(mark) if mark else px
            fee_bps = self.costs.spot_fee_bps if is_spot(o.symbol) else self.costs.maker_fee_bps
            o.meta["filled_as"] = "maker"
        else:
            b = self.book(o.symbol, mark)
            half = max(0.0, b["spread_bps"] / 2.0) / 1e4
            slip = self.costs.slippage_bps / 1e4
            mid = b["mid"]
            px = mid * (1 + half + slip) if o.side == "buy" else mid * (1 - half - slip)
            fee_bps = self.costs.spot_fee_bps if is_spot(o.symbol) else self.costs.fee_bps
            o.meta["filled_as"] = "taker"
        fee = qty * px * fee_bps / 1e4
        signed = qty if o.side == "buy" else -qty
        prev = self.pos.get(o.symbol, 0.0)
        new = prev + signed
        if prev != 0 and (signed * prev) < 0:
            closed = min(abs(signed), abs(prev))
            pnl = closed * (px - self.avg_cost.get(o.symbol, px)) * (1 if prev > 0 else -1) - fee * (closed / qty)
            self.realised[o.symbol] = self.realised.get(o.symbol, 0.0) + pnl
            # a "round trip" ends when the position is back to zero or flips; until then reductions only add to its running pnl
            self.trip_pnl[o.symbol] = self.trip_pnl.get(o.symbol, 0.0) + pnl
            full = abs(new) < 1e-12 or (new * prev) < 0
            self.closed_trades.append({"ts": now.isoformat(timespec="seconds"), "symbol": o.symbol, "qty": closed, "pnl": pnl, "reason": o.reason, "order_id": o.id,
                                       "full_close": bool(full), "trip_pnl": float(self.trip_pnl[o.symbol]) if full else None})
            if full:
                self.trip_pnl.pop(o.symbol, None)
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
        self.last_marks[o.symbol] = mid
        o.fills.append(Fill(now.isoformat(timespec="milliseconds"), qty, px, fee, mid))
        if o.remaining <= 1e-12:
            o.state, o.closed_at = OrderState.FILLED, now.isoformat(timespec="milliseconds")
        else:
            o.state = OrderState.PARTIAL

    def poll(self, now: datetime, marks: Dict[str, float], ranges: Optional[Dict[str, Any]] = None) -> List[Order]:
        """Market orders fill after the latency. Limit orders fill only when `ranges` shows the market traded
        THROUGH their price: ranges = {data symbol: (low, high, bar_open_time)} for a bar that began no earlier
        than the order was placed. Without a qualifying range a limit order simply keeps resting."""
        changed: List[Order] = []
        lat = timedelta(milliseconds=int(self.x.latency_ms))
        ttl = timedelta(seconds=int(self.x.order_ttl_s))
        thr = self.costs.maker_through_bps / 1e4
        for o in list(self.orders.values()):
            if not o.open:
                continue
            acked = datetime.fromisoformat(o.acked_at) if o.acked_at else now
            if now < acked + lat:
                continue
            if o.type == "limit":
                rg = (ranges or {}).get(leg_symbol(o.symbol))
                if rg is not None:
                    lo, hi, bar_open = rg
                    fresh = bar_open is None or bar_open >= acked - timedelta(seconds=120)
                    if fresh and lo is not None and hi is not None and math.isfinite(float(lo)) and math.isfinite(float(hi)):
                        hit = (o.side == "buy" and float(lo) <= o.limit_price * (1 - thr)) or (o.side == "sell" and float(hi) >= o.limit_price * (1 + thr))
                        if hit:
                            self._fill(o, o.remaining, mark_for(marks, o.symbol) or float(o.limit_price), now, limit=True)
                            changed.append(o)
                continue
            mark = mark_for(marks, o.symbol)
            if mark is None:
                if now > acked + ttl:
                    o.state, o.closed_at, o.error = OrderState.EXPIRED, now.isoformat(timespec="milliseconds"), "no usable mark"
                    changed.append(o)
                continue
            rng = random.Random(f"{o.id}:{len(o.fills)}")
            partial = o.state == OrderState.ACKED and rng.random() < float(self.x.partial_fill_prob)
            qty = o.remaining * (rng.uniform(float(self.x.partial_fill_min), 0.95) if partial else 1.0)
            self._fill(o, qty, mark, now)
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

    def _mark(self, marks: Dict[str, float], sym: str) -> Optional[float]:
        m = mark_for(marks, sym)
        if m is not None:
            self.last_marks[sym] = m
            return m
        return self.last_marks.get(sym)       # last known, never the average cost (that would hide a loss)

    def equity(self, marks: Dict[str, float]) -> float:
        val = self.cash
        for sym, q in self.pos.items():
            m = self._mark(marks, sym)
            if m is None:
                return float("nan")           # an unpriced position: the risk engine locks on a NaN equity
            val += q * m
        return float(val)

    def weights(self, marks: Dict[str, float]) -> Dict[str, float]:
        eq = self.equity(marks)
        if not math.isfinite(eq) or eq <= 0:
            return {}
        return {sym: q * (self._mark(marks, sym) or 0.0) / eq for sym, q in self.pos.items()}

    def apply_funding(self, rates: Dict[str, float], marks: Dict[str, float], now: datetime, key: Optional[str] = None) -> float:
        """Perp legs: long pays when the rate is positive. Idempotent per `key` (the settlement time).
        Returns the amount paid (negative = received)."""
        if key is not None and key in self.funding_keys:
            return 0.0
        paid = 0.0
        for sym, q in self.pos.items():
            if is_spot(sym):
                continue
            rate = rates.get(sym)
            m = mark_for(marks, sym)
            if rate is None or m is None or not math.isfinite(float(rate)):
                continue
            paid += q * m * float(rate)
        self.cash -= paid
        self.funding_paid += paid
        if key is not None:
            self.funding_keys.append(key)
        self.save()
        return paid
