"""Alpaca adapter (US equities and crypto via Alpaca's own paper endpoint), behind the same live
guard as Binance. The repo already has a read-only Alpaca market-data source
(`itrader/inputs/market_data/alpaca.py`); this is the order side, and it is off by default.
Alpaca's "paper" endpoint still needs the user's keys, so it sits behind the guard too.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..data.feeds import _CTX, UA
from ..data.universe import leg_symbol
from ..settings import Settings, settings as default_settings
from .broker import Broker, Fill, LiveDisabled, Order, OrderState, live_guard

PAPER = "https://paper-api.alpaca.markets/v2"
LIVE = "https://api.alpaca.markets/v2"


class AlpacaBroker(Broker):
    name = "alpaca"

    def __init__(self, s: Optional[Settings] = None, root: Optional[Path] = None, live: bool = False, paper_endpoint: bool = True) -> None:
        self.s = s or default_settings()
        self.root = root
        self.live = bool(live)
        self.base = PAPER if paper_endpoint else LIVE
        self.guard = live_guard(self.s, root, "alpaca")
        self._orders: Dict[str, Order] = {}

    def _assert_live(self) -> None:
        self.guard = live_guard(self.s, self.root, "alpaca")
        if not (self.live and self.guard["open"]):
            closed = [k for k, v in self.guard["locks"].items() if not v]
            raise LiveDisabled("alpaca adapter is off; closed locks: " + (", ".join(closed) if closed else "adapter constructed with live=False"))

    def _headers(self) -> Dict[str, str]:
        return {**UA, "APCA-API-KEY-ID": os.environ.get(self.s.live.alpaca_key_env, ""),
                "APCA-API-SECRET-KEY": os.environ.get(self.s.live.alpaca_secret_env, ""), "Content-Type": "application/json"}

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        self._assert_live()
        req = urllib.request.Request(f"{self.base}{path}", data=json.dumps(payload).encode() if payload else None, method=method, headers=self._headers())
        with urllib.request.urlopen(req, timeout=10, context=_CTX) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw else {}

    @staticmethod
    def venue_symbol(symbol: str) -> str:
        """Book legs are Binance style ("BTCUSDT", "BTCUSDT.SPOT"); Alpaca crypto wants "BTC/USD" and has no perps.
        Spot legs map; a perp leg is refused rather than silently turned into a spot trade."""
        from ..data.universe import base_asset, is_spot

        if symbol.endswith(".SPOT"):
            return f"{base_asset(symbol)}/USD"
        if symbol.endswith("USDT") or symbol.endswith("USD"):
            raise ValueError(f"{symbol} is a perpetual leg; Alpaca has no perpetual futures, refusing to route it as spot")
        return symbol        # an equity ticker passes through unchanged

    def dry_run(self, order: Order) -> Dict[str, Any]:
        return {"method": "POST", "path": "/orders", "payload": {"symbol": self.venue_symbol(order.symbol), "qty": f"{order.qty:.6f}", "side": order.side,
                                                                  "type": "market", "time_in_force": "day", "client_order_id": f"itr-{order.id}"},
                "endpoint": self.base, "guard": self.guard}

    def book(self, symbol: str, mark: Optional[float] = None) -> Dict[str, float]:
        m = float(mark or 0.0)
        return {"bid": m, "ask": m, "mid": m, "spread_bps": 0.0, "source": "mark"}

    def submit(self, order: Order, mark: float, now: datetime) -> Order:
        res = self._request("POST", "/orders", self.dry_run(order)["payload"])
        order.state = OrderState.ACKED if res.get("id") else OrderState.REJECTED
        order.acked_at = now.isoformat(timespec="milliseconds")
        order.meta["venue_order_id"] = res.get("id")
        self._orders[order.id] = order
        return order

    def poll(self, now: datetime, marks: Dict[str, float]) -> List[Order]:
        changed = []
        for o in list(self._orders.values()):
            if not o.open:
                continue
            res = self._request("GET", f"/orders:by_client_order_id?client_order_id=itr-{o.id}")
            st = res.get("status", "")
            done = float(res.get("filled_qty", 0) or 0)
            if done > o.filled_qty + 1e-12:
                avg = float(res.get("filled_avg_price", 0) or 0)
                o.fills.append(Fill(now.isoformat(timespec="milliseconds"), done - o.filled_qty, avg, 0.0, float(marks.get(leg_symbol(o.symbol)) or avg)))
            if st == "filled":
                o.state, o.closed_at = OrderState.FILLED, now.isoformat(timespec="milliseconds")
            elif st == "partially_filled":
                o.state = OrderState.PARTIAL
            elif st in ("canceled", "expired", "rejected"):
                o.state, o.closed_at = OrderState(st.replace("canceled", "cancelled")), now.isoformat(timespec="milliseconds")
            o.meta["venue_status"] = res
            changed.append(o)
        return changed

    def cancel(self, order_id: str, now: datetime) -> bool:
        o = self._orders.get(order_id)
        if not o or not o.open or not o.meta.get("venue_order_id"):
            return False
        try:
            self._request("DELETE", f"/orders/{o.meta['venue_order_id']}")
        except LiveDisabled:
            raise
        except Exception as exc:  # noqa: BLE001
            o.error = f"cancel failed: {type(exc).__name__}: {exc}"
            return False
        o.state, o.closed_at = OrderState.CANCELLED, now.isoformat(timespec="milliseconds")
        return True

    def cancel_all(self, now: datetime) -> int:
        n = 0
        for o in list(self._orders.values()):
            try:
                n += 1 if self.cancel(o.id, now) else 0
            except LiveDisabled:
                raise
            except Exception:  # noqa: BLE001
                continue
        return n

    def positions(self) -> Dict[str, float]:
        return {p["symbol"]: float(p["qty"]) for p in self._request("GET", "/positions")}

    def equity(self, marks: Dict[str, float]) -> float:
        return float(self._request("GET", "/account").get("equity", 0.0))

    def open_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.open]
