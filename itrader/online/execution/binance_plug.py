"""Binance USDT-M futures adapter, behind the live guard. Off by default and stays off.

Four locks must all be open before `submit` will send anything: `[live] enabled = true` in
settings, `ITRADER_LIVE=1` in the environment, both key env vars set, and the confirmation
file present. With any lock closed every order call raises `LiveDisabled`; `dry_run()` shows
the exact signed request that would have gone out, so the plumbing can be checked without
a key. Keys never touch the repo or the logs.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..data.feeds import _CTX, UA
from ..data.universe import leg_symbol
from ..settings import Settings, settings as default_settings
from .broker import Broker, LiveDisabled, Order, OrderState, live_guard

FAPI = "https://fapi.binance.com"


class BinanceBroker(Broker):
    name = "binance_futures"

    def __init__(self, s: Optional[Settings] = None, root: Optional[Path] = None, live: bool = False) -> None:
        self.s = s or default_settings()
        self.root = root
        self.live = bool(live)
        self.guard = live_guard(self.s, root, "binance")
        self._key = os.environ.get(self.s.live.binance_key_env, "")
        self._secret = os.environ.get(self.s.live.binance_secret_env, "")
        self._orders: Dict[str, Order] = {}

    # ------------------------------------------------------------------ guard
    def _assert_live(self) -> None:
        self.guard = live_guard(self.s, self.root, "binance")
        if not (self.live and self.guard["open"]):
            closed = [k for k, v in self.guard["locks"].items() if not v]
            raise LiveDisabled("live trading is off; closed locks: " + (", ".join(closed) if closed else "adapter constructed with live=False"))

    def _sign(self, params: Dict[str, Any]) -> str:
        q = urllib.parse.urlencode({**params, "timestamp": int(time.time() * 1000), "recvWindow": 5000})
        sig = hmac.new(self._secret.encode(), q.encode(), hashlib.sha256).hexdigest()
        return f"{q}&signature={sig}"

    def _request(self, method: str, path: str, params: Dict[str, Any]) -> Any:
        self._assert_live()
        body = self._sign(params)
        url = f"{FAPI}{path}" + (f"?{body}" if method == "GET" else "")
        req = urllib.request.Request(url, data=body.encode() if method != "GET" else None, method=method,
                                     headers={**UA, "X-MBX-APIKEY": self._key, "Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=10, context=_CTX) as r:
            return json.loads(r.read().decode())

    def dry_run(self, order: Order) -> Dict[str, Any]:
        """What would be sent, with the signature redacted. Never sends."""
        params = {"symbol": leg_symbol(order.symbol), "side": order.side.upper(), "type": "MARKET", "quantity": f"{order.qty:.6f}",
                  "newClientOrderId": f"itr-{order.id}"}
        return {"method": "POST", "path": "/fapi/v1/order", "params": params, "signed": bool(self._secret), "guard": self.guard}

    # ------------------------------------------------------------------ broker api
    def book(self, symbol: str, mark: Optional[float] = None) -> Dict[str, float]:
        d = json.loads(urllib.request.urlopen(urllib.request.Request(f"{FAPI}/fapi/v1/ticker/bookTicker?symbol={leg_symbol(symbol)}", headers=UA),
                                              timeout=10, context=_CTX).read().decode())
        bid, ask = float(d["bidPrice"]), float(d["askPrice"])
        return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2, "spread_bps": 1e4 * (ask - bid) / ((ask + bid) / 2), "source": "live"}

    def submit(self, order: Order, mark: float, now: datetime) -> Order:
        self._assert_live()
        res = self._request("POST", "/fapi/v1/order", self.dry_run(order)["params"])
        order.state = OrderState.ACKED if res.get("orderId") else OrderState.REJECTED
        order.acked_at = now.isoformat(timespec="milliseconds")
        order.meta["venue_order_id"] = res.get("orderId")
        self._orders[order.id] = order
        return order

    def poll(self, now: datetime, marks: Dict[str, float]) -> List[Order]:
        self._assert_live()
        changed = []
        for o in list(self._orders.values()):
            if not o.open:
                continue
            res = self._request("GET", "/fapi/v1/order", {"symbol": leg_symbol(o.symbol), "origClientOrderId": f"itr-{o.id}"})
            st = res.get("status")
            if st == "FILLED":
                o.state, o.closed_at = OrderState.FILLED, now.isoformat(timespec="milliseconds")
            elif st == "PARTIALLY_FILLED":
                o.state = OrderState.PARTIAL
            elif st in ("CANCELED", "EXPIRED", "REJECTED"):
                o.state, o.closed_at = OrderState(st.lower().replace("canceled", "cancelled")), now.isoformat(timespec="milliseconds")
            o.meta["venue_status"] = res
            changed.append(o)
        return changed

    def cancel(self, order_id: str, now: datetime) -> bool:
        o = self._orders.get(order_id)
        if not o or not o.open:
            return False
        self._request("DELETE", "/fapi/v1/order", {"symbol": leg_symbol(o.symbol), "origClientOrderId": f"itr-{o.id}"})
        o.state, o.closed_at = OrderState.CANCELLED, now.isoformat(timespec="milliseconds")
        return True

    def cancel_all(self, now: datetime) -> int:
        return sum(1 for o in list(self._orders.values()) if self.cancel(o.id, now))

    def positions(self) -> Dict[str, float]:
        res = self._request("GET", "/fapi/v2/positionRisk", {})
        return {p["symbol"]: float(p["positionAmt"]) for p in res if abs(float(p["positionAmt"])) > 0}

    def equity(self, marks: Dict[str, float]) -> float:
        res = self._request("GET", "/fapi/v2/account", {})
        return float(res.get("totalMarginBalance", 0.0))

    def open_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.open]
