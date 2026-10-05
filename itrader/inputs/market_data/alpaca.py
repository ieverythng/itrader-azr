from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from datetime import date
from typing import Any

import requests

from .contracts import HistoricalBarsRequest, MarketDataPage


class AlpacaMarketDataSource:
    """Read-only Alpaca historical-bars and exchange-calendar adapter."""

    provider = "alpaca"
    _DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
    _CALENDAR_URL = "https://paper-api.alpaca.markets/v2/calendar"

    def __init__(
        self,
        api_key_id: str,
        api_secret_key: str,
        *,
        session: requests.Session | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 6,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key_id.strip() or not api_secret_key.strip():
            raise ValueError("Alpaca credentials must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("Alpaca timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("Alpaca max_retries must be nonnegative")
        self._headers = {
            "APCA-API-KEY-ID": api_key_id.strip(),
            "APCA-API-SECRET-KEY": api_secret_key.strip(),
        }
        self._session = session or requests.Session()
        self._timeout = float(timeout_seconds)
        self._max_retries = int(max_retries)
        self._sleep = sleep

    def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        for attempt in range(self._max_retries + 1):
            response = self._session.get(
                url,
                params=params,
                headers=self._headers,
                timeout=self._timeout,
            )
            if response.status_code < 400:
                return response.json()
            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt >= self._max_retries:
                raise RuntimeError(
                    f"Alpaca request failed with HTTP {response.status_code}: {response.text[:500]}"
                )
            fallback_delay = min(2**attempt, 30)
            try:
                delay = float(response.headers.get("Retry-After", fallback_delay))
            except (TypeError, ValueError):
                delay = float(fallback_delay)
            self._sleep(max(0.0, delay))
        raise RuntimeError("Alpaca retry loop exhausted")

    def fetch_calendar(self, start: date, end: date) -> MarketDataPage:
        params = {"start": start.isoformat(), "end": end.isoformat()}
        return MarketDataPage(
            request={"endpoint": "calendar", **params},
            payload=self._get_json(self._CALENDAR_URL, params),
        )

    def iter_bar_pages(self, request: HistoricalBarsRequest) -> Iterator[MarketDataPage]:
        params: dict[str, Any] = {
            "symbols": request.symbol,
            "timeframe": request.timeframe,
            "start": request.start.isoformat().replace("+00:00", "Z"),
            "end": request.end.isoformat().replace("+00:00", "Z"),
            "adjustment": request.adjustment,
            "feed": request.feed,
            "sort": "asc",
            "limit": 10_000,
        }
        while True:
            payload = self._get_json(self._DATA_URL, params)
            request_record = {"endpoint": "bars", **params}
            yield MarketDataPage(request=request_record, payload=payload)
            token = payload.get("next_page_token") if isinstance(payload, dict) else None
            if not token:
                return
            params = {**params, "page_token": token}
