"""Public market-data feeds. No keys, no cost.

Primary: Binance's public data mirror (data-api.binance.vision for REST, data.binance.vision
for monthly archives). It serves spot candles back to 2017 and USDT-perpetual funding rates.
Fallback: OKX public REST (candles and funding).

Every function returns plain pandas frames with UTC timestamps, so the rest of the stack
never sees a vendor's field names.
"""
from __future__ import annotations

import csv
import io
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

import pandas as pd

UA = {"User-Agent": "itrader-online/0.1 (research; public data)"}
BAR_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
OHLCV_COLS = ["ts", "open", "high", "low", "close", "volume"]


class FeedError(RuntimeError):
    pass


def _ssl_context() -> Optional[ssl.SSLContext]:
    """python.org builds on macOS ship without root certificates until 'Install Certificates'
    is run. If certifi is installed (requests brings it) use its bundle; else the system default."""
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return None


_CTX = _ssl_context()


def _get(url: str, timeout: float = 30.0, tries: int = 4) -> bytes:
    last: Optional[Exception] = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (400, 404):
                raise FeedError(f"HTTP {exc.code} for {url}") from exc
            time.sleep(1.5 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise FeedError(f"gave up on {url}: {last}")


def _get_json(url: str, timeout: float = 30.0):
    return json.loads(_get(url, timeout=timeout).decode("utf-8"))


@dataclass
class Feed:
    """Interface every feed implements. Timestamps are bar OPEN times, UTC, tz-aware."""

    name: str = "feed"

    def klines(self, symbol: str, bar: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    def funding(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    def book_ticker(self, symbol: str) -> dict:
        raise NotImplementedError


class BinanceVision(Feed):
    REST = "https://data-api.binance.vision/api/v3"
    ARCHIVE = "https://data.binance.vision/data"

    def __init__(self, pause_s: float = 0.15) -> None:
        self.name = "binance_vision"
        self.pause_s = pause_s

    def klines(self, symbol: str, bar: str, start: datetime, end: datetime) -> pd.DataFrame:
        if bar not in BAR_MS:
            raise ValueError(f"unsupported bar {bar}")
        out: List[list] = []
        cur = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        step = BAR_MS[bar]
        while cur < end_ms:
            url = (f"{self.REST}/klines?symbol={symbol}&interval={bar}&startTime={cur}"
                   f"&endTime={min(end_ms, cur + step * 1000)}&limit=1000")
            rows = _get_json(url)
            if not rows:
                cur += step * 1000
                continue
            out.extend(rows)
            cur = int(rows[-1][0]) + step
            time.sleep(self.pause_s)
        return _klines_frame(out)

    def funding(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """USDT-perpetual funding from the monthly archives (8-hourly settlements)."""
        frames = []
        month = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
        while month <= end:
            url = f"{self.ARCHIVE}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month:%Y-%m}.zip"
            try:
                raw = _get(url, timeout=60, tries=2)
            except FeedError:
                raw = b""
            if raw:
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    for name in z.namelist():
                        text = z.read(name).decode("utf-8")
                        rdr = csv.reader(io.StringIO(text))
                        rows = [r for r in rdr if r and not r[0].startswith("calc_time") and not r[0].startswith("symbol")]
                        for r in rows:
                            try:
                                # formats seen: [calc_time, funding_interval_hours, last_funding_rate] or [symbol, time, rate]
                                if len(r) >= 3 and r[0].isdigit():
                                    frames.append((int(r[0]), float(r[2])))
                                elif len(r) >= 3:
                                    frames.append((int(r[1]), float(r[2])))
                            except ValueError:
                                continue
            month = (month + timedelta(days=32)).replace(day=1)
        if not frames:
            return pd.DataFrame(columns=["ts", "rate"])
        df = pd.DataFrame(frames, columns=["ts", "rate"]).drop_duplicates("ts").sort_values("ts")
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df[(df.ts >= start) & (df.ts <= end)].reset_index(drop=True)

    def book_ticker(self, symbol: str) -> dict:
        d = _get_json(f"{self.REST}/ticker/bookTicker?symbol={symbol}", timeout=10)
        bid, ask = float(d["bidPrice"]), float(d["askPrice"])
        return {"symbol": symbol, "bid": bid, "ask": ask, "bid_size": float(d["bidQty"]), "ask_size": float(d["askQty"]),
                "mid": (bid + ask) / 2, "spread_bps": 1e4 * (ask - bid) / ((ask + bid) / 2), "ts": datetime.now(timezone.utc)}

    def exchange_filters(self, symbol: str) -> dict:
        info = _get_json(f"{self.REST}/exchangeInfo?symbol={symbol}", timeout=15)
        s = info["symbols"][0]
        out = {"status": s.get("status"), "tick": None, "step": None, "min_notional": None}
        for f in s.get("filters", []):
            if f["filterType"] == "PRICE_FILTER":
                out["tick"] = float(f["tickSize"])
            elif f["filterType"] == "LOT_SIZE":
                out["step"] = float(f["stepSize"])
            elif f["filterType"] in ("NOTIONAL", "MIN_NOTIONAL"):
                out["min_notional"] = float(f.get("minNotional", 0) or 0)
        return out


class OKX(Feed):
    REST = "https://www.okx.com/api/v5"
    BARS = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}

    def __init__(self, pause_s: float = 0.2) -> None:
        self.name = "okx"
        self.pause_s = pause_s

    @staticmethod
    def inst(symbol: str) -> str:
        return symbol.replace("USDT", "-USDT")

    def klines(self, symbol: str, bar: str, start: datetime, end: datetime) -> pd.DataFrame:
        out: List[list] = []
        after = int(end.timestamp() * 1000)
        start_ms = int(start.timestamp() * 1000)
        for _ in range(4000):
            url = f"{self.REST}/market/history-candles?instId={self.inst(symbol)}&bar={self.BARS[bar]}&after={after}&limit=100"
            data = _get_json(url).get("data") or []
            if not data:
                break
            for r in data:
                out.append([int(r[0]), r[1], r[2], r[3], r[4], r[5]])
            after = int(data[-1][0])
            if after <= start_ms:
                break
            time.sleep(self.pause_s)
        df = _klines_frame(out)
        return df[df.ts >= start].reset_index(drop=True)

    def funding(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        out = []
        after = int(end.timestamp() * 1000)
        for _ in range(200):
            url = f"{self.REST}/public/funding-rate-history?instId={self.inst(symbol)}-SWAP&after={after}&limit=100"
            data = _get_json(url).get("data") or []
            if not data:
                break
            out.extend((int(r["fundingTime"]), float(r["realizedRate"] if r.get("realizedRate") else r["fundingRate"])) for r in data)
            after = int(data[-1]["fundingTime"])
            if after <= int(start.timestamp() * 1000):
                break
            time.sleep(self.pause_s)
        if not out:
            return pd.DataFrame(columns=["ts", "rate"])
        df = pd.DataFrame(out, columns=["ts", "rate"]).drop_duplicates("ts").sort_values("ts")
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df[(df.ts >= start) & (df.ts <= end)].reset_index(drop=True)

    def book_ticker(self, symbol: str) -> dict:
        d = (_get_json(f"{self.REST}/market/ticker?instId={self.inst(symbol)}", timeout=10).get("data") or [{}])[0]
        bid, ask = float(d["bidPx"]), float(d["askPx"])
        return {"symbol": symbol, "bid": bid, "ask": ask, "bid_size": float(d["bidSz"]), "ask_size": float(d["askSz"]),
                "mid": (bid + ask) / 2, "spread_bps": 1e4 * (ask - bid) / ((ask + bid) / 2), "ts": datetime.now(timezone.utc)}


def _klines_frame(rows: Iterable[list]) -> pd.DataFrame:
    recs = []
    for r in rows:
        try:
            recs.append((int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
        except (TypeError, ValueError, IndexError):
            continue
    if not recs:
        return pd.DataFrame(columns=OHLCV_COLS)
    df = pd.DataFrame(recs, columns=OHLCV_COLS).drop_duplicates("ts").sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.reset_index(drop=True)


def with_fallback(primary: Feed, fallback: Feed):
    """A feed that tries the primary and falls back on any error, recording which one answered."""

    class _F(Feed):
        name = f"{primary.name}+{fallback.name}"
        last_used = primary.name

        def _try(self, fn: str, *a):
            try:
                out = getattr(primary, fn)(*a)
                self.last_used = primary.name
                return out
            except Exception:  # noqa: BLE001
                out = getattr(fallback, fn)(*a)
                self.last_used = fallback.name
                return out

        def klines(self, *a):
            return self._try("klines", *a)

        def funding(self, *a):
            return self._try("funding", *a)

        def book_ticker(self, *a):
            return self._try("book_ticker", *a)

    return _F()


class Yahoo(Feed):
    """Daily adjusted bars for listed ETFs and shares from Yahoo's public chart endpoint. No key.

    Used for the multi-asset trend sleeve, where breadth (bonds, gold, oil, currencies, equity indices)
    is the point. Prices are total-return adjusted: open, high, low and close are all scaled by
    adjclose / close so splits and dividends do not look like price moves. Timestamps are the session
    date at 00:00 UTC. There is no funding and no order book here."""

    REST = "https://query1.finance.yahoo.com/v8/finance/chart"

    def __init__(self, pause_s: float = 0.4) -> None:
        self.name = "yahoo"
        self.pause_s = pause_s

    def klines(self, symbol: str, bar: str, start: datetime, end: datetime) -> pd.DataFrame:
        if bar != "1d":
            raise ValueError("the Yahoo feed serves daily bars only")
        url = (f"{self.REST}/{urllib.parse.quote(symbol)}?period1={int(start.timestamp())}&period2={int(end.timestamp())}"
               f"&interval=1d&events=div%2Csplit")
        raw = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (itrader-online research; public data)"})
        last: Optional[Exception] = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(raw, timeout=30, context=_CTX) as r:
                    d = json.loads(r.read().decode("utf-8"))
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(2.0 * (attempt + 1))
        else:
            raise FeedError(f"yahoo gave up on {symbol}: {last}")
        time.sleep(self.pause_s)
        res = (d.get("chart", {}).get("result") or [None])[0]
        if not res or not res.get("timestamp"):
            return pd.DataFrame(columns=OHLCV_COLS)
        q = res["indicators"]["quote"][0]
        adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose") or q["close"]
        df = pd.DataFrame({"ts": res["timestamp"], "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                           "volume": q["volume"], "adj": adj}).dropna(subset=["close", "adj"])
        k = df["adj"] / df["close"]
        for c in ("open", "high", "low", "close"):
            df[c] = df[c].astype(float) * k
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.floor("D")
        df["volume"] = df["volume"].fillna(0.0).astype(float)
        return df[OHLCV_COLS].drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    def funding(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return pd.DataFrame(columns=["ts", "rate"])

    def book_ticker(self, symbol: str) -> dict:
        raise FeedError("no order book on the Yahoo feed")
