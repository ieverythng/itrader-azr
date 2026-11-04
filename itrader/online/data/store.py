"""Local parquet store with incremental updates and quality checks.

Layout:  <root>/<bar>/<SYMBOL>.parquet   and   <root>/funding/<SYMBOL>.parquet

Quality is a first-class output. `Quality.ok` is False when data is stale, has gaps, or has
impossible prices, and the RiskEngine refuses to trade on it. Bad data is the most common
way a correct strategy loses money.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .feeds import BAR_MS, Feed, BinanceVision, OKX, with_fallback


@dataclass
class Quality:
    symbol: str
    bar: str
    rows: int
    first: Optional[str]
    last: Optional[str]
    stale_s: Optional[float]
    gaps: int
    largest_gap_bars: int
    bad_prices: int
    ok: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class DataStore:
    def __init__(self, root: str | Path, feed: Optional[Feed] = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.feed = feed or with_fallback(BinanceVision(), OKX())

    # ------------------------------------------------------------------ paths
    def path(self, symbol: str, bar: str) -> Path:
        p = self.root / bar / f"{symbol}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def funding_path(self, symbol: str) -> Path:
        p = self.root / "funding" / f"{symbol}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------ reads
    def load(self, symbol: str, bar: str) -> pd.DataFrame:
        p = self.path(symbol, bar)
        if not p.exists():
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df.sort_values("ts").reset_index(drop=True)

    def load_funding(self, symbol: str) -> pd.DataFrame:
        p = self.funding_path(symbol)
        if not p.exists():
            return pd.DataFrame(columns=["ts", "rate"])
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df.sort_values("ts").reset_index(drop=True)

    def panel(self, symbols: Iterable[str], bar: str, field: str = "close") -> pd.DataFrame:
        """One column per symbol, aligned on the union of timestamps, forward-filled at most 3 bars."""
        cols = {}
        for s in symbols:
            df = self.load(s, bar)
            if len(df):
                cols[s] = df.set_index("ts")[field]
        if not cols:
            return pd.DataFrame()
        out = pd.DataFrame(cols).sort_index()
        return out.ffill(limit=3)

    def funding_panel(self, symbols: Iterable[str]) -> pd.DataFrame:
        cols = {}
        for s in symbols:
            df = self.load_funding(s)
            if len(df):
                cols[s] = df.set_index("ts")["rate"]
        return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()

    # ------------------------------------------------------------------ writes
    def update(self, symbol: str, bar: str, since: Optional[datetime] = None, until: Optional[datetime] = None) -> int:
        """Fetch what is missing and append. Returns rows added."""
        until = until or datetime.now(timezone.utc)
        have = self.load(symbol, bar)
        if len(have):
            start = have.ts.iloc[-1].to_pydatetime() + timedelta(milliseconds=BAR_MS[bar])
        else:
            start = since or (until - timedelta(days=365 * 3))
        if start >= until:
            return 0
        new = self.feed.klines(symbol, bar, start, until)
        if not len(new):
            return 0
        # never keep the still-open bar
        new = new[new.ts + pd.Timedelta(milliseconds=BAR_MS[bar]) <= pd.Timestamp(until)]
        df = pd.concat([have, new]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        df.to_parquet(self.path(symbol, bar), index=False)
        return len(df) - len(have)

    def update_funding(self, symbol: str, since: Optional[datetime] = None) -> int:
        until = datetime.now(timezone.utc)
        have = self.load_funding(symbol)
        start = have.ts.iloc[-1].to_pydatetime() + timedelta(hours=1) if len(have) else (since or until - timedelta(days=365 * 3))
        if start >= until:
            return 0
        new = self.feed.funding(symbol, start, until)
        if not len(new):
            return 0
        df = pd.concat([have, new]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        df.to_parquet(self.funding_path(symbol), index=False)
        return len(df) - len(have)

    def sync(self, symbols: Iterable[str], bars: Iterable[str], years: float = 3.0, funding: bool = True, log=print) -> Dict[str, int]:
        since = datetime.now(timezone.utc) - timedelta(days=365 * years)
        added: Dict[str, int] = {}
        for s in symbols:
            for b in bars:
                try:
                    n = self.update(s, b, since=since)
                except Exception as exc:  # noqa: BLE001
                    log(f"  [!] {s} {b}: {exc}")
                    n = -1
                added[f"{s}:{b}"] = n
                log(f"  {s} {b}: +{n} rows ({self.feed.name}, via {getattr(self.feed, 'last_used', self.feed.name)})")
            if funding:
                try:
                    n = self.update_funding(s, since=since)
                except Exception as exc:  # noqa: BLE001
                    log(f"  [!] {s} funding: {exc}")
                    n = -1
                added[f"{s}:funding"] = n
                log(f"  {s} funding: +{n} rows")
        return added

    # ------------------------------------------------------------------ quality
    def quality(self, symbol: str, bar: str, max_stale_bars: float = 2.0, now: Optional[datetime] = None, recent_bars: int = 200) -> Quality:
        """Staleness is judged on the last bar; gaps, bad prints and jumps on the last `recent_bars`
        (a 70% day three years ago is history, not a reason to refuse to trade today)."""
        df = self.load(symbol, bar)
        now = now or datetime.now(timezone.utc)
        reasons: List[str] = []
        if not len(df):
            return Quality(symbol, bar, 0, None, None, None, 0, 0, 0, False, ["no data"])
        step = pd.Timedelta(milliseconds=BAR_MS[bar])
        recent = df.tail(int(recent_bars))
        d = recent.ts.diff().dropna()
        gap_bars = (d / step).round().astype(int) - 1
        gaps = int((gap_bars > 0).sum())
        largest = int(gap_bars.max()) if len(gap_bars) else 0
        bad = int(((recent.high < recent.low) | (recent.close <= 0) | (recent.open <= 0) | ~np.isfinite(recent.close)).sum())
        stale_s = (pd.Timestamp(now) - (df.ts.iloc[-1] + step)).total_seconds()
        if stale_s > max_stale_bars * step.total_seconds():
            reasons.append(f"stale by {stale_s / 60:.0f} min")
        if largest > 24:
            reasons.append(f"gap of {largest} bars")
        if bad:
            reasons.append(f"{bad} impossible prices")
        # a 1-bar move beyond 40% is almost always a bad print on these pairs
        jumps = int((recent.close.pct_change().abs() > 0.40).sum())
        if jumps:
            reasons.append(f"{jumps} moves over 40% in one bar")
        return Quality(symbol, bar, len(df), str(df.ts.iloc[0]), str(df.ts.iloc[-1]), stale_s, gaps, largest, bad, not reasons, reasons)

    def quality_report(self, symbols: Iterable[str], bar: str, now: Optional[datetime] = None) -> Dict[str, Quality]:
        return {s: self.quality(s, bar, now=now) for s in symbols}

    def write_quality(self, symbols: Iterable[str], bar: str) -> Path:
        rep = {s: q.to_dict() for s, q in self.quality_report(symbols, bar).items()}
        p = self.root / f"quality_{bar}.json"
        p.write_text(json.dumps(rep, indent=1))
        return p
