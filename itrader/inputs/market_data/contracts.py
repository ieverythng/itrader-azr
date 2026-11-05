from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


_STOCK_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9._-]{0,31}\Z")


@dataclass(frozen=True)
class HistoricalBarsRequest:
    symbol: str
    start: datetime
    end: datetime
    timeframe: str = "1Min"
    feed: str = "sip"
    adjustment: str = "split"

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("Historical bar bounds must be timezone-aware")
        if self.start >= self.end:
            raise ValueError("Historical bar start must precede end")
        if self.adjustment != "split":
            raise ValueError("dataset_v2 supports split-adjusted bars only")


@dataclass(frozen=True)
class MarketDataPage:
    request: dict[str, Any]
    payload: Any


class HistoricalMarketDataSource(Protocol):
    provider: str

    def fetch_calendar(self, start: date, end: date) -> MarketDataPage: ...

    def iter_bar_pages(
        self,
        request: HistoricalBarsRequest,
    ) -> Iterator[MarketDataPage]: ...


@dataclass(frozen=True)
class SnapshotBuildConfig:
    snapshot_root: Path
    symbols: tuple[str, ...]
    start: date
    end: date
    feed: str = "sip"
    adjustment: str = "split"
    timeframe: str = "1Min"
    timezone_name: str = "America/New_York"
    usage_classification: str = "unverified_personal_internal_research"
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("At least one symbol is required")
        if self.start > self.end:
            raise ValueError("Snapshot start must not exceed end")
        if self.schema_version != 2:
            raise ValueError("Only dataset_v2 schema version 2 is supported")
        if self.adjustment != "split":
            raise ValueError("dataset_v2 supports split-adjusted bars only")
        if self.timeframe != "1Min":
            raise ValueError("dataset_v2 supports one-minute bars only")
        if not self.feed.strip():
            raise ValueError("Snapshot feed must be non-empty")
        if not self.usage_classification.strip():
            raise ValueError("Snapshot usage classification must be non-empty")
        try:
            ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown snapshot timezone: {self.timezone_name}") from exc
        normalized = tuple(symbol.strip().upper() for symbol in self.symbols)
        if any(_STOCK_SYMBOL.fullmatch(symbol) is None for symbol in normalized):
            raise ValueError("Snapshot symbols must be safe stock ticker identifiers")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Snapshot symbols must be unique")
        object.__setattr__(self, "symbols", normalized)
        object.__setattr__(self, "snapshot_root", Path(self.snapshot_root).expanduser().resolve())

    @property
    def end_exclusive(self) -> datetime:
        from datetime import timedelta

        return datetime.combine(self.end + timedelta(days=1), datetime.min.time(), timezone.utc)
