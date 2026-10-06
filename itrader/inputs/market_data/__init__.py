"""Versioned historical market-data acquisition and snapshot interfaces."""

from .contracts import HistoricalBarsRequest, MarketDataPage, SnapshotBuildConfig
from .alpaca import AlpacaMarketDataSource
from .snapshot import SnapshotBuildResult, build_market_snapshot, open_snapshot_catalog

__all__ = [
    "HistoricalBarsRequest",
    "AlpacaMarketDataSource",
    "MarketDataPage",
    "SnapshotBuildConfig",
    "SnapshotBuildResult",
    "build_market_snapshot",
    "open_snapshot_catalog",
]
