"""
Resample ticks into 1-minute OHLCV bars for all symbols in itrader.duckdb
and add a small set of TA features.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import ta


DB_PATH = "itrader/data/itrader.duckdb"
OUT_PARQUET = Path(__file__).resolve().parents[1] / "data" / "ohlcv.parquet"


def _load_ticks(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.sql(
        """
        SELECT ts::TIMESTAMP AS ts,
               symbol,
               price,
               size
        FROM ticks
        """
    ).df()


def _build_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    return (
        ticks.groupby(["symbol", pd.Grouper(key="ts", freq="1min")])
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("size", "sum"),
        )
        .dropna()
        .reset_index()
    )


def _add_features(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    bars["rsi14"] = bars.groupby("symbol")["close"].transform(
        lambda series: ta.momentum.rsi(series, window=14)
    )
    bars["sma20"] = bars.groupby("symbol")["close"].transform(
        lambda series: ta.trend.sma_indicator(series, window=20)
    )
    return bars


def main() -> int:
    con = duckdb.connect(DB_PATH)
    ticks = _load_ticks(con)
    if ticks.empty:
        print("No data in ticks table yet.")
        return 1

    bars = _build_bars(ticks)
    bars = _add_features(bars)
    bars.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {len(bars)} rows -> {OUT_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
