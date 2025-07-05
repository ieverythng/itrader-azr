# itrader/p5/etl_job.py
"""
Resample ticks → 1-min OHLCV for *all* symbols found in data/itrader.duckdb
and add basic TA features.  Safe to re-run any time.
"""
import duckdb, pandas as pd, ta, pathlib, datetime as dt

DB_PATH = "itrader/data/itrader.duckdb"
OUT_PARQUET = pathlib.Path(__file__).parent.parent / "data" / "ohlcv.parquet"

con = duckdb.connect(DB_PATH)

ticks = con.sql("""
    SELECT ts::TIMESTAMP AS ts,  -- ensure timestamp type
           symbol, price, size
    FROM   ticks
""").df()

if ticks.empty:
    print("⚠️  No data in ticks table yet.")
    exit()

# ---------- resample per-symbol ----------
bars = (ticks
        .groupby(["symbol",
                  pd.Grouper(key="ts", freq="1min")])
        .agg(open=("price","first"),
             high=("price","max"),
             low=("price","min"),
             close=("price","last"),
             volume=("size","sum"))
        .dropna()
        .reset_index())

# ---------- TA features ----------
bars["rsi14"]  = bars.groupby("symbol")["close"].transform(
                    lambda s: ta.momentum.rsi(s, window=14))
bars["sma20"]  = bars.groupby("symbol")["close"].transform(
                    lambda s: ta.trend.sma_indicator(s, window=20))

# ---------- write parquet ----------
bars.to_parquet(OUT_PARQUET, index=False)
print("✓ Wrote", len(bars), "rows →", OUT_PARQUET)
