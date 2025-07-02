"""
Purpose: Resample ticks ➜ 1-min OHLCV, add TA features, and
materialise them into an ‘ohlcv’ table that RLlib can read fast.
Core Value: *Single source of truth* – everything derived from ticks.
"""
import duckdb, pandas as pd, ta

con = duckdb.connect("data/itrader.duckdb")
ticks = con.sql("SELECT * FROM ticks").df()

bar = (ticks.groupby(["symbol",
                      pd.Grouper(key="ts", freq="1min")])
             .agg(open=("price","first"),
                  high=("price","max"),
                  low=("price","min"),
                  close=("price","last"),
                  volume=("size","sum"))
             .dropna())

bar["rsi14"] = ta.momentum.rsi(bar.close, window=14)
bar["sma20"] = ta.trend.sma_indicator(bar.close, window=20)

con.execute("CREATE OR REPLACE TABLE ohlcv AS SELECT * FROM bar")
