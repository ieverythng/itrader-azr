"""
Purpose: Stream real-time trades (or quotes) into DuckDB.
Core Value: *Reproducibility* – raw tick data is immutable; every
subsequent ETL step can be rerun from exactly the same source.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os

import duckdb
import websockets  # async WS client

API_KEY_ENV = "POLYGON_API_KEY"
URL = os.getenv("POLYGON_WS_URL", "wss://socket.polygon.io/stocks")
DB_PATH = os.getenv("ITRADER_DUCKDB_PATH", "itrader/data/itrader.duckdb")
SYMBOLS = [s.strip().upper() for s in os.getenv("POLYGON_SYMBOLS", "AAPL,MSFT,AMZN").split(",") if s.strip()]

con = duckdb.connect(DB_PATH)
con.execute("""CREATE TABLE IF NOT EXISTS ticks(
                 ts TIMESTAMP, symbol VARCHAR,
                 price DOUBLE, size BIGINT)""")


def _require_api_key() -> str:
    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing {API_KEY_ENV}. Export it before running live_data.py."
        )
    return api_key


async def stream():
    api_key = _require_api_key()
    async with websockets.connect(URL) as ws:
        await ws.send(json.dumps({"action": "auth",
                                  "params": api_key}))
        await ws.send(json.dumps({"action": "subscribe",
                                  "params": ",".join(f"T.{s}" for s in SYMBOLS)}))
        while True:
            for msg in json.loads(await ws.recv()):
                if msg["ev"] == "T":            # ‘Trade’ event
                    row = [dt.datetime.fromtimestamp(msg["t"]/1e9),
                           msg["sym"], msg["p"], msg["s"]]
                    con.execute("INSERT INTO ticks VALUES (?,?,?,?)", row)


if __name__ == "__main__":
    asyncio.run(stream())
