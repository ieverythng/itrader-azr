"""
Purpose: Stream real-time trades into DuckDB.
Core value: raw tick data stays immutable so later ETL runs remain reproducible.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
from typing import Iterable

import duckdb
import websockets


API_KEY_ENV = "POLYGON_API_KEY"
DEFAULT_URL = "wss://socket.polygon.io/stocks"
DEFAULT_DB_PATH = "itrader/data/itrader.duckdb"
DEFAULT_SYMBOLS = "AAPL,MSFT,AMZN"


def _require_api_key() -> str:
    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing {API_KEY_ENV}. Export it before running live_data.py."
        )
    return api_key


def _configured_symbols() -> list[str]:
    raw_symbols = os.getenv("POLYGON_SYMBOLS", DEFAULT_SYMBOLS)
    return [symbol.strip().upper() for symbol in raw_symbols.split(",") if symbol.strip()]


def _connect_db() -> duckdb.DuckDBPyConnection:
    db_path = os.getenv("ITRADER_DUCKDB_PATH", DEFAULT_DB_PATH)
    con = duckdb.connect(db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ticks(
            ts TIMESTAMP,
            symbol VARCHAR,
            price DOUBLE,
            size BIGINT
        )
        """
    )
    return con


def _trade_rows(messages: Iterable[dict]) -> Iterable[list[object]]:
    for message in messages:
        if message.get("ev") != "T":
            continue
        yield [
            dt.datetime.fromtimestamp(message["t"] / 1e9),
            message["sym"],
            message["p"],
            message["s"],
        ]


async def stream() -> None:
    api_key = _require_api_key()
    url = os.getenv("POLYGON_WS_URL", DEFAULT_URL)
    symbols = _configured_symbols()
    con = _connect_db()
    subscribe_params = ",".join(f"T.{symbol}" for symbol in symbols)

    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"action": "auth", "params": api_key}))
        await ws.send(json.dumps({"action": "subscribe", "params": subscribe_params}))
        while True:
            rows = list(_trade_rows(json.loads(await ws.recv())))
            for row in rows:
                con.execute("INSERT INTO ticks VALUES (?,?,?,?)", row)


if __name__ == "__main__":
    asyncio.run(stream())
