"""
Minute-bar back-fill that respects Polygon’s 5-calls/min limit and
can be killed / resumed at any point.

Examples
--------
# one symbol, two years
python itrader/inputs/backfill.py --symbols AAPL --days 730

# three symbols, one year
python itrader/inputs/backfill.py --symbols AAPL,MSFT,AMZN --days 365
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time

import duckdb
import requests

DB_PATH  = "itrader/data/itrader.duckdb" # keep as-is
STEP_D   = 5                                           # Polygon free slice
SLEEP_S  = 13                                          # ≈ 4 calls/min < 5-cap
API_KEY_ENV = "POLYGON_API_KEY"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="AMZN", help="comma list e.g. AAPL,MSFT")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--api-key", default="", help=f"Polygon API key (or set {API_KEY_ENV})")
    return p.parse_args()


def _resolve_api_key(cli_key: str) -> str:
    api_key = (cli_key or os.getenv(API_KEY_ENV, "")).strip()
    if not api_key:
        raise RuntimeError(f"Missing Polygon API key. Pass --api-key or set {API_KEY_ENV}.")
    return api_key


def _ensure_cursor_table(con: duckdb.DuckDBPyConnection) -> None:
    cols = [r[1] for r in con.execute("PRAGMA table_info('_backfill_cursor')").fetchall()]
    if "symbol" not in cols or "d" not in cols:
        con.execute("DROP TABLE IF EXISTS _backfill_cursor;")
        con.execute(
            """
            CREATE TABLE _backfill_cursor(
              symbol VARCHAR PRIMARY KEY,
              d DATE
            );
            """
        )


def grab(sym: str, date_from: dt.date, date_to: dt.date, api_key: str):
    url = (f"https://api.polygon.io/v2/aggs/ticker/{sym}/"
           f"range/1/minute/{date_from}/{date_to}")
    out = requests.get(
        url,
        params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key},
        timeout=60,
    ).json()
    if "results" not in out:
        sys.exit(f"Polygon error for {sym} [{date_from} -> {date_to}] ➜ {out}")
    return out["results"]


def main() -> int:
    args = _parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=args.days)
    api_key = _resolve_api_key(args.api_key)

    con = duckdb.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ticks(
          ts TIMESTAMP, symbol VARCHAR, price DOUBLE, size BIGINT
        );
        """
    )
    _ensure_cursor_table(con)

    for sym in symbols:
        row = con.execute(
            "SELECT d FROM _backfill_cursor WHERE symbol = ?", [sym]
        ).fetchone()
        last_date = row[0] if row else None
        current = (last_date + dt.timedelta(days=1)) if last_date else start_date

        while current < end_date:
            slice_end = min(current + dt.timedelta(days=STEP_D), end_date)
            bars = grab(sym, current, slice_end, api_key)

            for b in bars:
                con.execute(
                    "INSERT INTO ticks VALUES (?,?,?,?)",
                    [dt.datetime.fromtimestamp(b["t"] / 1000), sym, b["c"], b["v"]],
                )

            con.execute("INSERT OR REPLACE INTO _backfill_cursor VALUES (?, ?)", [sym, slice_end])
            print(f"{sym}: {current} -> {slice_end} rows:{len(bars)}")
            current = slice_end
            time.sleep(SLEEP_S)

    print(
        "Back-fill done. Total rows:",
        con.sql("SELECT COUNT(*) FROM ticks").fetchone()[0],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
