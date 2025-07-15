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
import argparse, datetime as dt, duckdb, sys, time, requests

API_KEY  = "aYXChnAvjhfkTTfl6LSvYSBnlwu7FM5A"         # ← your real key
DB_PATH  = "itrader/data/itrader.duckdb" # keep as-is
STEP_D   = 5                                           # Polygon free slice
SLEEP_S  = 13                                          # ≈ 4 calls/min < 5-cap

# ── CLI args ──────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--symbols", default="AMZN", help="comma list e.g. AAPL,MSFT")
p.add_argument("--days", type=int, default=365)
args       = p.parse_args()
SYMBOLS    = [s.strip().upper() for s in args.symbols.split(",")]
end_date   = dt.date.today()
start_date = end_date - dt.timedelta(days=args.days)

# ── DB setup ──────────────────────────────────────────────────────────────
con = duckdb.connect(DB_PATH)
con.execute("""
CREATE TABLE IF NOT EXISTS ticks(
  ts TIMESTAMP, symbol VARCHAR, price DOUBLE, size BIGINT
);
""")

cols = [r[1] for r in con.execute(
            "PRAGMA table_info('_backfill_cursor')").fetchall()]

if "symbol" not in cols or "d" not in cols:
    # either table missing or wrong schema → recreate cleanly
    con.execute("DROP TABLE IF EXISTS _backfill_cursor;")
    con.execute("""
        CREATE TABLE _backfill_cursor(
          symbol VARCHAR PRIMARY KEY,
          d DATE
        );
    """)

# ── helpers ───────────────────────────────────────────────────────────────
def grab(sym: str, date_from, date_to):
    url = (f"https://api.polygon.io/v2/aggs/ticker/{sym}/"
           f"range/1/minute/{date_from}/{date_to}")
    out = requests.get(url, params={
        "adjusted": "true", "sort": "asc", "limit": 50000,
        "apiKey": API_KEY}).json()
    if "results" not in out:
        sys.exit(f"Polygon error ➜ {out}")
    return out["results"]

# ── main loop per symbol ──────────────────────────────────────────────────
for sym in SYMBOLS:
    row = con.execute(
            "SELECT d FROM _backfill_cursor WHERE symbol = ?", [sym]
        ).fetchone()
    last_date = row[0] if row else None
    current   = (last_date + dt.timedelta(days=1)) if last_date else start_date

    while current < end_date:
        slice_end = min(current + dt.timedelta(days=STEP_D), end_date)
        bars = grab(sym, current, slice_end)

        for b in bars:
            con.execute("INSERT INTO ticks VALUES (?,?,?,?)",
                        [dt.datetime.fromtimestamp(b["t"] / 1000),
                         sym, b["c"], b["v"]])

        con.execute("INSERT OR REPLACE INTO _backfill_cursor VALUES (?, ?)", [sym, slice_end])
        print(f"{sym}: {current} → {slice_end} rows:{len(bars)}")
        current = slice_end
        time.sleep(SLEEP_S)

print("✓ Back-fill done. Total rows:",
      con.sql("SELECT COUNT(*) FROM ticks").fetchone()[0])
