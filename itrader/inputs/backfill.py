@ -0,0 +1,54 @@
"""
Minute-bar back-fill that respects Polygon’s 5-calls/min limit
and can be killed / resumed at any point.

Usage
------
python itrader/inputs/backfill.py --symbol AAPL --days 730
"""
import argparse, datetime as dt, duckdb, sys, time, requests, json

API_KEY = "aYXChnAvjhfkTTfl6LSvYSBnlwu7FM5A"
DB_PATH = "itrader/data/itrader.duckdb"

parser = argparse.ArgumentParser()
parser.add_argument("--symbol", default="AAPL")
parser.add_argument("--days", type=int, default=730)
args = parser.parse_args()

con = duckdb.connect(DB_PATH)
con.execute("""CREATE TABLE IF NOT EXISTS ticks(
                 ts TIMESTAMP, symbol VARCHAR, price DOUBLE, size BIGINT)""")

end_date   = dt.date.today()
start_date = end_date - dt.timedelta(days=args.days)
step       = dt.timedelta(days=5)        # free-plan slice

# ——— helper so we can resume ———
cursor_tbl = "CREATE TABLE IF NOT EXISTS _backfill_cursor(d DATE)"
con.execute(cursor_tbl)
last_done = con.sql("SELECT max(d) FROM _backfill_cursor").fetchone()[0]
current   = (last_done + dt.timedelta(days=1)) if last_done else start_date

def grab(slice_from, slice_to):
    url = f"https://api.polygon.io/v2/aggs/ticker/{args.symbol}/range/1/minute/{slice_from}/{slice_to}"
    params = {"adjusted":"true", "sort":"asc", "limit":50000, "apiKey":API_KEY}
    out = requests.get(url, params=params).json()
    if "results" not in out:
        sys.exit(f"Polygon error ➜ {out}")
    return out["results"]

while current < end_date:
    slice_end = min(current + step, end_date)
    bars = grab(current, slice_end)
    for b in bars:
        con.execute("INSERT INTO ticks VALUES (?,?,?,?)",
                    [dt.datetime.fromtimestamp(b["t"]/1000),
                     args.symbol, b["c"], b["v"]])
    con.execute("INSERT INTO _backfill_cursor VALUES (?)", [slice_end])
    print(f"{current} → {slice_end} rows:{len(bars)}")
    current = slice_end
    time.sleep(13)                     #  ~4 requests/min  < 5-limit

print("✓ Done. Total rows:",
      con.sql("SELECT COUNT(*) FROM ticks").fetchone()[0])