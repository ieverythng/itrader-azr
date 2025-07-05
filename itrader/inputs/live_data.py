"""
Purpose: Stream real-time trades (or quotes) into DuckDB.
Core Value: *Reproducibility* – raw tick data is immutable; every
subsequent ETL step can be rerun from exactly the same source.
"""
import asyncio, duckdb, json, datetime as dt
import websockets                                   # async WS client

API_KEY = "YaYXChnAvjhfkTTfl6LSvYSBnlwu7FM5A"
URL = "wss://socket.polygon.io/stocks"

con = duckdb.connect("itrader/data/itrader.duckdb")
con.execute("""CREATE TABLE IF NOT EXISTS ticks(
                 ts TIMESTAMP, symbol VARCHAR,
                 price DOUBLE, size BIGINT)""")

SYMBOLS = ["AAPL","MSFT", "AMZN"]           # add/remove here

async def stream():
    async with websockets.connect(URL) as ws:
        await ws.send(json.dumps({"action": "auth",
                                  "params": API_KEY}))
        await ws.send(json.dumps({"action": "subscribe",
                                  "params": ",".join(f"T.{s}" for s in SYMBOLS)}))
        while True:
            for msg in json.loads(await ws.recv()):
                if msg["ev"] == "T":            # ‘Trade’ event
                    row = [dt.datetime.fromtimestamp(msg["t"]/1e9),
                           msg["sym"], msg["p"], msg["s"]]
                    con.execute("INSERT INTO ticks VALUES (?,?,?,?)", row)

asyncio.run(stream())
