# # tests/test_llm.py
# from itrader.llm.engine import ask
# def test_llm_runs():
#     assert "trading" in ask("Say one word connected to trading:").lower()

import pandas as pd, pathlib, pprint
df = pd.read_parquet("itrader/data/ohlcv.parquet")
pprint.pp(df.columns.tolist())