# itrader/inputs/backfill.py
# ────────────────────────────────────────────────────────────────────────
"""
Tiny loader utilities used *only* by Phase-A (solver-only PPO):
    • load_market_data(path)         →  np.ndarray  [T, 5]  (O-H-L-C-V)
    • compute_valid_indices(array, window_size)
                                    →  np.ndarray  [N]      start-indices
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]   # .../itrader-azr

def load_market_data(path: str | Path) -> np.ndarray:
    """Read the parquet produced by etl_job and return a float32 tensor.

    Expected columns: ['open','high','low','close','volume'] in time order.
    """
    p = Path(path)
    if not p.exists():
        alt = REPO_ROOT / p
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(
                f"Could not locate parquet at '{path}' "
                f"or '{alt}'.\n\n"
                "Pass --data_path explicitly or run the ETL job first."
            )
    df = pd.read_parquet(p).sort_index()
    # Pick the 5 core columns and cast to float32 for the GPU.
    arr = df[['open', 'high', 'low', 'close', 'volume']].values
    return arr.astype(np.float32)


def compute_valid_indices(array: np.ndarray, window_size: int) -> np.ndarray:
    """Return all starting indices where a full window fits."""
    return np.arange(0, len(array) - window_size, dtype=np.int32)
