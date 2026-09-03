"""
Tiny loader utilities used *only* by Phase-A (solver-only PPO):
    • load_market_data(path)         →  np.ndarray  [T, 5]  (O-H-L-C-V)
    • compute_valid_indices(array, window_size)
                                    →  np.ndarray  [N]      start-indices
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _resolve_input_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    alt = REPO_ROOT / candidate
    if alt.exists():
        return alt

    raise FileNotFoundError(
        f"Could not locate parquet at '{path}' or '{alt}'.\n\n"
        "Pass --data_path explicitly or run the ETL job first."
    )


def _core_columns(df: pd.DataFrame) -> list[str]:
    lower_to_actual = {column.lower(): column for column in df.columns}
    missing = [column for column in CORE_OHLCV_COLUMNS if column not in lower_to_actual]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
    return [lower_to_actual[column] for column in CORE_OHLCV_COLUMNS]

def load_market_data(path: str | Path) -> np.ndarray:
    """Read the parquet produced by etl_job and return a float32 tensor.

    Expected columns: ['open','high','low','close','volume'] in time order.
    """
    df = pd.read_parquet(_resolve_input_path(path)).sort_index()
    # Pick the 5 core columns and cast to float32 for the GPU.
    arr = df[_core_columns(df)].to_numpy(copy=False)
    return arr.astype(np.float32)


def compute_valid_indices(array: np.ndarray, window_size: int) -> np.ndarray:
    """Return all starting indices where a full window fits."""
    return np.arange(0, len(array) - window_size, dtype=np.int32)
