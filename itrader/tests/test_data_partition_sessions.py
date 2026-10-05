from __future__ import annotations

import numpy as np
import pandas as pd

from itrader.rl_lib.evaluation import DataPartition


def _session_frame() -> pd.DataFrame:
    frames = []
    for day in ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"):
        ts = pd.date_range(f"{day} 14:30:00+00:00", periods=10, freq="min")
        frames.append(
            pd.DataFrame(
                {
                    "symbol": "AAPL",
                    "ts": ts,
                    "session_id": day,
                    "CLOSE": np.arange(10, dtype=float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_partition_intersects_symbol_chronology_with_session_boundaries():
    frame = _session_frame()
    partition = DataPartition(
        frame,
        window_size=2,
        split="validation",
        split_fractions=(0.5, 0.25, 0.25),
        symbol_col="symbol",
        timestamp_col="ts",
        boundary_cols=("session_id",),
        expected_bar_interval="1min",
    )

    ranges = partition.valid_ranges(horizon=3)

    assert [(item.first, item.last) for item in ranges] == [(22, 27)]
    for seed in range(10):
        start = partition.sample(3, np.random.default_rng(seed))
        episode = frame.iloc[start - 2 : start + 3]
        assert episode["session_id"].nunique() == 1
        assert episode["ts"].diff().dropna().eq(pd.Timedelta(minutes=1)).all()


def test_partition_splits_on_missing_minute_inside_session():
    frame = _session_frame().iloc[:10].drop(index=5).reset_index(drop=True)
    partition = DataPartition(
        frame,
        window_size=2,
        split="all",
        symbol_col="symbol",
        timestamp_col="ts",
        boundary_cols=("session_id",),
        expected_bar_interval="1min",
    )

    ranges = partition.valid_ranges(horizon=2)

    assert [(item.first, item.last) for item in ranges] == [(2, 3), (7, 7)]
