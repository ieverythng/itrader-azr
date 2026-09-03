from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from itrader.rl_lib.tasks.market_summary import (
    build_market_snapshot,
    build_market_summary_from_ohlcv_path,
    render_market_summary,
)


def _sample_ohlcv(n: int = 120) -> pd.DataFrame:
    x = np.linspace(0.0, 1.0, n)
    close = 100.0 + 2.0 * x + 0.5 * np.sin(10.0 * x)
    open_ = close - 0.1
    high = close + 0.2
    low = close - 0.2
    vol = 1_000.0 + 50.0 * np.cos(8.0 * x)
    return pd.DataFrame(
        {
            "OPEN": open_,
            "HIGH": high,
            "LOW": low,
            "CLOSE": close,
            "VOLUME": vol,
        }
    )


def test_market_snapshot_builds_core_fields():
    df = _sample_ohlcv()
    snap = build_market_snapshot(df, window_size=64, end_idx=100)
    assert snap["n_rows"] == 64
    assert snap["start_idx"] == 37
    assert snap["end_idx"] == 100
    assert isinstance(snap["ret_window"], float)
    assert isinstance(snap["realized_vol"], float)
    assert snap["trend_regime"] in {"bullish", "bearish", "sideways"}
    assert snap["vol_regime"] in {"low", "normal", "high"}


def test_render_market_summary_contains_runtime_constraints():
    snap = build_market_snapshot(_sample_ohlcv(), window_size=32, end_idx=-1)
    txt = render_market_summary(snap)
    assert "Market snapshot (OHLCV window):" in txt
    assert 'family_id="00"' in txt
    assert 'reward_id="pnl_linear"' in txt


def test_build_market_summary_from_path(tmp_path: Path):
    path = tmp_path / "ohlcv.parquet"
    _sample_ohlcv().to_parquet(path, index=False)
    txt = build_market_summary_from_ohlcv_path(path, window_size=50, end_idx=80)
    assert "rows=50" in txt
    assert "end_idx=80" in txt
