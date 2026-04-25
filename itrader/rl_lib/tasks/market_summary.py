from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from itrader.rl_lib.tasks.families import PRIMARY_FAMILY_ID, PRIMARY_REWARD_ID


def _pick_col(df: pd.DataFrame, preferred: str) -> str:
    preferred_u = str(preferred).strip().upper()
    for col in df.columns:
        if str(col).strip().upper() == preferred_u:
            return str(col)
    raise KeyError(f"Missing column: {preferred}")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def build_market_snapshot(
    df: pd.DataFrame,
    *,
    window_size: int = 256,
    end_idx: int | None = None,
    close_col: str = "CLOSE",
    high_col: str = "HIGH",
    low_col: str = "LOW",
    volume_col: str = "VOLUME",
) -> Dict[str, Any]:
    if df is None or len(df) == 0:
        raise ValueError("Input dataframe is empty")

    close_name = _pick_col(df, close_col)
    high_name = _pick_col(df, high_col)
    low_name = _pick_col(df, low_col)
    volume_name = _pick_col(df, volume_col)

    n_rows_total = len(df)
    end = n_rows_total - 1 if end_idx is None else int(end_idx)
    if end < 0:
        end = n_rows_total - 1
    end = max(0, min(end, n_rows_total - 1))

    n = max(8, int(window_size))
    start = max(0, end - n + 1)
    w = df.iloc[start : end + 1].copy()
    if len(w) < 2:
        raise ValueError("Window too small to summarize")

    closes = w[close_name].astype(float).to_numpy()
    highs = w[high_name].astype(float).to_numpy()
    lows = w[low_name].astype(float).to_numpy()
    volumes = w[volume_name].astype(float).to_numpy()

    rets = np.diff(closes) / np.maximum(np.abs(closes[:-1]), 1e-8)
    ret_1 = _safe_float(rets[-1] if len(rets) > 0 else 0.0)
    ret_window = _safe_float((closes[-1] - closes[0]) / max(abs(closes[0]), 1e-8))
    realized_vol = _safe_float(np.std(rets) if len(rets) > 1 else 0.0)

    if len(closes) >= 2:
        x = np.arange(len(closes), dtype=float)
        slope = np.polyfit(x, closes, deg=1)[0]
    else:
        slope = 0.0
    trend_slope = _safe_float(slope / max(abs(closes[-1]), 1e-8))

    base = max(abs(closes[0]), 1e-8)
    equity = closes / base
    peaks = np.maximum.accumulate(equity)
    drawdowns = (peaks - equity) / np.maximum(peaks, 1e-8)
    max_drawdown = _safe_float(np.max(drawdowns))

    range_ratio = (highs - lows) / np.maximum(np.abs(closes), 1e-8)
    mean_range_ratio = _safe_float(np.mean(range_ratio))

    v_mu = _safe_float(np.mean(volumes))
    v_sigma = _safe_float(np.std(volumes))
    volume_z_last = _safe_float((volumes[-1] - v_mu) / max(v_sigma, 1e-8))

    if realized_vol < 0.003:
        vol_regime = "low"
    elif realized_vol > 0.015:
        vol_regime = "high"
    else:
        vol_regime = "normal"

    if ret_window > 0.01:
        trend_regime = "bullish"
    elif ret_window < -0.01:
        trend_regime = "bearish"
    else:
        trend_regime = "sideways"

    if volume_z_last > 1.0:
        liq_regime = "volume_surge"
    elif volume_z_last < -1.0:
        liq_regime = "thin"
    else:
        liq_regime = "normal"

    return {
        "start_idx": int(start),
        "end_idx": int(end),
        "n_rows": int(len(w)),
        "close_last": _safe_float(closes[-1]),
        "ret_1": ret_1,
        "ret_window": ret_window,
        "realized_vol": realized_vol,
        "trend_slope": trend_slope,
        "max_drawdown": max_drawdown,
        "mean_range_ratio": mean_range_ratio,
        "volume_z_last": volume_z_last,
        "trend_regime": trend_regime,
        "vol_regime": vol_regime,
        "liquidity_regime": liq_regime,
    }


def render_market_summary(snapshot: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "Market snapshot (OHLCV window):",
            f"- rows={int(snapshot.get('n_rows', 0))}, start_idx={int(snapshot.get('start_idx', 0))}, end_idx={int(snapshot.get('end_idx', 0))}",
            f"- close_last={_safe_float(snapshot.get('close_last')):.6f}",
            f"- return_1={_safe_float(snapshot.get('ret_1')):.6f}",
            f"- return_window={_safe_float(snapshot.get('ret_window')):.6f}",
            f"- realized_vol={_safe_float(snapshot.get('realized_vol')):.6f}",
            f"- trend_slope={_safe_float(snapshot.get('trend_slope')):.6f}",
            f"- max_drawdown_window={_safe_float(snapshot.get('max_drawdown')):.6f}",
            f"- intrabar_range_ratio_mean={_safe_float(snapshot.get('mean_range_ratio')):.6f}",
            f"- volume_z_last={_safe_float(snapshot.get('volume_z_last')):.6f}",
            f"- regimes: trend={snapshot.get('trend_regime', 'unknown')}, vol={snapshot.get('vol_regime', 'unknown')}, liquidity={snapshot.get('liquidity_regime', 'unknown')}",
            (
                "Task-generation runtime constraints: "
                f'family_id="{PRIMARY_FAMILY_ID}", reward_id="{PRIMARY_REWARD_ID}".'
            ),
            "Generate valid task JSON only.",
        ]
    )


def build_market_summary_from_ohlcv_path(
    df_path: str | Path,
    *,
    window_size: int = 256,
    end_idx: int | None = None,
) -> str:
    path = Path(df_path).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[3] / path).resolve()
    df = pd.read_parquet(path)
    snap = build_market_snapshot(
        df,
        window_size=int(window_size),
        end_idx=end_idx,
    )
    return render_market_summary(snap)
