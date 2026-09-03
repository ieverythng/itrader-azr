"""Tag task pools with market-regime labels derived from OHLCV windows.

Regime tags are computed per-task over the market window that the episode
actually covers (``start_idx .. start_idx + horizon``), so capability reports
can later group held-out performance by regime, not just horizon.

Tag families (mirroring ``itrader/rl_lib/tasks/market_summary.py`` semantics):

- vol_regime:    low | normal | high      (realized vol vs dataset median)
- trend_regime:  bullish | bearish | sideways  (window return vs band)
- liq_regime:    volume_surge | thin | normal  (volume vs dataset median)

Tasks with ``start_idx is None`` are tagged from the full dataset distribution
and receive a ``dataset_wide`` marker tag so downstream consumers can tell
them apart from window-local tags.

Usage:
    python -m itrader.scripts.tag_pool_regimes --pool _tmp/task_pool_x.jsonl \
        [--df-path itrader/data/ohlcv.parquet] [--out <path>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tag a task pool JSONL with market regime labels.")
    parser.add_argument("--pool", type=str, required=True, help="Task pool JSONL file (modified in place unless --out given)")
    parser.add_argument(
        "--df-path",
        type=str,
        default="itrader/data/ohlcv.parquet",
        help="OHLCV parquet used to compute regime statistics",
    )
    parser.add_argument("--out", type=str, default=None, help="Output JSONL path (default: overwrite --pool)")
    parser.add_argument(
        "--vol-window",
        type=int,
        default=60,
        help="Bars used for realized-vol estimation when start_idx is None",
    )
    return parser.parse_args()


def _window_stats(df: pd.DataFrame, start_idx: int, horizon: int) -> Dict[str, float]:
    """Return (ret, vol, volume_mean) over df rows [start_idx, start_idx + horizon)."""
    end = min(start_idx + horizon, len(df))
    window = df.iloc[start_idx:end]
    if len(window) < 2:
        return {"ret": 0.0, "vol": 0.0, "volume_mean": 0.0}
    close = window["CLOSE"].to_numpy(dtype=np.float64)
    ret = float(close[-1] / close[0] - 1.0) if close[0] != 0 else 0.0
    log_ret = np.log(np.maximum(close, 1e-9))
    vol = float(np.std(np.diff(log_ret))) if len(log_ret) > 2 else 0.0
    volume_mean = float(window["VOLUME"].mean())
    return {"ret": ret, "vol": vol, "volume_mean": volume_mean}


def _tag_from_stats(stats: Dict[str, float], dataset: Dict[str, float]) -> List[str]:
    """Map window stats onto discrete regime tags using dataset medians."""
    tags: List[str] = []

    # Volatility regime (relative to dataset median realized vol)
    med_vol = max(dataset["vol_median"], 1e-9)
    ratio = stats["vol"] / med_vol
    if ratio < 0.75:
        tags.append("vol_low")
    elif ratio > 1.5:
        tags.append("vol_high")
    else:
        tags.append("vol_normal")

    # Trend regime (absolute window return band)
    ret = stats["ret"]
    if ret > 0.02:
        tags.append("trend_bullish")
    elif ret < -0.02:
        tags.append("trend_bearish")
    else:
        tags.append("trend_sideways")

    # Liquidity regime (relative to dataset median volume)
    med_volu = max(dataset["volume_median"], 1e-9)
    volu_ratio = stats["volume_mean"] / med_volu
    if volu_ratio > 1.5:
        tags.append("liq_surge")
    elif volu_ratio < 0.75:
        tags.append("liq_thin")
    else:
        tags.append("liq_normal")

    return tags


def tag_pool(
    pool_path: str,
    df_path: str,
    out_path: Optional[str] = None,
    vol_window: int = 60,
) -> Dict[str, Any]:
    """Tag every task in a pool JSONL with regime labels. Returns a summary."""
    df = pd.read_parquet(df_path)
    close_all = df["CLOSE"].to_numpy(dtype=np.float64)

    # Dataset-level reference stats: realized vol over rolling windows of `vol_window`
    if len(close_all) > vol_window + 1:
        log_ret = np.diff(np.log(np.maximum(close_all, 1e-9)))
        chunk = max(vol_window - 1, 2)
        n_chunks = max(1, (len(log_ret) - 1) // chunk)
        sample_vols = [
            float(np.std(log_ret[i * chunk : i * chunk + chunk])) for i in range(n_chunks)
        ]
        vol_median = float(np.median(sample_vols))
    else:
        vol_median = 0.0
    volume_median = float(df["VOLUME"].median())
    dataset = {"vol_median": vol_median, "volume_median": volume_median}

    out_path = out_path or pool_path
    tagged: List[Dict[str, Any]] = []
    tag_counts: Dict[str, int] = {}
    n_window_local = 0
    n_dataset_wide = 0

    with open(pool_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            task = entry.get("task", {})
            horizon = int(task.get("horizon", 128))
            start_idx = task.get("start_idx")

            if start_idx is None:
                # Dataset-wide task: tag from a reference window at the end of data.
                stats = _window_stats(df, max(0, len(df) - vol_window), horizon)
                tags = _tag_from_stats(stats, dataset) + ["dataset_wide"]
                n_dataset_wide += 1
            else:
                start_idx = int(start_idx)
                start_idx = max(0, min(start_idx, len(df) - 2))
                stats = _window_stats(df, start_idx, horizon)
                tags = _tag_from_stats(stats, dataset)
                n_window_local += 1

            task["regime_tags"] = tags
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            tagged.append(entry)

    with open(out_path, "w", encoding="utf-8") as f:
        for entry in tagged:
            f.write(json.dumps(entry) + "\n")

    return {
        "pool": str(pool_path),
        "out": str(out_path),
        "tasks_tagged": len(tagged),
        "window_local_tasks": n_window_local,
        "dataset_wide_tasks": n_dataset_wide,
        "tag_counts": tag_counts,
    }


def main() -> int:
    args = _parse_args()
    summary = tag_pool(
        pool_path=args.pool,
        df_path=args.df_path,
        out_path=args.out,
        vol_window=args.vol_window,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
