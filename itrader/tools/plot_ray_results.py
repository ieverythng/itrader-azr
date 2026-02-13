from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

# headless / non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd


def _columns_look_numeric(cols) -> bool:
    if cols is None:
        return False
    if hasattr(cols, "empty"):
        if cols.empty:
            return False
    elif len(cols) == 0:
        return False
    for c in cols:
        try:
            float(c)
        except Exception:
            return False
    return True


def _load_progress_csv(trial_dir: Path) -> pd.DataFrame | None:
    csv_path = trial_dir / "progress.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return None
    if df.empty or _columns_look_numeric(df.columns):
        return None
    return df


def _flatten_record(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(_flatten_record(v, key))
        elif isinstance(v, (int, float, bool, str)) or v is None:
            out[key] = v
    return out


def _load_result_json(trial_dir: Path) -> pd.DataFrame | None:
    candidates = [trial_dir / "result.json", trial_dir / "rllib" / "result.json"]
    for result_path in candidates:
        if not result_path.exists() or result_path.stat().st_size == 0:
            continue
        rows = []
        with result_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(_flatten_record(json.loads(line)))
                except Exception:
                    continue
        if rows:
            return pd.DataFrame(rows)
    return None


def _load_metrics_jsonl(trial_dir: Path) -> pd.DataFrame | None:
    candidates = [trial_dir / "metrics_full.jsonl", trial_dir / "rllib" / "metrics_full.jsonl"]
    for metrics_path in candidates:
        if not metrics_path.exists() or metrics_path.stat().st_size == 0:
            continue
        rows = []
        with metrics_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        if rows:
            return pd.DataFrame(rows)
    return None


def load_trial_df(trial_dir: Path) -> pd.DataFrame:
    df = _load_progress_csv(trial_dir)
    if df is not None:
        return df
    df = _load_metrics_jsonl(trial_dir)
    if df is not None:
        return df
    df = _load_result_json(trial_dir)
    if df is not None:
        return df
    raise SystemExit(f"No usable progress.csv/result.json/metrics_full.jsonl in: {trial_dir}")


def pick_x_axis(df: pd.DataFrame) -> str | None:
    """
    Pick a robust x-axis column. Falls back to index when none are present.
    """
    for c in ("training_iteration", "iteration", "iterations_since_restore", "time_total_s"):
        if c in df.columns:
            return c
    return None


def _plot_series(df: pd.DataFrame, ax, y_cols, xcol: str | None, title: str):
    if not y_cols:
        return False
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    y_cols = [c for c in y_cols if c in df.columns]
    if not y_cols:
        return False
    if xcol:
        df.plot(x=xcol, y=y_cols, ax=ax)
    else:
        df[y_cols].plot(ax=ax)
    ax.set_title(title)
    return True


def _hide_axis_if_empty(plotted: bool, ax, title: str):
    if plotted:
        return
    ax.set_title(f"{title} (no data)")
    ax.axis("off")


def plot_original_metrics(df: pd.DataFrame, out: Path, xcol: str | None):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    col_list = list(df.columns)
    col_set = set(col_list)

    # Episode return (custom) or built-in reward
    y = "custom_metrics/episode_return_mean" if "custom_metrics/episode_return_mean" in col_set else None
    if not y:
        y = "custom_metrics/episode_return_mean_mean" if "custom_metrics/episode_return_mean_mean" in col_set else None
    if not y:
        y = "episode_reward_mean" if "episode_reward_mean" in col_set else None
    _hide_axis_if_empty(_plot_series(df, axes[0], y, xcol, "Episode Return / Reward"), axes[0], "Episode Return / Reward")

    # Entropy (or episode length)
    ent_cols = [c for c in col_list if c.lower().endswith("entropy")]
    if ent_cols:
        _hide_axis_if_empty(_plot_series(df, axes[1], ent_cols[0], xcol, "Policy Entropy"), axes[1], "Policy Entropy")
    else:
        _hide_axis_if_empty(_plot_series(df, axes[1], "episode_len_mean", xcol, "Episode Length Mean"), axes[1], "Episode Length Mean")

    # Losses
    loss_cols = [c for c in col_list if any(t in c.lower() for t in ["policy_loss", "vf_loss", "total_loss", "loss"])]
    _hide_axis_if_empty(_plot_series(df, axes[2], loss_cols[:2] or loss_cols, xcol, "Losses"), axes[2], "Losses")

    # Action fractions
    act_cols = [c for c in col_list if c.startswith("custom_metrics/act_") and c.endswith("_mean")]
    _hide_axis_if_empty(_plot_series(df, axes[3], act_cols, xcol, "Action Fractions"), axes[3], "Action Fractions")

    plt.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fp = out / "original_metrics.png"
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fp}")


def plot_enhanced_metrics(df: pd.DataFrame, out: Path, xcol: str | None):
    fig, axes = plt.subplots(5, 1, figsize=(12, 15))
    col_list = list(df.columns)

    _hide_axis_if_empty(_plot_series(df, axes[0], [
        "custom_metrics/episode_return_mean",
        "custom_metrics/episode_return_mean_mean",
        "episode_reward_mean",
    ], xcol, "Episode Return / Reward"), axes[0], "Episode Return / Reward")

    _hide_axis_if_empty(_plot_series(df, axes[1], "episode_len_mean", xcol, "Episode Length Mean"), axes[1], "Episode Length Mean")

    act_cols = [c for c in col_list if c.startswith("custom_metrics/act_") and c.endswith("_mean")]
    _hide_axis_if_empty(_plot_series(df, axes[3], act_cols, xcol, "Action Fractions"), axes[3], "Action Fractions")

    rc_cols = [c for c in col_list if c.startswith("custom_metrics/rc_") and c.endswith("_mean")]
    _hide_axis_if_empty(
        _plot_series(df, axes[4], rc_cols, xcol, "Reward Components (PnL, Drawdown, Turnover)"),
        axes[4],
        "Reward Components (PnL, Drawdown, Turnover)",
    )

    # Entropy moved to a dedicated pane for readability
    ent_cols = [c for c in col_list if c.lower().endswith("entropy")]
    _hide_axis_if_empty(_plot_series(df, axes[2], ent_cols[:1], xcol, "Policy Entropy"), axes[2], "Policy Entropy")

    plt.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fp = out / "enhanced_metrics.png"
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fp}")


def plot_task_sampler_metrics(df: pd.DataFrame, out: Path, xcol: str | None):
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    col_list = list(df.columns)

    _hide_axis_if_empty(
        _plot_series(df, axes[0], ["custom_metrics/success_mean", "custom_metrics/success_mean_mean"], xcol, "Success Rate"),
        axes[0],
        "Success Rate",
    )
    _hide_axis_if_empty(
        _plot_series(df, axes[1], ["custom_metrics/task_horizon_mean", "custom_metrics/task_horizon_mean_mean"], xcol, "Task Horizon Mean"),
        axes[1],
        "Task Horizon Mean",
    )

    rc_cols = [c for c in col_list if c.startswith("custom_metrics/rc_") and c.endswith("_mean")]
    _hide_axis_if_empty(
        _plot_series(df, axes[2], rc_cols, xcol, "Reward Components (PnL, Drawdown, Turnover)"),
        axes[2],
        "Reward Components (PnL, Drawdown, Turnover)",
    )

    _hide_axis_if_empty(
        _plot_series(df, axes[3], "num_env_steps_trained_throughput_per_sec", xcol, "Training Throughput (steps/sec)"),
        axes[3],
        "Training Throughput (steps/sec)",
    )

    plt.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fp = out / "task_sampler_metrics.png"
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fp}")


def main():
    ap = argparse.ArgumentParser(description="Plot Ray Tune results")
    ap.add_argument("trial_dir", help="Path to Ray Tune trial directory")
    ap.add_argument("--out", help="Output directory (optional)")
    args = ap.parse_args()

    trial_dir = Path(args.trial_dir)
    df = load_trial_df(trial_dir)
    xcol = pick_x_axis(df)

    out = Path(args.out) if args.out else (trial_dir / "plots")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Creating plots in: {out}")
    plot_original_metrics(df, out, xcol)
    plot_enhanced_metrics(df, out, xcol)
    plot_task_sampler_metrics(df, out, xcol)

    summary_path = out / "plot_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Plot Generation Summary\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Source Trial: {trial_dir}\n")
        # Use row count as a fallback when there's no training_iteration column
        total_iters = df["training_iteration"].iloc[-1] if "training_iteration" in df.columns else len(df)
        f.write(f"Total Iterations: {total_iters}\n")
        f.write(f"Columns ({len(df.columns)}):\n")
        for col in df.columns:
            f.write(f"  - {col}\n")

    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
