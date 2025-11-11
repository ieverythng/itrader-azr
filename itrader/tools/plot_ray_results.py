from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

# headless / non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd


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
        return
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    y_cols = [c for c in y_cols if c in df.columns]
    if not y_cols:
        return
    if xcol:
        df.plot(x=xcol, y=y_cols, ax=ax)
    else:
        df[y_cols].plot(ax=ax)
    ax.set_title(title)


def plot_original_metrics(df: pd.DataFrame, out: Path, xcol: str | None):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))

    # Episode return (custom) or built-in reward
    y = "custom_metrics/episode_return_mean" if "custom_metrics/episode_return_mean" in df.columns else None
    if not y:
        y = "custom_metrics/episode_return_mean_mean" if "custom_metrics/episode_return_mean_mean" in df.columns else None
    if not y:
        y = "episode_reward_mean" if "episode_reward_mean" in df.columns else None
    _plot_series(df, axes[0], y, xcol, "Episode Return / Reward")

    # Entropy (or episode length)
    ent_cols = [c for c in df.columns if c.lower().endswith("entropy")]
    if ent_cols:
        _plot_series(df, axes[1], ent_cols[0], xcol, "Policy Entropy")
    else:
        _plot_series(df, axes[1], "episode_len_mean", xcol, "Episode Length Mean")

    # Losses
    loss_cols = [c for c in df.columns if any(t in c.lower() for t in ["policy_loss", "vf_loss", "total_loss", "loss"])]
    _plot_series(df, axes[2], loss_cols[:2] or loss_cols, xcol, "Losses")

    # Action fractions
    act_cols = [c for c in df.columns if c.startswith("custom_metrics/act_") and c.endswith("_mean")]
    _plot_series(df, axes[3], act_cols, xcol, "Action Fractions")

    plt.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fp = out / "original_metrics.png"
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fp}")


def plot_enhanced_metrics(df: pd.DataFrame, out: Path, xcol: str | None):
    fig, axes = plt.subplots(5, 1, figsize=(12, 15))

    _plot_series(df, axes[0], [
        "custom_metrics/episode_return_mean",
        "custom_metrics/episode_return_mean_mean",
        "episode_reward_mean",
    ], xcol, "Episode Return / Reward")

    _plot_series(df, axes[1], "episode_len_mean", xcol, "Episode Length Mean")

    act_cols = [c for c in df.columns if c.startswith("custom_metrics/act_") and c.endswith("_mean")]
    _plot_series(df, axes[3], act_cols, xcol, "Action Fractions")

    rc_cols = [c for c in df.columns if c.startswith("custom_metrics/rc_") and c.endswith("_mean")]
    _plot_series(df, axes[4], rc_cols, xcol, "Reward Components (PnL, Drawdown, Turnover)")

    # Entropy moved to a dedicated pane for readability
    ent_cols = [c for c in df.columns if c.lower().endswith("entropy")]
    _plot_series(df, axes[2], ent_cols[:1], xcol, "Policy Entropy")

    plt.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fp = out / "enhanced_metrics.png"
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fp}")


def plot_task_sampler_metrics(df: pd.DataFrame, out: Path, xcol: str | None):
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    _plot_series(df, axes[0], "custom_metrics/success_mean", xcol, "Success Rate")
    _plot_series(df, axes[1], "custom_metrics/task_horizon_mean", xcol, "Task Horizon Mean")

    rc_cols = [c for c in df.columns if c.startswith("custom_metrics/rc_") and c.endswith("_mean")]
    _plot_series(df, axes[2], rc_cols, xcol, "Reward Components (PnL, Drawdown, Turnover)")

    _plot_series(df, axes[3], "num_env_steps_trained_throughput_per_sec", xcol, "Training Throughput (steps/sec)")

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
    csv_path = trial_dir / "progress.csv"
    if not csv_path.exists():
        raise SystemExit(f"progress.csv not found at: {csv_path}")

    # Be tolerant of partially-written CSVs; Pandas will handle.
    df = pd.read_csv(csv_path, encoding="utf-8")
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
