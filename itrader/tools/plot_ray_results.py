from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def pick_x_axis(df: pd.DataFrame) -> str | None:
    for c in ("training_iteration", "iteration", "iterations_since_restore", "time_total_s"):
        if c in df.columns:
            return c
    return None  # will fallback to index


def plot_original_metrics(df: pd.DataFrame, out: Path, xcol: str | None):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))

    # Episode Return Mean
    y = "custom_metrics/episode_return_mean" if "custom_metrics/episode_return_mean" in df.columns else \
        ("episode_reward_mean" if "episode_reward_mean" in df.columns else None)
    if y:
        df.plot(x=xcol, y=y, ax=axes[0]) if xcol else df[y].plot(ax=axes[0])
        axes[0].set_title("Episode Return Mean" if "custom_metrics" in y else "Episode Reward Mean")

    # Entropy (or episode length)
    ent_cols = [c for c in df.columns if "entropy" in c.lower()]
    if ent_cols:
        df.plot(x=xcol, y=ent_cols[0], ax=axes[1]) if xcol else df[ent_cols[0]].plot(ax=axes[1])
        axes[1].set_title("Policy Entropy")
    elif "episode_len_mean" in df.columns:
        df.plot(x=xcol, y="episode_len_mean", ax=axes[1]) if xcol else df["episode_len_mean"].plot(ax=axes[1])
        axes[1].set_title("Episode Length Mean")

    # Losses
    loss_cols = [c for c in df.columns if any(t in c.lower() for t in ["policy_loss", "vf_loss", "total_loss", "loss"])]
    if loss_cols:
        cols = loss_cols[:2] if len(loss_cols) >= 2 else loss_cols
        df.plot(x=xcol, y=cols, ax=axes[2]) if xcol else df[cols].plot(ax=axes[2])
        axes[2].set_title("Losses")

    # Action fractions
    act_cols = [c for c in df.columns if c.startswith("custom_metrics/act_") and c.endswith("_mean")]
    if act_cols:
        df.plot(x=xcol, y=act_cols, ax=axes[3]) if xcol else df[act_cols].plot(ax=axes[3])
        axes[3].set_title("Action Fractions")

    plt.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fp = out / "original_metrics.png"
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fp}")


def plot_enhanced_metrics(df: pd.DataFrame, out: Path, xcol: str | None):
    fig, axes = plt.subplots(5, 1, figsize=(12, 15))

    # Episode Return & Reward
    if "custom_metrics/episode_return_mean" in df.columns:
        df.plot(x=xcol, y="custom_metrics/episode_return_mean", ax=axes[0]) if xcol else \
            df["custom_metrics/episode_return_mean"].plot(ax=axes[0])
        axes[0].set_title("Episode Return Mean")
    if "episode_reward_mean" in df.columns:
        df.plot(x=xcol, y="episode_reward_mean", ax=axes[1]) if xcol else df["episode_reward_mean"].plot(ax=axes[1])
        axes[1].set_title("Episode Reward Mean")

    # Episode length
    if "episode_len_mean" in df.columns:
        df.plot(x=xcol, y="episode_len_mean", ax=axes[2]) if xcol else df["episode_len_mean"].plot(ax=axes[2])
        axes[2].set_title("Episode Length Mean")

    # Action frac
    act_cols = [c for c in df.columns if c.startswith("custom_metrics/act_") and c.endswith("_mean")]
    if act_cols:
        df.plot(x=xcol, y=act_cols, ax=axes[3]) if xcol else df[act_cols].plot(ax=axes[3])
        axes[3].set_title("Action Fractions")

    # Reward components
    rc_cols = [c for c in df.columns if c.startswith("custom_metrics/rc_") and c.endswith("_mean")]
    if rc_cols:
        df.plot(x=xcol, y=rc_cols, ax=axes[4]) if xcol else df[rc_cols].plot(ax=axes[4])
        axes[4].set_title("Reward Components (PnL, Drawdown, Turnover)")

    plt.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fp = out / "enhanced_metrics.png"
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fp}")


def plot_task_sampler_metrics(df: pd.DataFrame, out: Path, xcol: str | None):
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    if "custom_metrics/success_mean" in df.columns:
        df.plot(x=xcol, y="custom_metrics/success_mean", ax=axes[0]) if xcol else \
            df["custom_metrics/success_mean"].plot(ax=axes[0])
        axes[0].set_title("Success Rate")

    if "custom_metrics/task_horizon_mean" in df.columns:
        df.plot(x=xcol, y="custom_metrics/task_horizon_mean", ax=axes[1]) if xcol else \
            df["custom_metrics/task_horizon_mean"].plot(ax=axes[1])
        axes[1].set_title("Task Horizon Mean")

    rc_cols = [c for c in df.columns if c.startswith("custom_metrics/rc_") and c.endswith("_mean")]
    if rc_cols:
        df.plot(x=xcol, y=rc_cols, ax=axes[2]) if xcol else df[rc_cols].plot(ax=axes[2])
        axes[2].set_title("Reward Components (PnL, Drawdown, Turnover)")

    if "num_env_steps_trained_throughput_per_sec" in df.columns:
        df.plot(x=xcol, y="num_env_steps_trained_throughput_per_sec", ax=axes[3]) if xcol else \
            df["num_env_steps_trained_throughput_per_sec"].plot(ax=axes[3])
        axes[3].set_title("Training Throughput (steps/sec)")

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

    df = pd.read_csv(csv_path)
    xcol = pick_x_axis(df)

    out = Path(args.out) if args.out else (Path("plots") / trial_dir.name)
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
        f.write(f"Total Iterations: {df.get('training_iteration', pd.Series([len(df)])).iloc[-1]}\n")
        f.write(f"Columns ({len(df.columns)}):\n")
        for col in df.columns:
            f.write(f"  - {col}\n")

    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()