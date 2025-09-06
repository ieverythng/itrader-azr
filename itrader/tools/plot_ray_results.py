from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

def create_output_dir(trial_dir):
    """Create output directory with trial name and timestamp"""
    trial_name = Path(trial_dir).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("plots") / f"{trial_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def plot_original_metrics(df, output_dir):
    """Plot the original metrics that were intended"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    
    # Try to find the best available columns for original metrics
    # Episode Return Mean
    if "custom_metrics/episode_return_mean" in df.columns:
        ax = axes[0]; df.plot(x="training_iteration", y="custom_metrics/episode_return_mean", ax=ax); ax.set_title("Episode Return Mean")
    elif "episode_reward_mean" in df.columns:
        ax = axes[0]; df.plot(x="training_iteration", y="episode_reward_mean", ax=ax); ax.set_title("Episode Reward Mean (Fallback)")
    
    # Policy Entropy (if available)
    entropy_cols = [c for c in df.columns if "entropy" in c.lower()]
    if entropy_cols:
        ax = axes[1]; df.plot(x="training_iteration", y=entropy_cols[0], ax=ax); ax.set_title("Policy Entropy")
    else:
        # Fallback to episode length
        ax = axes[1]; df.plot(x="training_iteration", y="episode_len_mean", ax=ax); ax.set_title("Episode Length Mean")
    
    # Losses (if available)
    loss_cols = [c for c in df.columns if any(loss in c.lower() for loss in ["policy_loss", "value_loss", "loss"])]
    if loss_cols:
        ax = axes[2]; df.plot(x="training_iteration", y=loss_cols[:2], ax=ax); ax.set_title("Losses")
    else:
        # Fallback to throughput metrics
        throughput_cols = [c for c in df.columns if "throughput" in c.lower()]
        if throughput_cols:
            ax = axes[2]; df.plot(x="training_iteration", y=throughput_cols[0], ax=ax); ax.set_title("Training Throughput")
        else:
            ax = axes[2]; df.plot(x="training_iteration", y="timesteps_total", ax=ax); ax.set_title("Total Timesteps")
    
    # Action Fractions
    cols = [c for c in df.columns if c.startswith("custom_metrics/act_") and c.endswith("_mean")]
    ax = axes[3]; df.plot(x="training_iteration", y=cols, ax=ax); ax.set_title("Action Fractions")
    
    plt.tight_layout()
    output_path = output_dir / "original_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved original metrics: {output_path}")

def plot_enhanced_metrics(df, output_dir):
    """Plot enhanced metrics including reward components"""
    fig, axes = plt.subplots(5, 1, figsize=(12, 15))
    
    # Episode Return Mean
    ax = axes[0]; df.plot(x="training_iteration", y="custom_metrics/episode_return_mean", ax=ax); ax.set_title("Episode Return Mean")
    
    # Episode Reward Mean
    ax = axes[1]; df.plot(x="training_iteration", y="episode_reward_mean", ax=ax); ax.set_title("Episode Reward Mean")
    
    # Episode Length Mean
    ax = axes[2]; df.plot(x="training_iteration", y="episode_len_mean", ax=ax); ax.set_title("Episode Length Mean")
    
    # Action Fractions
    cols = [c for c in df.columns if c.startswith("custom_metrics/act_") and c.endswith("_mean")]
    ax = axes[3]; df.plot(x="training_iteration", y=cols, ax=ax); ax.set_title("Action Fractions")
    
    # Reward Components (PnL, Drawdown, Turnover)
    reward_cols = [c for c in df.columns if c.startswith("custom_metrics/rc_") and c.endswith("_mean")]
    ax = axes[4]; df.plot(x="training_iteration", y=reward_cols, ax=ax); ax.set_title("Reward Components (PnL, Drawdown, Turnover)")
    
    plt.tight_layout()
    output_path = output_dir / "enhanced_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved enhanced metrics: {output_path}")

def plot_task_sampler_metrics(df, output_dir):
    """Plot metrics relevant for TaskSampler/Proposer analysis"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    
    # Success rate
    if "custom_metrics/success_mean" in df.columns:
        ax = axes[0]; df.plot(x="training_iteration", y="custom_metrics/success_mean", ax=ax); ax.set_title("Success Rate")
    
    # Task horizon distribution
    if "custom_metrics/task_horizon_mean" in df.columns:
        ax = axes[1]; df.plot(x="training_iteration", y="custom_metrics/task_horizon_mean", ax=ax); ax.set_title("Task Horizon Mean")
    
    # Reward components breakdown
    reward_cols = [c for c in df.columns if c.startswith("custom_metrics/rc_") and c.endswith("_mean")]
    ax = axes[2]; df.plot(x="training_iteration", y=reward_cols, ax=ax); ax.set_title("Reward Components (PnL, Drawdown, Turnover)")
    
    # Training efficiency
    if "num_env_steps_trained_throughput_per_sec" in df.columns:
        ax = axes[3]; df.plot(x="training_iteration", y="num_env_steps_trained_throughput_per_sec", ax=ax); ax.set_title("Training Throughput (steps/sec)")
    
    plt.tight_layout()
    output_path = output_dir / "task_sampler_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved task sampler metrics: {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot Ray Tune results with organized output")
    ap.add_argument("trial_dir", help="Path to Ray Tune trial directory")
    ap.add_argument("--out", help="Output filename (optional, will use auto-generated folder structure)")
    args = ap.parse_args()
    
    # Load data
    df = pd.read_csv(Path(args.trial_dir)/"progress.csv")
    
    # Create output directory
    output_dir = create_output_dir(args.trial_dir)
    print(f"Creating plots in: {output_dir}")
    
    # Generate all plot types
    plot_original_metrics(df, output_dir)
    plot_enhanced_metrics(df, output_dir)
    plot_task_sampler_metrics(df, output_dir)
    
    # Save a summary CSV for reference
    summary_path = output_dir / "plot_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Plot Generation Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source Trial: {args.trial_dir}\n")
        f.write(f"Total Iterations: {df['training_iteration'].max()}\n")
        f.write(f"Available Columns: {len(df.columns)}\n")
        f.write(f"\nKey Metrics Available:\n")
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['return', 'reward', 'success', 'entropy', 'loss']):
                f.write(f"  - {col}\n")
    
    print(f"Saved summary: {summary_path}")
    print(f"All plots saved in: {output_dir}")