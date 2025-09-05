from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trial_dir")
    ap.add_argument("--out", default="plots.png")
    args = ap.parse_args()
    df = pd.read_csv(Path(args.trial_dir)/"progress.csv")
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    ax = axes[0]; df.plot(x="training_iteration", y="episode_return_mean", ax=ax); ax.set_title("Episode Return Mean")
    ax = axes[1]; df.plot(x="training_iteration", y="entropy", ax=ax); ax.set_title("Policy Entropy")
    ax = axes[2]; df.plot(x="training_iteration", y=["policy_loss","value_loss"], ax=ax); ax.set_title("Losses")
    cols = [c for c in df.columns if c.startswith("custom_metrics/act_") and c.endswith("_mean")]
    ax = axes[3]; df.plot(x="training_iteration", y=cols, ax=ax); ax.set_title("Action Fractions")
    plt.tight_layout(); plt.savefig(args.out); print("Saved:", args.out)