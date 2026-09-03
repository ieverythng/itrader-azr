from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_COLUMNS = [
    "training_iteration",
    "episode_return_mean",
    "entropy",
    "policy_loss",
    "value_loss",
    "custom_metrics/success_mean",
]


def _selected_columns(df: pd.DataFrame) -> list[str]:
    action_cols = sorted(
        col for col in df.columns if col.startswith("custom_metrics/act_") and "_mean" in col
    )
    reward_cols = sorted(
        col for col in df.columns if col.startswith("custom_metrics/rc_") and "_mean" in col
    )
    return [col for col in BASE_COLUMNS if col in df.columns] + action_cols + reward_cols


def last_trial_dir(root: Path) -> Path:
    trials = sorted((path for path in root.glob("*/") if path.is_dir()), key=lambda path: path.stat().st_mtime)
    if not trials:
        raise SystemExit("No trials found.")
    return trials[-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ray_dir", help="path to ray_results/Phase_A or similar")
    args = ap.parse_args()
    root = Path(args.ray_dir)
    trial = last_trial_dir(root)
    csv = trial / "progress.csv"
    df = pd.read_csv(csv)
    df = df[_selected_columns(df)].tail(30)
    print("\n== Last 30 iters ==\n")
    print(df.to_string(index=False))
    print("\nQuick hints:")
    ent = df["entropy"].iloc[-1] if "entropy" in df else None
    if ent is not None and ent < 0.05:
        print("- Entropy is ~0: policy collapsed. Increase entropy_coeff or check obs scaling.")
    er = df["episode_return_mean"].iloc[-1] if "episode_return_mean" in df else None
    if er is not None and er < 0:
        print("- Negative returns: try zeroing penalties/fees briefly to sanity-check learnability.")
