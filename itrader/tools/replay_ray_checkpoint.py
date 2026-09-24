from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import ray
from ray.rllib.algorithms.algorithm import Algorithm

from itrader.rl_lib.train.train_solo_V2 import (
    _atomic_write_json,
    _extract_custom_metrics,
    _register_models,
    deterministic_evaluate,
)


METRICS = {
    "optimization_reward": "evaluation/optimization_reward_mean",
    "economic_gross_return": "evaluation/gross_return_mean",
    "economic_net_return": "evaluation/net_return_mean",
    "economic_net_sharpe_per_sqrt_bar": "evaluation/net_sharpe_per_sqrt_bar_mean",
}


def _checkpoint_record(stable_trial_dir: Path, iteration: int | None) -> dict:
    index_path = stable_trial_dir / "checkpoint_index.jsonl"
    records = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if iteration is not None:
        records = [
            record
            for record in records
            if int(record["training_iteration"]) == int(iteration)
        ]
    if not records:
        requested = "latest" if iteration is None else str(iteration)
        raise ValueError(f"No checkpoint record found for iteration {requested}")
    return max(records, key=lambda record: int(record["training_iteration"]))


def _report_row(stable_trial_dir: Path, iteration: int) -> dict:
    with (stable_trial_dir / "report_rows.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if int(row["training_iteration"]) == int(iteration)
        ]
    if not matches:
        raise ValueError(f"No report row found for iteration {iteration}")
    return matches[-1]


def _metric_value(custom_metrics: dict, name: str) -> float:
    for key in (f"{name}_mean", name):
        value = custom_metrics.get(key)
        if value is not None:
            return float(value)
    raise KeyError(f"Evaluation did not emit {name!r}")


def replay(stable_trial_dir: Path, iteration: int | None, tolerance: float) -> dict:
    checkpoint_record = _checkpoint_record(stable_trial_dir, iteration)
    checkpoint_iteration = int(checkpoint_record["training_iteration"])
    checkpoint_path = Path(checkpoint_record["stable_checkpoint"]).resolve()
    expected_row = _report_row(stable_trial_dir, checkpoint_iteration)

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Stable checkpoint is missing: {checkpoint_path}")

    _register_models()
    ray.init(include_dashboard=False)
    algorithm = None
    try:
        algorithm = Algorithm.from_checkpoint(str(checkpoint_path))
        evaluation = deterministic_evaluate(algorithm)
        custom_metrics = _extract_custom_metrics(evaluation)
        comparisons = {}
        for custom_name, report_name in METRICS.items():
            expected = float(expected_row[report_name])
            observed = _metric_value(custom_metrics, custom_name)
            delta = abs(observed - expected)
            comparisons[report_name] = {
                "expected": expected,
                "observed": observed,
                "absolute_delta": delta,
                "within_tolerance": math.isfinite(delta) and delta <= tolerance,
            }
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()

    passed = all(item["within_tolerance"] for item in comparisons.values())
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "fresh_process": True,
        "stable_trial_dir": str(stable_trial_dir.resolve()),
        "checkpoint_path": str(checkpoint_path),
        "training_iteration": checkpoint_iteration,
        "tolerance": tolerance,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay an immutable iTrader RLlib checkpoint and compare held-out metrics."
    )
    parser.add_argument("stable_trial_dir", type=Path)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    receipt = replay(args.stable_trial_dir, args.iteration, args.tolerance)
    output_path = args.output or args.stable_trial_dir / "fresh_process_replay.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_path, receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
