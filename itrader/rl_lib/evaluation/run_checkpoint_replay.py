"""Deterministic per-episode replay of a retained Phase A PBT checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def load_validation_env(resolved_config: Path) -> dict:
    payload = json.loads(resolved_config.read_text(encoding="utf-8"))
    env_config = payload["config"]["evaluation_config"]["env_config"]
    if env_config.get("data_split") != "validation":
        raise ValueError("Checkpoint replay requires a validation split; final test stays sealed")
    if not env_config.get("task_sampler", {}).get("fixed_tasks"):
        raise ValueError("Checkpoint replay requires a frozen evaluation task panel")
    dataset = Path(env_config["df_path"])
    sha256 = hashlib.sha256()
    with dataset.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)
    if sha256.hexdigest() != payload["dataset_sha256"]:
        raise ValueError(f"Dataset hash differs from checkpoint provenance: {dataset}")
    return env_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-trial-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    stable_dir = args.stable_trial_dir.resolve(strict=True)
    env_config = load_validation_env(stable_dir / "resolved_config.json")
    selection = json.loads((stable_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    checkpoint = Path(selection["stable_checkpoint"]).resolve(strict=True)

    import ray
    from ray.rllib.algorithms.algorithm import Algorithm

    from itrader.envs.itrader_env import ITraderEnv
    from itrader.rl_lib.evaluation.policy_baselines import (
        FrozenCheckpointPolicy,
        evaluate_policy_control,
        summarize_control_episodes,
    )
    from itrader.rl_lib.train.train_solo_V2 import _register_models

    ray.init(include_dashboard=False)
    algo = None
    try:
        _register_models()
        algo = Algorithm.from_checkpoint(str(checkpoint))
        env = ITraderEnv(env_config)
        episodes = evaluate_policy_control(
            env,
            FrozenCheckpointPolicy(algo),
            episode_count=len(env_config["task_sampler"]["fixed_tasks"]),
        )
        summary = summarize_control_episodes(episodes)
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()

    expected = float(selection["score"])
    observed = float(summary["evaluation/net_return_mean"])
    delta = observed - expected
    report = {
        "schema_version": 1,
        "trial": stable_dir.name,
        "checkpoint": str(checkpoint),
        "selection_metric": selection["metric"],
        "selected_score": expected,
        "replayed_score": observed,
        "replay_delta": delta,
        "checkpoint_parity": math.isfinite(delta) and abs(delta) <= 1e-8,
        "episode_count": len(episodes),
        "summary": summary,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = stable_dir.name
    (args.out_dir / f"{stem}.episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8"
    )
    (args.out_dir / f"{stem}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["checkpoint_parity"]:
        raise RuntimeError(f"Checkpoint replay did not reproduce its selected score: {delta}")


if __name__ == "__main__":
    main()
