"""State-conditioned replay diagnostics for a retained R4-v1 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from itrader.rl_lib.evaluation.run_checkpoint_replay import (
    load_replay_inputs,
    replay_contract,
    replay_execution_settings,
)
from itrader.rl_lib.evaluation.replay_evidence import ReplayInvocation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-trial-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("best", "terminal"),
        default="best",
    )
    parser.add_argument(
        "--checkpoint-progress-steps",
        type=int,
        help="Diagnose the unique retained checkpoint at these target-owned steps.",
    )
    parser.add_argument(
        "--evaluation-task-panel-path",
        type=Path,
        help="Use a cardinality-matched, non-overlapping validation panel.",
    )
    args = parser.parse_args()

    stable_dir = args.stable_trial_dir.resolve(strict=True)
    invocation = ReplayInvocation(args.out_dir / f"{stable_dir.name}.state_diagnostics.json", operation="state_diagnostics")
    selection, checkpoint, resolved_config, validation, digest = load_replay_inputs(
        stable_dir,
        args.checkpoint_selection,
        progress_steps=args.checkpoint_progress_steps,
        evaluation_task_panel_path=args.evaluation_task_panel_path,
    )
    env_config = validation.env_config

    import ray
    from ray.rllib.algorithms.algorithm import Algorithm

    from itrader.envs.itrader_env import ITraderEnv
    from itrader.rl_lib.evaluation.policy_baselines import (
        FrozenCheckpointPolicy,
        evaluate_checkpoint_state_diagnostics,
        summarize_counterfactual_state_steps,
        summarize_state_conditioned_steps,
    )
    ray.init(include_dashboard=False)
    algorithm = None
    try:
        algorithm = Algorithm.from_checkpoint(str(checkpoint))
        from itrader.models.registry import validate_native_restored_model

        validate_native_restored_model(algorithm, resolved_config["config"])
        execution = replay_execution_settings(algorithm)
        env = ITraderEnv(env_config)
        rows = evaluate_checkpoint_state_diagnostics(
            env,
            FrozenCheckpointPolicy(algorithm),
            episode_count=len(env_config["task_sampler"]["fixed_tasks"]),
        )
        env.close()
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()

    report = {
        "schema_version": 3,
        "trial": stable_dir.name,
        "checkpoint": str(checkpoint),
        "checkpoint_selection": (
            f"progress:{args.checkpoint_progress_steps}"
            if args.checkpoint_progress_steps is not None
            else args.checkpoint_selection
        ),
        "diagnostic_only": validation.diagnostic_only,
        "evaluation_panel_path": validation.evaluation_panel_path,
        "source_panel_sha256": validation.source_panel_sha256,
        "evaluation_panel_sha256": validation.evaluation_panel_sha256,
        "solver_state_mode": env_config["solver_state_mode"],
        "observation_schema": resolved_config["observation_schema"],
        "episode_count": len(env_config["task_sampler"]["fixed_tasks"]),
        "diagnostics": summarize_state_conditioned_steps(rows),
        "counterfactual_state_sensitivity": summarize_counterfactual_state_steps(rows),
    }
    stem = stable_dir.name
    artifact = args.out_dir / f"{stem}.state_steps.jsonl"
    with artifact.open("x", encoding="utf-8") as stream:
        stream.write("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    report = invocation.complete(
        report, contract=replay_contract(selection, resolved_config, validation, digest, execution), artifacts=[artifact],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
