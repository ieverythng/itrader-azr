"""Deterministic per-episode replay of a retained Phase A PBT checkpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

from itrader.rl_lib.api_stack import NATIVE_API_STACK_ID
from itrader.rl_lib.evaluation.provenance import (
    canonical_json_sha256,
    dataset_sha256,
    verify_checkpoint_digest,
    source_tree_fingerprint,
    runtime_identity,
    require_indexed_integer,
)
from itrader.rl_lib.evaluation.replay_evidence import ReplayInvocation, replay_parity
from itrader.rl_lib.tasks.spec import TaskSpec
from itrader.rl_lib.train.pbt_contract import (
    economic_search_score,
    economic_selection_score,
)


def _replay_metric_value(metric: str, summary: dict, config: dict) -> float:
    if metric == "evaluation/net_return_mean":
        return float(summary[metric])
    if metric == "evaluation/economic_selection_score":
        return economic_selection_score(
            net_return=float(summary["evaluation/net_return_mean"]),
            net_sharpe=float(summary["evaluation/net_sharpe_per_sqrt_bar_mean"]),
            flat_action_fraction=float(
                summary["evaluation/custom_metrics/act_flat_frac_mean"]
            ),
            min_net_sharpe=float(config.get("pbt_selection_min_sharpe", 0.0)),
            max_flat_action_fraction=float(
                config.get("pbt_selection_max_flat_action_fraction", 0.98)
            ),
        )
    if metric == "evaluation/economic_search_score":
        return economic_search_score(
            net_return=float(summary["evaluation/net_return_mean"]),
            flat_action_fraction=float(
                summary["evaluation/custom_metrics/act_flat_frac_mean"]
            ),
            max_flat_action_fraction=float(
                config.get("pbt_selection_max_flat_action_fraction", 0.98)
            ),
        )
    raise ValueError(f"Unsupported checkpoint selection metric: {metric}")


@dataclass(frozen=True)
class ValidationReplayContext:
    env_config: dict
    diagnostic_only: bool
    evaluation_panel_path: str | None
    source_panel_sha256: str
    evaluation_panel_sha256: str


def _load_panel_tasks(path: Path, *, expected_size: int) -> list[dict]:
    panel_path = path.expanduser().resolve(strict=True)
    try:
        payload = json.loads(panel_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid evaluation task panel JSON: {panel_path}") from exc
    tasks = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(tasks, list):
        raise ValueError("evaluation task panel must be a list or an object with tasks")
    if len(tasks) != expected_size:
        raise ValueError(
            "diagnostic evaluation panel must preserve source panel cardinality: "
            f"{len(tasks)} != {expected_size}"
        )
    normalized = [TaskSpec.from_dict(task).to_dict() for task in tasks]
    if any(task.get("start_idx") is None for task in normalized):
        raise ValueError("diagnostic evaluation tasks require explicit start_idx values")
    return normalized


def _task_interval(task: dict, *, window_size: int) -> tuple[int, int]:
    start = int(task["start_idx"])
    return start - int(window_size), start + int(task["horizon"])


def _assert_panels_do_not_overlap(
    source_tasks: list[dict],
    diagnostic_tasks: list[dict],
    *,
    window_size: int,
) -> None:
    source_intervals = [
        _task_interval(task, window_size=window_size) for task in source_tasks
    ]
    for diagnostic in diagnostic_tasks:
        diagnostic_interval = _task_interval(diagnostic, window_size=window_size)
        if any(
            diagnostic_interval[0] <= source_end
            and source_start <= diagnostic_interval[1]
            for source_start, source_end in source_intervals
        ):
            raise ValueError(
                "Diagnostic evaluation panel overlaps the source selection panel: "
                f"start_idx={diagnostic['start_idx']}, horizon={diagnostic['horizon']}"
            )


def load_validation_context(
    resolved_config: Path,
    *,
    evaluation_task_panel_path: Path | None = None,
) -> ValidationReplayContext:
    payload = json.loads(resolved_config.read_text(encoding="utf-8"))
    env_config = deepcopy(payload["config"]["evaluation_config"]["env_config"])
    if env_config.get("data_split") != "validation":
        raise ValueError("Checkpoint replay requires a validation split; final test stays sealed")
    source_tasks = env_config.get("task_sampler", {}).get("fixed_tasks")
    if not source_tasks:
        raise ValueError("Checkpoint replay requires a frozen evaluation task panel")
    dataset = Path(env_config["df_path"])
    if dataset_sha256(dataset) != payload["dataset_sha256"]:
        raise ValueError(f"Dataset hash differs from checkpoint provenance: {dataset}")
    source_panel_sha256 = canonical_json_sha256(source_tasks)
    panel_path = None
    diagnostic_only = evaluation_task_panel_path is not None
    if diagnostic_only:
        panel_path = str(evaluation_task_panel_path.expanduser().resolve(strict=True))
        diagnostic_tasks = _load_panel_tasks(
            Path(panel_path),
            expected_size=len(source_tasks),
        )
        _assert_panels_do_not_overlap(
            source_tasks,
            diagnostic_tasks,
            window_size=int(env_config["window_size"]),
        )
        env_config["task_sampler"] = {
            **env_config["task_sampler"],
            "fixed_tasks": diagnostic_tasks,
        }
    evaluation_tasks = env_config["task_sampler"]["fixed_tasks"]
    return ValidationReplayContext(
        env_config=env_config,
        diagnostic_only=diagnostic_only,
        evaluation_panel_path=panel_path,
        source_panel_sha256=source_panel_sha256,
        evaluation_panel_sha256=canonical_json_sha256(evaluation_tasks),
    )


def load_validation_env(
    resolved_config: Path,
    *,
    evaluation_task_panel_path: Path | None = None,
) -> dict:
    return load_validation_context(
        resolved_config,
        evaluation_task_panel_path=evaluation_task_panel_path,
    ).env_config


def load_checkpoint_selection(
    stable_dir: Path,
    checkpoint_selection: str,
    *,
    progress_steps: int | None = None,
) -> dict:
    """Load either the trajectory-best or the final retained checkpoint receipt."""
    if progress_steps is not None:
        progress_steps = require_indexed_integer(progress_steps, "requested progress_steps")
        rows = [
            json.loads(line)
            for line in (stable_dir / "checkpoint_index.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        matches = [
            row
            for row in rows
            if require_indexed_integer(row.get("pbt_progress_steps"), "pbt_progress_steps") == progress_steps
        ]
        if len(matches) != 1:
            raise ValueError(
                "Progress checkpoint selection requires exactly one indexed row: "
                f"steps={progress_steps}, matches={len(matches)}"
            )
        selected = matches[0]
        return {
            **selected,
            "metric": selected["selection_metric"],
            "score": selected["selection_score"],
            "selection_scope": "indexed_progress_checkpoint",
        }
    if checkpoint_selection == "best":
        return json.loads(
            (stable_dir / "best_checkpoint.json").read_text(encoding="utf-8")
        )
    if checkpoint_selection != "terminal":
        raise ValueError(f"Unsupported checkpoint selection: {checkpoint_selection}")

    rows = [
        json.loads(line)
        for line in (stable_dir / "checkpoint_index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Terminal checkpoint selection requires checkpoint-index rows")
    terminal = max(rows, key=lambda row: require_indexed_integer(row.get("pbt_progress_updates"), "pbt_progress_updates"))
    return {
        **terminal,
        "metric": terminal["selection_metric"],
        "score": terminal["selection_score"],
        "selection_scope": "terminal_checkpoint",
    }


def validate_checkpoint_replay_inputs(
    selection: dict,
    checkpoint: Path,
    resolved_path: Path,
    *,
    evaluation_task_panel_path: Path | None = None,
):
    """Validate indexed checkpoint identity and its actual replay inputs.

    Both episode replay and the fresh-process parity command use this contract,
    so copied provenance fields cannot substitute for hashing the dataset or
    reconstructing the frozen source panel.
    """
    resolved_path = Path(resolved_path)
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    api_stack_id = selection.get("api_stack_id")
    if (
        not api_stack_id
        or resolved.get("api_stack_id") != api_stack_id
        or resolved.get("config", {}).get("api_stack_id") != api_stack_id
    ):
        raise ValueError("Checkpoint/resolved provenance mismatch: api_stack_id")
    if api_stack_id != NATIVE_API_STACK_ID:
        raise ValueError(
            "Native checkpoint replay only supports "
            f"{NATIVE_API_STACK_ID}; use the historical compatibility workflow "
            "for legacy API-stack parity"
        )
    expected_config = selection.get("config_hash")
    if not expected_config or canonical_json_sha256(resolved["config"]) != expected_config:
        raise ValueError("Checkpoint resolved config does not match its indexed config identity")
    for key in ("dataset_sha256", "evaluation_panel_sha256", "observation_schema"):
        if key not in selection or selection[key] != resolved.get(key):
            raise ValueError(f"Checkpoint/resolved provenance mismatch: {key}")

    digest = verify_checkpoint_digest(
        checkpoint,
        selection.get("checkpoint_sha256"),
        api_stack_id=api_stack_id,
    )
    validation = load_validation_context(
        resolved_path,
        evaluation_task_panel_path=evaluation_task_panel_path,
    )
    if validation.source_panel_sha256 != selection["evaluation_panel_sha256"]:
        raise ValueError("Resolved source panel does not match indexed panel")
    return resolved, validation, digest


def load_replay_inputs(stable_dir: Path, checkpoint_selection: str, *,
                       progress_steps: int | None = None,
                       evaluation_task_panel_path: Path | None = None):
    selection = load_checkpoint_selection(stable_dir, checkpoint_selection, progress_steps=progress_steps)
    checkpoint = Path(selection["stable_checkpoint"]).resolve(strict=True)
    snapshot = selection.get("config_snapshot")
    resolved_path = Path(snapshot) if snapshot else stable_dir / "resolved_config.json"
    if snapshot and not resolved_path.is_absolute():
        resolved_path = stable_dir / resolved_path
    resolved, validation, digest = validate_checkpoint_replay_inputs(
        selection,
        checkpoint,
        resolved_path,
        evaluation_task_panel_path=evaluation_task_panel_path,
    )
    return selection, checkpoint, resolved, validation, digest


def replay_contract(selection: dict, resolved: dict, validation: ValidationReplayContext,
                    checkpoint_digest: str, execution: dict) -> dict:
    return {
        "version": "deterministic_replay_v1",
        "api_stack_id": selection.get("api_stack_id"),
        "checkpoint_sha256": checkpoint_digest,
        "progress_steps": selection.get("pbt_progress_steps"),
        "restore_generation": selection.get("pbt_restore_generation"),
        "dataset_sha256": resolved["dataset_sha256"],
        "source_panel_sha256": validation.source_panel_sha256,
        "evaluation_panel_sha256": validation.evaluation_panel_sha256,
        "effective_env_config_sha256": canonical_json_sha256(validation.env_config),
        "checkpoint_config_sha256": selection["config_hash"],
        "observation_schema": resolved["observation_schema"],
        "runtime": runtime_identity(),
        "replay_source": source_tree_fingerprint(Path(__file__).resolve().parents[3]),
        "explore": False, "execution": execution,
        "evaluation_seed": resolved["config"]["evaluation_config"].get("seed"),
    }


def replay_execution_settings(algorithm) -> dict:
    import torch

    from itrader.rl_lib.evaluation.ray_native_inference import algorithm_device

    return {"device": algorithm_device(algorithm),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_num_threads": torch.get_num_threads()}


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
        help="Replay the unique retained checkpoint at these target-owned steps.",
    )
    parser.add_argument(
        "--evaluation-task-panel-path",
        type=Path,
        help=(
            "Use a cardinality-matched, non-overlapping validation panel for a "
            "diagnostic-only replay. This disables source-panel parity scoring."
        ),
    )
    args = parser.parse_args()

    stable_dir = args.stable_trial_dir.resolve(strict=True)
    invocation = ReplayInvocation(args.out_dir / f"{stable_dir.name}.json", operation="episode_replay")
    selection, checkpoint, resolved, validation, digest = load_replay_inputs(
        stable_dir, args.checkpoint_selection,
        progress_steps=args.checkpoint_progress_steps,
        evaluation_task_panel_path=args.evaluation_task_panel_path,
    )
    env_config = validation.env_config
    training_config = resolved["config"]

    import ray
    from ray.rllib.algorithms.algorithm import Algorithm

    from itrader.envs.itrader_env import ITraderEnv
    from itrader.rl_lib.evaluation.policy_baselines import (
        FrozenCheckpointPolicy,
        evaluate_policy_control,
        summarize_control_episodes,
    )
    ray.init(include_dashboard=False)
    algo = None
    try:
        algo = Algorithm.from_checkpoint(str(checkpoint))
        from itrader.models.registry import validate_native_restored_model

        validate_native_restored_model(algo, training_config)
        execution = replay_execution_settings(algo)
        env = ITraderEnv(env_config)
        episodes = evaluate_policy_control(
            env,
            FrozenCheckpointPolicy(algo),
            episode_count=len(env_config["task_sampler"]["fixed_tasks"]),
        )
        summary = summarize_control_episodes(episodes)
        env.close()
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()

    expected = float(selection["score"])
    observed = _replay_metric_value(selection["metric"], summary, training_config)
    report = {
        "schema_version": 3,
        "trial": stable_dir.name,
        "checkpoint": str(checkpoint),
        "checkpoint_selection": (
            f"progress:{args.checkpoint_progress_steps}"
            if args.checkpoint_progress_steps is not None
            else args.checkpoint_selection
        ),
        "selection_metric": selection["metric"],
        "diagnostic_only": validation.diagnostic_only,
        "evaluation_panel_path": validation.evaluation_panel_path,
        "source_panel_sha256": validation.source_panel_sha256,
        "evaluation_panel_sha256": validation.evaluation_panel_sha256,
        "selected_score": expected,
        "replayed_score": observed,
        **replay_parity(selection, observed_score=observed, episodes=episodes,
                        diagnostic_only=validation.diagnostic_only),
        "episode_count": len(episodes),
        "summary": summary,
    }
    stem = stable_dir.name
    artifact = args.out_dir / f"{stem}.episodes.jsonl"
    with artifact.open("x", encoding="utf-8") as stream:
        stream.write("".join(json.dumps(row, allow_nan=False) + "\n" for row in episodes))
    report = invocation.complete(
        report, contract=replay_contract(selection, resolved, validation, digest, execution), artifacts=[artifact],
    )
    print(json.dumps(report, indent=2))
    if report["checkpoint_parity"] is False or report["selection_score_parity"] is False:
        raise RuntimeError("Checkpoint replay did not reproduce its selection evidence")


if __name__ == "__main__":
    main()
