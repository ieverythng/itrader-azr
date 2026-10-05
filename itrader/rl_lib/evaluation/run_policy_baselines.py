from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from itrader.envs.itrader_env import ITraderEnv
from itrader.rl_lib.evaluation.policy_baselines import (
    ControlPolicy,
    UntrainedFiLMPolicy,
    evaluate_policy_control,
    summarize_control_episodes,
)
from itrader.rl_lib.evaluation.provenance import (
    canonical_json_sha256, dataset_sha256, evaluation_contract_sha256,
    source_tree_fingerprint, runtime_identity,
)
from itrader.rl_lib.train.train_solo_V2 import (
    _build_fixed_evaluation_tasks,
    _load_evaluation_task_panel,
    _model_config,
    _parse_split_fractions,
    _task_sampler_config,
)


def _resolve_results_root(value: str) -> Path:
    root = Path(value).expanduser() if value.strip() else Path.home() / "ray_results"
    return root.resolve()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _baseline_trial_name(args) -> str:
    state_mode = str(getattr(args, "solver_state_mode", "off"))
    residual_init = str(getattr(args, "solver_state_residual_init", "random_v1"))
    if state_mode == "off" and residual_init == "random_v1":
        return f"training_fn_baseline_{args.policy_control}_00000"
    return (
        f"training_fn_baseline_{args.policy_control}_"
        f"{state_mode}_{residual_init}_00000"
    )


def _write_artifacts(
    args,
    report: dict,
    episodes: list[dict],
    env_config: dict,
    evaluation_panel_sha256: str,
) -> None:
    root = _resolve_results_root(args.results_root)
    trial_name = _baseline_trial_name(args)
    trial_dir = root / args.experiment_name / trial_name
    stable_dir = root / "_stable_metrics" / args.experiment_name / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    stable_dir.mkdir(parents=True, exist_ok=True)

    with (stable_dir / "report_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report))
        writer.writeheader()
        writer.writerow(report)

    metrics = dict(report)
    metrics.update(
        {
            "evaluation/custom_metrics/economic_gross_return_mean": report[
                "evaluation/gross_return_mean"
            ],
            "evaluation/custom_metrics/economic_net_return_mean": report[
                "evaluation/net_return_mean"
            ],
            "evaluation/custom_metrics/economic_net_sharpe_per_sqrt_bar_mean": report[
                "evaluation/net_sharpe_per_sqrt_bar_mean"
            ],
            "evaluation/custom_metrics/economic_turnover_mean": report[
                "evaluation/turnover_mean"
            ],
        }
    )
    (stable_dir / "metrics_full.jsonl").write_text(
        json.dumps(metrics) + "\n",
        encoding="utf-8",
    )
    (stable_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes),
        encoding="utf-8",
    )
    dataset_hash = dataset_sha256(Path(args.data_path))
    provenance = {
        "dataset_sha256": dataset_hash,
        "evaluation_panel_sha256": evaluation_panel_sha256,
        **source_tree_fingerprint(Path(__file__).resolve().parents[3]),
        "runtime_sha256": canonical_json_sha256(runtime_identity()),
        "evaluation_contract_sha256": evaluation_contract_sha256(env_config, dataset_hash),
    }
    comparison_env = copy.deepcopy(env_config)
    comparison_env["df_path"] = f"sha256:{dataset_hash}"
    model_config = _model_config(args) if args.policy_control == "untrained" else None
    comparison_config_hash = canonical_json_sha256(
        {
            "policy_control": args.policy_control,
            "seed": args.seed,
            "env_config": comparison_env,
            "model_config": model_config,
            "provenance": provenance,
        }
    )
    _write_json(
        stable_dir / "resolved_config.json",
        {
            "schema_version": 1,
            "policy_control": args.policy_control,
            "evaluation_num_episodes": args.evaluation_num_episodes,
            "seed": args.seed,
            "data_path": str(Path(args.data_path).resolve()),
            "dataset_sha256": dataset_hash,
            "comparison_config_hash": comparison_config_hash,
            "evaluation_panel_sha256": evaluation_panel_sha256,
            "data_sessions": args.data_sessions,
            "data_symbols": args.data_symbols,
            "episode_boundary_cols": args.episode_boundary_cols,
            "expected_bar_interval": args.expected_bar_interval,
            "evidence_cost_bps": args.evidence_cost_bps,
            "task_reward_ids": args.task_reward_ids,
            "task_horizons": args.task_horizons,
            "evaluation_seed": args.evaluation_seed,
            "evaluation_task_panel_path": str(
                getattr(args, "evaluation_task_panel_path", "") or ""
            ),
            "solver_state_mode": str(getattr(args, "solver_state_mode", "off")),
            "solver_state_residual_init": str(
                getattr(args, "solver_state_residual_init", "random_v1")
            ),
            "model_config": model_config,
        },
    )
    _write_json(
        trial_dir / "baseline_result.json",
        {
            **report,
            "provenance": provenance,
            "policy_control": args.policy_control,
            "solver_state_mode": str(getattr(args, "solver_state_mode", "off")),
            "solver_state_residual_init": str(
                getattr(args, "solver_state_residual_init", "random_v1")
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic Phase A policy controls.")
    parser.add_argument("--policy_control", required=True, choices=["flat", "long", "short", "random", "untrained"])
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--data_path", default="itrader/data/ohlcv.parquet")
    parser.add_argument("--data_sessions", default="")
    parser.add_argument("--data_symbols", default="")
    parser.add_argument("--episode_boundary_cols", default="")
    parser.add_argument("--expected_bar_interval", default="")
    parser.add_argument("--results_root", default="")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--evaluation_seed", type=int, default=None)
    parser.add_argument("--evaluation_num_episodes", type=int, default=16)
    parser.add_argument("--evaluation_task_panel_path", default="")
    parser.add_argument("--task_horizons", default="96,128,160,200")
    parser.add_argument("--task_reward_ids", default="net_return")
    parser.add_argument("--evidence_cost_bps", type=float, default=1.0)
    parser.add_argument("--optimization_cost_bps", type=float, default=None)
    parser.add_argument("--net_return_reward_scale", type=float, default=1000.0)
    parser.add_argument("--evidence_periods_per_year", type=float, default=0.0)
    parser.add_argument("--split_fractions", default="0.7,0.15,0.15")
    parser.add_argument("--split_purge_rows", type=int, default=200)
    parser.add_argument("--observation_transform", default="market_relative_v1", choices=["raw", "market_relative_v1"])
    parser.add_argument("--solver_state_mode", default="off", choices=["off", "masked", "position", "time", "both"])
    parser.add_argument("--solver_state_residual_init", default="random_v1", choices=["random_v1", "zero_v1"])
    parser.add_argument("--solver_model", default="solver_mlp_film", choices=["solver_mlp_film", "solver_window_lstm_film"])
    parser.add_argument("--representation_profile", default="legacy", choices=["legacy", "r0_r1_v1"])
    parser.add_argument("--observation_filter", default="NoFilter", choices=["NoFilter"])
    args = parser.parse_args()

    sampler_config = _task_sampler_config(args)
    evaluation_seed = (
        int(args.seed) + 10_000_019
        if args.evaluation_seed is None
        else int(args.evaluation_seed)
    )
    args.evaluation_seed = evaluation_seed
    sampler_config["seed"] = evaluation_seed
    if args.evaluation_task_panel_path:
        sampler_config["fixed_tasks"] = _load_evaluation_task_panel(
            args.evaluation_task_panel_path,
            int(args.evaluation_num_episodes),
        )
        evaluation_start_idx_mode = "task_spec"
    else:
        sampler_config["fixed_tasks"] = _build_fixed_evaluation_tasks(
            sampler_config,
            int(args.evaluation_num_episodes),
        )
        evaluation_start_idx_mode = "random_episode"
    evaluation_panel_sha256 = canonical_json_sha256(sampler_config["fixed_tasks"])
    env_config = {
        "df_path": str(Path(args.data_path).resolve()),
        "window_size": args.window,
        "start_idx_mode": evaluation_start_idx_mode,
        "ret_norm_reset": "episode",
        "observation_transform": args.observation_transform,
        "solver_state_mode": args.solver_state_mode,
        "t_embed_mode": "concat",
        "task_encoder_version": "semantic_v2",
        "data_split": "validation",
        "data_sessions": [
            value.strip() for value in args.data_sessions.split(",") if value.strip()
        ],
        "data_symbols": [
            value.strip().upper()
            for value in args.data_symbols.split(",")
            if value.strip()
        ],
        "episode_boundary_cols": [
            value.strip()
            for value in args.episode_boundary_cols.split(",")
            if value.strip()
        ],
        "expected_bar_interval": args.expected_bar_interval or None,
        "split_fractions": _parse_split_fractions(args.split_fractions),
        "split_purge_rows": args.split_purge_rows,
        "evidence_cost_bps": args.evidence_cost_bps,
        "evidence_periods_per_year": args.evidence_periods_per_year,
        "task_sampler": sampler_config,
    }
    model_config = _model_config(args)
    semantic = model_config["custom_model_config"].get("semantic_observation_contract")
    if semantic is not None:
        env_config.update({"semantic_observation_contract": semantic,
                           "feature_cols": semantic["feature_cols"],
                           "observation_filter": args.observation_filter})
    env_config["capture_replay_trace"] = True
    env = ITraderEnv(env_config)
    if args.policy_control == "untrained":
        policy = UntrainedFiLMPolicy(
            env.observation_space,
            env.action_space,
            seed=args.seed,
            model_config=model_config,
        )
    else:
        policy = ControlPolicy(args.policy_control, args.seed)

    try:
        episodes = evaluate_policy_control(
            env,
            policy,
            episode_count=args.evaluation_num_episodes,
        )
    finally:
        env.close()
    report = summarize_control_episodes(episodes)
    _write_artifacts(args, report, episodes, env_config, evaluation_panel_sha256)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
