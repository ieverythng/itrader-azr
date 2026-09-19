from __future__ import annotations

import argparse
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
from itrader.rl_lib.train.train_solo_V2 import (
    _build_fixed_evaluation_tasks,
    _model_config,
    _parse_split_fractions,
    _sha256_file,
    _task_sampler_config,
)


def _resolve_results_root(value: str) -> Path:
    root = Path(value).expanduser() if value.strip() else Path.home() / "ray_results"
    return root.resolve()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_artifacts(args, report: dict, episodes: list[dict]) -> None:
    root = _resolve_results_root(args.results_root)
    trial_name = f"training_fn_baseline_{args.policy_control}_00000"
    trial_dir = root / args.experiment_name / trial_name
    stable_dir = root / "_stable_metrics" / trial_name
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
    _write_json(
        stable_dir / "resolved_config.json",
        {
            "schema_version": 1,
            "policy_control": args.policy_control,
            "evaluation_num_episodes": args.evaluation_num_episodes,
            "seed": args.seed,
            "data_path": str(Path(args.data_path).resolve()),
            "dataset_sha256": _sha256_file(Path(args.data_path).resolve()),
            "evidence_cost_bps": args.evidence_cost_bps,
            "task_reward_ids": args.task_reward_ids,
            "task_horizons": args.task_horizons,
        },
    )
    _write_json(trial_dir / "baseline_result.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic Phase A policy controls.")
    parser.add_argument("--policy_control", required=True, choices=["flat", "long", "short", "random", "untrained"])
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--data_path", default="itrader/data/ohlcv.parquet")
    parser.add_argument("--results_root", default="")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--evaluation_num_episodes", type=int, default=16)
    parser.add_argument("--task_horizons", default="96,128,160,200")
    parser.add_argument("--task_reward_ids", default="net_return")
    parser.add_argument("--evidence_cost_bps", type=float, default=1.0)
    parser.add_argument("--optimization_cost_bps", type=float, default=None)
    parser.add_argument("--net_return_reward_scale", type=float, default=1000.0)
    parser.add_argument("--evidence_periods_per_year", type=float, default=0.0)
    parser.add_argument("--split_fractions", default="0.7,0.15,0.15")
    parser.add_argument("--split_purge_rows", type=int, default=200)
    parser.add_argument("--observation_transform", default="market_relative_v1", choices=["raw", "market_relative_v1"])
    args = parser.parse_args()

    sampler_config = _task_sampler_config(args)
    evaluation_seed = int(args.seed) + 10_000_019
    sampler_config["seed"] = evaluation_seed
    sampler_config["fixed_tasks"] = _build_fixed_evaluation_tasks(
        sampler_config,
        int(args.evaluation_num_episodes),
    )
    env = ITraderEnv(
        {
            "df_path": str(Path(args.data_path).resolve()),
            "window_size": args.window,
            "start_idx_mode": "random_episode",
            "ret_norm_reset": "episode",
            "observation_transform": args.observation_transform,
            "t_embed_mode": "concat",
            "task_encoder_version": "semantic_v2",
            "data_split": "validation",
            "split_fractions": _parse_split_fractions(args.split_fractions),
            "split_purge_rows": args.split_purge_rows,
            "evidence_cost_bps": args.evidence_cost_bps,
            "evidence_periods_per_year": args.evidence_periods_per_year,
            "task_sampler": sampler_config,
        }
    )
    if args.policy_control == "untrained":
        policy = UntrainedFiLMPolicy(
            env.observation_space,
            env.action_space,
            seed=args.seed,
            model_config=_model_config(args),
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
    _write_artifacts(args, report, episodes)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
