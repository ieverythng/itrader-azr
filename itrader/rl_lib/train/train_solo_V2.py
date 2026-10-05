# itrader/rl_lib/train_solo.py

import argparse
import copy
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ray
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.stopper import TimeoutStopper
from ray.tune import PlacementGroupFactory
from ray.train import Checkpoint
# your env & model imports
from itrader.envs.itrader_env import (
    ITraderEnv,
    resolve_solver_state_mode,
    solver_state_observation_contract,
)
from itrader.rl_lib.tasks.families import DEFAULT_BAR_RESOLUTION
from itrader.rl_lib.train.ray_native_config import NATIVE_API_STACK_ID
from itrader.rl_lib.tasks.rewards import RewardEngine
from itrader.rl_lib.tasks.samplers import TaskSampler00
from itrader.rl_lib.tasks.spec import TaskSpec
from itrader.rl_lib.callbacks.auto_plotter import AutoPlotterCallback
from itrader.rl_lib.callbacks.metrics_contract import mean_from_totals
from itrader.rl_lib.callbacks.ps_metrics_native import NativePSMetrics
from itrader.rl_lib.evaluation.provenance import (
    canonical_json_sha256,
    checkpoint_sha256,
    dataset_sha256 as _dataset_sha256,
    source_tree_fingerprint,
)
from itrader.rl_lib.train.pbt_contract import (
    PBTProgressClock,
    checkpoint_retention_decision,
    constrain_trial_config,
    crossed_interval,
    economic_search_score,
    economic_selection_score,
    parse_step_schedule,
    scheduled_value,
)
from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

from gymnasium.envs.registration import EnvSpec

EXPERIMENT_NAME = "Phase_B"
DEFAULT_T_EMBED_DIM = 32
DEFAULT_FCNET_HIDDENS = [256, 256]
DEFAULT_TASK_HORIZONS = [96, 128, 160, 200]
DEFAULT_TURNOVER_W_CHOICES = [0.0, 0.0005, 0.001, 0.002]
DEFAULT_DD_W_RANGE = (0.0, 0.04)
BASE_METRIC_COLUMNS = [
    "pbt_progress_steps",
    "training/optimization_cost_bps",
    "training/optimization_reward_mean",
    "training/net_return_mean",
    "evaluation/net_return_mean",
    "evaluation/net_sharpe_per_sqrt_bar_mean",
    "evaluation/economic_search_score",
    "evaluation/economic_selection_score",
    "episode_len_mean",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
]
PBT_PRIMARY_METRIC = "evaluation/net_return_mean"
PBT_SEARCH_METRIC = "evaluation/economic_search_score"
PBT_PROMOTION_METRIC = "evaluation/economic_selection_score"
PBT_ENTROPY_SCHEDULES = [
    [[0, 0.0], [96_000, 0.0], [240_000, 0.0]],
    [[0, 0.001], [96_000, 0.0005], [240_000, 0.0]],
    [[0, 0.005], [96_000, 0.002], [240_000, 0.0005]],
    [[0, 0.01], [96_000, 0.005], [240_000, 0.001]],
]
def _optimizer_key(name: str) -> str:
    """Map stable CLI option names to Ray's native PPO config vocabulary."""
    return {
        "sgd_minibatch_size": "minibatch_size",
        "num_sgd_iter": "num_epochs",
        "lambda": "lambda_",
        "entropy_coeff_schedule": "entropy_coeff",
        "train_batch_size": "train_batch_size_per_learner",
    }.get(name, name)


PBT_MUTATIONS = {
    "lr": tune.loguniform(5e-6, 3e-4),
    "entropy_coeff": PBT_ENTROPY_SCHEDULES,
    "clip_param": [0.1, 0.2, 0.3],
    "lambda_": [0.90, 0.95, 0.97],
    "gamma": [0.98, 0.99, 0.995],
    _optimizer_key("sgd_minibatch_size"): [128, 256, 512],
    _optimizer_key("num_sgd_iter"): [10, 20, 30],
    "vf_clip_param": [5.0, 10.0, 20.0],
    "vf_loss_coeff": [0.5, 1.0, 2.0],
}


def _pbt_learning_rate_bounds(args) -> tuple[float, float]:
    lower = float(getattr(args, "pbt_lr_min", 5e-6))
    upper = float(getattr(args, "pbt_lr_max", 3e-4))
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower <= 0.0
        or upper <= lower
    ):
        raise ValueError("pbt_lr_min and pbt_lr_max must be finite, positive, and ordered")
    return lower, upper


def _pbt_mutations(args) -> dict:
    lower, upper = _pbt_learning_rate_bounds(args)
    return {**PBT_MUTATIONS, "lr": tune.loguniform(lower, upper)}


def _pbt_mutation_manifest(args) -> dict:
    """Serialize the resolved PBT search space without process-local reprs."""
    lower, upper = _pbt_learning_rate_bounds(args)
    return {
        **{key: value for key, value in PBT_MUTATIONS.items() if key != "lr"},
        "lr": {
            "distribution": "loguniform",
            "lower": lower,
            "upper": upper,
        },
    }


def _cli_metric_columns(action_dim: int) -> list[str]:
    columns = list(BASE_METRIC_COLUMNS)
    columns.extend(f"custom_metrics/act_{index}_frac_mean" for index in range(max(0, int(action_dim))))
    if int(action_dim) >= 3:
        columns.extend(
            [
                "custom_metrics/act_short_frac_mean",
                "custom_metrics/act_flat_frac_mean",
                "custom_metrics/act_long_frac_mean",
            ]
        )
    return columns


def _task_sampler_config(args) -> dict:
    task_pool_path = str(getattr(args, "task_pool_path", "") or "").strip()
    reward_ids = _parse_task_reward_ids(getattr(args, "task_reward_ids", ""))
    cfg = {
        "window_size": args.window,
        "n_features": 5,
        "t_embed_dim": DEFAULT_T_EMBED_DIM,
        "action_dim": 3,
        "seed": int(args.seed),
        "bar_resolution": DEFAULT_BAR_RESOLUTION,
        "horizons": _parse_task_horizons(getattr(args, "task_horizons", "")),
        "reward_ids": reward_ids,
        "turnover_w_choices": DEFAULT_TURNOVER_W_CHOICES,
        "dd_modes": ["delta"],
        "dd_w_range": DEFAULT_DD_W_RANGE,
        "turnover_cap": 0.02,
        "dd_cap": 0.10,
    }
    if task_pool_path:
        cfg["task_pool_path"] = task_pool_path
        cfg["p_pool"] = float(getattr(args, "p_pool", 0.7))
    if "net_return" in reward_ids:
        cost_schedule = parse_step_schedule(
            getattr(args, "optimization_cost_schedule", "")
        )
        configured_cost = getattr(args, "optimization_cost_bps", None)
        cost_bps = (
            cost_schedule[0][1]
            if cost_schedule
            else float(configured_cost)
            if configured_cost is not None
            else float(getattr(args, "evidence_cost_bps", 1.0))
        )
        cfg["reward_params_by_id"] = {
            "net_return": {
                "cost_bps": cost_bps,
                "reward_scale": float(getattr(args, "net_return_reward_scale", 1000.0)),
            }
        }
    return cfg


def _training_optimization_cost_bps(
    config: dict,
    *,
    completed_steps: int,
) -> float | None:
    """Resolve the cost used by the training reward for report provenance."""
    schedule = config.get("optimization_cost_schedule", ())
    if schedule:
        return float(scheduled_value(schedule, completed_steps))

    configured_cost = config.get("optimization_cost_bps")
    if configured_cost is not None:
        return float(configured_cost)

    task_sampler = config.get("env_config", {}).get("task_sampler", {})
    reward_params = task_sampler.get("reward_params_by_id", {}).get(
        "net_return",
        {},
    )
    static_cost = reward_params.get("cost_bps")
    return None if static_cost is None else float(static_cost)


def _parse_task_horizons(value) -> list[int]:
    if value is None or value == "":
        return list(DEFAULT_TASK_HORIZONS)
    if isinstance(value, (list, tuple)):
        raw_values = value
    else:
        raw_values = str(value).split(",")
    horizons = [int(str(item).strip()) for item in raw_values if str(item).strip()]
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("task_horizons must contain positive comma-separated integers")
    return horizons


def _parse_task_reward_ids(value) -> list[str]:
    if value is None or value == "":
        return ["pnl_linear"]
    reward_ids = [item.strip() for item in str(value).split(",") if item.strip()]
    unknown = sorted(set(reward_ids).difference(RewardEngine.registered_ids()))
    if not reward_ids or unknown:
        detail = f": {', '.join(unknown)}" if unknown else ""
        raise ValueError(f"task_reward_ids contains unknown reward IDs{detail}")
    return reward_ids


def _parse_split_fractions(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        fractions = [float(item) for item in value]
    else:
        fractions = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(fractions) != 3 or any(item <= 0.0 for item in fractions):
        raise ValueError("split_fractions must contain three positive values")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split_fractions must sum to 1.0")
    return fractions


def _pbt_intervals(args) -> tuple[int, int]:
    perturb = int(getattr(args, "pbt_interval", 3))
    checkpoint = int(getattr(args, "checkpoint_interval", perturb))
    if perturb <= 0 or checkpoint <= 0:
        raise ValueError("pbt_interval and checkpoint_interval must be positive")
    if checkpoint != perturb:
        raise ValueError(
            "checkpoint_interval must equal pbt_interval so every exploit boundary has a checkpoint"
        )
    return perturb, checkpoint


def _model_config(args) -> dict:
    from itrader.models.registry import validate_solver_selector

    selector = str(getattr(args, "solver_model", "solver_mlp_film"))
    validate_solver_selector(selector)
    solver_state_mode = resolve_solver_state_mode(
        getattr(args, "solver_state_mode", "off")
    )
    solver_state_dim = 0 if solver_state_mode == "off" else 2
    model = {
        "custom_model": selector,
        "custom_model_config": {
            "window_size": args.window,
            "t_embed_dim": DEFAULT_T_EMBED_DIM,
            "solver_state_dim": solver_state_dim,
            "solver_state_width": 8,
            "solver_state_residual_init": str(
                getattr(args, "solver_state_residual_init", "random_v1")
            ),
            "fcnet_hiddens": DEFAULT_FCNET_HIDDENS,
            "conditioning": "film",
            "t_keep_p": 0.7,
            "t_proj": 64,
        },
    }
    if getattr(args, "representation_profile", "legacy") == "r0_r1_v1":
        from itrader.models.observation_contract import semantic_observation_contract

        contract = semantic_observation_contract(solver_state_mode)
        model["custom_model_config"].update({
            **contract,
            "window_size": int(args.window),
            "observation_transform": getattr(args, "observation_transform", "raw"),
            "observation_filter": getattr(args, "observation_filter", "NoFilter"),
            "semantic_observation_contract": contract,
        })
    return model


def _observation_contract(args) -> dict[str, int | str]:
    return solver_state_observation_contract(
        window_size=int(args.window),
        n_features=5,
        task_embed_dim=DEFAULT_T_EMBED_DIM,
        solver_state_mode=getattr(args, "solver_state_mode", "off"),
    )


def _observation_schema_manifest(config: dict) -> dict[str, int | str]:
    keys = (
        "observation_schema_id",
        "base_obs_dim",
        "solver_state_dim",
        "agent_obs_dim",
    )
    train_env = config.get("env_config", {})
    schema = {key: train_env[key] for key in keys if key in train_env}
    if not schema:
        return {}
    if len(schema) != len(keys):
        raise ValueError("Training observation schema evidence is incomplete")
    evaluation_env = config.get("evaluation_config", {}).get("env_config", {})
    evaluation_schema = {key: evaluation_env.get(key) for key in keys}
    if evaluation_schema != schema:
        raise ValueError("Training and evaluation observation schemas must match")
    return schema


def _env_runner_gpu_settings(value: float) -> dict[str, float]:
    """Use the native EnvRunner resource key."""
    return {"num_gpus_per_env_runner": value}


def _evaluation_runner_settings(count: int) -> dict[str, int]:
    """Use the native evaluation EnvRunner count key."""
    return {"evaluation_num_env_runners": count}


def _first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None

# ────────────────────────────────────────────────────────────────────────────────
# 1) Ray init utilities + run metadata
# ────────────────────────────────────────────────────────────────────────────────
def init_ray(use_pbt_mode: bool, num_cpus: int | None = None):
    # PBT can run many trials; avoid a shared Prometheus port collision.
    init_kwargs = {} if use_pbt_mode else {"_metrics_export_port": 8888}
    if num_cpus is not None:
        init_kwargs["num_cpus"] = int(num_cpus)
    # The minimal Ray installation cannot host the dashboard. Attempting it
    # first spawns a partial Windows node and can leave ports unavailable for
    # the fallback startup, so reliability takes precedence for experiments.
    ray.init(**init_kwargs, include_dashboard=False)


def _source_tree_fingerprint(repo_root: Path | None = None) -> dict[str, object]:
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    return source_tree_fingerprint(root)


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _config_hash(config: dict) -> str:
    return canonical_json_sha256(config)


def _runtime_manifest() -> dict[str, str]:
    """Capture the platform/dependency identity that produced run evidence."""
    from itrader.rl_lib.evaluation.provenance import runtime_identity

    return runtime_identity()


def _comparison_config_hash(config: dict, dataset_hash: str) -> str:
    """Hash behavior-affecting config while normalizing branch-local paths."""
    comparable = copy.deepcopy(config)
    comparable.pop("source_provenance", None)
    comparable.pop("stable_metrics_root", None)
    comparable.pop("pbt_experiment_name", None)
    env_configs = [comparable.get("env_config", {})]
    env_configs.append(comparable.get("evaluation_config", {}).get("env_config", {}))
    for env_config in env_configs:
        if "df_path" in env_config:
            env_config["df_path"] = f"sha256:{dataset_hash}"
    return _config_hash(comparable)


def _evaluation_panel_sha256(config: dict) -> str:
    tasks = config["evaluation_config"]["env_config"]["task_sampler"]["fixed_tasks"]
    return canonical_json_sha256(tasks)


def _freeze_run_provenance(config: dict, data_path: Path) -> dict[str, str]:
    """Embed immutable source, data, and panel identities in a launch config."""
    frozen = {
        "dataset_sha256": _dataset_sha256(data_path),
        "evaluation_panel_sha256": _evaluation_panel_sha256(config),
    }
    # Keep application provenance namespaced so PPO receives only supported
    # evaluation settings.
    config["frozen_provenance"] = frozen
    config["provenance_enforced"] = True
    return frozen


def _assert_frozen_run_provenance(
    config: dict,
    *,
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Reject an actor start when launch-critical inputs changed in place."""
    if not bool(config.get("provenance_enforced", False)):
        return {}
    expected_source = str(
        (config.get("source_provenance") or {}).get("source_tree_sha256", "")
    )
    frozen = dict(config.get("frozen_provenance", {}) or {})
    expected_dataset = str(
        frozen.get("dataset_sha256") or config.get("dataset_sha256", "")
    )
    expected_panel = str(
        frozen.get("evaluation_panel_sha256")
        or config.get("evaluation_panel_sha256", "")
    )
    if not all((expected_source, expected_dataset, expected_panel)):
        raise RuntimeError("Frozen run provenance is incomplete")

    current_source = _source_tree_fingerprint(repo_root)["source_tree_sha256"]
    current_dataset = _dataset_sha256(Path(str(config["env_config"]["df_path"])))
    current_panel = _evaluation_panel_sha256(config)
    if current_source != expected_source:
        raise RuntimeError("Frozen run source tree changed before actor start")
    if current_dataset != expected_dataset:
        raise RuntimeError("Frozen run dataset changed before actor start")
    if current_panel != expected_panel:
        raise RuntimeError("Frozen run evaluation panel changed before actor start")
    return {
        "source_tree_sha256": current_source,
        "dataset_sha256": current_dataset,
        "evaluation_panel_sha256": current_panel,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Replace a JSON evidence file only after a complete durable write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _results_root(args) -> Path:
    configured = str(getattr(args, "results_root", "") or "").strip()
    root = Path(configured).expanduser() if configured else Path.home() / "ray_results"
    return root.resolve()


def _build_fixed_evaluation_tasks(task_sampler_cfg: dict, panel_size: int) -> list[dict]:
    """Materialize one immutable, task-balanced panel for every PBT member."""
    if panel_size <= 0:
        raise ValueError("evaluation_num_episodes must be positive")
    panel_cfg = {**task_sampler_cfg}
    panel_cfg.pop("fixed_tasks", None)
    panel_cfg.pop("task_pool_path", None)
    panel_cfg["p_pool"] = 0.0
    horizons = [int(value) for value in panel_cfg.get("horizons", DEFAULT_TASK_HORIZONS)]
    reward_ids = [str(value) for value in panel_cfg.get("reward_ids", ["pnl_linear"])]
    base_seed = int(panel_cfg.get("seed", 0))
    tasks = []
    for index in range(panel_size):
        task_cfg = {
            **panel_cfg,
            "seed": base_seed + index * 104_729,
            "horizons": [horizons[index % len(horizons)]],
            "reward_ids": [reward_ids[(index // len(horizons)) % len(reward_ids)]],
        }
        tasks.append(TaskSampler00(task_cfg).sample().to_dict())
    return tasks


def _load_evaluation_task_panel(path: str | Path, panel_size: int) -> list[dict]:
    """Load an immutable evaluation panel whose starts are part of the contract."""
    panel_path = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(panel_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid evaluation task panel JSON: {panel_path}") from exc
    tasks = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(tasks, list):
        raise ValueError("evaluation task panel must be a list or an object with tasks")
    if len(tasks) != panel_size:
        raise ValueError(
            "evaluation task panel size must equal evaluation_num_episodes: "
            f"{len(tasks)} != {panel_size}"
        )
    normalized = [TaskSpec.from_dict(task).to_dict() for task in tasks]
    if any(task.get("start_idx") is None for task in normalized):
        raise ValueError("evaluation task panel tasks require explicit start_idx values")
    return normalized


def build_base_config(args, data_path: Path, use_pbt_mode: bool) -> dict:
    if hasattr(args, "rllib_api_stack"):
        raise ValueError(
            "rllib_api_stack is retired; the active trainer always uses the native "
            "Ray 2.58 RLModule/Learner/EnvRunner API"
        )
    auto_plot_on_end = bool(getattr(args, "auto_plot_on_end", True))
    task_sampler_cfg = _task_sampler_config(args)
    _, checkpoint_interval = _pbt_intervals(args)
    split_fractions = _parse_split_fractions(getattr(args, "split_fractions", "0.7,0.15,0.15"))
    evidence_cost_bps = float(getattr(args, "evidence_cost_bps", 1.0))
    evidence_periods = float(getattr(args, "evidence_periods_per_year", 0.0))
    purge_rows = int(getattr(args, "split_purge_rows", 0))
    evaluation_episodes = int(getattr(args, "evaluation_num_episodes", 8))
    configured_evaluation_seed = getattr(args, "evaluation_seed", None)
    evaluation_seed = (
        int(args.seed) + 10_000_019
        if configured_evaluation_seed is None
        else int(configured_evaluation_seed)
    )
    train_batch_size = int(getattr(args, "train_batch_size", 4800))
    pbt_interval, _ = _pbt_intervals(args)
    optimization_cost_schedule = parse_step_schedule(
        getattr(args, "optimization_cost_schedule", "")
    )
    if optimization_cost_schedule and "net_return" not in task_sampler_cfg["reward_ids"]:
        raise ValueError("optimization_cost_schedule requires the net_return reward")
    source_provenance = {
        "git_commit": _git_commit(),
        **_source_tree_fingerprint(),
    }
    configured_checkpoint_evaluation = getattr(
        args,
        "checkpoint_consistent_evaluation",
        None,
    )
    data_sessions = [
        value.strip() for value in str(getattr(args, "data_sessions", "")).split(",") if value.strip()
    ]
    data_symbols = [
        value.strip().upper()
        for value in str(getattr(args, "data_symbols", "")).split(",")
        if value.strip()
    ]
    episode_boundary_cols = [
        value.strip()
        for value in str(getattr(args, "episode_boundary_cols", "")).split(",")
        if value.strip()
    ]
    solver_state_mode = resolve_solver_state_mode(
        getattr(args, "solver_state_mode", "off")
    )
    observation_contract = _observation_contract(args)

    env_config = {
        "df_path": data_path.as_posix(),
        "window_size": args.window,
        "start_idx_mode": args.start_idx_mode,
        "stream_jump_every_episodes": args.stream_jump_every_episodes,
        "stream_jump_on_wrap": args.stream_jump_on_wrap,
        "stream_stride": args.stream_stride,
        "ret_norm_reset": args.ret_norm_reset,
        "observation_transform": str(getattr(args, "observation_transform", "raw")),
        "solver_state_mode": solver_state_mode,
        **observation_contract,
        "t_embed_mode": "concat",
        "t_scale": 1.0,
        "task_encoder_version": "semantic_v2",
        "data_split": "train",
        "data_sessions": data_sessions,
        "data_symbols": data_symbols,
        "episode_boundary_cols": episode_boundary_cols,
        "expected_bar_interval": str(getattr(args, "expected_bar_interval", "") or "") or None,
        "split_fractions": split_fractions,
        "split_purge_rows": purge_rows,
        "evidence_cost_bps": evidence_cost_bps,
        "evidence_periods_per_year": evidence_periods,
        "task_sampler": task_sampler_cfg,
    }
    model_config = _model_config(args)
    semantic = model_config["custom_model_config"].get("semantic_observation_contract")
    if semantic is not None:
        env_config.update({
            "semantic_observation_contract": semantic,
            "feature_cols": semantic["feature_cols"],
            "observation_filter": str(getattr(args, "observation_filter", "NoFilter")),
        })
    evaluation_task_sampler_cfg = {
        **task_sampler_cfg,
        "seed": evaluation_seed,
    }
    evaluation_panel_path = str(
        getattr(args, "evaluation_task_panel_path", "") or ""
    ).strip()
    if evaluation_panel_path:
        evaluation_task_sampler_cfg["fixed_tasks"] = _load_evaluation_task_panel(
            evaluation_panel_path,
            evaluation_episodes,
        )
        evaluation_start_idx_mode = "task_spec"
    else:
        evaluation_task_sampler_cfg["fixed_tasks"] = _build_fixed_evaluation_tasks(
            evaluation_task_sampler_cfg,
            evaluation_episodes,
        )
        evaluation_start_idx_mode = "random_episode"
    evaluation_env_config = {
        **env_config,
        "data_split": "validation",
        "start_idx_mode": evaluation_start_idx_mode,
        "ret_norm_reset": "episode",
        "task_sampler": evaluation_task_sampler_cfg,
        "capture_replay_trace": True,
    }

    pbt_lr_min, pbt_lr_max = _pbt_learning_rate_bounds(args)
    config = {
        "env": "ITraderEnv",
        "solver_model": str(getattr(args, "solver_model", "solver_mlp_film")),
        "representation_profile": str(getattr(args, "representation_profile", "legacy")),
        "api_stack_id": NATIVE_API_STACK_ID,
        "enable_rl_module_and_learner": True,
        "enable_env_runner_and_connector_v2": True,
        "source_provenance": source_provenance,
        "optimization_cost_schedule": optimization_cost_schedule,
        "env_config": env_config,
        "evaluation_interval": int(getattr(args, "evaluation_interval", 1)),
        **_evaluation_runner_settings(0),
        "evaluation_duration": evaluation_episodes,
        "evaluation_duration_unit": "episodes",
        # The post-sync evaluation must replace Ray's pre-sync episode history.
        # A larger default smoothing window would rank a blend of old policies.
        "metrics_num_episodes_for_smoothing": evaluation_episodes,
        "evaluation_config": {
            "explore": False,
            "seed": evaluation_seed,
            "env_config": evaluation_env_config,
        },
        # PBT must rank the post-sync state that its checkpoint will contain.
        "checkpoint_consistent_evaluation": (
            bool(use_pbt_mode)
            if configured_checkpoint_evaluation is None
            else bool(configured_checkpoint_evaluation)
        ),
        "model": model_config,
        # Native resource selection replaces this CPU-safe default.
        "num_gpus_per_learner": 0,
        **_env_runner_gpu_settings(0),
        "num_env_runners": args.workers,
        # reproducibility
        "seed": int(args.seed),
        # core RL settings
        "framework": "torch",
        # swap callbacks for PBT to reduce collisions/overhead
        "callbacks": (
            NativeMetricsOnlyCallback
            if use_pbt_mode or not auto_plot_on_end
            else NativeCombinedCallback
        ),
        "observation_filter": str(getattr(args, "observation_filter", "NoFilter")),
        "grad_clip": 10.0,
        "clip_param": float(getattr(args, "clip_param", 0.2)),
        "lambda_": float(getattr(args, "gae_lambda", 0.95)),
        "vf_loss_coeff": float(getattr(args, "vf_loss_coeff", 1.0)),
        # RLlib linearly interpolates entropy on its native sampled-step clock.
        "entropy_coeff": [list(point) for point in parse_step_schedule(
            getattr(args, "entropy_coeff_schedule", ""), name="entropy_coeff_schedule"
        )] or [
            [0,        0.02],
            [200_000,  0.01],
            [800_000,  0.003],
        ],
        "train_batch_size_per_learner": train_batch_size,
        "batch_mode": "complete_episodes",
        "minibatch_size": int(getattr(args, "sgd_minibatch_size", 128)),
        "num_epochs": int(getattr(args, "num_sgd_iter", 30)),
        "gamma": float(getattr(args, "gamma", 0.99)),
        "vf_clip_param": float(getattr(args, "vf_clip_param", 10.0)),
        "lr": float(getattr(args, "lr", 5e-5)),
        # PBT runs multiple trials -> shared port conflicts. Enable only for single-trial runs.
        "metrics_sinks": ([] if use_pbt_mode else ["prometheus"]),
        "prometheus_export_port": (None if use_pbt_mode else 8888),
        # pass the stop criterion to training_fn
        "stop_iters": args.iters,
        "checkpoint_interval": checkpoint_interval,
        "pbt_synch": bool(
            use_pbt_mode and getattr(args, "pbt_synch", False)
        ),
        "pbt_target_steps": int(args.iters) * train_batch_size,
        "pbt_interval_steps": pbt_interval * train_batch_size,
        "checkpoint_interval_steps": checkpoint_interval * train_batch_size,
        "stable_metrics_root": str(_results_root(args) / "_stable_metrics"),
        "pbt_experiment_name": str(getattr(args, "experiment_name", EXPERIMENT_NAME)),
        "auto_plot_on_end": auto_plot_on_end,
        "pbt_promotion_eligible": bool(
            getattr(args, "pbt_promotion_eligible", False)
        ),
        "pbt_selection_mode": _pbt_selection_mode(args),
        "pbt_selection_min_sharpe": float(
            getattr(args, "pbt_selection_min_sharpe", 0.0)
        ),
        "pbt_selection_max_flat_action_fraction": float(
            getattr(args, "pbt_selection_max_flat_action_fraction", 0.98)
        ),
        "pbt_lr_min": pbt_lr_min,
        "pbt_lr_max": pbt_lr_max,
        "resume_rewind_to_checkpoint": bool(
            getattr(args, "resume_rewind_to_checkpoint", False)
        ),
        "restore_progress_steps": int(getattr(args, "restore_progress_steps", 0)),
        "restore_progress_updates": int(
            getattr(args, "restore_progress_updates", 0)
        ),
    }
    from itrader.models.registry import validate_representation_config

    validate_representation_config(config)
    return config


def write_run_manifest(base_config: dict, args, data_path: Path):
    from itrader.models.registry import model_identity

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_name = str(getattr(args, "experiment_name", EXPERIMENT_NAME))
    source_provenance = dict(base_config.get("source_provenance", {}))
    dataset_hash = _dataset_sha256(data_path)
    manifest = {
        "run_id": run_id,
        "api_stack_id": str(base_config["api_stack_id"]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "runtime": _runtime_manifest(),
        "git_commit": source_provenance.get("git_commit", _git_commit()),
        "source_tree_fingerprint": source_provenance,
        "seed": int(args.seed),
        "pbt_population_seed": _pbt_population_seed(args),
        "resume_existing": bool(getattr(args, "resume", False)),
        "resume_rewind_to_checkpoint": bool(
            getattr(args, "resume_rewind_to_checkpoint", False)
        ),
        "config_hash": _config_hash(base_config),
        "solver_identity": model_identity(base_config),
        "comparison_config_hash": _comparison_config_hash(base_config, dataset_hash),
        "observation_schema": _observation_schema_manifest(base_config),
        "evaluation_panel_sha256": _evaluation_panel_sha256(base_config),
        "dataset_path": str(data_path),
        "dataset_sha256": dataset_hash,
        "results_root": str(_results_root(args)),
        "start_idx_mode": args.start_idx_mode,
        "ret_norm_reset": args.ret_norm_reset,
        "is_pbt": bool(args.pbt or args.cpu_pbt),
        "is_cpu_pbt": bool(args.cpu_pbt),
        "pbt_primary_metric": _pbt_metric_name(args) if (args.pbt or args.cpu_pbt) else None,
        "pbt_time_attr": "pbt_progress_steps" if (args.pbt or args.cpu_pbt) else None,
        "pbt_interval": int(getattr(args, "pbt_interval", 3)),
        "pbt_synch": bool(getattr(args, "pbt_synch", False)),
        "pbt_interval_steps": int(getattr(args, "pbt_interval", 3))
        * int(getattr(args, "train_batch_size", 4800)),
        "checkpoint_interval": int(getattr(args, "checkpoint_interval", 3)),
        "target_environment_steps": int(args.iters)
        * int(getattr(args, "train_batch_size", 4800)),
        "pbt_mutations": _pbt_mutation_manifest(args),
    }
    # Tune 2.9 can truncate files created in its experiment directory on
    # Windows. Repository-owned provenance therefore lives in a sibling root.
    out_dir = _results_root(args) / "_stable_runs" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_manifest_{run_id}.json"
    _atomic_write_json(out_path, manifest)
    print(f"Run manifest written to: {out_path}")
    return manifest, out_path


def _run_completion_status(
    trials: list[dict],
    *,
    target_steps: int,
) -> tuple[str, list[str]]:
    """Reject Tune's TERMINATED label when the declared exposure was not reached."""
    incomplete_trial_ids = []
    for trial in trials:
        last_result = trial.get("last_result") or {}
        completed_steps = int(last_result.get("pbt_progress_steps", 0) or 0)
        if completed_steps < int(target_steps):
            incomplete_trial_ids.append(str(trial.get("trial_id", "")))
    return (
        ("completed" if not incomplete_trial_ids else "incomplete"),
        incomplete_trial_ids,
    )

# ────────────────────────────────────────────────────────────────────────────────
# 2) Environment factory (with proper .spec for TimeLimit)
# ────────────────────────────────────────────────────────────────────────────────
def make_env(cfg):
    e = ITraderEnv(cfg)
    e.spec = EnvSpec("ITraderEnv")
    return e

tune.register_env("ITraderEnv", make_env)
# ────────────────────────────────────────────────────────────────────────────────
# 3) Custom callback to expose samples/sec
# ────────────────────────────────────────────────────────────────────────────────
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class NativeCombinedCallback(DefaultCallbacks):
    def __init__(self, plot_interval: int = 50):
        super().__init__()
        self.ps_metrics = NativePSMetrics()
        self.auto_plotter = AutoPlotterCallback(
            plot_interval=plot_interval,
            plot_at_end_of_loop=False,
        )

    def on_episode_end(self, *, episode, metrics_logger=None, env=None, env_runner=None, **kwargs):
        return self.ps_metrics.on_episode_end(
            episode=episode,
            metrics_logger=metrics_logger,
            env=env,
            env_runner=env_runner,
            **kwargs,
        )

    def on_train_result(self, *, algorithm, result, **kwargs):
        self.ps_metrics.on_train_result(algorithm=algorithm, result=result)
        return self.auto_plotter.on_train_result(
            algorithm=algorithm,
            result=result,
            **kwargs,
        )


class NativeMetricsOnlyCallback(DefaultCallbacks):
    """MetricsLogger adapter for native-stack PBT without plot/file collisions."""

    def __init__(self):
        super().__init__()
        self.ps_metrics = NativePSMetrics()

    def on_episode_end(self, *, episode, metrics_logger=None, env=None, env_runner=None, **kwargs):
        return self.ps_metrics.on_episode_end(
            episode=episode,
            metrics_logger=metrics_logger,
            env=env,
            env_runner=env_runner,
            **kwargs,
        )

    def on_train_result(self, *, algorithm, result, **kwargs):
        return self.ps_metrics.on_train_result(
            algorithm=algorithm,
            result=result,
            **kwargs,
        )

#  ────────────────────────────────────────────────────────────────────────────────
# 4) The “training function” for Tune
#    This builds the PPOAlgo, loops train(), and reports key scalars
# ────────────────────────────────────────────────────────────────────────────────
def _extract_custom_metrics(result: dict) -> dict:
    """Prefer native EnvRunner metrics; do not merge RLlib nestings."""
    candidates = (
        result.get("env_runners", {}).get("custom_metrics"),
        result.get("custom_metrics"),
    )
    for custom_metrics in candidates:
        if isinstance(custom_metrics, dict) and custom_metrics:
            return custom_metrics
    return {}


def _numeric_custom_metric_items(custom_metrics: dict) -> list[tuple[str, float]]:
    """Project native per-episode sums/counts into stable report means."""
    metric_names: set[str] = set()
    for key in custom_metrics:
        key_text = str(key)
        if key_text.endswith("__sum"):
            metric_names.add(key_text[:-5])
        elif key_text.endswith("__count"):
            metric_names.add(key_text[:-7])

    items: list[tuple[str, float]] = []
    for name in sorted(metric_names):
        if not (
            name.startswith(
                ("act_", "rc_", "economic_", "optimization_", "family_", "horizon_", "regime_")
            )
            or name in {"success", "task_horizon"}
        ):
            continue
        mean = mean_from_totals(custom_metrics, name, required=True)
        items.append((f"{name}_mean", mean))
    return items


def _extract_report(result: dict, default_iteration: int) -> dict:
    learner_stats = result.get("learners", {}).get("default_policy", {})
    if not isinstance(learner_stats, dict):
        learner_stats = {}
    training_custom = _extract_custom_metrics(result)
    evaluation_result = result.get("evaluation", {}) or {}
    evaluation_custom = _extract_custom_metrics(evaluation_result)
    training_means = dict(_numeric_custom_metric_items(training_custom))
    evaluation_means = dict(_numeric_custom_metric_items(evaluation_custom))

    def custom_value(metrics: dict, key: str):
        return metrics.get(f"{key}_mean")

    def custom_number(means: dict, key: str) -> float:
        value = custom_value(means, key)
        return float(value) if value is not None else float("nan")

    optimization_reward = custom_value(training_means, "optimization_reward")
    episode_len_mean = result.get("env_runners", {}).get("episode_len_mean")

    report = {
        "training_iteration": int(result.get("training_iteration", default_iteration)),
        "time_this_iter_s": result.get("time_this_iter_s"),
        "time_total_s": result.get("time_total_s"),
        "training/optimization_reward_mean": (
            float(optimization_reward) if optimization_reward is not None else float("nan")
        ),
        "training/gross_return_mean": custom_number(
            training_means, "economic_gross_return"
        ),
        "training/net_return_mean": custom_number(
            training_means, "economic_net_return"
        ),
        "training/net_sharpe_per_sqrt_bar_mean": custom_number(
            training_means, "economic_net_sharpe_per_sqrt_bar"
        ),
        "evaluation/optimization_reward_mean": custom_number(
            evaluation_means, "optimization_reward"
        ),
        "evaluation/gross_return_mean": custom_number(
            evaluation_means, "economic_gross_return"
        ),
        "evaluation/net_return_mean": custom_number(
            evaluation_means, "economic_net_return"
        ),
        "evaluation/net_sharpe_per_sqrt_bar_mean": custom_number(
            evaluation_means, "economic_net_sharpe_per_sqrt_bar"
        ),
        "episode_len_mean": float(episode_len_mean) if episode_len_mean is not None else float("nan"),
        "policy_loss": float(learner_stats.get("policy_loss", float("nan"))),
        "value_loss": float(learner_stats.get("vf_loss", float("nan"))),
        "entropy": float(learner_stats.get("entropy", float("nan"))),
        "approx_kl": float(learner_stats.get("mean_kl_loss", float("nan"))),
        "clip_fraction": float(learner_stats.get("clip_fraction", float("nan"))),
        "clip_fraction_clipped_count": learner_stats.get("clip_fraction_clipped_count"),
        "clip_fraction_valid_count": learner_stats.get("clip_fraction_valid_count"),
        "clip_fraction_version": learner_stats.get("clip_fraction_version"),
        "clip_fraction_scope": learner_stats.get("clip_fraction_scope"),
        "explained_variance": float(learner_stats.get("vf_explained_var", float("nan"))),
        "total_loss": float(learner_stats.get("total_loss", float("nan"))),
        "grad_norm": float(
            learner_stats.get("gradients_default_optimizer_global_norm", float("nan"))
        ),
        "learning_rate": float(
            learner_stats.get("default_optimizer_learning_rate", float("nan"))
        ),
        "kl_coeff": float(learner_stats.get("curr_kl_coeff", float("nan"))),
    }
    if report.get("time_this_iter_s"):
        report["iters_per_sec"] = 1.0 / max(float(report["time_this_iter_s"]), 1e-8)

    for key, value in training_means.items():
        report[f"custom_metrics/{key}"] = value
    for key, value in evaluation_means.items():
        report[f"evaluation/custom_metrics/{key}"] = value
    for key in ("episode_len_mean", "iters_per_sec"):
        report.setdefault(key, float("nan"))
    return report


def _pbt_metric_name(source) -> str:
    mode = _pbt_selection_mode(source)
    if mode == "nonflat_net":
        return PBT_SEARCH_METRIC
    if mode == "promotion":
        return PBT_PROMOTION_METRIC
    return PBT_PRIMARY_METRIC


def _pbt_selection_mode(source) -> str:
    if isinstance(source, dict):
        promotion_eligible = bool(source.get("pbt_promotion_eligible", False))
        configured = source.get("pbt_selection_mode", "raw_net")
    else:
        promotion_eligible = bool(getattr(source, "pbt_promotion_eligible", False))
        configured = getattr(source, "pbt_selection_mode", "raw_net")
    mode = "promotion" if promotion_eligible else str(configured or "raw_net")
    if mode not in {"raw_net", "nonflat_net", "promotion"}:
        raise ValueError(
            "pbt_selection_mode must be raw_net, nonflat_net, or promotion"
        )
    return mode


def _flatten_scalar_metrics(result: dict) -> dict:
    flat_metrics = {}
    stack = [("", result)]
    while stack:
        prefix, obj = stack.pop()
        if not isinstance(obj, dict):
            continue
        for key, value in obj.items():
            full_key = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(value, dict):
                stack.append((full_key, value))
            elif isinstance(value, (int, float, bool, str, np.floating, np.integer)):
                flat_metrics[full_key] = (
                    float(value) if isinstance(value, (np.floating, np.integer)) else value
                )
    return flat_metrics


def _persisted_scalar_metrics(result: dict, report: dict) -> dict:
    """Merge RLlib metrics with scalar values computed by the reporting seam."""
    persisted = _flatten_scalar_metrics(result)
    persisted.update(_flatten_scalar_metrics(report))
    return persisted


def _append_json_line(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _existing_report_fieldnames(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            return next(reader)
        except StopIteration:
            return []


def _report_fieldnames(existing: list[str], row: dict) -> list[str]:
    ordered = list(existing)
    for key in row.keys():
        if key not in ordered:
            ordered.append(key)
    return ordered


def _rewrite_report_csv(path: Path, fieldnames: list[str]) -> None:
    rows: list[dict[str, object]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _append_report_row(path: Path, row: dict) -> None:
    existing = _existing_report_fieldnames(path)
    fieldnames = _report_fieldnames(existing, row)
    if path.exists() and existing != fieldnames:
        _rewrite_report_csv(path, fieldnames)
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if fh.tell() == 0:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _rewind_iteration_artifacts(
    restored_iteration: int,
    report_path: Path,
    jsonl_paths: list[Path],
) -> None:
    """Drop rows superseded when PBT replaces a trial with an older checkpoint."""
    boundary = int(restored_iteration)
    if report_path.exists():
        with report_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [
                row
                for row in reader
                if int(row.get("training_iteration", 0)) <= boundary
            ]
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, report_path)

    for path in jsonl_paths:
        if not path.exists():
            continue
        retained = [
            payload
            for line in path.read_text(encoding="utf-8").splitlines()
            if (payload := json.loads(line)).get("training_iteration", 0) <= boundary
        ]
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for payload in retained:
                handle.write(json.dumps(payload, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def _rewind_resume_progress(
    *,
    progress: PBTProgressClock,
    restored_iteration: int,
    stable_trial_dir: Path,
    local_metrics_path: Path,
) -> dict[str, object]:
    """Align an interrupted target clock with Tune's restored checkpoint.

    This is deliberately opt-in. Ordinary PBT exploitation must preserve the
    target-owned clock even when it restores an older donor iteration.
    """
    boundary = int(restored_iteration)
    index_path = stable_trial_dir / "checkpoint_index.jsonl"
    stable_metrics_path = stable_trial_dir / "metrics_full.jsonl"
    report_path = stable_trial_dir / "report_rows.csv"

    checkpoint_rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    checkpoint = next(
        (
            row
            for row in reversed(checkpoint_rows)
            if int(row.get("training_iteration", -1)) == boundary
            and bool(row.get("reported_to_tune", False))
        ),
        None,
    )
    if checkpoint is None:
        raise RuntimeError(
            "Resume recovery requires a reported checkpoint-index row for "
            f"restored iteration {boundary}"
        )

    metric_rows = [
        json.loads(line)
        for line in stable_metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    metric = next(
        (
            row
            for row in reversed(metric_rows)
            if int(row.get("training_iteration", -1)) == boundary
        ),
        None,
    )
    if metric is None:
        raise RuntimeError(
            "Resume recovery requires a stable metric row for "
            f"restored iteration {boundary}"
        )

    checkpoint_steps = int(checkpoint["pbt_progress_steps"])
    checkpoint_updates = int(metric["pbt_progress_updates"])
    before = {
        "training_iteration": boundary,
        "pbt_progress_steps": int(progress.completed_steps),
        "pbt_progress_updates": int(progress.updates),
    }
    if (
        progress.completed_steps == checkpoint_steps
        and progress.updates == checkpoint_updates
    ):
        return {"rewound": False, "before": before, "after": before}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot_dir = stable_trial_dir / "_resume_recovery" / timestamp
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    snapshot_sources = {
        "pbt_progress.json": stable_trial_dir / "pbt_progress.json",
        "stable_metrics_full.jsonl": stable_metrics_path,
        "report_rows.csv": report_path,
        "checkpoint_index.jsonl": index_path,
        "trial_status.json": stable_trial_dir / "trial_status.json",
        "local_metrics_full.jsonl": local_metrics_path,
    }
    for destination_name, source in snapshot_sources.items():
        if source.exists():
            shutil.copy2(source, snapshot_dir / destination_name)

    _rewind_iteration_artifacts(
        restored_iteration=boundary,
        report_path=report_path,
        jsonl_paths=[stable_metrics_path, local_metrics_path, index_path],
    )
    progress.completed_steps = checkpoint_steps
    progress.updates = checkpoint_updates
    progress.save(stable_trial_dir / "pbt_progress.json")
    after = {
        "training_iteration": boundary,
        "pbt_progress_steps": checkpoint_steps,
        "pbt_progress_updates": checkpoint_updates,
    }
    receipt = {
        "schema_version": 1,
        "rewound": True,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_dir": str(snapshot_dir),
        "before": before,
        "after": after,
    }
    _append_json_line(stable_trial_dir / "resume_recovery.jsonl", receipt)
    return receipt


def _seed_restored_progress(
    progress: PBTProgressClock,
    *,
    restore_source: str | None,
    progress_state_exists: bool,
    configured_steps: int,
    configured_updates: int,
) -> bool:
    """Seed a new recovery trial with the target exposure of its checkpoint."""
    if not restore_source or progress_state_exists:
        return False
    steps = int(configured_steps)
    updates = int(configured_updates)
    if steps == 0 and updates == 0:
        return False
    if steps < 0 or steps >= progress.target_steps or updates < 0:
        raise ValueError(
            "restore progress must be nonnegative and below the target step budget"
        )
    progress.completed_steps = steps
    progress.updates = updates
    return True


def _checkpoint_from_save_result(saved):
    if isinstance(saved, str):
        return Checkpoint.from_directory(saved), saved
    if isinstance(saved, Checkpoint):
        return saved, getattr(saved, "path", "Checkpoint")
    if hasattr(saved, "checkpoint") and isinstance(saved.checkpoint, Checkpoint):
        checkpoint = saved.checkpoint
        return checkpoint, getattr(checkpoint, "path", "Checkpoint")
    return None, str(saved)


def _trial_context_values() -> tuple[str, str, str]:
    """Read the active trial identity from Ray 2.58's Tune context."""
    context = tune.get_context()
    return (
        context.get_trial_dir(),
        context.get_trial_id(),
        context.get_trial_name(),
    )


def _incoming_tune_checkpoint():
    """Read the incoming checkpoint through Ray Tune's function API."""
    return tune.get_checkpoint()


def _report_to_tune(metrics: dict, checkpoint=None) -> None:
    """Report metrics and checkpoints through Ray Tune's function API."""
    tune.report(metrics, checkpoint=checkpoint)


def _copy_stable_checkpoint(
    checkpoint_dir: str | Path,
    *,
    stable_trial_dir: Path,
    training_iteration: int,
    api_stack_id: str,
) -> Path:
    """Copy a checkpoint before Ray AIR takes ownership of its source path."""
    source = Path(checkpoint_dir)
    destination = (
        stable_trial_dir
        / "checkpoints"
        / f"iter_{int(training_iteration):06d}_{datetime.now(timezone.utc):%H%M%S%f}"
    )
    shutil.copytree(source, destination)
    try:
        checkpoint_sha256(destination, api_stack_id=api_stack_id)
    except ValueError as exc:
        raise RuntimeError(f"Stable checkpoint is incomplete: {exc}") from exc
    return destination


def _validate_pbt_restore_checkpoint(
    checkpoint_dir: str | Path,
    provenance: dict,
) -> dict[str, object]:
    """Prove a Tune restore copy is byte-identical to its selected donor."""
    api_stack_id = str(provenance.get("api_stack_id", ""))
    observed = checkpoint_sha256(checkpoint_dir, api_stack_id=api_stack_id)
    expected = str(provenance.get("donor_checkpoint_sha256", ""))
    if not expected or observed != expected:
        raise RuntimeError(
            "PBT restore checkpoint does not match selected donor checkpoint"
        )
    return {
        "schema_version": 1,
        "api_stack_id": api_stack_id,
        "exploit_id": str(provenance.get("exploit_id", "")),
        "observed_checkpoint_path": str(Path(checkpoint_dir).resolve()),
        "observed_checkpoint_sha256": observed,
        "expected_checkpoint_sha256": expected,
        "digest_match": True,
    }


def _validate_tune_restore_checkpoint(
    checkpoint_dir: str | Path,
    provenance: dict,
    *,
    stable_trial_dir: Path,
    target_trial_id: str,
    restore_generation: int,
    api_stack_id: str,
) -> dict[str, object]:
    """Distinguish a fresh exploit restore from an indexed target self-resume."""
    if provenance and provenance.get("api_stack_id") != api_stack_id:
        raise RuntimeError("PBT restore API stack does not match the active trainer")
    observed = checkpoint_sha256(checkpoint_dir, api_stack_id=api_stack_id)
    if not provenance:
        return {
            "schema_version": 1,
            "receipt_kind": "tune_self_resume",
            "api_stack_id": api_stack_id,
            "observed_checkpoint_path": str(Path(checkpoint_dir).resolve()),
            "observed_checkpoint_sha256": observed,
            "expected_checkpoint_sha256": observed,
            "digest_match": True,
            "lineage_verified": False,
        }

    exploit_id = str(provenance.get("exploit_id", ""))
    expected_target = str(provenance.get("target_trial_id", ""))
    expected_donor = str(provenance.get("donor_checkpoint_sha256", ""))
    if not exploit_id or not expected_target or expected_target != target_trial_id:
        raise RuntimeError("PBT restore receipt target does not match the active Tune trial")

    if expected_donor and observed == expected_donor:
        receipt = _validate_pbt_restore_checkpoint(checkpoint_dir, provenance)
        receipt.update(
            {
                "receipt_kind": "pbt_exploit",
                "lineage_verified": True,
            }
        )
        return receipt

    prior_receipt_path = stable_trial_dir / "restore_receipts" / f"{exploit_id}.json"
    try:
        prior_receipt = json.loads(prior_receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "PBT restore checkpoint does not match selected donor checkpoint"
        ) from exc
    prior_receipt_valid = (
        prior_receipt.get("receipt_kind") == "pbt_exploit"
        and prior_receipt.get("exploit_id") == exploit_id
        and prior_receipt.get("target_trial_id") == target_trial_id
        and prior_receipt.get("digest_match") is True
        and prior_receipt.get("observed_checkpoint_sha256") == expected_donor
        and prior_receipt.get("expected_checkpoint_sha256") == expected_donor
    )
    if not prior_receipt_valid:
        raise RuntimeError("PBT restore checkpoint does not match selected donor checkpoint")

    index_path = stable_trial_dir / "checkpoint_index.jsonl"
    try:
        indexed_rows = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "PBT restore checkpoint does not match selected donor checkpoint"
        ) from exc
    indexed_match = next(
        (
            row
            for row in reversed(indexed_rows)
            if row.get("trial_id") == target_trial_id
                and row.get("api_stack_id") == api_stack_id
                and row.get("checkpoint_sha256") == observed
                and int(row.get("pbt_restore_generation", -1)) == int(restore_generation)
                and row.get("pbt_parent_exploit_id") == exploit_id
                and (
                    row.get("reported_to_tune") is True
                    or row.get("post_sync_evaluation") is True
                )
            ),
        None,
    )
    if indexed_match is None:
        raise RuntimeError("PBT restore checkpoint does not match selected donor checkpoint")

    return {
        "schema_version": 1,
        "receipt_kind": "tune_self_resume",
        "api_stack_id": api_stack_id,
        "observed_checkpoint_path": str(Path(checkpoint_dir).resolve()),
        "observed_checkpoint_sha256": observed,
        "expected_checkpoint_sha256": observed,
        "digest_match": True,
        "lineage_verified": True,
        "lineage_exploit_id": exploit_id,
        "lineage_donor_checkpoint_sha256": expected_donor,
        "indexed_training_iteration": indexed_match.get("training_iteration"),
        "indexed_progress_steps": indexed_match.get("pbt_progress_steps"),
    }


def _update_best_checkpoint_record(
    path: Path,
    *,
    metric: str,
    report: dict,
    stable_checkpoint: Path,
    score: float,
    economically_eligible: bool,
    selection_scope: str = "reported_checkpoint_boundaries",
    provenance: dict | None = None,
) -> bool:
    """Point to the best retained cadence checkpoint for fresh-process replay."""
    candidate = float(score)
    if not math.isfinite(candidate):
        return False
    existing_score = _best_checkpoint_score(path)
    if candidate <= existing_score:
        return False
    payload = {
        "schema_version": 1,
        "metric": metric,
        "score": candidate,
        "economically_eligible": bool(economically_eligible),
        "training_iteration": int(report["training_iteration"]),
        "pbt_progress_steps": int(report.get("pbt_progress_steps", 0)),
        "stable_checkpoint": str(stable_checkpoint),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": selection_scope,
        "requires_fresh_process_replay": True,
    }
    if provenance:
        payload.update(provenance)
    _atomic_write_json(path, payload)
    return True


def _checkpoint_selection_scope(
    *,
    report_to_tune: bool,
    selection_score: float,
) -> str:
    """Describe why a best-checkpoint pointer was retained without overstating it."""
    if report_to_tune:
        return "reported_checkpoint_boundary"
    if math.isfinite(float(selection_score)) and float(selection_score) > 0.0:
        return "eligible_evaluation_improvement"
    return "ineligible_evaluation_improvement_for_forensics"


def _best_checkpoint_score(path: Path) -> float:
    if not path.exists():
        return float("-inf")
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))["score"])
    except (OSError, ValueError, TypeError, KeyError):
        return float("-inf")


def _report_checkpoint_with_index(
    *,
    index_path: Path,
    record: dict,
    report: dict,
    checkpoint,
) -> None:
    """Persist checkpoint provenance before Tune can suspend the function."""
    _append_json_line(index_path, record)
    _report_to_tune(report, checkpoint=checkpoint)


def _remaining_training_steps(current_iteration: int, stop_iteration: int) -> range:
    """Return local steps still owed after an RLlib/PBT checkpoint restore."""
    remaining = max(0, int(stop_iteration) - int(current_iteration))
    return range(remaining)


def _training_env_runner_group(algo):
    """Return the required native training EnvRunner group."""
    training_group = getattr(algo, "env_runner_group", None)
    if training_group is None:
        raise TypeError("Active PPO requires a native training EnvRunner group")
    return training_group


def _env_runner_groups(algo):
    """Return the required native EnvRunner groups without legacy fallback."""
    training_group = _training_env_runner_group(algo)
    evaluation_group = getattr(algo, "eval_env_runner_group", None)
    if evaluation_group is None:
        raise TypeError("Native checkpoint-consistent evaluation requires an evaluation EnvRunner group")
    return training_group, evaluation_group


def deterministic_evaluate(algo):
    """Synchronize current training weights and run greedy held-out evaluation."""
    _, evaluation_group = _env_runner_groups(algo)
    evaluation_group.sync_weights(
        from_worker_or_learner_group=algo.learner_group,
        inference_only=True,
    )
    return algo.evaluate()


def train_with_checkpoint_consistent_evaluation(algo):
    """Score the exact learner weights that the next checkpoint will persist."""
    result = algo.train()
    if "evaluation" not in result:
        return result

    post_sync_result = deterministic_evaluate(algo)
    evaluation = post_sync_result.get("evaluation", post_sync_result)
    if isinstance(evaluation, dict):
        result["evaluation"] = evaluation
    return result


def _set_training_reward_overrides(algo, overrides: dict[str, float]) -> None:
    """Update training environments without changing the evaluation task panel."""
    training_group = _training_env_runner_group(algo)

    def update_runner(env_runner):
        vector_env = getattr(env_runner, "env", None)
        if vector_env is None:
            raise RuntimeError("Native EnvRunner does not expose its vectorized training env")
        # Ray 2.58 wraps the Gymnasium VectorEnv in DictInfoToList. VectorWrapper
        # exposes the wrapped vector env as `.env`, whose `call` method dispatches
        # through each sub-env's wrapper chain to the ITraderEnv method.
        wrapped_vector_env = getattr(vector_env, "env", vector_env)
        call_each = getattr(wrapped_vector_env, "call", None)
        if not callable(call_each):
            raise RuntimeError("Native EnvRunner vector env does not support per-env calls")
        updated = call_each("set_task_sampler_overrides", overrides)
        if not updated:
            raise RuntimeError("Native EnvRunner has no training envs to update")

    training_group.foreach_env_runner(update_runner)


def training_fn(config):
    from itrader.models.registry import validate_representation_config

    validate_representation_config(config)
    _assert_frozen_run_provenance(config)
    # Always use Tune trial dir as the single source of truth for ALL outputs.
    trial_dir_raw, trial_id, trial_name = _trial_context_values()
    trial_dir = Path(trial_dir_raw)
    stable_root = Path(
        str(
            config.get(
                "stable_metrics_root",
                Path.home() / "ray_results" / "_stable_metrics",
            )
        )
    )
    stable_trial_dir = stable_root / trial_dir.name

    from itrader.rl_lib.train.ray_native_config import build_native_ppo_config

    from itrader.rl_lib.train.ray_native_ppo import DiagnosticPPO

    algo = DiagnosticPPO(config=build_native_ppo_config(config))

    # --- Restore for PBT exploit/explore (weight inheritance) ---------------
    ckpt = _incoming_tune_checkpoint()
    requested_restore = str(config.get("restore_ckpt", "") or "").strip()
    pbt_restore_provenance = dict(config.get("pbt_restore_provenance", {}) or {})
    restore_generation = int(config.get("pbt_restore_generation", 0) or 0)
    parent_exploit_id = str(pbt_restore_provenance.get("exploit_id", ""))
    restore_source = None
    restore_receipt = None
    if requested_restore:
        if pbt_restore_provenance:
            raise RuntimeError(
                "Manual --restore_ckpt cannot impersonate a PBT exploit receipt"
            )
        algo.restore(requested_restore)
        restore_source = requested_restore
        print(f"Restored checkpoint from --restore_ckpt: {requested_restore}")
    elif ckpt is not None:
        restore_dir = ckpt.to_directory()
        restore_receipt = _validate_tune_restore_checkpoint(
            restore_dir,
            pbt_restore_provenance,
            stable_trial_dir=stable_trial_dir,
            target_trial_id=trial_id,
            restore_generation=restore_generation,
            api_stack_id=str(config["api_stack_id"]),
        )
        algo.restore(restore_dir)
        restore_source = restore_dir
        print(f"Restored checkpoint from Tune checkpoint: {restore_dir}")
    elif pbt_restore_provenance:
        raise RuntimeError("PBT restore provenance is present without a Tune checkpoint")

    if restore_source:
        from itrader.models.registry import validate_native_restored_model

        validate_native_restored_model(algo, config)


    checkpoint_interval = int(config.get("checkpoint_interval", 3))
    full_metrics_path = trial_dir / "rllib" / "metrics_full.jsonl"
    full_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    stable_trial_dir.mkdir(parents=True, exist_ok=True)
    stable_metrics_path = stable_trial_dir / "metrics_full.jsonl"
    stable_report_csv = stable_trial_dir / "report_rows.csv"
    trial_status_path = stable_trial_dir / "trial_status.json"
    progress_state_path = stable_trial_dir / "pbt_progress.json"
    target_steps = int(
        config.get(
            "pbt_target_steps",
            int(config["stop_iters"])
            * int(config.get("train_batch_size_per_learner", 4800)),
        )
    )
    progress_state_exists = progress_state_path.exists()
    progress = PBTProgressClock.load(progress_state_path, target_steps=target_steps)
    if _seed_restored_progress(
        progress,
        restore_source=restore_source,
        progress_state_exists=progress_state_exists,
        configured_steps=int(config.get("restore_progress_steps", 0)),
        configured_updates=int(config.get("restore_progress_updates", 0)),
    ):
        progress.save(progress_state_path)
    if restore_source and bool(config.get("resume_rewind_to_checkpoint", False)):
        _rewind_resume_progress(
            progress=progress,
            restored_iteration=int(getattr(algo, "iteration", 0)),
            stable_trial_dir=stable_trial_dir,
            local_metrics_path=full_metrics_path,
        )
    if restore_source:
        from itrader.rl_lib.train.ppo_native_restore import (
            reconcile_native_ppo_after_restore,
        )

        learning_settings = reconcile_native_ppo_after_restore(
            algo,
            config,
            progress_steps=progress.completed_steps,
        )
        learning_settings.update(
            {
                "schedule_application_scope": "first_update_after_restore_at_target_progress",
                "subsequent_schedule_clock": "rllib_native_sampled_env_steps",
            }
        )
        if restore_receipt is None:
            restore_receipt = {
                "schema_version": 1,
                "receipt_kind": "manual_restore",
                "api_stack_id": config["api_stack_id"],
                "observed_checkpoint_path": str(Path(restore_source).resolve()),
                "observed_checkpoint_sha256": checkpoint_sha256(
                    restore_source, api_stack_id=str(config["api_stack_id"])
                ),
                "lineage_verified": False,
            }
        restore_receipt["learning_settings_reconciliation"] = learning_settings
    if restore_receipt is not None:
        if pbt_restore_provenance:
            expected_target = str(pbt_restore_provenance.get("target_trial_id", ""))
            if expected_target and expected_target != trial_id:
                raise RuntimeError(
                    "PBT restore receipt target does not match the active Tune trial"
                )
        restore_receipt.update(
            {
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "target_trial_id": trial_id,
                "target_trial_name": trial_name,
                "restore_generation": restore_generation,
                "parent_exploit_id": parent_exploit_id or None,
                "target_clock_at_restore": int(progress.completed_steps),
                "target_updates_at_restore": int(progress.updates),
                "donor_checkpoint_path": pbt_restore_provenance.get(
                    "donor_checkpoint_path"
                ),
                "donor_trial_id": pbt_restore_provenance.get("donor_trial_id"),
                "donor_progress_steps": pbt_restore_provenance.get(
                    "donor_progress_steps"
                ),
                "source_tree_sha256": str(
                    (config.get("source_provenance") or {}).get(
                        "source_tree_sha256",
                        "",
                    )
                ),
                "dataset_sha256": str(
                    (config.get("frozen_provenance") or {}).get(
                        "dataset_sha256",
                        config.get("dataset_sha256", ""),
                    )
                ),
                "evaluation_panel_sha256": str(
                    (config.get("frozen_provenance") or {}).get(
                        "evaluation_panel_sha256",
                        config.get("evaluation_panel_sha256", ""),
                    )
                ),
            }
        )
        receipt_prefix = "manual_restore" if requested_restore else "self_resume"
        receipt_name = (
            f"{parent_exploit_id}.json"
            if restore_receipt.get("receipt_kind") == "pbt_exploit"
            else f"{receipt_prefix}_{restore_generation:04d}_{datetime.now(timezone.utc):%H%M%S%f}.json"
        )
        _atomic_write_json(
            stable_trial_dir / "restore_receipts" / receipt_name,
            restore_receipt,
        )
    data_path = Path(str(config["env_config"]["df_path"]))
    source_provenance = dict(config.get("source_provenance", {}))
    dataset_hash = _dataset_sha256(data_path)
    trial_identity = {
        "schema_version": 1,
        "api_stack_id": str(config["api_stack_id"]),
        "runtime": _runtime_manifest(),
        "trial_id": trial_id,
        "trial_name": trial_name,
        "trial_dir": str(trial_dir),
        "stable_trial_dir": str(stable_trial_dir),
        "git_commit": source_provenance.get("git_commit", _git_commit()),
        "source_tree_fingerprint": source_provenance,
        "config_hash": _config_hash(config),
        "comparison_config_hash": _comparison_config_hash(config, dataset_hash),
        "observation_schema": _observation_schema_manifest(config),
        "solver_identity": validate_representation_config(config),
        "evaluation_panel_sha256": _evaluation_panel_sha256(config),
        "dataset_path": str(data_path),
        "dataset_sha256": dataset_hash,
        "pbt_restore_generation": restore_generation,
        "pbt_parent_exploit_id": parent_exploit_id or None,
    }
    generation_label = (
        f"generation_{restore_generation:04d}_{parent_exploit_id[:12]}"
        if parent_exploit_id
        else f"generation_{restore_generation:04d}_initial"
    )
    config_snapshot_path = stable_trial_dir / "resolved_configs" / f"{generation_label}.json"
    resolved_payload = {**trial_identity, "config": config}
    _atomic_write_json(config_snapshot_path, resolved_payload)
    _atomic_write_json(
        stable_trial_dir / "resolved_config.json",
        {**resolved_payload, "config_snapshot": str(config_snapshot_path)},
    )
    _append_json_line(
        stable_trial_dir / "config_history.jsonl",
        {
            **trial_identity,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "event": "checkpoint_restore" if restore_source else "initial_start",
            "restored_iteration": int(getattr(algo, "iteration", 0)),
            "pbt_progress_steps": progress.completed_steps,
            "restore_source": restore_source,
            "restore_generation": restore_generation,
            "parent_exploit_id": parent_exploit_id or None,
            "api_stack_id": str(config["api_stack_id"]),
            "config_snapshot": str(config_snapshot_path),
            "config": config,
        },
    )
    _atomic_write_json(
        trial_status_path,
        {
            **trial_identity,
            "status": "running",
            "last_iteration": int(getattr(algo, "iteration", 0)),
            "pbt_progress_steps": progress.completed_steps,
        },
    )

    current_iteration = int(getattr(algo, "iteration", 0))
    last_iteration = current_iteration
    train_batch_size = int(config.get("train_batch_size_per_learner", 4800))
    checkpoint_interval_steps = int(
        config.get(
            "checkpoint_interval_steps",
            int(config.get("checkpoint_interval", 3)) * train_batch_size,
        )
    )
    while progress.remaining:
        cost_schedule = config.get("optimization_cost_schedule", ())
        optimization_cost_bps = _training_optimization_cost_bps(
            config,
            completed_steps=progress.completed_steps,
        )
        if cost_schedule:
            _set_training_reward_overrides(
                algo,
                {"cost_bps": optimization_cost_bps},
            )
        if config.get("checkpoint_consistent_evaluation", False):
            res = train_with_checkpoint_consistent_evaluation(algo)
        else:
            res = algo.train()
        report = _extract_report(res, last_iteration + 1)
        last_iteration = int(report["training_iteration"])
        previous_progress = progress.completed_steps
        sampled_steps = progress.advance(res)
        report["pbt_progress_steps"] = progress.completed_steps
        report["pbt_progress_updates"] = progress.updates
        report["pbt_restore_generation"] = restore_generation
        report["pbt_parent_exploit_id"] = parent_exploit_id
        report["env_steps_this_update"] = sampled_steps
        if optimization_cost_bps is not None:
            report["training/optimization_cost_bps"] = optimization_cost_bps
        report[PBT_PROMOTION_METRIC] = economic_selection_score(
            net_return=report["evaluation/net_return_mean"],
            net_sharpe=report["evaluation/net_sharpe_per_sqrt_bar_mean"],
            flat_action_fraction=report.get(
                "evaluation/custom_metrics/act_flat_frac_mean",
                1.0,
            ),
            min_net_sharpe=float(config.get("pbt_selection_min_sharpe", 0.0)),
            max_flat_action_fraction=float(
                config.get("pbt_selection_max_flat_action_fraction", 0.98)
            ),
        )
        report[PBT_SEARCH_METRIC] = economic_search_score(
            net_return=report["evaluation/net_return_mean"],
            flat_action_fraction=report.get(
                "evaluation/custom_metrics/act_flat_frac_mean",
                1.0,
            ),
            max_flat_action_fraction=float(
                config.get("pbt_selection_max_flat_action_fraction", 0.98)
            ),
        )
        progress.save(progress_state_path)

        checkpoint = None
        ckpt_label = None
        stable_checkpoint = None
        stable_checkpoint_digest = None
        cadence_due = (
            checkpoint_interval > 0
            and crossed_interval(
                previous_progress,
                progress.completed_steps,
                checkpoint_interval_steps,
            )
        )
        terminal_due = checkpoint_interval > 0 and not progress.remaining
        best_checkpoint_path = stable_trial_dir / "best_checkpoint.json"
        retention = checkpoint_retention_decision(
            cadence_due=cadence_due,
            terminal_due=terminal_due,
            selection_score=float(report[PBT_PROMOTION_METRIC]),
            current_best_score=_best_checkpoint_score(best_checkpoint_path),
            synchronous_pbt=bool(config.get("pbt_synch", False)),
        )
        retained_as_best = False
        if retention.retain_stable:
            # Never reuse a mutable checkpoint directory. Tune may copy the
            # artifact after tune.report returns, and PBT must receive an
            # immutable snapshot of the state whose metrics it ranked.
            ckpt_root = trial_dir / "rllib_checkpoints"
            ckpt_root.mkdir(parents=True, exist_ok=True)
            ckpt_dir = ckpt_root / f"iter_{last_iteration:06d}_{datetime.now(timezone.utc):%H%M%S%f}"
            saved = algo.save(str(ckpt_dir))
            checkpoint, ckpt_label = _checkpoint_from_save_result(saved)
            if checkpoint is None:
                raise RuntimeError(f"Checkpoint produced unexpected type: {ckpt_label}")
            stable_checkpoint = _copy_stable_checkpoint(
                ckpt_label,
                stable_trial_dir=stable_trial_dir,
                training_iteration=last_iteration,
                api_stack_id=str(config["api_stack_id"]),
            )
            stable_checkpoint_digest = checkpoint_sha256(
                stable_checkpoint, api_stack_id=str(config["api_stack_id"])
            )
            from itrader.rl_lib.evaluation.replay_evidence import evaluation_payload_from_result

            evaluation_payload_digest = evaluation_payload_from_result(res, int(config["evaluation_duration"]))
            retained_as_best = _update_best_checkpoint_record(
                best_checkpoint_path,
                metric=_pbt_metric_name(config),
                report=report,
                stable_checkpoint=stable_checkpoint,
                score=float(report.get(_pbt_metric_name(config), float("nan"))),
                economically_eligible=float(report[PBT_PROMOTION_METRIC]) > 0.0,
                selection_scope=_checkpoint_selection_scope(
                    report_to_tune=retention.report_to_tune,
                    selection_score=float(report[PBT_PROMOTION_METRIC]),
                ),
                provenance={
                    "schema_version": 2,
                    "api_stack_id": str(config["api_stack_id"]),
                    "trial_id": trial_id,
                    "pbt_progress_updates": int(report["pbt_progress_updates"]),
                    "pbt_restore_generation": restore_generation,
                    "pbt_parent_exploit_id": parent_exploit_id or None,
                    "checkpoint_sha256": stable_checkpoint_digest,
                    "evaluation_payload_sha256": evaluation_payload_digest,
                    "config_hash": trial_identity["config_hash"],
                    "comparison_config_hash": trial_identity[
                        "comparison_config_hash"
                    ],
                    "observation_schema": trial_identity["observation_schema"],
                    "source_tree_sha256": str(
                        source_provenance.get("source_tree_sha256", "")
                    ),
                    "dataset_sha256": dataset_hash,
                    "evaluation_panel_sha256": trial_identity[
                        "evaluation_panel_sha256"
                    ],
                    "config_snapshot": str(config_snapshot_path),
                },
            )

        # Persist scalar-only flattened metrics for robust plotting fallback.
        full = _persisted_scalar_metrics(res, report)
        _append_json_line(full_metrics_path, full)
        _append_json_line(stable_metrics_path, full)
        _append_report_row(stable_report_csv, dict(report))
        _atomic_write_json(
            trial_status_path,
            {
                **trial_identity,
                "status": "running",
                "last_iteration": report["training_iteration"],
                "pbt_progress_steps": report["pbt_progress_steps"],
                "last_report": report,
            },
        )

        # --- Checkpointing for PBT (so exploit can copy weights) ------------
        checkpoint_record = None
        if checkpoint is not None:
            checkpoint_record = {
                "schema_version": 2,
                "api_stack_id": str(config["api_stack_id"]),
                "trial_id": trial_id,
                "training_iteration": report["training_iteration"],
                "pbt_progress_steps": report["pbt_progress_steps"],
                "pbt_progress_updates": report["pbt_progress_updates"],
                "pbt_restore_generation": restore_generation,
                "pbt_parent_exploit_id": parent_exploit_id or None,
                "tune_checkpoint": str(ckpt_label),
                "stable_checkpoint": str(stable_checkpoint),
                "checkpoint_sha256": stable_checkpoint_digest,
                "evaluation_payload_sha256": evaluation_payload_digest,
                "config_hash": trial_identity["config_hash"],
                "comparison_config_hash": trial_identity[
                    "comparison_config_hash"
                ],
                "observation_schema": trial_identity["observation_schema"],
                "source_tree_sha256": str(
                    source_provenance.get("source_tree_sha256", "")
                ),
                "dataset_sha256": dataset_hash,
                "evaluation_panel_sha256": trial_identity[
                    "evaluation_panel_sha256"
                ],
                "selection_metric": _pbt_metric_name(config),
                "selection_score": float(
                    report.get(_pbt_metric_name(config), float("nan"))
                ),
                "config_snapshot": str(config_snapshot_path),
                "reported_to_tune": retention.report_to_tune,
                "retained_as_best": retained_as_best,
                "post_sync_evaluation": bool(
                    config.get("checkpoint_consistent_evaluation", False)
                ),
            }
        if checkpoint is not None and retention.report_to_tune:
            _report_checkpoint_with_index(
                index_path=stable_trial_dir / "checkpoint_index.jsonl",
                record=checkpoint_record,
                report=report,
                checkpoint=checkpoint,
            )
            print(
                "Immutable checkpoint saved "
                f"@ env steps {progress.completed_steps} / RLlib iter {last_iteration}: {ckpt_label}"
            )
        else:
            if checkpoint_record is not None:
                _append_json_line(
                    stable_trial_dir / "checkpoint_index.jsonl",
                    checkpoint_record,
                )
                print(
                    (
                        "Eligible best checkpoint retained "
                        if float(report[PBT_PROMOTION_METRIC]) > 0.0
                        else "Best ineligible checkpoint retained for forensic replay "
                    )
                    + f"@ env steps {progress.completed_steps} / RLlib iter {last_iteration}: {ckpt_label}"
                )
            _report_to_tune(report)

    if bool(config.get("auto_plot_on_end", True)):
        try:
            plot_script = Path(__file__).resolve().parents[2] / "tools" / "plot_ray_results.py"
            metrics_jsonl = stable_trial_dir / "metrics_full.jsonl"
            out_dir = stable_trial_dir / "plots"
            cmd = [
                sys.executable,
                str(plot_script),
                "--metrics-jsonl",
                str(metrics_jsonl),
                "--out",
                str(out_dir),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if proc.returncode == 0:
                print(f"[FinalPlot] Generated final plots in: {out_dir}")
            else:
                print(f"[FinalPlot] Plot generation failed (code={proc.returncode})")
                if proc.stderr:
                    print(proc.stderr)
        except Exception as e:
            print(f"[FinalPlot] Suppressed final plotting error: {e}")

    _atomic_write_json(
        trial_status_path,
        {
            **trial_identity,
            "status": "completed",
            "last_iteration": last_iteration,
            "pbt_progress_steps": progress.completed_steps,
        },
    )
    algo.stop()
# ────────────────────────────────────────────────────────────────────────────────
# 5) Entrypoint: assemble Tune config & kick off the experiment
# ────────────────────────────────────────────────────────────────────────────────
def _resources_per_trial(args, base_config: dict) -> PlacementGroupFactory:
    requested_workers = int(args.workers)
    if requested_workers < 0:
        raise ValueError("workers must be non-negative")

    if args.cpu_pbt or bool(getattr(args, "cpu_only", False)):
        rollout_workers = min(requested_workers, 4)
        base_config.update({
            "num_gpus_per_learner": 0,
            "num_env_runners": rollout_workers,
        })
        return PlacementGroupFactory(
            [{"CPU": 1}] + [{"CPU": 1}] * rollout_workers
        )

    base_config.update({
        "num_gpus_per_learner": 1,
        "num_env_runners": requested_workers,
    })
    return PlacementGroupFactory(
        [{"GPU": 1, "CPU": 1}] + [{"CPU": 1}] * requested_workers
    )


def _apply_best_params(base_config: dict, best: dict) -> None:
    if base_config.get("representation_profile") == "r0_r1_v1":
        current_model = base_config["model"]
        requested_model = best.get("model", {})
        changed = any(current_model.get(key) != value for key, value in requested_model.items() if key != "custom_model_config")
        changed |= any(current_model.get("custom_model_config", {}).get(key) != value
                       for key, value in requested_model.get("custom_model_config", {}).items())
        protected = ("solver_model", "representation_profile", "observation_filter", "env_config", "evaluation_config")
        changed |= any(key in best and best[key] != base_config.get(key) for key in protected)
        if changed:
            raise ValueError("Best-parameter overrides cannot change fixed-recipe architecture or observation semantics")
    for key in [
        "lr", "clip_param", "lambda_", "gamma", "entropy_coeff",
        "minibatch_size", "num_epochs",
        "vf_clip_param", "vf_loss_coeff", "train_batch_size_per_learner",
    ]:
        if key in best:
            base_config[key] = best[key]

    model_cfg = best.get("model", {}).get("custom_model_config")
    if isinstance(model_cfg, dict):
        base_config["model"]["custom_model_config"].update(model_cfg)

    task_sampler_overrides = best.get("env_config", {}).get("task_sampler_overrides")
    if isinstance(task_sampler_overrides, dict):
        base_config["env_config"]["task_sampler_overrides"] = task_sampler_overrides


def _trial_training_seed(spec) -> int:
    return int(spec.config["seed"])


def _pbt_population_seed(args) -> int:
    configured = getattr(args, "pbt_population_seed", None)
    return int(args.seed if configured is None else configured)


def _configure_pbt_population(base_config: dict, args) -> None:
    """Configure diverse training seeds with a matched validation stream."""
    for key in PBT_MUTATIONS:
        base_config.pop(key, None)
    base_seed = _pbt_population_seed(args)
    base_config["seed"] = tune.randint(base_seed, base_seed + 1_000_000)
    base_config["env_config"]["task_sampler"]["seed"] = tune.sample_from(_trial_training_seed)


def _build_pbt_scheduler(args) -> SeededPopulationBasedTraining:
    perturbation_interval, _ = _pbt_intervals(args)
    train_batch_size = int(getattr(args, "train_batch_size", 4800))
    selection_mode = _pbt_selection_mode(args)
    return SeededPopulationBasedTraining(
        random_seed=_pbt_population_seed(args),
        require_positive_donor=selection_mode == "promotion",
        # This trial-owned clock survives donor checkpoint restores. RLlib's
        # training_iteration does not and caused repeated late perturbations.
        time_attr="pbt_progress_steps",
        metric=_pbt_metric_name(args),
        mode="max",
        perturbation_interval=perturbation_interval * train_batch_size,
        burn_in_period=float(
            getattr(args, "pbt_burn_in", perturbation_interval) * train_batch_size
        ),
        quantile_fraction=0.25,
        resample_probability=0.25,
        hyperparam_mutations=_pbt_mutations(args),
        custom_explore_fn=constrain_trial_config,
        log_config=True,
        require_attrs=True,
        # Synchronous ordering ensures upper donors checkpoint before lower
        # trials exploit at a perturbation boundary.
        synch=bool(getattr(args, "pbt_synch", False)),
    )


def _tune_objective_kwargs(use_pbt_mode: bool, args=None) -> dict[str, str]:
    # The scheduler owns the objective when present; otherwise Tune needs it.
    if use_pbt_mode:
        return {}
    return {"metric": _pbt_metric_name(args), "mode": "max"}


def _tune_storage_kwargs(results_root: Path) -> dict[str, str]:
    """Route Tune output through Ray 2.58's storage interface."""
    return {"storage_path": str(results_root)}


def _trial_output_path(trial) -> str:
    """Read a trial output path from Ray 2.58's Trial contract."""
    try:
        return str(trial.local_path)
    except AttributeError as exc:
        raise TypeError("Ray 2.58 Tune Trial must expose local_path") from exc


def main(args):
    use_pbt_mode = bool(args.pbt or args.cpu_pbt)
    configured_ray_cpus = int(getattr(args, "ray_num_cpus", 0))
    init_ray(
        use_pbt_mode=use_pbt_mode,
        num_cpus=(configured_ray_cpus or None),
    )

    data_path = Path(args.data_path).expanduser()
    if not data_path.exists() or not (data_path.is_file() or data_path.is_dir()):
        raise FileNotFoundError(f"Parquet dataset not found ➜ {data_path}")
    data_path = data_path.resolve(strict=True)
    if args.restore_ckpt:
        ckpt_path = Path(args.restore_ckpt).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Restore checkpoint path does not exist ➜ {ckpt_path}")
    elif int(getattr(args, "restore_progress_steps", 0)) or int(
        getattr(args, "restore_progress_updates", 0)
    ):
        raise ValueError("restore progress requires --restore_ckpt")

    # shared RLlib config
    base_config = build_base_config(args=args, data_path=data_path, use_pbt_mode=use_pbt_mode)
    if use_pbt_mode and int(getattr(args, "evaluation_interval", 1)) <= 0:
        raise ValueError("PBT requires held-out evaluation_interval > 0")
    pbt = _build_pbt_scheduler(args)
    if use_pbt_mode:
        _configure_pbt_population(base_config, args)

# ────────────────────────────────────────────────────────────────────────────────
# 5.1) Kick-off training algo
# ────────────────────────────────────────────────────────────────────────────────

    print("Starting training with automatic plotting...")
    print("Plots will be generated every 50 iterations in non-PBT mode.")

    resources = _resources_per_trial(args, base_config)

    if args.best_params_json:
        params_path = Path(args.best_params_json).expanduser()
        print(f"Loading best params from: {params_path}")
        best = json.loads(params_path.read_text(encoding="utf-8"))
        _apply_best_params(base_config, best)

    if args.restore_ckpt:
        base_config["restore_ckpt"] = args.restore_ckpt

    from itrader.models.registry import validate_representation_config

    validate_representation_config(base_config)

    _freeze_run_provenance(base_config, data_path)

    run_manifest, run_manifest_path = write_run_manifest(
        base_config=base_config,
        args=args,
        data_path=data_path,
    )
    run_status_path = run_manifest_path.with_name(
        f"run_status_{run_manifest['run_id']}.json"
    )
    _atomic_write_json(run_status_path, {**run_manifest, "status": "starting"})

    try:
        analysis = tune.run(
            training_fn,
            name=args.experiment_name,
            scheduler=(pbt if use_pbt_mode else None),
            num_samples=(args.pop if use_pbt_mode else 1),
            resources_per_trial=resources,
            stop=TimeoutStopper(args.timeout_hours * 3600),
            search_alg=(
                BasicVariantGenerator(random_state=_pbt_population_seed(args))
                if use_pbt_mode
                else None
            ),
            config=base_config,
            **_tune_storage_kwargs(_results_root(args)),
            progress_reporter=tune.CLIReporter(
                metric_columns=_cli_metric_columns(
                    base_config["env_config"]["task_sampler"]["action_dim"]
                ),
            ),
            reuse_actors=False,
            max_failures=3,
            **({"resume": "AUTO"} if bool(getattr(args, "resume", False)) else {}),
            **_tune_objective_kwargs(use_pbt_mode, args),
        )
        trials = [
            {
                "trial_id": trial.trial_id,
                "status": trial.status,
                "logdir": _trial_output_path(trial),
                "last_result": trial.last_result,
            }
            for trial in analysis.trials
        ]
        completion_status, incomplete_trial_ids = _run_completion_status(
            trials,
            target_steps=int(run_manifest["target_environment_steps"]),
        )
        _atomic_write_json(
            run_status_path,
            {
                **run_manifest,
                "status": completion_status,
                "trial_count": len(trials),
                "incomplete_trial_ids": incomplete_trial_ids,
                "trials": trials,
            },
        )
    except Exception as exc:
        _atomic_write_json(
            run_status_path,
            {**run_manifest, "status": "failed", "error": repr(exc)},
        )
        raise
    finally:
        ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200, help="Nominal PPO-update budget; converted to a sampled-environment-step target")
    parser.add_argument("--window",    type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--ray_num_cpus", type=int, default=0, help="Optional CPU capacity exposed to the local Ray node; 0 uses Ray's host default")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=4800)
    parser.add_argument("--sgd_minibatch_size", type=int, default=128)
    parser.add_argument("--num_sgd_iter", type=int, default=30)
    parser.add_argument("--vf_clip_param", type=float, default=10.0)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--vf_loss_coeff", type=float, default=1.0)
    parser.add_argument("--pbt", action="store_true", help="Enable Population-Based Training")
    parser.add_argument("--cpu_pbt", action="store_true", help="Enable PBT with CPU-only trial resources")
    parser.add_argument("--cpu_only", action="store_true", help="Use CPU-only resources without enabling PBT")
    parser.add_argument("--pop", type=int, default=10, help="Population size if --pbt")
    parser.add_argument("--timeout_hours", type=int, default=5, help="Global timeout for the whole run")
    parser.add_argument("--restore_ckpt", type=str, default="", help="Path to an RLlib checkpoint_XXXXXX dir to restore from")
    parser.add_argument("--restore_progress_steps", type=int, default=0, help="Target-owned sampled steps represented by --restore_ckpt for a new recovery trial")
    parser.add_argument("--restore_progress_updates", type=int, default=0, help="Target-owned PPO update count represented by --restore_ckpt for a new recovery trial")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False, help="Resume the existing Tune experiment with the same name and results root")
    parser.add_argument("--resume_rewind_to_checkpoint", action=argparse.BooleanOptionalAction, default=False, help="On Tune resume only, rewind target-owned progress and metrics to the restored target checkpoint")
    parser.add_argument("--checkpoint_consistent_evaluation", action=argparse.BooleanOptionalAction, default=None, help="Force post-train weight sync and deterministic evaluation, including checkpoint recovery runs")
    parser.add_argument("--best_params_json", type=str, default="", help="Path to params.json from best PBT trial (optional)")
    parser.add_argument("--start_idx_mode", type=str, default="random_episode", choices=["random_episode", "stream_worker", "task_spec"], help="Episode start index policy for env resets")
    parser.add_argument("--stream_jump_every_episodes", type=int, default=0, help="For stream_worker: random jump cadence in episodes (0 disables)")
    parser.add_argument("--stream_jump_on_wrap", action=argparse.BooleanOptionalAction, default=True, help="For stream_worker: randomize on stream wrap-around")
    parser.add_argument("--stream_stride", type=int, default=0, help="For stream_worker: stride in rows between consecutive episodes (0 uses horizon)")
    parser.add_argument("--ret_norm_reset", type=str, default="episode", choices=["episode", "stream"], help="Reset reward normalization every episode or only when stream changes")
    parser.add_argument("--observation_transform", type=str, default="raw", choices=["raw", "market_relative_v1"], help="Deterministic market observation transform")
    parser.add_argument("--solver_state_mode", type=str, default="off", choices=["off", "masked", "position", "time", "both"], help="Fixed two-value solver-state diagnostic mask")
    parser.add_argument("--solver_state_residual_init", type=str, default="random_v1", choices=["random_v1", "zero_v1"], help="Initialization profile for the R4-v1 actor/value residual heads")
    parser.add_argument("--observation_filter", type=str, default="NoFilter", choices=["NoFilter"], help="Native RLlib stack uses explicit unnormalized observations")
    parser.add_argument("--solver_model", default="solver_mlp_film", choices=["solver_mlp_film", "solver_window_lstm_film"])
    parser.add_argument("--representation_profile", default="legacy", choices=["legacy", "r0_r1_v1"])
    parser.add_argument("--task_pool_path", type=str, default="", help="Optional JSONL task pool path for sampler")
    parser.add_argument("--p_pool", type=float, default=0.7, help="Probability of sampling from task pool when configured")
    parser.add_argument("--task_horizons", type=str, default="", help="Comma-separated task horizons; empty uses the Phase A defaults")
    parser.add_argument("--task_reward_ids", type=str, default="", help="Comma-separated registered reward IDs; empty uses pnl_linear")
    parser.add_argument("--experiment_name", type=str, default="Phase_A", help="Ray Tune experiment folder name under ~/ray_results")
    parser.add_argument("--results_root", type=str, default="", help="Optional Ray/stable artifact root; defaults to ~/ray_results")
    parser.add_argument("--auto_plot_on_end", action=argparse.BooleanOptionalAction, default=True, help="Generate plots automatically at trial end")
    parser.add_argument("--pbt_interval", type=int, default=3, help="PBT cadence in PPO updates; converted to sampled environment steps")
    parser.add_argument("--pbt_burn_in", type=int, default=3, help="PPO updates before first PBT event; converted to sampled environment steps")
    parser.add_argument("--pbt_synch", action=argparse.BooleanOptionalAction, default=False, help="Synchronize PBT boundaries so upper donors checkpoint before lower trials exploit")
    parser.add_argument("--pbt_promotion_eligible", action=argparse.BooleanOptionalAction, default=False, help="Rank positive-net, positive-Sharpe, non-all-flat PBT donors ahead of ineligible donors, then rank by net return")
    parser.add_argument("--pbt_selection_mode", choices=["raw_net", "nonflat_net", "promotion"], default="raw_net", help="PBT donor ranking: raw net, non-flat net search, or strict promotion eligibility; the promotion-eligible flag selects promotion mode")
    parser.add_argument("--pbt_selection_min_sharpe", type=float, default=0.0, help="Strict held-out Sharpe floor for promotion-eligible PBT selection")
    parser.add_argument("--pbt_selection_max_flat_action_fraction", type=float, default=0.98, help="Exclusive all-flat ceiling for promotion-eligible PBT selection")
    parser.add_argument("--pbt_lr_min", type=float, default=5e-6, help="Inclusive lower bound for PBT learning-rate sampling and mutation")
    parser.add_argument("--pbt_lr_max", type=float, default=3e-4, help="Inclusive upper bound for PBT learning-rate sampling and mutation")
    parser.add_argument("--checkpoint_interval", type=int, default=3, help="Checkpoint cadence in PPO updates; must equal --pbt_interval")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="Run held-out evaluation every N training iterations")
    parser.add_argument("--evaluation_num_episodes", type=int, default=8, help="Deterministic validation episodes per evaluation")
    parser.add_argument("--evaluation_task_panel_path", default="", help="Optional predeclared JSON panel with explicit TaskSpec start_idx values")
    parser.add_argument("--evidence_cost_bps", type=float, default=1.0, help="Explicit per-turnover evaluation cost in basis points")
    parser.add_argument("--optimization_cost_bps", type=float, default=None, help="Cost embedded by net_return tasks; defaults to --evidence_cost_bps")
    parser.add_argument("--optimization_cost_schedule", type=str, default="", help="Comma-separated sampled-step:cost-bps curriculum for net_return tasks")
    parser.add_argument("--entropy_coeff_schedule", type=str, default="", help="Comma-separated step:entropy-coefficient points; RLlib interpolates on its native sampled-step clock. Empty preserves the historical default; PBT may mutate it.")
    parser.add_argument("--net_return_reward_scale", type=float, default=1000.0, help="Constant PPO scale for net_return tasks")
    parser.add_argument("--evidence_periods_per_year", type=float, default=0.0, help="Optional stated annualization periods; 0 reports per-sqrt-bar Sharpe only")
    parser.add_argument("--split_fractions", type=str, default="0.7,0.15,0.15", help="Chronological train,validation,test fractions")
    parser.add_argument("--split_purge_rows", type=int, default=200, help="Rows purged on each side of split boundaries")
    parser.add_argument(
        "--data_path",
        default="itrader/data/ohlcv.parquet",
    )
    parser.add_argument("--data_sessions", type=str, default="", help="Comma-separated Parquet session partitions to load")
    parser.add_argument("--data_symbols", type=str, default="", help="Comma-separated symbol partitions to load")
    parser.add_argument("--episode_boundary_cols", type=str, default="", help="Columns that episodes may not cross")
    parser.add_argument("--expected_bar_interval", type=str, default="", help="Required timestamp cadence within episodes, e.g. 1min")
    parser.add_argument("--seed", type=int, default=1234, help="Global run seed for RLlib and task sampler")
    parser.add_argument("--evaluation_seed", type=int, default=None, help="Optional fixed validation seed, independent of the training/member seed")
    parser.add_argument("--pbt_population_seed", type=int, default=None, help="Optional PBT initialization/search seed; evaluation starts remain governed by --seed")
    args = parser.parse_args()
    main(args)
