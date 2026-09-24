# itrader/rl_lib/train_solo.py

import argparse
import csv
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ray
from ray import train, tune
from ray.air import session
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.stopper import TimeoutStopper
from ray.tune import PlacementGroupFactory
from ray.train import Checkpoint
try:
    from ray.tune.logger import UnifiedLogger
except ImportError:
    UnifiedLogger = None

# your env & model imports
from itrader.envs.itrader_env import ITraderEnv
from itrader.models.solver_lstm import SolverLSTMModel
from itrader.models.solver_MLP import SolverMLPModel
from itrader.models.solver_mlp_film import SolverMLPFiLM
from itrader.rl_lib.tasks.families import DEFAULT_BAR_RESOLUTION
from itrader.rl_lib.tasks.rewards import RewardEngine
from itrader.rl_lib.tasks.samplers import TaskSampler00
from itrader.rl_lib.callbacks.auto_plotter import AutoPlotterCallback
from itrader.rl_lib.callbacks.ps_metrics import PSMetrics
from itrader.rl_lib.train.pbt_contract import constrain_trial_config

from gymnasium.envs.registration import EnvSpec

EXPERIMENT_NAME = "Phase_B"
DEFAULT_T_EMBED_DIM = 32
DEFAULT_FCNET_HIDDENS = [256, 256]
DEFAULT_TASK_HORIZONS = [96, 128, 160, 200]
DEFAULT_TURNOVER_W_CHOICES = [0.0, 0.0005, 0.001, 0.002]
DEFAULT_DD_W_RANGE = (0.0, 0.04)
BASE_METRIC_COLUMNS = [
    "training/optimization_reward_mean",
    "training/net_return_mean",
    "evaluation/net_return_mean",
    "evaluation/net_sharpe_per_sqrt_bar_mean",
    "episode_len_mean",
    "policy_loss",
    "value_loss",
    "entropy",
]
PBT_PRIMARY_METRIC = "evaluation/net_return_mean"
PBT_MUTATIONS = {
    "lr": tune.loguniform(1e-5, 5e-3),
    "clip_param": [0.1, 0.2, 0.3],
    "lambda": [0.90, 0.95, 0.97],
    "gamma": [0.98, 0.99, 0.995],
    "sgd_minibatch_size": [128, 256, 512],
    "num_sgd_iter": [10, 20, 30],
    "vf_clip_param": [5.0, 10.0, 20.0],
    "vf_loss_coeff": [0.5, 1.0, 2.0],
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
    cfg = {
        "window_size": args.window,
        "n_features": 5,
        "t_embed_dim": DEFAULT_T_EMBED_DIM,
        "action_dim": 3,
        "seed": int(args.seed),
        "bar_resolution": DEFAULT_BAR_RESOLUTION,
        "horizons": _parse_task_horizons(getattr(args, "task_horizons", "")),
        "reward_ids": _parse_task_reward_ids(getattr(args, "task_reward_ids", "")),
        "turnover_w_choices": DEFAULT_TURNOVER_W_CHOICES,
        "dd_modes": ["delta"],
        "dd_w_range": DEFAULT_DD_W_RANGE,
        "turnover_cap": 0.02,
        "dd_cap": 0.10,
    }
    if task_pool_path:
        cfg["task_pool_path"] = task_pool_path
        cfg["p_pool"] = float(getattr(args, "p_pool", 0.7))
    return cfg


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
    return {
        "custom_model": "solver_mlp_film",
        "custom_model_config": {
            "window_size": args.window,
            "t_embed_dim": DEFAULT_T_EMBED_DIM,
            "fcnet_hiddens": DEFAULT_FCNET_HIDDENS,
            "conditioning": "film",
            "t_keep_p": 0.7,
            "t_proj": 64,
        },
    }


def _legacy_api_stack_settings() -> dict[str, bool]:
    """Select ModelV2/RolloutWorker explicitly when Ray exposes both stacks."""
    if not hasattr(PPOConfig, "api_stack"):
        return {}
    return {
        "enable_rl_module_and_learner": False,
        "enable_env_runner_and_connector_v2": False,
    }


def _env_runner_gpu_settings(value: float) -> dict[str, float]:
    """Bridge the Ray 2.9 worker key to the Ray 2.58 EnvRunner key."""
    key = "num_gpus_per_env_runner" if hasattr(PPOConfig, "api_stack") else "num_gpus_per_worker"
    return {key: value}


def _evaluation_runner_settings(count: int) -> dict[str, int]:
    """Bridge Ray's renamed evaluation worker/EnvRunner count."""
    key = "evaluation_num_env_runners" if hasattr(PPOConfig, "api_stack") else "evaluation_num_workers"
    return {key: count}


def _first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None

# ────────────────────────────────────────────────────────────────────────────────
# 1) Ray init utilities + run metadata
# ────────────────────────────────────────────────────────────────────────────────
def init_ray(use_pbt_mode: bool):
    # PBT can run many trials; avoid a shared Prometheus port collision.
    init_kwargs = {} if use_pbt_mode else {"_metrics_export_port": 8888}
    # The minimal Ray installation cannot host the dashboard. Attempting it
    # first spawns a partial Windows node and can leave ports unavailable for
    # the fallback startup, so reliability takes precedence for experiments.
    ray.init(**init_kwargs, include_dashboard=False)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    """Materialize one immutable TaskSpec panel shared by every PBT member."""
    if panel_size <= 0:
        raise ValueError("evaluation_num_episodes must be positive")
    panel_cfg = {**task_sampler_cfg}
    panel_cfg.pop("fixed_tasks", None)
    sampler = TaskSampler00(panel_cfg)
    return [sampler.sample().to_dict() for _ in range(panel_size)]


def build_base_config(args, data_path: Path, use_pbt_mode: bool) -> dict:
    auto_plot_on_end = bool(getattr(args, "auto_plot_on_end", True))
    task_sampler_cfg = _task_sampler_config(args)
    _, checkpoint_interval = _pbt_intervals(args)
    split_fractions = _parse_split_fractions(getattr(args, "split_fractions", "0.7,0.15,0.15"))
    evidence_cost_bps = float(getattr(args, "evidence_cost_bps", 1.0))
    evidence_periods = float(getattr(args, "evidence_periods_per_year", 0.0))
    purge_rows = int(getattr(args, "split_purge_rows", 0))
    evaluation_episodes = int(getattr(args, "evaluation_num_episodes", 8))
    evaluation_seed = int(args.seed) + 10_000_019

    env_config = {
        "df_path": data_path.as_posix(),
        "window_size": args.window,
        "start_idx_mode": args.start_idx_mode,
        "stream_jump_every_episodes": args.stream_jump_every_episodes,
        "stream_jump_on_wrap": args.stream_jump_on_wrap,
        "stream_stride": args.stream_stride,
        "ret_norm_reset": args.ret_norm_reset,
        "observation_transform": str(getattr(args, "observation_transform", "raw")),
        "t_embed_mode": "concat",
        "t_scale": 1.0,
        "task_encoder_version": "semantic_v2",
        "data_split": "train",
        "split_fractions": split_fractions,
        "split_purge_rows": purge_rows,
        "evidence_cost_bps": evidence_cost_bps,
        "evidence_periods_per_year": evidence_periods,
        "task_sampler": task_sampler_cfg,
    }
    evaluation_task_sampler_cfg = {
        **task_sampler_cfg,
        "seed": evaluation_seed,
    }
    evaluation_task_sampler_cfg["fixed_tasks"] = _build_fixed_evaluation_tasks(
        evaluation_task_sampler_cfg,
        evaluation_episodes,
    )
    evaluation_env_config = {
        **env_config,
        "data_split": "validation",
        "start_idx_mode": "random_episode",
        "ret_norm_reset": "episode",
        "task_sampler": evaluation_task_sampler_cfg,
    }

    return {
        **_legacy_api_stack_settings(),
        "env": "ITraderEnv",
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
        # The legacy stack can evaluate before synchronising rollout filters.
        # PBT must rank the post-sync state that its checkpoint will contain.
        "checkpoint_consistent_evaluation": bool(use_pbt_mode),
        "model": _model_config(args),
        # resources
        "num_gpus": 1,
        **_env_runner_gpu_settings(0),
        "num_workers": args.workers,
        # reproducibility
        "seed": int(args.seed),
        # core RL settings
        "framework": "torch",
        # swap callbacks for PBT to reduce collisions/overhead
        "callbacks": (MetricsOnlyCallback if use_pbt_mode else CombinedCallback),
        "observation_filter": str(getattr(args, "observation_filter", "MeanStdFilter")),
        "grad_clip": 10.0,
        "clip_param": float(getattr(args, "clip_param", 0.2)),
        "lambda": float(getattr(args, "gae_lambda", 0.95)),
        "vf_loss_coeff": float(getattr(args, "vf_loss_coeff", 1.0)),
        # Stabilise exploration → exploitation
        "entropy_coeff_schedule": [
            [0,        0.02],
            [200_000,  0.01],
            [800_000,  0.003],
        ],
        "train_batch_size": int(getattr(args, "train_batch_size", 4800)),
        "batch_mode": "complete_episodes",
        "sgd_minibatch_size": int(getattr(args, "sgd_minibatch_size", 128)),
        "num_sgd_iter": int(getattr(args, "num_sgd_iter", 30)),
        "gamma": float(getattr(args, "gamma", 0.99)),
        "vf_clip_param": float(getattr(args, "vf_clip_param", 10.0)),
        "lr": float(getattr(args, "lr", 5e-5)),
        # PBT runs multiple trials -> shared port conflicts. Enable only for single-trial runs.
        "metrics_sinks": ([] if use_pbt_mode else ["prometheus"]),
        "prometheus_export_port": (None if use_pbt_mode else 8888),
        # pass the stop criterion to training_fn
        "stop_iters": args.iters,
        "checkpoint_interval": checkpoint_interval,
        "stable_metrics_root": str(_results_root(args) / "_stable_metrics"),
        "auto_plot_on_end": auto_plot_on_end,
    }


def write_run_manifest(base_config: dict, args, data_path: Path):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_name = str(getattr(args, "experiment_name", EXPERIMENT_NAME))
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "git_commit": _git_commit(),
        "seed": int(args.seed),
        "config_hash": _config_hash(base_config),
        "dataset_path": str(data_path),
        "dataset_sha256": _sha256_file(data_path),
        "results_root": str(_results_root(args)),
        "start_idx_mode": args.start_idx_mode,
        "ret_norm_reset": args.ret_norm_reset,
        "is_pbt": bool(args.pbt or args.cpu_pbt),
        "is_cpu_pbt": bool(args.cpu_pbt),
        "pbt_primary_metric": PBT_PRIMARY_METRIC if (args.pbt or args.cpu_pbt) else None,
        "pbt_interval": int(getattr(args, "pbt_interval", 3)),
        "checkpoint_interval": int(getattr(args, "checkpoint_interval", 3)),
        "pbt_mutations": {key: str(value) for key, value in PBT_MUTATIONS.items()},
    }
    # Tune 2.9 can truncate files created in its experiment directory on
    # Windows. Repository-owned provenance therefore lives in a sibling root.
    out_dir = _results_root(args) / "_stable_runs" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_manifest_{run_id}.json"
    _atomic_write_json(out_path, manifest)
    print(f"Run manifest written to: {out_path}")
    return manifest, out_path

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


class CombinedCallback(DefaultCallbacks):
    def __init__(self, plot_interval: int = 50):
        super().__init__()
        self.ps_metrics = PSMetrics()
        self.auto_plotter = AutoPlotterCallback(
            plot_interval=plot_interval,
            plot_at_end_of_loop=False,
        )

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        return self.ps_metrics.on_episode_end(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)

    def on_train_result(self, *, algorithm, result, **kwargs):
        return self.auto_plotter.on_train_result(algorithm=algorithm, result=result, **kwargs)

class MetricsOnlyCallback(DefaultCallbacks):
    """Use for PBT to avoid plot/file collisions across many trials."""
    def __init__(self):
        super().__init__()
        self.ps_metrics = PSMetrics()

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        return self.ps_metrics.on_episode_end(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)

#  ────────────────────────────────────────────────────────────────────────────────
# 4) The “training function” for Tune
#    This builds the PPOAlgo, loops train(), and reports key scalars
# ────────────────────────────────────────────────────────────────────────────────
def _extract_custom_metrics(result: dict) -> dict:
    custom_metrics = result.get("custom_metrics")
    if isinstance(custom_metrics, dict):
        return custom_metrics
    sampler_custom_metrics = result.get("sampler_results", {}).get("custom_metrics")
    if isinstance(sampler_custom_metrics, dict):
        return sampler_custom_metrics
    env_runner_custom_metrics = result.get("env_runners", {}).get("custom_metrics")
    return env_runner_custom_metrics if isinstance(env_runner_custom_metrics, dict) else {}


def _numeric_custom_metric_items(custom_metrics: dict) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for key, value in custom_metrics.items():
        if not isinstance(value, (int, float, np.floating, np.integer)):
            continue
        key_text = str(key)
        if key_text.startswith(
            ("act_", "rc_", "economic_", "optimization_", "family_", "horizon_", "regime_")
        ) and key_text.endswith("_mean"):
            items.append((key_text, float(value)))
            continue
        if key_text in {"success_mean", "task_horizon_mean"}:
            items.append((key_text, float(value)))
    return sorted(items)


def _extract_report(result: dict, default_iteration: int) -> dict:
    learner_stats = result["info"]["learner"]["default_policy"]["learner_stats"]
    training_custom = _extract_custom_metrics(result)
    evaluation_result = result.get("evaluation", {}) or {}
    evaluation_custom = _extract_custom_metrics(evaluation_result)

    def custom_value(metrics: dict, key: str):
        return _first_non_none(metrics.get(f"{key}_mean"), metrics.get(key))

    optimization_reward = _first_non_none(
        custom_value(training_custom, "optimization_reward"),
        result.get("episode_reward_mean"),
        result.get("env_runners", {}).get("episode_reward_mean"),
    )
    episode_len_mean = _first_non_none(
        result.get("episode_len_mean"),
        result.get("sampler_results", {}).get("episode_len_mean"),
        result.get("env_runners", {}).get("episode_len_mean"),
    )

    report = {
        "training_iteration": int(result.get("training_iteration", default_iteration)),
        "time_this_iter_s": result.get("time_this_iter_s"),
        "time_total_s": result.get("time_total_s"),
        "timesteps_total": result.get("timesteps_total"),
        "training/optimization_reward_mean": (
            float(optimization_reward) if optimization_reward is not None else float("nan")
        ),
        "training/gross_return_mean": float(
            custom_value(training_custom, "economic_gross_return") or 0.0
        ),
        "training/net_return_mean": float(
            custom_value(training_custom, "economic_net_return") or 0.0
        ),
        "training/net_sharpe_per_sqrt_bar_mean": float(
            custom_value(training_custom, "economic_net_sharpe_per_sqrt_bar") or 0.0
        ),
        "evaluation/optimization_reward_mean": float(
            custom_value(evaluation_custom, "optimization_reward") or 0.0
        ),
        "evaluation/gross_return_mean": float(
            custom_value(evaluation_custom, "economic_gross_return") or 0.0
        ),
        "evaluation/net_return_mean": float(
            custom_value(evaluation_custom, "economic_net_return") or 0.0
        ),
        "evaluation/net_sharpe_per_sqrt_bar_mean": float(
            custom_value(evaluation_custom, "economic_net_sharpe_per_sqrt_bar") or 0.0
        ),
        "episode_len_mean": float(episode_len_mean) if episode_len_mean is not None else float("nan"),
        "policy_loss": float(learner_stats.get("policy_loss", 0.0)),
        "value_loss": float(learner_stats.get("vf_loss", learner_stats.get("value_loss", 0.0))),
        "entropy": float(learner_stats.get("entropy", 0.0)),
    }
    if report.get("time_this_iter_s"):
        report["iters_per_sec"] = 1.0 / max(float(report["time_this_iter_s"]), 1e-8)

    for key, value in _numeric_custom_metric_items(training_custom):
        report[f"custom_metrics/{key}"] = value
    for key, value in _numeric_custom_metric_items(evaluation_custom):
        report[f"evaluation/custom_metrics/{key}"] = value
    for key in ("episode_len_mean", "iters_per_sec"):
        report.setdefault(key, float("nan"))
    return report


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


def _checkpoint_from_save_result(saved):
    checkpoint_type = getattr(tune, "Checkpoint", Checkpoint)
    if isinstance(saved, str):
        return checkpoint_type.from_directory(saved), saved
    if isinstance(saved, checkpoint_type):
        return saved, getattr(saved, "path", "Checkpoint")
    if isinstance(saved, Checkpoint):
        checkpoint_dir = saved.to_directory()
        return checkpoint_type.from_directory(checkpoint_dir), checkpoint_dir
    if hasattr(saved, "checkpoint") and saved.checkpoint is not None:
        checkpoint = saved.checkpoint
        checkpoint_dir = getattr(checkpoint, "path", None) or checkpoint.to_directory()
        return checkpoint_type.from_directory(checkpoint_dir), checkpoint_dir
    return None, str(saved)


def _trial_context_values() -> tuple[str, str, str]:
    if hasattr(tune, "get_context"):
        context = tune.get_context()
        return context.get_trial_dir(), context.get_trial_id(), context.get_trial_name()
    return session.get_trial_dir(), session.get_trial_id(), session.get_trial_name()


def _incoming_tune_checkpoint():
    if hasattr(tune, "get_checkpoint"):
        return tune.get_checkpoint()
    return train.get_checkpoint()


def _report_to_tune(metrics: dict, checkpoint=None) -> None:
    if hasattr(tune, "get_context"):
        tune.report(metrics, checkpoint=checkpoint)
    elif checkpoint is not None:
        session.report(metrics, checkpoint=checkpoint)
    else:
        session.report(metrics)


def _copy_stable_checkpoint(
    checkpoint_dir: str | Path,
    *,
    stable_trial_dir: Path,
    training_iteration: int,
) -> Path:
    """Copy a checkpoint before Ray AIR takes ownership of its source path."""
    source = Path(checkpoint_dir)
    destination = (
        stable_trial_dir
        / "checkpoints"
        / f"iter_{int(training_iteration):06d}_{datetime.now(timezone.utc):%H%M%S%f}"
    )
    shutil.copytree(source, destination)
    required = ("algorithm_state.pkl", "rllib_checkpoint.json")
    invalid = [
        name
        for name in required
        if not (destination / name).is_file()
        or not (destination / name).stat().st_size
    ]
    if invalid:
        raise RuntimeError(f"Stable checkpoint is incomplete: {', '.join(invalid)}")
    return destination


def _remaining_training_steps(current_iteration: int, stop_iteration: int) -> range:
    """Return local steps still owed after an RLlib/PBT checkpoint restore."""
    remaining = max(0, int(stop_iteration) - int(current_iteration))
    return range(remaining)


def _env_runner_groups(algo):
    training_group = getattr(algo, "env_runner_group", None)
    evaluation_group = getattr(algo, "eval_env_runner_group", None)
    if training_group is None:
        training_group = algo.workers
    if evaluation_group is None:
        evaluation_group = algo.evaluation_workers
    return training_group, evaluation_group


def _local_env_runner(group):
    if hasattr(type(group), "local_env_runner"):
        return group.local_env_runner
    return group.local_worker()


def deterministic_evaluate(algo) -> dict:
    """Evaluate the current learner weights with exploration disabled."""
    training_group, evaluation_group = _env_runner_groups(algo)
    evaluation_group.foreach_policy(
        lambda policy, _policy_id: policy.config.__setitem__("explore", False)
    )
    evaluation_group.sync_weights(
        from_worker_or_learner_group=_local_env_runner(training_group)
    )
    result = algo.evaluate()
    return result.get("evaluation", result)


def train_with_checkpoint_consistent_evaluation(algo):
    """Score the exact learner weights that the next checkpoint will persist."""
    result = algo.train()
    if "evaluation" not in result:
        return result

    # Ray 2.9 Algorithm.restore() copies the training Policy config onto the
    # evaluation Policy and can silently flip explore back to True.
    evaluation = deterministic_evaluate(algo)
    if isinstance(evaluation, dict):
        result["evaluation"] = evaluation
    return result


def training_fn(config):
    # Always use Tune trial dir as the single source of truth for ALL outputs.
    trial_dir_raw, trial_id, trial_name = _trial_context_values()
    trial_dir = Path(trial_dir_raw)

    # Force RLlib's UnifiedLogger to write into the Tune trial dir (prevents PPO_ITraderEnv_* dirs).
    def logger_creator(cfg):
        # Keep RLlib logger artifacts isolated from Tune's own progress files.
        rllib_dir = trial_dir / "rllib"
        rllib_dir.mkdir(parents=True, exist_ok=True)
        return UnifiedLogger(cfg, str(rllib_dir), loggers=None)
    algo_kwargs = {"config": config}
    if UnifiedLogger is not None:
        algo_kwargs["logger_creator"] = logger_creator
    algo = PPO(**algo_kwargs)

    # --- Restore for PBT exploit/explore (weight inheritance) ---------------
    ckpt = _incoming_tune_checkpoint()
    requested_restore = str(config.get("restore_ckpt", "") or "").strip()
    restore_source = None
    if requested_restore:
        algo.restore(requested_restore)
        restore_source = requested_restore
        print(f"Restored checkpoint from --restore_ckpt: {requested_restore}")
    elif ckpt is not None:
        restore_dir = ckpt.to_directory()
        algo.restore(restore_dir)
        restore_source = restore_dir
        print(f"Restored checkpoint from Tune checkpoint: {restore_dir}")


    checkpoint_interval = int(config.get("checkpoint_interval", 3))
    full_metrics_path = trial_dir / "rllib" / "metrics_full.jsonl"
    full_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    stable_root = Path(str(config.get("stable_metrics_root", Path.home() / "ray_results" / "_stable_metrics")))
    stable_trial_dir = stable_root / trial_dir.name
    stable_trial_dir.mkdir(parents=True, exist_ok=True)
    stable_metrics_path = stable_trial_dir / "metrics_full.jsonl"
    stable_report_csv = stable_trial_dir / "report_rows.csv"
    trial_status_path = stable_trial_dir / "trial_status.json"
    if restore_source:
        _rewind_iteration_artifacts(
            restored_iteration=int(getattr(algo, "iteration", 0)),
            report_path=stable_report_csv,
            jsonl_paths=[
                full_metrics_path,
                stable_metrics_path,
                stable_trial_dir / "checkpoint_index.jsonl",
            ],
        )
    data_path = Path(str(config["env_config"]["df_path"]))
    trial_identity = {
        "schema_version": 1,
        "trial_id": trial_id,
        "trial_name": trial_name,
        "trial_dir": str(trial_dir),
        "stable_trial_dir": str(stable_trial_dir),
        "git_commit": _git_commit(),
        "config_hash": _config_hash(config),
        "dataset_path": str(data_path),
        "dataset_sha256": _sha256_file(data_path),
    }
    _atomic_write_json(stable_trial_dir / "resolved_config.json", {**trial_identity, "config": config})
    _append_json_line(
        stable_trial_dir / "config_history.jsonl",
        {
            **trial_identity,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "event": "checkpoint_restore" if restore_source else "initial_start",
            "restored_iteration": int(getattr(algo, "iteration", 0)),
            "restore_source": restore_source,
            "config": config,
        },
    )
    _atomic_write_json(trial_status_path, {**trial_identity, "status": "running", "last_iteration": 0})

    current_iteration = int(getattr(algo, "iteration", 0))
    last_iteration = current_iteration
    for _ in _remaining_training_steps(current_iteration, config["stop_iters"]):
        if config.get("checkpoint_consistent_evaluation", False):
            res = train_with_checkpoint_consistent_evaluation(algo)
        else:
            res = algo.train()
        report = _extract_report(res, last_iteration + 1)
        last_iteration = int(report["training_iteration"])

        checkpoint = None
        ckpt_label = None
        stable_checkpoint = None
        should_checkpoint = checkpoint_interval > 0 and last_iteration % checkpoint_interval == 0
        if should_checkpoint:
            # Never reuse a mutable checkpoint directory. Tune may copy the
            # artifact after session.report returns, and PBT must receive an
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
            )

        # Persist scalar-only flattened metrics for robust plotting fallback.
        full = _flatten_scalar_metrics(res)
        full["training_iteration"] = report["training_iteration"]
        _append_json_line(full_metrics_path, full)
        _append_json_line(stable_metrics_path, full)
        _append_report_row(stable_report_csv, dict(report))
        _atomic_write_json(
            trial_status_path,
            {
                **trial_identity,
                "status": "running",
                "last_iteration": report["training_iteration"],
                "last_report": report,
            },
        )

        # --- Checkpointing for PBT (so exploit can copy weights) ------------
        if checkpoint is not None:
            _report_to_tune(report, checkpoint=checkpoint)
            _append_json_line(
                stable_trial_dir / "checkpoint_index.jsonl",
                {
                    "training_iteration": report["training_iteration"],
                    "tune_checkpoint": str(ckpt_label),
                    "stable_checkpoint": str(stable_checkpoint),
                    "reported_to_tune": True,
                    "post_sync_evaluation": bool(config.get("checkpoint_consistent_evaluation", False)),
                },
            )
            print(f"Immutable checkpoint saved @ iter {last_iteration}: {ckpt_label}")
        else:
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
        },
    )
    algo.stop()
# ────────────────────────────────────────────────────────────────────────────────
# 5) Entrypoint: assemble Tune config & kick off the experiment
# ────────────────────────────────────────────────────────────────────────────────
def _register_models() -> None:
    ModelCatalog.register_custom_model("solver_lstm_v2", SolverLSTMModel)
    ModelCatalog.register_custom_model("solver_mlp", SolverMLPModel)
    ModelCatalog.register_custom_model("solver_mlp_film", SolverMLPFiLM)


def _resources_per_trial(args, base_config: dict) -> PlacementGroupFactory:
    if args.cpu_pbt or bool(getattr(args, "cpu_only", False)):
        base_config.update({
            "num_gpus": 0,
            "num_workers": max(2, min(args.workers, 4)),
        })
        return PlacementGroupFactory(
            [{"CPU": 1}] + [{"CPU": 1}] * base_config["num_workers"]
        )

    base_config.update({
        "num_gpus": 1,
        "num_workers": args.workers,
    })
    return PlacementGroupFactory(
        [{"GPU": 1, "CPU": 1}] + [{"CPU": 1}] * base_config["num_workers"]
    )


def _apply_best_params(base_config: dict, best: dict) -> None:
    for key in [
        "lr", "clip_param", "lambda", "gamma",
        "sgd_minibatch_size", "num_sgd_iter",
        "vf_clip_param", "vf_loss_coeff", "train_batch_size",
    ]:
        if key in best:
            base_config[key] = best[key]

    model_cfg = best.get("model", {}).get("custom_model_config")
    if isinstance(model_cfg, dict):
        base_config["model"]["custom_model_config"].update(model_cfg)

    task_sampler_overrides = best.get("env_config", {}).get("task_sampler_overrides")
    if isinstance(task_sampler_overrides, dict):
        base_config["env_config"]["task_sampler_overrides"] = task_sampler_overrides


def _trial_training_seed(config) -> int:
    return int(config["seed"])


def _configure_pbt_population(base_config: dict, args) -> None:
    """Configure diverse training seeds with a matched validation stream."""
    for key in PBT_MUTATIONS:
        base_config.pop(key, None)
    base_seed = int(args.seed)
    base_config["seed"] = tune.randint(base_seed, base_seed + 1_000_000)
    base_config["env_config"]["task_sampler"]["seed"] = tune.sample_from(_trial_training_seed)


def _build_pbt_scheduler(args) -> PopulationBasedTraining:
    perturbation_interval, _ = _pbt_intervals(args)
    return PopulationBasedTraining(
        time_attr="training_iteration",
        metric=PBT_PRIMARY_METRIC,
        mode="max",
        perturbation_interval=perturbation_interval,
        burn_in_period=float(getattr(args, "pbt_burn_in", perturbation_interval)),
        quantile_fraction=0.25,
        resample_probability=0.25,
        hyperparam_mutations=PBT_MUTATIONS,
        custom_explore_fn=constrain_trial_config,
        log_config=True,
        require_attrs=True,
        synch=False,
    )


def _tune_objective_kwargs(use_pbt_mode: bool) -> dict[str, str]:
    # PBT owns metric/mode; duplicating them at tune.run is rejected by Ray.
    if use_pbt_mode:
        return {}
    return {"metric": PBT_PRIMARY_METRIC, "mode": "max"}


def _tune_storage_kwargs(results_root: Path) -> dict[str, str]:
    """Use Ray's supported Tune storage keyword without changing the path contract."""
    key = "storage_path" if "storage_path" in inspect.signature(tune.run).parameters else "local_dir"
    return {key: str(results_root)}


def _trial_output_path(trial) -> str:
    if hasattr(type(trial), "local_path"):
        return str(trial.local_path)
    return str(trial.logdir)


def _validate_pbt_observation_contract(config: dict, use_pbt_mode: bool) -> None:
    """Fail closed when legacy-stack PBT depends on filter state across restores."""
    if not use_pbt_mode:
        return
    if config.get("observation_filter") != "NoFilter":
        raise ValueError(
            "Legacy-stack PBT requires observation_filter=NoFilter; MeanStdFilter "
            "checkpoint restores do not reproduce the ranked evaluation policy"
        )
    train_transform = config.get("env_config", {}).get("observation_transform")
    eval_transform = (
        config.get("evaluation_config", {}).get("env_config", {}).get("observation_transform")
    )
    if train_transform != "market_relative_v1" or eval_transform != train_transform:
        raise ValueError(
            "PBT with NoFilter requires the matched market_relative_v1 observation transform"
        )


def main(args):
    use_pbt_mode = bool(args.pbt or args.cpu_pbt)
    init_ray(use_pbt_mode=use_pbt_mode)

    data_path = Path(args.data_path).expanduser()
    if not data_path.is_file():
        raise FileNotFoundError(f"Parquet not found ➜ {data_path}")
    data_path = data_path.resolve(strict=True)
    if args.restore_ckpt:
        ckpt_path = Path(args.restore_ckpt).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Restore checkpoint path does not exist ➜ {ckpt_path}")

    _register_models()

    # shared RLlib config
    base_config = build_base_config(args=args, data_path=data_path, use_pbt_mode=use_pbt_mode)
    _validate_pbt_observation_contract(base_config, use_pbt_mode)
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
            search_alg=(BasicVariantGenerator(random_state=int(args.seed)) if use_pbt_mode else None),
            config=base_config,
            **_tune_storage_kwargs(_results_root(args)),
            progress_reporter=tune.CLIReporter(
                metric_columns=_cli_metric_columns(
                    base_config["env_config"]["task_sampler"]["action_dim"]
                ),
            ),
            reuse_actors=False,
            max_failures=3,
            **_tune_objective_kwargs(use_pbt_mode),
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
        _atomic_write_json(
            run_status_path,
            {**run_manifest, "status": "completed", "trial_count": len(trials), "trials": trials},
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
    parser.add_argument("--iters",     type=int, default=200)
    parser.add_argument("--window",    type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
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
    parser.add_argument("--best_params_json", type=str, default="", help="Path to params.json from best PBT trial (optional)")
    parser.add_argument("--start_idx_mode", type=str, default="random_episode", choices=["random_episode", "stream_worker", "task_spec"], help="Episode start index policy for env resets")
    parser.add_argument("--stream_jump_every_episodes", type=int, default=0, help="For stream_worker: random jump cadence in episodes (0 disables)")
    parser.add_argument("--stream_jump_on_wrap", action=argparse.BooleanOptionalAction, default=True, help="For stream_worker: randomize on stream wrap-around")
    parser.add_argument("--stream_stride", type=int, default=0, help="For stream_worker: stride in rows between consecutive episodes (0 uses horizon)")
    parser.add_argument("--ret_norm_reset", type=str, default="episode", choices=["episode", "stream"], help="Reset reward normalization every episode or only when stream changes")
    parser.add_argument("--observation_transform", type=str, default="raw", choices=["raw", "market_relative_v1"], help="Deterministic market observation transform")
    parser.add_argument("--observation_filter", type=str, default="MeanStdFilter", choices=["NoFilter", "MeanStdFilter"], help="RLlib observation filter; legacy-stack PBT requires NoFilter")
    parser.add_argument("--task_pool_path", type=str, default="", help="Optional JSONL task pool path for sampler")
    parser.add_argument("--p_pool", type=float, default=0.7, help="Probability of sampling from task pool when configured")
    parser.add_argument("--task_horizons", type=str, default="", help="Comma-separated task horizons; empty uses the Phase A defaults")
    parser.add_argument("--task_reward_ids", type=str, default="", help="Comma-separated registered reward IDs; empty uses pnl_linear")
    parser.add_argument("--experiment_name", type=str, default="Phase_A", help="Ray Tune experiment folder name under ~/ray_results")
    parser.add_argument("--results_root", type=str, default="", help="Optional Ray/stable artifact root; defaults to ~/ray_results")
    parser.add_argument("--auto_plot_on_end", action=argparse.BooleanOptionalAction, default=True, help="Generate plots automatically at trial end")
    parser.add_argument("--pbt_interval", type=int, default=3, help="PBT exploit/explore cadence in training iterations")
    parser.add_argument("--pbt_burn_in", type=int, default=3, help="Iterations before the first PBT exploit/explore")
    parser.add_argument("--checkpoint_interval", type=int, default=3, help="Checkpoint cadence; must equal --pbt_interval")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="Run held-out evaluation every N training iterations")
    parser.add_argument("--evaluation_num_episodes", type=int, default=8, help="Deterministic validation episodes per evaluation")
    parser.add_argument("--evidence_cost_bps", type=float, default=1.0, help="Explicit per-turnover evaluation cost in basis points")
    parser.add_argument("--evidence_periods_per_year", type=float, default=0.0, help="Optional stated annualization periods; 0 reports per-sqrt-bar Sharpe only")
    parser.add_argument("--split_fractions", type=str, default="0.7,0.15,0.15", help="Chronological train,validation,test fractions")
    parser.add_argument("--split_purge_rows", type=int, default=200, help="Rows purged on each side of split boundaries")
    parser.add_argument(
        "--data_path",
        default="itrader/data/ohlcv.parquet",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Global run seed for RLlib and task sampler")
    args = parser.parse_args()
    main(args)
