# itrader/rl_lib/train_solo.py

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ray
from ray import train, tune
from ray.air import session
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.stopper import TimeoutStopper
from ray.tune import PlacementGroupFactory
from ray.train import Checkpoint
from ray.tune.logger import UnifiedLogger

# your env & model imports
from itrader.envs.itrader_env import ITraderEnv
from itrader.models.solver_lstm import SolverLSTMModel
from itrader.models.solver_MLP import SolverMLPModel
from itrader.models.solver_mlp_film import SolverMLPFiLM
from itrader.rl_lib.tasks.families import DEFAULT_BAR_RESOLUTION
from itrader.rl_lib.tasks.rewards import RewardEngine
from itrader.rl_lib.callbacks.auto_plotter import AutoPlotterCallback
from itrader.rl_lib.callbacks.ps_metrics import PSMetrics

from gymnasium.envs.registration import EnvSpec

EXPERIMENT_NAME = "Phase_B"
DEFAULT_T_EMBED_DIM = 32
DEFAULT_FCNET_HIDDENS = [256, 256]
DEFAULT_TASK_HORIZONS = [96, 128, 160, 200]
DEFAULT_TURNOVER_W_CHOICES = [0.0, 0.0005, 0.001, 0.002]
DEFAULT_DD_W_RANGE = (0.0, 0.04)
BASE_METRIC_COLUMNS = [
    "custom_metrics/episode_return_mean",
    "episode_len_mean",
    "policy_loss",
    "value_loss",
    "entropy",
]


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
    if use_pbt_mode:
        ray.init(include_dashboard=True, dashboard_host="0.0.0.0")
    else:
        ray.init(
            _metrics_export_port=8888,
            include_dashboard=True,
            dashboard_host="0.0.0.0",
        )


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


def build_base_config(args, data_path: Path, use_pbt_mode: bool) -> dict:
    auto_plot_on_end = bool(getattr(args, "auto_plot_on_end", True))
    task_sampler_cfg = _task_sampler_config(args)

    return {
        "env": "ITraderEnv",
        "env_config": {
            "df_path": data_path.as_posix(),
            "window_size": args.window,
            "start_idx_mode": args.start_idx_mode,
            "stream_jump_every_episodes": args.stream_jump_every_episodes,
            "stream_jump_on_wrap": args.stream_jump_on_wrap,
            "stream_stride": args.stream_stride,
            "ret_norm_reset": args.ret_norm_reset,
            "t_embed_mode": "concat",   # off|scale|concat
            "t_scale": 1.0,
            "task_sampler": task_sampler_cfg,
        # ← NEW: allows Tune/PBT to tweak tiny reward weights *without* editing the sampler.
            # Your env.reset() should merge this dict into current_task.reward_params (you already added this).
            # "task_sampler_overrides": {
            #     "turnover_w": 0.001,
            #     "dd_w": 0.04,
            #     "dd_T_weight": 0.0,
            # },
        },
        "model": _model_config(args),
        # resources
        "num_gpus": 1,
        "num_gpus_per_worker": 0,
        "num_workers": args.workers,
        # reproducibility
        "seed": int(args.seed),
        # core RL settings
        "framework": "torch",
        # swap callbacks for PBT to reduce collisions/overhead
        "callbacks": (MetricsOnlyCallback if use_pbt_mode else CombinedCallback),
        "observation_filter": "MeanStdFilter",
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
        "checkpoint_interval": 10,  # Save checkpoint every N iterations for PBT
        "stable_metrics_root": str(Path.home() / "ray_results" / "_stable_metrics"),
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
        "start_idx_mode": args.start_idx_mode,
        "ret_norm_reset": args.ret_norm_reset,
        "is_pbt": bool(args.pbt),
        "is_cpu_pbt": bool(args.cpu_pbt),
    }
    out_dir = Path.home() / "ray_results" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_manifest_{run_id}.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Run manifest written to: {out_path}")

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
    return sampler_custom_metrics if isinstance(sampler_custom_metrics, dict) else {}


def _numeric_custom_metric_items(custom_metrics: dict) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for key, value in custom_metrics.items():
        if not isinstance(value, (int, float, np.floating, np.integer)):
            continue
        key_text = str(key)
        if key_text.startswith(("act_", "rc_")) and key_text.endswith("_mean"):
            items.append((key_text, float(value)))
            continue
        if key_text in {"success_mean", "task_horizon_mean"}:
            items.append((key_text, float(value)))
    return sorted(items)


def _extract_report(result: dict, default_iteration: int) -> dict:
    learner_stats = result["info"]["learner"]["default_policy"]["learner_stats"]
    episode_return_mean = _first_non_none(
        result.get("custom_metrics/episode_return_mean_mean"),
        result.get("sampler_results", {}).get("custom_metrics", {}).get("episode_return_mean_mean"),
        result.get("episode_reward_mean"),
    )
    episode_len_mean = _first_non_none(
        result.get("episode_len_mean"),
        result.get("sampler_results", {}).get("episode_len_mean"),
    )

    report = {
        "training_iteration": int(result.get("training_iteration", default_iteration)),
        "time_this_iter_s": result.get("time_this_iter_s"),
        "time_total_s": result.get("time_total_s"),
        "timesteps_total": result.get("timesteps_total"),
        "custom_metrics/episode_return_mean": (
            float(episode_return_mean) if episode_return_mean is not None else float("nan")
        ),
        "episode_len_mean": float(episode_len_mean) if episode_len_mean is not None else float("nan"),
        "policy_loss": float(learner_stats.get("policy_loss", 0.0)),
        "value_loss": float(learner_stats.get("vf_loss", learner_stats.get("value_loss", 0.0))),
        "entropy": float(learner_stats.get("entropy", 0.0)),
    }
    if report.get("time_this_iter_s"):
        report["iters_per_sec"] = 1.0 / max(float(report["time_this_iter_s"]), 1e-8)

    custom_metrics = _extract_custom_metrics(result)
    for key, value in _numeric_custom_metric_items(custom_metrics):
        report[f"custom_metrics/{key}"] = value
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
        fh.write(json.dumps(payload) + "\n")


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


def _checkpoint_from_save_result(saved):
    if isinstance(saved, str):
        return Checkpoint.from_directory(saved), saved
    if isinstance(saved, Checkpoint):
        return saved, getattr(saved, "path", "Checkpoint")
    if hasattr(saved, "checkpoint") and isinstance(saved.checkpoint, Checkpoint):
        checkpoint = saved.checkpoint
        return checkpoint, getattr(checkpoint, "path", "Checkpoint")
    return None, str(saved)


def training_fn(config):
    # Always use Tune trial dir as the single source of truth for ALL outputs.
    trial_dir = Path(session.get_trial_dir())

    # Force RLlib's UnifiedLogger to write into the Tune trial dir (prevents PPO_ITraderEnv_* dirs).
    def logger_creator(cfg):
        # Keep RLlib logger artifacts isolated from Tune's own progress files.
        rllib_dir = trial_dir / "rllib"
        rllib_dir.mkdir(parents=True, exist_ok=True)
        return UnifiedLogger(cfg, str(rllib_dir), loggers=None)
    algo = PPO(config=config, logger_creator=logger_creator)

    # --- Restore for PBT exploit/explore (weight inheritance) ---------------
    ckpt = train.get_checkpoint()
    requested_restore = str(config.get("restore_ckpt", "") or "").strip()
    if requested_restore:
        algo.restore(requested_restore)
        print(f"Restored checkpoint from --restore_ckpt: {requested_restore}")
    elif ckpt is not None:
        restore_dir = ckpt.to_directory()
        algo.restore(restore_dir)
        print(f"Restored checkpoint from Tune checkpoint: {restore_dir}")


    checkpoint_interval = int(config.get("checkpoint_interval", 3))
    full_metrics_path = trial_dir / "rllib" / "metrics_full.jsonl"
    full_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    stable_root = Path(str(config.get("stable_metrics_root", Path.home() / "ray_results" / "_stable_metrics")))
    stable_trial_dir = stable_root / trial_dir.name
    stable_trial_dir.mkdir(parents=True, exist_ok=True)
    stable_metrics_path = stable_trial_dir / "metrics_full.jsonl"
    stable_report_csv = stable_trial_dir / "report_rows.csv"

    for it in range(config["stop_iters"]):
        res = algo.train()
        report = _extract_report(res, it + 1)

        # Persist scalar-only flattened metrics for robust plotting fallback.
        full = _flatten_scalar_metrics(res)
        full["training_iteration"] = report["training_iteration"]
        _append_json_line(full_metrics_path, full)
        _append_json_line(stable_metrics_path, full)
        _append_report_row(stable_report_csv, dict(report))

        # --- Checkpointing for PBT (so exploit can copy weights) ------------
        should_checkpoint = checkpoint_interval > 0 and (it + 1) % checkpoint_interval == 0
        if should_checkpoint:
            try:
                ckpt_root = trial_dir / "rllib_checkpoints"
                ckpt_root.mkdir(parents=True, exist_ok=True)

                saved = algo.save(str(ckpt_root))
                checkpoint, ckpt_label = _checkpoint_from_save_result(saved)

                if checkpoint is not None:
                    session.report(report, checkpoint=checkpoint)
                    print(f"Checkpoint saved @ iter {it+1}: {ckpt_label}")
                else:
                    session.report(report)
                    print(f"Checkpoint produced unexpected type; reported metrics only. save()={ckpt_label}")

            except Exception as e:
                # If checkpointing fails, still report metrics
                session.report(report)
                print(f"Checkpoint save failed @ iter {it+1}: {e}")
        else:
            session.report(report)

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

    algo.stop()
# ────────────────────────────────────────────────────────────────────────────────
# 5) Entrypoint: assemble Tune config & kick off the experiment
# ────────────────────────────────────────────────────────────────────────────────
def _register_models() -> None:
    ModelCatalog.register_custom_model("solver_lstm_v2", SolverLSTMModel)
    ModelCatalog.register_custom_model("solver_mlp", SolverMLPModel)
    ModelCatalog.register_custom_model("solver_mlp_film", SolverMLPFiLM)


def _resources_per_trial(args, base_config: dict) -> PlacementGroupFactory:
    if args.cpu_pbt:
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
    write_run_manifest(base_config=base_config, args=args, data_path=data_path)

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        # metric="custom_metrics/episode_return_mean",
        # mode="max",
        perturbation_interval=3,           # exploit/explore every 3 iters
        quantile_fraction=0.25,            # bottom 25% replaced by top 25%
        resample_probability=0.25,
        hyperparam_mutations={
            # PPO core knobs
            # Include aggressive LR regime seen in ablations while keeping conservative floor.
            "lr": tune.loguniform(1e-5, 5e-3),
            "clip_param": tune.choice([0.1, 0.2]),
            "lambda": tune.choice([0.90, 0.95, 0.97]),
            "gamma": tune.choice([0.98, 0.99]),
            "sgd_minibatch_size": tune.choice([128, 256, 512]),
            "num_sgd_iter": tune.choice([10, 20, 30]),
            "vf_clip_param": tune.choice([5.0, 10.0]),
            "vf_loss_coeff": tune.choice([0.5, 1.0, 2.0]),
            "train_batch_size": tune.choice([4096, 4800, 6144]),
            # If you drop entropy schedule, you can mutate entropy_coeff instead.
            # "entropy_coeff": tune.uniform(0.005, 0.02),
            # NOTE: Avoid mutating list/dict-valued nested keys here; Ray PBT may
            # attempt arithmetic perturbation and crash on non-scalars.
        },
    )

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

    tune.run(
        training_fn,                        # <--- use the class directly
        name=args.experiment_name,
        scheduler=(pbt if args.pbt else None),
        num_samples=(args.pop if args.pbt else 1),
        resources_per_trial=resources,
        stop=TimeoutStopper(args.timeout_hours * 3600),
        # stop={"training_iteration": args.iters},
        metric="custom_metrics/episode_return_mean",    # tune on our custom metric
        mode="max",
        config=base_config,
        local_dir=str(Path.home() / "ray_results"),
        progress_reporter=tune.CLIReporter(
            metric_columns=_cli_metric_columns(base_config["env_config"]["task_sampler"]["action_dim"]),
        ),
        reuse_actors=False,  # safer for PBT/long runs; flip True only if you're 100% stable
        max_failures=3,  # Allow up to 3 failures before giving up on a trial
        )



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
    parser.add_argument("--cpu_pbt", action="store_true", help="Run PBT on CPU (many trials), then retrain best on GPU")
    parser.add_argument("--pop", type=int, default=10, help="Population size if --pbt")
    parser.add_argument("--timeout_hours", type=int, default=5, help="Global timeout for the whole run")
    parser.add_argument("--restore_ckpt", type=str, default="", help="Path to an RLlib checkpoint_XXXXXX dir to restore from")
    parser.add_argument("--best_params_json", type=str, default="", help="Path to params.json from best PBT trial (optional)")
    parser.add_argument("--start_idx_mode", type=str, default="random_episode", choices=["random_episode", "stream_worker", "task_spec"], help="Episode start index policy for env resets")
    parser.add_argument("--stream_jump_every_episodes", type=int, default=0, help="For stream_worker: random jump cadence in episodes (0 disables)")
    parser.add_argument("--stream_jump_on_wrap", action=argparse.BooleanOptionalAction, default=True, help="For stream_worker: randomize on stream wrap-around")
    parser.add_argument("--stream_stride", type=int, default=0, help="For stream_worker: stride in rows between consecutive episodes (0 uses horizon)")
    parser.add_argument("--ret_norm_reset", type=str, default="episode", choices=["episode", "stream"], help="Reset reward normalization every episode or only when stream changes")
    parser.add_argument("--task_pool_path", type=str, default="", help="Optional JSONL task pool path for sampler")
    parser.add_argument("--p_pool", type=float, default=0.7, help="Probability of sampling from task pool when configured")
    parser.add_argument("--task_horizons", type=str, default="", help="Comma-separated task horizons; empty uses the Phase A defaults")
    parser.add_argument("--task_reward_ids", type=str, default="", help="Comma-separated registered reward IDs; empty uses pnl_linear")
    parser.add_argument("--experiment_name", type=str, default="Phase_A", help="Ray Tune experiment folder name under ~/ray_results")
    parser.add_argument("--auto_plot_on_end", action=argparse.BooleanOptionalAction, default=True, help="Generate plots automatically at trial end")
    parser.add_argument(
        "--data_path",
        default="itrader/data/ohlcv.parquet",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Global run seed for RLlib and task sampler")
    args = parser.parse_args()
    main(args)
