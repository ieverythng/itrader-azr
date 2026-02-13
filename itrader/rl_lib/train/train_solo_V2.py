# itrader/rl_lib/train_solo.py

import argparse
import json
from pathlib import Path

import torch
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
from itrader.rl_lib.callbacks.ps_metrics import PSMetrics
from itrader.rl_lib.callbacks.auto_plotter import AutoPlotterCallback

# TimeLimit wrapper
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import EnvSpec
# ────────────────────────────────────────────────────────────────────────────────
# 1) Ray init: dashboard + Prometheus
# ────────────────────────────────────────────────────────────────────────────────
ray.init(
    _metrics_export_port=(8888),
    include_dashboard=True,
    dashboard_host="0.0.0.0",
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

class CombinedCallback(DefaultCallbacks):
    def __init__(self, plot_interval: int = 50):
        super().__init__()
        self.ps_metrics = PSMetrics()
        self.auto_plotter = AutoPlotterCallback(plot_at_end_of_loop=True)
    
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
def training_fn(config):
    # Always use Tune trial dir as the single source of truth for ALL outputs.
    trial_dir = Path(session.get_trial_dir())

    # Force RLlib's UnifiedLogger to write into the Tune trial dir (prevents PPO_ITraderEnv_* dirs).
    def logger_creator(cfg):
        # Keep RLlib logs separate to avoid clobbering Tune's progress.csv/result.json.
        rllib_dir = trial_dir / "rllib"
        rllib_dir.mkdir(parents=True, exist_ok=True)
        return UnifiedLogger(cfg, str(rllib_dir), loggers=None)
    algo = PPO(config=config, logger_creator=logger_creator)
    
    # --- Restore for PBT exploit/explore (weight inheritance) ---------------
    ckpt = train.get_checkpoint()
    # Check if user requested a specific checkpoint restore
    if args.restore_ckpt:
        restore_dir = str(config.get("restore_ckpt", "") or "") 
        algo.restore(restore_dir)
        print(f"✅ Restored checkpoint from: {restore_dir}")
    elif ckpt is not None:
        restore_dir = ckpt.to_directory() 
        algo.restore(restore_dir)
        print(f"✅ Restored checkpoint from: {restore_dir}")

    
    checkpoint_interval = int(config.get("checkpoint_interval", 3))
    full_metrics_path = trial_dir / "metrics_full.jsonl"
    
    for it in range(config["stop_iters"]):
        res = algo.train()

        # push scalars into Tune’s result store
        ls   = res["info"]["learner"]["default_policy"]["learner_stats"]

        # robust metric extraction
        policy_loss = float(ls.get("policy_loss", 0.0))
        vf_loss = float(ls.get("vf_loss", ls.get("value_loss", 0.0)))
        entropy = float(ls.get("entropy", 0.0))

        # Minimal payload for console + CSV (avoid dumping all min/max).
        report = {
            "training_iteration": int(res.get("training_iteration", it + 1)),
            "time_this_iter_s": res.get("time_this_iter_s"),
            "time_total_s": res.get("time_total_s"),
            "timesteps_total": res.get("timesteps_total"),
            "policy_loss": policy_loss,
            "value_loss": vf_loss,
            "entropy": entropy,
        }
        if res.get("time_this_iter_s"):
            report["iters_per_sec"] = 1.0 / float(res["time_this_iter_s"])

        # Custom metrics (means only for console)
        cm = res.get("custom_metrics") or res.get("sampler_results", {}).get("custom_metrics")
        if isinstance(cm, dict):
            for k in ("act_long_frac_mean", "act_short_frac_mean", "act_flat_frac_mean"):
                if k in cm and isinstance(cm[k], (int, float, np.floating, np.integer)):
                    report[f"custom_metrics/{k}"] = float(cm[k])

        # Full metrics dump for plotting (scalar-only, JSONL).
        full = {}
        stack = [("", res)]
        while stack:
            prefix, obj = stack.pop()
            if not isinstance(obj, dict):
                continue
            for k, v in obj.items():
                key = f"{prefix}/{k}" if prefix else str(k)
                if isinstance(v, dict):
                    stack.append((key, v))
                elif isinstance(v, (int, float, bool, str, np.floating, np.integer)):
                    full[key] = float(v) if isinstance(v, (np.floating, np.integer)) else v
        full["training_iteration"] = report["training_iteration"]
        full["custom_metrics/episode_return_mean"] = report["custom_metrics/episode_return_mean"]
        with full_metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(full) + "\n")
        
        # --- Checkpointing for PBT (so exploit can copy weights) ------------
        should_checkpoint = checkpoint_interval > 0 and (it + 1) % checkpoint_interval == 0
        if should_checkpoint:
            try:
                ckpt_root = trial_dir / "rllib_checkpoints"
                ckpt_root.mkdir(parents=True, exist_ok=True)

                saved = algo.save(str(ckpt_root))
                # RLlib/Ray versions differ: save() may return str path, Checkpoint, or a Result-like wrapper.
                if isinstance(saved, str):
                    checkpoint = Checkpoint.from_directory(saved)
                    ckpt_label = saved
                elif isinstance(saved, Checkpoint):
                    checkpoint = saved
                    ckpt_label = getattr(saved, "path", "Checkpoint")
                elif hasattr(saved, "checkpoint") and isinstance(saved.checkpoint, Checkpoint):
                    checkpoint = saved.checkpoint
                    ckpt_label = getattr(checkpoint, "path", "Checkpoint")
                else:
                    # Last-resort: stringify and avoid crashing training; still report metrics.
                    checkpoint = None
                    ckpt_label = str(saved)

                if checkpoint is not None:
                    session.report(report, checkpoint=checkpoint)
                    print(f"💾 Checkpoint saved @ iter {it+1}: {ckpt_label}")
                else:
                    session.report(report)
                    print(f"⚠️  Checkpoint produced unexpected type; reported metrics only. save()={ckpt_label}")

            except Exception as e:
                # If checkpointing fails, still report metrics
                session.report(report)
                print(f"⚠️  Checkpoint save failed @ iter {it+1}: {e}")
        else:
            session.report(report)
    
    algo.stop()
# ────────────────────────────────────────────────────────────────────────────────
# 5) Entrypoint: assemble Tune config & kick off the experiment
# ────────────────────────────────────────────────────────────────────────────────
def main(args):
    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Parquet not found ➜ {data_path}")

    # register the v2 model
    ModelCatalog.register_custom_model("solver_mlp_film", SolverMLPFiLM)

    # shared RLlib config
    base_config = {
        "env": "ITraderEnv",
        "env_config": {
            "df_path": data_path.as_posix(),
            "window_size": args.window,
            "t_embed_mode": "concat",   # off|scale|concat
            "t_scale": 1.0,
            "task_sampler": {
                "window_size": args.window,
                "n_features": 5,  # OHLCV features
                "t_embed_dim": 32,
                "action_dim": 3,
                # benign defaults; sampler gets per-worker seed automatically
                "horizons": [96, 128, 160, 200],
                "turnover_w_choices": [0.0, 0.0005, 0.001, 0.002],
                "dd_modes": ["delta"],
                "dd_w_range": (0.0, 0.04),
                "turnover_cap": 0.02,
                "dd_cap": 0.10,
            },
        # ← NEW: allows Tune/PBT to tweak tiny reward weights *without* editing the sampler.
            # Your env.reset() should merge this dict into current_task.reward_params (you already added this).
            # "task_sampler_overrides": {
            #     "turnover_w": 0.001,
            #     "dd_w": 0.04,
            #     "dd_T_weight": 0.0,
            # },
        },
        "model": {
            "custom_model": "solver_mlp_film",
            "custom_model_config": {
                "window_size": args.window,
                "t_embed_dim": 32,
                "fcnet_hiddens": [256, 256],
                "conditioning": "film",
                "t_keep_p": 0.7,
                "t_proj": 64,
            },
        },
        # resources
        "num_gpus": 1,
        "num_gpus_per_worker": 0,
        "num_workers": args.workers,
        # core RL settings
        "framework": "torch",
        # swap callbacks for PBT to reduce collisions/overhead
        "callbacks": (MetricsOnlyCallback if args.pbt else CombinedCallback),
        "observation_filter": "MeanStdFilter",
        "grad_clip": 10.0,
        # Stabilise exploration → exploitation
        "entropy_coeff_schedule": [
            [0,        0.02],
            [200_000,  0.01],
            [800_000,  0.003],
        ],
        "train_batch_size": 4800,
        "batch_mode": "complete_episodes",
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 30,
        "gamma": 0.99,
        "vf_clip_param": 10.0,
        "lr": 5e-5,
        # PBT runs multiple trials -> shared port conflicts. Enable only for single-trial runs.
        "metrics_sinks": ([] if args.pbt else ["prometheus"]),
        "prometheus_export_port": (None if args.pbt else 8888),
        # pass the stop criterion to training_fn
        "stop_iters": args.iters,
        "checkpoint_interval": 10,  # Save checkpoint every N iterations for PBT
    }

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        # metric="custom_metrics/episode_return_mean",
        # mode="max",
        perturbation_interval=3,           # exploit/explore every 3 iters
        quantile_fraction=0.25,            # bottom 25% replaced by top 25%
        resample_probability=0.25,
        hyperparam_mutations={
            # PPO core knobs
            "lr": tune.loguniform(1e-5, 3e-4),  # Wider range including lower LR for stability
            "clip_param": tune.choice([0.1, 0.2]),
            "lambda": tune.choice([0.90, 0.95, 0.97]),
            "gamma": tune.choice([0.98, 0.99]),
            "sgd_minibatch_size": tune.choice([128, 256, 512]),
            "num_sgd_iter": tune.choice([10, 20, 30]),
            "vf_clip_param": tune.choice([5.0, 10.0]),
            # If you drop entropy schedule, you can mutate entropy_coeff instead.
            # "entropy_coeff": tune.uniform(0.005, 0.02),
            # Small shaping weights via env passthrough
            "env_config/task_sampler_overrides": {
                "turnover_w": tune.loguniform(5e-4, 2e-2),  # Lower minimum for fine-tuning
                "dd_w": tune.loguniform(5e-3, 1e-1),  # Lower minimum, same max
                "dd_T_weight": tune.loguniform(1e-3, 2e-1),
                "norm_mode": tune.choice(["all", "penalties"])
            },
            "model/custom_model_config/fcnet_hiddens": tune.choice([
                [256, 256], [512, 256], [256, 256, 128]  # Test capacity
            ]),
        },
    )  

# ────────────────────────────────────────────────────────────────────────────────
# 5.1) Kick-off training algo
# ────────────────────────────────────────────────────────────────────────────────    
  
    print("🚀 Starting training with automatic plotting...")
    print(f"📊 Plots will be generated every 50 iterations")

    # CPU-PBT mode: run many trials in parallel on CPU, then retrain the best on GPU.
    if args.cpu_pbt:
        base_config.update({
            "num_gpus": 0,
            "num_workers": max(2, min(args.workers, 4)),   # keep workers modest per trial
        })
        resources = PlacementGroupFactory(
            # driver
            [{"CPU": 1}] +
            # rollout workers per trial
            [{"CPU": 1}] * base_config["num_workers"]
        )
    else:
        # single GPU mode (default): one trial uses your GPU
        base_config.update({
            "num_gpus": 1,
            "num_workers": args.workers,
        })
        resources = PlacementGroupFactory(
            [{ "GPU": 1, "CPU": 1}] + [{"CPU": 1}] * base_config["num_workers"]
        )
    
    # This avoids breaking callbacks/env wiring by blindly replacing the whole config.
    if args.best_params_json:
        import json
        p = Path(args.best_params_json).expanduser()
        print(f"🔁 Loading best_params_json: {p}")
        best = json.loads(p.read_text(encoding="utf-8"))

        # Top-level PPO knobs
        for k in ["lr", "clip_param", "lambda", "gamma",
                  "sgd_minibatch_size", "num_sgd_iter",
                  "vf_clip_param", "vf_loss_coeff", "train_batch_size"]:
            if k in best:
                base_config[k] = best[k]

        # Model custom config (FiLM)
        try:
            if "model" in best and "custom_model_config" in best["model"]:
                base_config["model"]["custom_model_config"].update(best["model"]["custom_model_config"])
        except Exception:
            pass

        # Env overrides (reward knobs)
        try:
            if "env_config" in best and "task_sampler_overrides" in best["env_config"]:
                base_config["env_config"]["task_sampler_overrides"] = best["env_config"]["task_sampler_overrides"]
        except Exception:
            pass

    if args.restore_ckpt:
        base_config["restore_ckpt"] = args.restore_ckpt
    
    metric_columns=[
        "episode_return_mean",
        "policy_loss",
        "value_loss",
        "entropy",
        "custom_metrics/act_short_frac_mean",
        "custom_metrics/act_flat_frac_mean",
        "custom_metrics/act_long_frac_mean",
    ]
    analysis = tune.run(
        training_fn,                        # <--- use the class directly
        name="Phase_A",
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
            metric_columns=metric_columns,
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
    parser.add_argument("--pbt", action="store_true", help="Enable Population-Based Training")
    parser.add_argument("--cpu_pbt", action="store_true", help="Run PBT on CPU (many trials), then retrain best on GPU")
    parser.add_argument("--pop", type=int, default=10, help="Population size if --pbt")
    parser.add_argument("--timeout_hours", type=int, default=5, help="Global timeout for the whole run")
    parser.add_argument("--restore_ckpt", type=str, default="", help="Path to an RLlib checkpoint_XXXXXX dir to restore from")
    parser.add_argument("--best_params_json", type=str, default="", help="Path to params.json from best PBT trial (optional)")
    parser.add_argument(
        "--data_path",
        default="itrader/data/ohlcv.parquet",
    )
    args = parser.parse_args()
    main(args)
