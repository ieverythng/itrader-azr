# itrader/rl_lib/train_solo.py

import argparse
from pathlib import Path

import torch
import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.logger import TBXLoggerCallback
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import PlacementGroupFactory

# your env & model imports
from itrader.envs.itrader_env import ITraderEnv
from itrader.models.solver_lstm import SolverLSTMModel
from itrader.models.solver_MLP import SolverMLPModel
from itrader.models.solver_mlp_film import SolverMLPFiLM
from itrader.rl_lib.callbacks.ps_metrics import PSMetrics

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
    e.spec = EnvSpec("ITraderEnv", max_episode_steps=200)
    return TimeLimit(e, max_episode_steps=200)  

tune.register_env("ITraderEnv", make_env)
# ────────────────────────────────────────────────────────────────────────────────
# 3) Custom callback to expose samples/sec
# ────────────────────────────────────────────────────────────────────────────────
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# class ReturnLoggingCallback(DefaultCallbacks):
#     def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
#         # This fires whenever an episode actually terminates.
#         total_reward = episode.total_reward
#         episode.custom_metrics["episode_return"] = total_reward

# class StatsCallback(DefaultCallbacks):
#     def on_learn_on_batch_end(self, *, results, **__):
#         ms = results["learner"]["learner_stats"].get("learner_time_ms", 0.0)
#         results["custom_metrics"]["iters_per_sec"] = (1000.0 / ms) if ms > 0 else 0.0
#  ────────────────────────────────────────────────────────────────────────────────
# 4) The “training function” for Tune
#    This builds the PPOAlgo, loops train(), and reports key scalars
# ────────────────────────────────────────────────────────────────────────────────
def training_fn(config):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # train.report(device=device)

    algo = PPO(config=config)
    print("Learner on:", next(algo.workers.local_worker().policy_map["default_policy"].model.parameters()).device)
    for it in range(config["stop_iters"]):
        res = algo.train()

        # push scalars into Tune’s result store
        ls   = res["info"]["learner"]["default_policy"]["learner_stats"]
        cm   = res.get("custom_metrics", {})
        train.report({
            "episode_return_mean": cm.get("episode_return_mean", float("nan")),
            "policy_loss":        ls["policy_loss"],
            "value_loss":         ls["vf_loss"],
            "entropy":            ls["entropy"],
        })

        if it % 10 == 0:
            r   = cm.get("episode_return", None)
            ips = cm.get("iters_per_sec",  float("nan"))
            print(f"[{it:04d}] return={r if r is not None else 'None':>7}"
                  f"  π_loss={ls['policy_loss']:+8.4f}"
                  f"  V_loss={ls['vf_loss']:+8.4f}"
                  f"  ent={ls['entropy']:.3f}"
                  f"  it/s={ips:5.2f}")
    algo.stop()
# ────────────────────────────────────────────────────────────────────────────────
# 5) Entrypoint: assemble Tune config & kick off the experiment
# ────────────────────────────────────────────────────────────────────────────────
def main(args):
    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Parquet not found ➜ {data_path}")

    # register the v2 model
    ModelCatalog.register_custom_model("solver_lstm_v2", SolverLSTMModel)
    ModelCatalog.register_custom_model("solver_mlp", SolverMLPModel)
    ModelCatalog.register_custom_model("solver_mlp_film", SolverMLPFiLM)

    num_workers  = args.workers

    # shared RLlib config
    base_config = {
        "env": "ITraderEnv",
        "env_config": {
            "df_path": data_path.as_posix(),
            "window_size": args.window,
            "t_embed_mode": "concat",   # off|scale|concat
            "t_scale": 1.0,
        },
        "model": {
            "custom_model": "solver_mlp_film",
            "custom_model_config": {
                "window_size": args.window,
                "t_embed_dim": 32,
                "fcnet_hiddens": [256, 256],
                "conditioning": "film",
                "t_keep_p": 0.9,
                "t_proj": 64,
            },
        },
        # resources
        "num_gpus": 1,
        "num_gpus_per_worker": 0,
        "num_workers": num_workers,
        # core RL settings
        "framework": "torch",
        "callbacks": PSMetrics,
        "observation_filter": "MeanStdFilter",
        "grad_clip": 10.0,
        # Stabilise exploration → exploitation
        "entropy_coeff_schedule": [
            [0,       0.03],
            [500_000, 0.01],
            [1_500_000, 0.003],
        ],
        "train_batch_size": 4800,
        "batch_mode": "complete_episodes",
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 30,
        "gamma": 0.99,
        "vf_clip_param": 10.0,
        "lr": 5e-5,
        # hook up Prometheus
        "metrics_sinks": ["prometheus"],
        "prometheus_export_port": 8888,
        # pass the stop criterion to training_fn
        "stop_iters": args.iters,
    }

    
    # optional: PBT scheduler placeholder
    # pbt = PopulationBasedTraining(
    #     time_attr="training_iteration",
    #     metric="episode_return_mean", mode="max",
    #     perturbation_interval=5,
    #     hyperparam_mutations={"lr": [1e-4, 5e-5, 1e-5]},
    # )
# ────────────────────────────────────────────────────────────────────────────────
# 5.1) Kick-off training algo
# ────────────────────────────────────────────────────────────────────────────────    
  
    analysis = tune.run(
        training_fn,                        # <--- use the class directly
        name="Phase_A",
        stop={"training_iteration": args.iters},
        metric="episode_return_mean",    # tune on our custom metric
        mode="max",
        config=base_config,
        local_dir=str(Path.home() / "ray_results"),
        progress_reporter=tune.CLIReporter(
        metric_columns=["episode_return_mean", "policy_loss","value_loss", "iters_per_sec"],
        ),
        reuse_actors=False,
        resources_per_trial= PlacementGroupFactory(
            # 1 slot for the main driver (will need 1 GPU + 1 CPU)
            [{ "GPU": 1, "CPU": 1}] +
            # N rollout workers, each with 1 CPU, 0 GPUs
            [{"CPU": 1}] * base_config["num_workers"]
        )
        
    )
    print("Best hyperparams found:", analysis.best_config)

    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters",     type=int, default=200)
    parser.add_argument("--window",    type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--data_path",
        default="itrader/data/ohlcv.parquet",
    )
    args = parser.parse_args()
    main(args)