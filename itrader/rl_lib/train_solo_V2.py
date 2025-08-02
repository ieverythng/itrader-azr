# itrader/rl_lib/train_solo.py

import argparse
from pathlib import Path

import torch
import ray
from ray import tune
from ray import train
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.logger import TBXLoggerCallback
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import PlacementGroupFactory

# your env & model imports
from itrader.envs.itrader_env import ITraderEnv
from itrader.models.solver_lstm import SolverLSTMModel

# TimeLimit wrapper
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import EnvSpec
# ────────────────────────────────────────────────────────────────────────────────
# 1) Ray init: dashboard + Prometheus
# ────────────────────────────────────────────────────────────────────────────────
ray.init(
    address=None,
    _metrics_export_port=8888,
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
class StatsCallback(DefaultCallbacks):
    def on_learn_on_batch_end(self, *, results, **__):
        # after each learner step, grab its timing and expose a custom metric
        learner_stats = results["learner"]["learner_stats"]
        iters_per_sec = 1000.0 / learner_stats["learner_time_ms"]
        results["custom_metrics"]["iters_per_sec"] = iters_per_sec
# ────────────────────────────────────────────────────────────────────────────────
# 4) The “training function” for Tune
#    This builds the PPOAlgo, loops train(), and reports key scalars
# ────────────────────────────────────────────────────────────────────────────────
def training_fn(config):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # train.report(device=device)

    algo = PPO(config=config)
    print("Learner on:", next(algo.workers.local_worker().policy_map["default_policy"].model.parameters()).device)
    for it in range(config["stop"]["training_iteration"]):
        result = algo.train()
        # report all the scalars we care about into Tune’s metrics store:
        train.report(
            episode_return_mean=result.get("episode_return_mean", float("nan")),
            iters_per_sec=result["custom_metrics"]["iters_per_sec"],
            policy_loss=result["info"]["learner"]["default_policy"]["learner_stats"]["policy_loss"],
            value_loss=result["info"]["learner"]["default_policy"]["learner_stats"]["vf_loss"],
            entropy=result["info"]["learner"]["default_policy"]["learner_stats"]["entropy"],
            timesteps_total=result["timesteps_total"],
        )
        if it % 10 == 0:
            ret = result.get("episode_return_mean", None)
            sps = result["custom_metrics"]["samples_per_sec"]
            if ret is not None:
                print(f"[{it:04d}] return={ret:+7.3f}  samples/sec={sps:5.2f}")
            else:
                print(f"[{it:04d}] return=None  samples/sec={sps:5.2f}")        
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

    # shared RLlib config
    base_config = {
        "env": "ITraderEnv",
        "env_config": {
            "df_path": data_path.as_posix(),
            "window_size": args.window,
        },
        "model": {
            "custom_model": "solver_lstm_v2",
            "custom_model_config": {"window_size": args.window},
        },
        # resources
        "num_gpus": 1,
        "num_gpus_per_worker": 0,
        "num_workers": 6,
        # core RL settings
        "framework": "torch",
        "callbacks": StatsCallback,
        "train_batch_size": 10_000,
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
    tune.run(
        training_fn,
        name="Phase_A",
        config=base_config,
        stop={"training_iteration": args.iters},
        metric="episode_return_mean",
        mode="max",
        local_dir=str(Path.home() / "ray_results"),
        progress_reporter=tune.CLIReporter(metric_columns=["episode_return_mean","samples_per_sec"]),
        callbacks=[],               # (you could add TBXLoggerCallback here)
        reuse_actors=False,         # each trial builds fresh actors
        resources_per_trial= PlacementGroupFactory(
            # 1 slot for the main driver (will need 1 GPU + 1 CPU)
            [{"CPU": 1, "GPU": 1}] +
            # N rollout workers, each with 1 CPU, 0 GPUs
            [{"CPU": 1}] * base_config["num_workers"]
        )
    )

    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters",     type=int, default=200)
    parser.add_argument("--window",    type=int, default=30)
    parser.add_argument(
        "--data_path",
        default="itrader/data/ohlcv.parquet",
    )
    args = parser.parse_args()
    main(args)