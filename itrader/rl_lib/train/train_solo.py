# itrader/rl_lib/train_solo.py

import argparse
from pathlib import Path

import torch
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog

# your env & model imports
from itrader.envs.itrader_env import ITraderEnv
from itrader.models.solver_lstm import SolverLSTMModel

# TimeLimit wrapper
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import EnvSpec

ray.init(
    address=None,
    _metrics_export_port=8888,
    include_dashboard=True,
    dashboard_host="0.0.0.0",
)

def make_env(cfg):
    e = ITraderEnv(cfg)
    e.spec = EnvSpec("ITraderEnv", max_episode_steps=200)
    return TimeLimit(e, max_episode_steps=200)  

tune.register_env("ITraderEnv", make_env)

# throughput callback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
class StatsCallback(DefaultCallbacks):
    def on_learn_on_batch_end(self, *, results, **__):
        ls = results["learner"]["learner_stats"]
        results["custom_metrics"]["iters_per_sec"] = 1000.0 / ls["learner_time_ms"]

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==> Training on device: {device}")

    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Parquet not found ➜ {data_path}")

    # register the v2 model
    ModelCatalog.register_custom_model("solver_lstm_v2", SolverLSTMModel)

    config = {
        "env": "ITraderEnv",
        "env_config": {
            "df_path": data_path.as_posix(),
            "window_size": args.window,
        },
        "model": {
            "custom_model": "solver_lstm_v2",
            "custom_model_config": {"window_size": args.window},
        },
        "num_gpus": 1,
        "num_gpus_per_worker": 0,         
        "num_workers": 6,
        "framework": "torch",
        "callbacks": StatsCallback,
        "metrics_sinks": ["prometheus"],
        "prometheus_export_port": 8888,
        "train_batch_size": 10000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 30,
        "gamma": 0.99,
        "vf_clip_param": 10.0,
        "lr": 5e-5,
    }

    algo = PPO(config=config)
    for it in range(args.iters):
        res = algo.train()
        if it % 10 == 0:
            # grab metrics
            ret = res.get("episode_return_mean", None)
            ips = res["custom_metrics"].get("iters_per_sec", 0.0)

            if ret is not None:
                # safe to format as float
                print(f"[{it:04d}] return={ret:+7.3f}  samples/sec={ips:5.2f}")
            else:
                # skip the float formatting for `ret`
                print(f"[{it:04d}] return=None  samples/sec={ips:5.2f}")
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iters",     type=int, default=200)
    p.add_argument("--window",    type=int, default=30)
    p.add_argument("--data_path", default="itrader/data/ohlcv.parquet")
    main(p.parse_args())
