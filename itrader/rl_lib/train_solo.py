import argparse, random, numpy as np, torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from gymnasium.wrappers import TimeLimit
from pathlib import Path
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Phase-A helpers  (path now pinned to PA_helpers)
from itrader.envs.itrader_env import ITraderEnv
from itrader.models.solver_lstm import SolverLSTM

# ----------------------------------------------------------------------
tune.register_env(
    "ITraderEnv",
    lambda cfg: TimeLimit(ITraderEnv(cfg), max_episode_steps=200),
)

# ----------------------------------------------------------------------

class StatsCallback(DefaultCallbacks):
    def on_learn_on_batch_end(self, *, results, **__):
        # Ray 2.9 stores learner stats under results["learners"]["default_policy"]
        ls = results["learners"]["default_policy"]["learner_stats"]
        results["custom_metrics"]["iters_per_sec"] = 1000.0 / ls["learner_time_ms"]


# ----------------------------------------------------------------------
def build_config(window:int, parquet:str)->PPOConfig:
    """
    Pure-new-stack PPOConfig that runs ITraderEnv and uses SolverLSTM.
    """
    env_cfg = {
        "df_path"    : parquet,
        "window_size": window,
        # (anything else ITraderEnv.__init__ expects)
    }

    # Get spaces once so we can feed them into RLModuleSpec
    tmp_env      = TimeLimit(ITraderEnv(env_cfg), max_episode_steps=200)
    obs_space    = tmp_env.observation_space
    act_space    = tmp_env.action_space
    tmp_env.close()

     # --- simple custom callback to surface throughput -----------------

    cfg = (
        PPOConfig()
        .experimental(_enable_new_api_stack=True)
        .environment(env="ITraderEnv", env_config=env_cfg)
        .callbacks(StatsCallback)                       
        .framework("torch", eager_tracing=True)
        .rollouts(num_rollout_workers=6)   # Ryzen-5500 = 6 cores
        .debugging(log_level="WARN")
        .resources(num_gpus=1, num_cpus_per_worker=1)
        .rl_module(
            rl_module_spec=SingleAgentRLModuleSpec(
                module_class      = SolverLSTM,
                observation_space = obs_space,
                action_space      = act_space,
                model_config_dict = {"window_size": window},
            )
        )                 # local sampling
        .training(
            train_batch_size=10000,
            sgd_minibatch_size=128,        # <-- replaces sgd_minibatch_size 
            num_sgd_iter=30,             # replaces num_sgd_iter
            gamma=0.99,
            vf_clip_param=10.0,
            lr=5e-5,
        )       
    )
    cfg.postprocess_inputs = True

    return cfg


# ----------------------------------------------------------------------
def main(args):
    parquet = Path(args.data_path).as_posix()
    if not Path(parquet).is_file():
        raise FileNotFoundError(f"Parquet not found ➜ {parquet}")

    cfg  = build_config(args.window, parquet)
    algo = cfg.build()

    for it in range(args.iters):
        res = algo.train()
        if it % 10 == 0:
            r   = res.get("episode_return_mean", None)
            ips = res["custom_metrics"].get("iters_per_sec", 0.0)
            if r is not None:
                print(f"[{it:04d}] return={r:+7.3f} | iters/s={ips:5.2f}")
            else:
                print(f"[{it:04d}] keys →", list(res.keys()))
    algo.stop()

# -------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iters",      type=int, default=200)
    p.add_argument("--window",     type=int, default=30)
    p.add_argument("--data_path",  default="itrader/data/ohlcv.parquet")
    args = p.parse_args()
    main(args)