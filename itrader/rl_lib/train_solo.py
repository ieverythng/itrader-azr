import argparse, random, numpy as np, torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
import gymnasium as gym
from pathlib import Path

# Phase-A helpers  (path now pinned to PA_helpers)
from itrader.envs.itrader_env import ITraderEnv
from itrader.inputs.PA_helpers import load_market_data, compute_valid_indices
from itrader.models.solver_lstm import SolverLSTM


# ----------------------------------------------------------------------
def build_config(window:int, parquet:str)->PPOConfig:
    """
    Pure-new-stack PPOConfig that runs ITraderEnv and uses SolverLSTM.
    """
    dummy_env = ITraderEnv                 # hand class to RLlib

    env_cfg = {
        "df_path"    : parquet,
        "window_size": window,
        # (anything else ITraderEnv.__init__ expects)
    }

    # Get spaces once so we can feed them into RLModuleSpec
    tmp_env      = dummy_env(env_cfg)
    obs_space    = tmp_env.observation_space
    act_space    = tmp_env.action_space
    tmp_env.close()

    return (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True,
                   enable_env_runner_and_connector_v2=True)   # 100 % new API
        .environment(env=dummy_env, env_config=env_cfg)                       
        .framework("torch", eager_tracing=True)
        .env_runners(num_env_runners=0)
        .debugging(log_level="WARN")
        .resources(num_gpus=1)
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class      = SolverLSTM,
                observation_space = obs_space,
                action_space      = act_space,
                model_config = {"window_size": window},
            )
        )                 # local sampling
        .training(
            train_batch_size=4000,
            minibatch_size=128,        # <-- replaces sgd_minibatch_size 
            num_epochs=30,             # replaces num_sgd_iter
            gamma=0.99,
            vf_clip_param=10.0,
            lr=5e-5,
        )
    )


# ----------------------------------------------------------------------
def main(args):
    parquet = Path(args.data_path).as_posix()
    if not Path(parquet).is_file():
        raise FileNotFoundError(f"Parquet not found ➜ {parquet}")

    cfg  = build_config(args.window, parquet)
    algo = cfg.build_algo()

    for it in range(args.iters):
        result = algo.train()                # standard PPO step
        if it % 10 == 0:
            r = result["episode_reward_mean"]
            print(f"[{it:04d}] mean episode reward = {r:.3f}")

    algo.stop()

# -------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iters",      type=int, default=200)
    p.add_argument("--window",     type=int, default=30)
    p.add_argument("--data_path",  default="itrader/data/ohlcv.parquet")
    args = p.parse_args()
    main(args)