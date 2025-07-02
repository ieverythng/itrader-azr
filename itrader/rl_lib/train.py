# p4/rllib/train.py
import ray, argparse, gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from itrader_env import ITraderEnv

ray.init()

cfg = (PPOConfig()
       .environment(env="iTraderEnv-v0",
                    env_config={"df_path":"data/sp500.parquet"})
       .framework("torch")
       .rollouts(num_rollout_workers=3)
       .training(train_batch_size=4000,
                 lambda_=0.95,
                 vf_clip_param=10.0))
algo = cfg.build()
for _ in range(100):
    result = algo.train()
    print(result["episode_reward_mean"])
