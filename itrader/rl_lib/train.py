import ray
from ray.rllib.algorithms.ppo import PPOConfig
from itrader.envs.itrader_env import ITraderEnv

ray.init()

config = (
    PPOConfig()
    .environment(
        ITraderEnv,                         # just pass the class
        env_config={
            "df_path": "data/ohlcv.parquet",
            "fee": 0.001,
        },
    )
    .framework("torch")
    .env_runners(num_env_runners=3)
    .training(train_batch_size=4000)
)

algo = config.build_algo()
