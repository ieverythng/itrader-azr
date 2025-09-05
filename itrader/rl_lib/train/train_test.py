# itrader/rl_lib/train.py
from pathlib import Path
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import (
    JsonLoggerCallback, CSVLoggerCallback, TBXLoggerCallback)

from itrader.envs.itrader_env import ITraderEnv

ROOT = Path(__file__).resolve().parents[1]          # …/itrader-azr
DF   = ROOT / "data" / "ohlcv.parquet"              # absolute path

# This is all the workers need – no Ray calls here!
def make_env(cfg=None):
    cfg = cfg or {}
    cfg.setdefault("df_path", str(DF))
    return ITraderEnv(cfg)

tune.register_env("iTraderEnv", make_env)


def main()-> None:
    # Start Ray **once**, on the driver only
    if not ray.is_initialized():          # safe-guard
        ray.init(                          # or `ray.init(address="auto")` for a cluster
            log_to_driver=False,
            ignore_reinit_error=True       # shuts the duplicate-init warning up
    )
               #  add ignore_reinit_error=True if you often re-run in the same process

    config = (
        PPOConfig()
        .environment("iTraderEnv", env_config={"df_path": str(DF)})
        .framework("torch")  # or "tf" – whatever you use
        .env_runners(num_env_runners=3)
        .resources(num_gpus=1)
    )

    config.logger_config = {
        "callbacks": [
            JsonLoggerCallback(),
            CSVLoggerCallback(),
            TBXLoggerCallback(),        # tensorboard-x
        ]
    }

    algo = config.build()          # new API – avoids the deprecation warning
    
    for _ in range(100):
        result = algo.train()
        print(result)               # first training iteration


#  >>> workers import this file, so we *must* protect the entry point
if __name__ == "__main__":
    main()
