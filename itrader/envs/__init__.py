import gymnasium as gym
from .itrader_env import ITraderEnv
import pandas as pd

def _creator(df_path: str, fee: float = 0.001, **_):
    df = pd.read_parquet(df_path)
    return ITraderEnv(df=df, fee=fee)

gym.register(
    id="iTraderEnv-v0",
    entry_point=_creator,
    max_episode_steps=390,          # 1 NYSE trading day (6.5h * 60)
)
