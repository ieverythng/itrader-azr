import gymnasium as gym
from .itrader_env import ITraderEnv
import pandas as pd

def _itrader_env_factory(df_path: str, fee: float = 0.001, **_):
    df = pd.read_parquet(df_path)
    return ITraderEnv(df=df, fee=fee)

# Register one time on import
if "iTraderEnv-v0" not in gym.registry:          # prevent double-register
    gym.register(
        id="iTraderEnv-v0",
        entry_point=_itrader_env_factory,
        max_episode_steps=390,
    )
