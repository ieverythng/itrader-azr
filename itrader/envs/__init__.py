import gymnasium as gym
from .itrader_env import ITraderEnv

def _itrader_env_creator(**kwargs):
    import pandas as pd
    df = pd.read_parquet(kwargs["df_path"])
    return ITraderEnv(df=df, fee=kwargs.get("fee", 0.001))

gym.register(
    id="iTraderEnv-v0",          # ← string other code will use
    entry_point=_itrader_env_creator,
    max_episode_steps=390,       # one US equity trading day (6.5 h * 60 min)
    )
    
