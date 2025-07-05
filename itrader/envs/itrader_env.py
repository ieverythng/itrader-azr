import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import Env
from pathlib import Path

class ITraderEnv(gym.Env):
    """
    RL environment for trading.
    Expects `env_config` keys:
        df_path : path to parquet/csv   OR
        df      : already-loaded DataFrame
        fee     : commission in %
    """
    def __init__(self, env_config: dict | None = None):
        env_config = env_config or {}

        # 1. explicit absolute path wins
        df_path = env_config.get("df_path")

        # 2. if caller gave a relative path, resolve it against *this* file
        if df_path and not Path(df_path).is_absolute():
            df_path = Path(__file__).resolve().parents[2] / df_path   # repo/ + data/...

        # 3. fallback to package-bundled file
        if df_path is None:
            df_path = Path(__file__).resolve().parents[1] / "data" / "ohlcv.parquet"

        df_path = Path(df_path).expanduser().resolve(strict=True)     # ← raises if missing
        self.df = pd.read_parquet(df_path).reset_index(drop=True)

        default_feats = [c for c in self.df.columns
                         if c.upper() in ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME")]
        self.feature_cols = env_config.get("feature_cols", default_feats)

        missing = [c for c in self.feature_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Feature columns not in dataframe: {missing}")

        self.fee = env_config.get("fee", 0.001)

        # -------- spaces -------------
        self.action_space = gym.spaces.Discrete(3)          # short / flat / long
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_cols),),
            dtype=np.float32,
        )

        self._reset_state()

    # --------------------------------
    def _reset_state(self):
        self.pos = 0            # -1 short, 0 flat, 1 long
        self.idx = 0

    # --------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.idx][self.feature_cols].values.astype(np.float32)
        return row

    def step(self, action: int):
        price_now  = self.df.loc[self.idx, "close"]
        price_next = self.df.loc[self.idx + 1, "close"]

        reward = 0.0
        if action != self.pos:
            reward -= self.fee                       # commission
        reward += self.pos * (price_next - price_now) / price_now

        self.pos = {0: -1, 1: 0, 2: 1}[action]
        self.idx += 1
        done = self.idx >= len(self.df) - 2
        info = {"pos": self.pos}

        return self._get_obs(), reward, done, False, info
