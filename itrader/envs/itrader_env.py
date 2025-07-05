import gymnasium as gym
import numpy as np
import pandas as pd

class ITraderEnv(gym.Env):
    """
    Minimal single-asset trading env.
    Action: 0 short, 1 flat, 2 long
    Observation: vector of engineered features per bar
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, fee: float = 0.001):
        self.df = df.reset_index(drop=True)

        # -------- feature set --------
        raw_cols = ["open", "high", "low", "close", "volume"]
        tech_cols = ["rsi14", "sma20"]              # produced by etl_job.py
        self.feature_cols = raw_cols + tech_cols
        self.fee = fee

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
