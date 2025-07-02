import gymnasium as gym
import numpy as np
import pandas as pd

class ITraderEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, df: pd.DataFrame, fee=0.001):
        self.df = df.reset_index(drop=True)
        self.fee = fee
        self.action_space = gym.spaces.Discrete(3)   # short / hold / long
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_cols),), dtype=np.float32
        )
        self._reset_state()

    def _reset_state(self):
        self.pos = 0  # -1 short, 0 flat, 1 long
        self.cash = 1.0
        self.idx = 0

    def reset(self, seed=None, options=None):
        self._reset_state()
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.idx][self.feature_cols].values.astype(np.float32)
        return row

    def step(self, action):
        price = self.df.loc[self.idx, "close"]
        reward = 0.0

        # simple PnL calc
        if action != self.pos:
            reward -= self.fee   # pay fee when changing position
        reward += self.pos * (self.df.loc[self.idx+1, "close"] - price) / price

        self.pos = {0:-1,1:0,2:1}[action]
        self.idx += 1
        done = self.idx >= len(self.df)-2
        info = {"cash": self.cash, "pos": self.pos}
        return self._get_obs(), reward, done, False, info
