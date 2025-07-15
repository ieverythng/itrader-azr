import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque


# ────────────────────────────────────────────────────────────────────────────
class ITraderEnv(gym.Env):
    """
    Long / flat / short trading toy-env that returns a *flattened* rolling
    OHLCV window (length = `window_size × n_features`).

    ── env_config keys ───────────────────────────────────────────────────────
    df_path      : str | Path   – parquet file to load (default: repo/data/…)
    window_size  : int          – timesteps per observation      (default 30)
    feature_cols : list[str]    – override default OHLCV columns
    fee          : float        – commission (%)                (default 0.001)
    """

    # --------------------------------------------------------------------- #
    def __init__(self, env_config: dict | None = None):
        env_config   = env_config or {}
        self.T       = int(env_config.get("window_size", 30))
        self.fee     = float(env_config.get("fee", 0.001))

        # ── load dataframe ------------------------------------------------
        df_path = env_config.get("df_path")
        if df_path and not Path(df_path).is_absolute():
            df_path = Path(__file__).resolve().parents[2] / df_path
        if df_path is None:
            df_path = Path(__file__).resolve().parents[1] / "data" / "ohlcv.parquet"

        df_path       = Path(df_path).expanduser().resolve(strict=True)
        self.df: pd.DataFrame = pd.read_parquet(df_path).reset_index(drop=True)

        default_cols  = [c for c in self.df.columns
                         if c.upper() in ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME")]
        self.cols     = env_config.get("feature_cols", default_cols)

        missing = [c for c in self.cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Feature columns not in dataframe: {missing}")

        # ── spaces --------------------------------------------------------
        feat_dim = len(self.cols)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.T * feat_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)  # 0=short, 1=flat, 2=long

        # rolling buffer holding last T rows
        self._buf: deque[np.ndarray] = deque(maxlen=self.T)
        self._reset_state()

    # ===================================================================== #
    # internal helpers
    # --------------------------------------------------------------------- #
    def _row(self, i: int) -> np.ndarray:
        return self.df.loc[i, self.cols].to_numpy(np.float32)

    def _get_obs(self) -> np.ndarray:
        """Return flattened window; pad (repeat first row) until buffer fills."""
        while len(self._buf) < self.T:
            self._buf.appendleft(self._buf[0])
        return np.concatenate(list(self._buf), axis=0)

    def _reset_state(self):
        self.pos  = 0          # -1 short | 0 flat | 1 long
        self.idx  = self.T     # start after first complete window
        self._buf = deque(
            (self._row(i) for i in range(self.idx - self.T, self.idx)),
            maxlen=self.T
        )

    # ===================================================================== #
    # gymnasium.Env interface
    # --------------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        price_now  = self.df.loc[self.idx - 1, "close"]
        price_next = self.df.loc[self.idx,     "close"]

        reward  = -self.fee if action != self.pos else 0.0
        reward += self.pos * (price_next - price_now) / price_now

        self.pos = {0: -1, 1: 0, 2: 1}[action]
        self._buf.append(self._row(self.idx))
        self.idx += 1

        done   = self.idx >= len(self.df) - 1
        trunc  = False
        info   = {"position": self.pos}
        return self._get_obs(), reward, done, trunc, info
