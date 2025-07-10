import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
# --------------------------------------------------------------------------- #
class ITraderEnv(gym.Env):
    """
    Simple long/flat/short environment that returns a *window* of OHLCV rows
    flattened to 1-D, so it lines up with `SolverLSTM` (fc→LSTM) expectations.

    `env_config` keys
    -----------------
    df_path      : str | Path     path to your parquet file
    window_size  : int            how many timesteps per observation  (default 30)
    feature_cols : list[str]      override default OHLCV column set
    fee          : float          commission in pct (default 0.001)
    """
    def __init__(self, env_config: dict | None = None):
        env_config     = env_config or {}
        self.win       = int(env_config.get("window_size", 30))
        fee            = env_config.get("fee", 0.001)
        self.fee       = float(fee)

        # -------------------------------------------------- load dataframe
        df_path = env_config.get("df_path")
        if df_path and not Path(df_path).is_absolute():
            df_path = Path(__file__).resolve().parents[2] / df_path
        if df_path is None:
            df_path = Path(__file__).resolve().parents[1] / "data" / "ohlcv.parquet"

        df_path = Path(df_path).expanduser().resolve(strict=True)
        self.df = pd.read_parquet(df_path).reset_index(drop=True)

        default_feats    = [c for c in self.df.columns
                            if c.upper() in ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME")]
        self.feature_cols = env_config.get("feature_cols", default_feats)

        missing = [c for c in self.feature_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Feature columns not in dataframe: {missing}")

        # -------------------------------------------------- spaces
        n_feat = len(self.feature_cols)
        self.observation_space = gym.spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape = (self.win * n_feat,),
            dtype = np.float32
        )
        self.action_space = gym.spaces.Discrete(3)   # 0 short, 1 flat, 2 long

        # buffer to build rolling windows
        self._buf: deque[np.ndarray] = deque(maxlen=self.win)
        self._reset_state()

    # ====================================================================== #
    # internal helpers
    # ---------------------------------------------------------------------- #
    def _row(self, idx: int) -> np.ndarray:
        return self.df.loc[idx, self.feature_cols].to_numpy(np.float32)

    def _get_obs(self) -> np.ndarray:
        """Return flattened window. If not enough history yet, pad with first row."""
        while len(self._buf) < self.win:
            self._buf.appendleft(self._buf[0])
        return np.concatenate(list(self._buf), axis=0)

    def _reset_state(self):
        self.pos  = 0                   # -1 short | 0 flat | 1 long
        self.idx  = self.win            # start after first full window
        self._buf = deque((self._row(i) for i in range(self.idx - self.win, self.idx)),
                          maxlen=self.win)

    # ====================================================================== #
    # Gymnasium API
    # ---------------------------------------------------------------------- #
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
        self._buf.append(self._row(self.idx))   # slide window
        self.idx += 1

        done  = self.idx >= len(self.df) - 1
        trunc = False
        info  = {"position": self.pos}
        return self._get_obs(), reward, done, trunc, info
