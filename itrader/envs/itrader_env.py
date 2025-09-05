import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque

from rl_lib.tasks.samplers import TaskSampler00
from rl_lib.tasks.encoder import TaskEncoder, DEFAULT_T_EMBED_DIM
from rl_lib.tasks.rewards import RewardEngine

def _sanitize(x: np.ndarray) -> np.ndarray:
    """Hard guard against NaN/Inf to keep PPO stable."""
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

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

        # ── Task sampling / embedding --------------------------------------------------------
        self.task_sampler = TaskSampler00(env_config.get("task_sampler", {}))
        self.t_embed_dim  = int(env_config.get("t_embed_dim", DEFAULT_T_EMBED_DIM))
        self.reward_engine = RewardEngine
        self.current_task = None
        self.t_embed_mode = env_config.get("t_embed_mode", "concat")  # off|scale|concat
        self.t_scale = float(env_config.get("t_scale", 1.0))
        
        feat_dim = len(self.cols)
        self.price_obs_dim = self.T * feat_dim
        obs_dim = self.price_obs_dim + self.t_embed_dim
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)  # 0=short, 1=flat, 2=long

        # rolling buffer holding last T rows
        self._buf: deque[np.ndarray] = deque(maxlen=self.T)
        self._reset_state()

    # ===================================================================== #
    # internal helpers
    # --------------------------------------------------------------------- #
    def _row(self, i: int) -> np.ndarray:
        row = self.df.loc[i, self.cols].to_numpy(np.float32)
        return _sanitize(row)

    def _get_obs(self) -> np.ndarray:
        """Return flattened window; pad (repeat first row) until buffer fills."""
        while len(self._buf) < self.T:
            self._buf.appendleft(self._buf[0])
        return np.concatenate(list(self._buf), axis=0)
    
    # def _sanitize(x):
    #     return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    def _reset_state(self):
        self.pos  = 0          # -1 short | 0 flat | 1 long
        self.idx  = self.T     # start after first complete window
        self._buf = deque(
            (self._row(i) for i in range(self.idx - self.T, self.idx)),
            maxlen=self.T
        )
        self._t = 0
        self._episode_pnl = 0.0
        self._max_equity  = 0.0
    # ===================================================================== #
    # gymnasium.Env interface
    # --------------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        # sample task for the episode, create deterministic embedding
        self.current_task = self.task_sampler.sample()
        self._t_embed = TaskEncoder.embed(self.current_task, dim=self.t_embed_dim)
        if self.t_embed_mode == "off":
            t = np.zeros_like(self._t_embed)
        elif self.t_embed_mode == "scale":
            t = self._t_embed * self.t_scale
        else:
            t = self._t_embed
        # build first obs
        price_obs = _sanitize(self._get_obs())
        obs = _sanitize(np.concatenate([price_obs, t]).astype(np.float32))
        info = {"task": self.current_task.to_dict()}
        return obs, info

    def step(self, action: int):
        # map discrete action to position target {-1,0,+1}
        action = int(action)
        target_pos = {0: -1, 1: 0, 2: 1}[action]

        # previous price & position
        price_t  = float(self.df.loc[self.idx - 1, "close"])
        prev_pos = float(self.pos)

        # apply action -> new position (instant fill in this toy env)
        self.pos = target_pos

        # advance one step
        price_next = float(self.df.loc[self.idx, "close"])
        self._buf.append(self._row(self.idx))
        self.idx += 1

        # scale-invariant return & turnover
        ret = (price_next - price_t) / max(price_t, 1e-8)
        turnover_t = abs(float(self.pos) - prev_pos)  # 0, 1 or 2

        # mark-to-market PnL (in return space)
        pnl_t  = self._episode_pnl
        mtm    = float(self.pos) * ret
        pnl_t1 = pnl_t + mtm
        self._episode_pnl = pnl_t1
        self._max_equity  = max(self._max_equity, pnl_t1)
        dd_t1  = max(0.0, self._max_equity - pnl_t1)

        # compute reward via registry
        r, comps = self.reward_engine.compute(
            self.current_task.reward_id,
            self.current_task.reward_params,
            {
                "pnl_t": pnl_t, "pnl_t1": pnl_t1,
                "pnl_delta": pnl_t1 - pnl_t,
                "dd_t1": dd_t1, "turnover_t": turnover_t,
                "price_t": price_t, "price_t1": price_next,
                "position_t": float(self.pos), "action_t": action,
                "ret": ret,  # optional if you compose on raw returns
            }
        )

        self._t += 1
        ep_horizon = int(self.current_task.horizon)
        done_by_horizon = (self._t >= ep_horizon)
        done_by_data    = (self.idx >= len(self.df) - 1)
        done = bool(done_by_horizon or done_by_data)
        trunc = False

        # baseline success flag for TRR++/curriculum stats (Phase-A)
        success = bool(pnl_t1 > 0.0) if done else False

        price_obs_next = _sanitize(self._get_obs())
        if self.t_embed_mode == "off":
            t = np.zeros_like(self._t_embed)
        elif self.t_embed_mode == "scale":
            t = self._t_embed * self.t_scale
        else:
            t = self._t_embed
        obs_next = _sanitize(np.concatenate([price_obs_next, t]).astype(np.float32))
        info = {"position": self.pos, "reward_comps": comps, "success": success,
                "task": self.current_task.to_dict()}
        return obs_next, float(r), done, trunc, info
    
    
