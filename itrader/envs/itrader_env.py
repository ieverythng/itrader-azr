import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from typing import Dict, Any

from itrader.rl_lib.tasks.samplers import TaskSampler00
from itrader.rl_lib.tasks.encoder import TaskEncoder, DEFAULT_T_EMBED_DIM
from itrader.rl_lib.tasks.rewards import RewardEngine

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
    """

    # --------------------------------------------------------------------- #
    def __init__(self, env_config: dict | None = None):
        env_config   = env_config or {}
        cfg = dict(env_config) if not isinstance(env_config, dict) else env_config
        self.T       = int(env_config.get("window_size", 30))

        # Worker-aware offsets for task sampler (safe defaults if not present)
        self.worker_index = int(getattr(env_config, "worker_index", env_config.get("worker_index", 0)))
        self.vector_index = int(getattr(env_config, "vector_index", env_config.get("vector_index", 0)))

        # If you want per-task control, you can move alpha into TaskSpec later.
        self.ret_norm_alpha = float(env_config.get("ret_norm_alpha", 0.01))
        self._ret_ema = 1e-3  # typical per-step abs return; reset each episode

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
        
        # --- Task sampling / embedding -------------------------------------
        ts_cfg = dict(cfg.get("task_sampler", {}))
        base_seed = int(ts_cfg.get("seed", 1234))
        ts_cfg["seed"] = int(base_seed + 1_000_003 * self.worker_index + 97 * self.vector_index)
        self.task_sampler = TaskSampler00(ts_cfg)

        # ── Task sampling / embedding --------------------------------------------------------
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

        # --- episode-level accumulators (used by callbacks) ----------------
        # MUST exist before the first _reset_state() call.
        self._act_counts = np.zeros(3, dtype=np.int64)
        self._rc_sums: Dict[str, float] = {"pnl_delta": 0.0, "dd": 0.0, "turnover": 0.0}
        self._last_info: Dict[str, Any] = {}

        # initialize state
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
        self._max_dd = 0.0     # maximum drawdown tracking

        # episode aggregates (for plotting via callback)

        # Defensive: ensure fields exist even if __init__ order changes.
        if not hasattr(self, "_act_counts"):
            self._act_counts = np.zeros(3, dtype=np.int64)
        else:
            self._act_counts[:] = 0
        if not hasattr(self, "_rc_sums"):
            self._rc_sums = {"pnl_delta": 0.0, "dd": 0.0, "turnover": 0.0}
        else:
            for k in self._rc_sums:
                self._rc_sums[k] = 0.0
        if not hasattr(self, "_last_info"):
            self._last_info = {}

        # reset EMA for this episode
        self._ret_ema = 1e-3
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
        self._last_info = info
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

        # update EMA(|ret|) and derive a unit scale so "typical" return ≈ 1.0
        a = self.ret_norm_alpha
        self._ret_ema = (1.0 - a) * self._ret_ema + a * abs(ret)
        ret_unit = 1.0 / max(self._ret_ema, 1e-8)  # multiply reward terms by this

        # mark-to-market PnL (in return space)
        pnl_t  = self._episode_pnl
        mtm    = float(self.pos) * ret
        pnl_t1 = pnl_t + mtm

        # drawdown (level and delta) BEFORE updating max_equity
        dd_t = max(0.0, self._max_equity - pnl_t)

        self._episode_pnl = pnl_t1
        self._max_equity  = max(self._max_equity, pnl_t1)

        dd_t1  = max(0.0, self._max_equity - pnl_t1)
        self._max_dd = max(self._max_dd, dd_t1)

        dd_inc = max(0.0, dd_t1 - dd_t)

        ctx = {
        "pnl_t": pnl_t, "pnl_t1": pnl_t1,
        "pnl_delta": pnl_t1 - pnl_t,
        "dd_t": dd_t, "dd_t1": dd_t1, "dd_inc": dd_inc,
        "max_dd": self._max_dd,
        "turnover_t": turnover_t,
        "price_t": price_t, "price_t1": price_next,
        "position_t": float(self.pos), "action_t": action,
        "ret": ret,
        "ret_unit": ret_unit,   # <- dynamic normalization factor for rewards
        }

        # compute reward via registry
        r, comps = self.reward_engine.compute(
            self.current_task.reward_id,
            self.current_task.reward_params,
            ctx,
        )

        # accumulate episode aggregates
        self._act_counts[action] += 1
        self._rc_sums["pnl_delta"] += float(comps.get("pnl_delta", 0.0))
        self._rc_sums["dd"] += float(comps.get("dd", 0.0))
        self._rc_sums["turnover"] += float(comps.get("turnover", 0.0))

        self._t += 1
        ep_horizon = int(self.current_task.horizon)
        done_by_horizon = (self._t >= ep_horizon)
        done_by_data    = (self.idx >= len(self.df) - 1)
        done = bool(done_by_horizon or done_by_data)
        trunc = False

        # baseline success flag for TRR++/curriculum stats (Phase-A)
        success = bool(pnl_t1 > 0.0) if done else False
        comps["success"] = float(success)  # for TRR++/curriculum stats

        price_obs_next = _sanitize(self._get_obs())
        if self.t_embed_mode == "off":
            t = np.zeros_like(self._t_embed)
        elif self.t_embed_mode == "scale":
            t = self._t_embed * self.t_scale
        else:
            t = self._t_embed
        obs_next = _sanitize(np.concatenate([price_obs_next, t]).astype(np.float32))

        info = {
            "position": self.pos,
            "reward_comps": comps,
            "success": success,
            "task": self.current_task.to_dict(),
        }

        if done:
            steps = max(1, int(self._t))
            info["ep_sums"] = {
                "steps": steps,
                "horizon": ep_horizon,
                "act_counts": self._act_counts.copy(),
                "rc_sums": dict(self._rc_sums),
            }

        self._last_info = info
        return obs_next, float(r), done, trunc, info
    
