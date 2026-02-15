import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from typing import Dict, Any, Optional

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
    def __init__(self, env_config: Optional[dict] = None):
        env_config   = env_config or {}
        self.cfg = dict(env_config) if not isinstance(env_config, dict) else env_config
        cfg = self.cfg  # Keep local variable for backward compatibility
        self.T = int(cfg.get("window_size", 30))

        # Worker-aware offsets for task sampler (safe defaults if not present)
        self.worker_index = int(getattr(cfg, "worker_index", cfg.get("worker_index", 0)))
        self.vector_index = int(getattr(cfg, "vector_index", cfg.get("vector_index", 0)))

        # If you want per-task control, you can move alpha into TaskSpec later.
        self.ret_norm_alpha = float(cfg.get("ret_norm_alpha", 0.01))
        self._ret_ema = 1e-3  # typical per-step abs return; reset each episode
        self.start_idx_mode = str(cfg.get("start_idx_mode", "random_episode")).lower()
        if self.start_idx_mode not in ("random_episode", "stream_worker", "task_spec"):
            self.start_idx_mode = "random_episode"
        self.stream_jump_every_episodes = int(cfg.get("stream_jump_every_episodes", 0))
        self.stream_jump_on_wrap = bool(cfg.get("stream_jump_on_wrap", True))
        self.stream_stride_cfg = int(cfg.get("stream_stride", 0))
        self.ret_norm_reset_mode = str(cfg.get("ret_norm_reset", "episode")).lower()
        if self.ret_norm_reset_mode not in ("episode", "stream"):
            self.ret_norm_reset_mode = "episode"
        self._stream_pos: Optional[int] = None
        self._stream_horizon: Optional[int] = None
        self._stream_episode_count = 0

        # ── load dataframe ------------------------------------------------
        df_path = cfg.get("df_path")
        if df_path and not Path(df_path).is_absolute():
            df_path = Path(__file__).resolve().parents[2] / df_path
        if df_path is None:
            df_path = Path(__file__).resolve().parents[1] / "data" / "ohlcv.parquet"

        df_path       = Path(df_path).expanduser().resolve(strict=True)
        self.df: pd.DataFrame = pd.read_parquet(df_path).reset_index(drop=True)

        default_cols  = [c for c in self.df.columns
                         if c.upper() in ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME")]
        self.cols = cfg.get("feature_cols", default_cols)

        missing = [c for c in self.cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Feature columns not in dataframe: {missing}")
        close_cols = [c for c in self.df.columns if c.upper() == "CLOSE"]
        if not close_cols:
            raise ValueError("Dataframe does not contain a CLOSE column")
        self.close_col = close_cols[0]
        self._feature_values = self.df[self.cols].to_numpy(np.float32, copy=True)
        self._close_values = self.df[self.close_col].to_numpy(np.float64, copy=False)
        
        # --- Task sampling / embedding -------------------------------------
        ts_cfg = dict(cfg.get("task_sampler", {}))
        base_seed = int(ts_cfg.get("seed", 1234))
        ts_cfg["seed"] = int(base_seed + 1_000_003 * self.worker_index + 97 * self.vector_index)
        self.task_sampler = TaskSampler00(ts_cfg)
        stream_seed = int(base_seed + 7 * self.worker_index + 13 * self.vector_index + 99)
        self._stream_rng = np.random.default_rng(stream_seed)

        # ── Task sampling / embedding --------------------------------------------------------
        self.t_embed_dim = int(cfg.get("t_embed_dim", DEFAULT_T_EMBED_DIM))
        self.reward_engine = RewardEngine
        self.current_task = None
        self.t_embed_mode = cfg.get("t_embed_mode", "concat")  # off|scale|concat
        self.t_scale = float(cfg.get("t_scale", 1.0))
        self.debug_resets = bool(cfg.get("debug_resets", False))
        self.debug_reset_every = int(cfg.get("debug_reset_every", 100))
        
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
        row = self._feature_values[i]
        return _sanitize(row)

    def _get_obs(self) -> np.ndarray:
        """Return flattened window; pad (repeat first row) until buffer fills."""
        while len(self._buf) < self.T:
            self._buf.appendleft(self._buf[0])
        return np.concatenate(self._buf, axis=0)

    def _task_embed_for_obs(self) -> np.ndarray:
        if self.t_embed_mode == "off":
            return np.zeros_like(self._t_embed)
        if self.t_embed_mode == "scale":
            return self._t_embed * self.t_scale
        return self._t_embed

    def _reset_state(self, start_idx: Optional[int] = None, reset_ret_norm: bool = True):
        """
        Reset episode state. If start_idx is None, uses default (self.T).
        
        Args:
            start_idx: Starting index in dataframe. If None, defaults to self.T (after first window).
            reset_ret_norm: Whether to reset return normalization state.
        """
        self.pos  = 0          # -1 short | 0 flat | 1 long
        
        if start_idx is None:
            # Default: start after first complete window (backward compatibility)
            self.idx = self.T
        else:
            # Use provided start_idx, clamped to valid range
            # Must be >= T (to fill buffer) and < len(df) - 1 (to have next price)
            min_idx = self.T
            max_idx = max(min_idx, len(self.df) - 2)
            self.idx = max(min_idx, min(start_idx, max_idx))
        self._episode_start_idx = int(self.idx)
        
        # Fill buffer with rows [idx - T, ..., idx - 1]
        self._buf = deque(
            (self._row(i) for i in range(self.idx - self.T, self.idx)),
            maxlen=self.T
        )
        self._t = 0
        self._episode_pnl = 0.0
        self._max_equity  = 0.0
        self._max_dd = 0.0     # maximum drawdown tracking

        # episode aggregates (for plotting via callback)

        self._act_counts[:] = 0
        for k in self._rc_sums:
            self._rc_sums[k] = 0.0
        self._last_info = {}

        # reset EMA for this episode
        if reset_ret_norm:
            self._ret_ema = 1e-3

    def _start_idx_bounds(self, horizon: int) -> tuple[int, int]:
        min_idx = self.T
        max_idx = max(min_idx, len(self.df) - int(horizon) - 2)
        return min_idx, max_idx

    def _sample_random_start_idx(self, horizon: int, seed: int) -> int:
        min_idx, max_idx = self._start_idx_bounds(horizon)
        if max_idx > min_idx:
            rng = np.random.default_rng(seed)
            return int(rng.integers(min_idx, max_idx + 1))
        return min_idx

    def _select_stream_start_idx(self, horizon: int) -> tuple[int, bool]:
        stride = int(self.stream_stride_cfg) if self.stream_stride_cfg != 0 else int(horizon)
        stride = max(1, stride)
        min_idx, max_idx = self._start_idx_bounds(horizon)
        stream_changed = False

        if self._stream_pos is None or self._stream_horizon != int(horizon):
            self._stream_pos = (
                int(self._stream_rng.integers(min_idx, max_idx + 1))
                if max_idx > min_idx else min_idx
            )
            self._stream_episode_count = 0
            stream_changed = True

        if (
            self.stream_jump_every_episodes > 0
            and self._stream_episode_count >= self.stream_jump_every_episodes
        ):
            self._stream_pos = (
                int(self._stream_rng.integers(min_idx, max_idx + 1))
                if max_idx > min_idx else min_idx
            )
            self._stream_episode_count = 0
            stream_changed = True

        if self._stream_pos > max_idx:
            if self.stream_jump_on_wrap and max_idx > min_idx:
                self._stream_pos = int(self._stream_rng.integers(min_idx, max_idx + 1))
            else:
                self._stream_pos = min_idx
            self._stream_episode_count = 0
            stream_changed = True

        start_idx = int(self._stream_pos)
        self._stream_pos = start_idx + stride
        self._stream_horizon = int(horizon)
        self._stream_episode_count += 1
        return start_idx, stream_changed

    def _choose_start_idx(self, horizon: int) -> tuple[int, bool]:
        if self.start_idx_mode == "stream_worker":
            return self._select_stream_start_idx(horizon)

        task_start = self.current_task.start_idx if self.start_idx_mode == "task_spec" else None
        if task_start is None:
            start_idx = self._sample_random_start_idx(horizon, self.current_task.seed)
        else:
            start_idx = int(task_start)
        min_idx, max_idx = self._start_idx_bounds(horizon)
        return max(min_idx, min(start_idx, max_idx)), True
    # ===================================================================== #
    # gymnasium.Env interface
    # --------------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # sample task for the episode first
        self.current_task = self.task_sampler.sample()
        horizon = int(self.current_task.horizon)
        start_idx, stream_changed = self._choose_start_idx(horizon)

        # Reset state with determined start_idx
        reset_ret_norm = True
        if self.ret_norm_reset_mode == "stream" and self.start_idx_mode == "stream_worker":
            reset_ret_norm = stream_changed
        self._reset_state(start_idx=start_idx, reset_ret_norm=reset_ret_norm)
        
        # Create deterministic embedding
        self._t_embed = TaskEncoder.embed(self.current_task, dim=self.t_embed_dim)
        t = self._task_embed_for_obs()
        # build first obs
        price_obs = _sanitize(self._get_obs())
        obs = _sanitize(np.concatenate([price_obs, t]).astype(np.float32))
        task_info = self.current_task.to_dict()
        task_info["start_idx"] = start_idx
        info = {"task": task_info}
        self._last_info = info
        over = self.cfg.get("task_sampler_overrides", {})
        if over:
            self.current_task.reward_params.update(over)
        
        # Debug: log start_idx occasionally (every ~100 episodes)
        if hasattr(self, "_reset_count"):
            self._reset_count += 1
        else:
            self._reset_count = 0
        if self.debug_resets and self._reset_count % max(1, self.debug_reset_every) == 0:
            print(f"[Worker {self.worker_index}] Reset #{self._reset_count}: start_idx={start_idx}, horizon={self.current_task.horizon}, df_len={len(self.df)}")
        
        return obs, info

    def step(self, action: int):
        # map discrete action to position target {-1,0,+1}
        action = int(action)
        target_pos = {0: -1, 1: 0, 2: 1}[action]

        # previous price & position
        price_t = float(self._close_values[self.idx - 1])
        prev_pos = float(self.pos)

        # apply action -> new position 
        self.pos = target_pos

        # advance one step
        price_next = float(self._close_values[self.idx])
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
        "ret_unit": ret_unit,      # multiply terms by this if normalize=True
        "ret_ema": self._ret_ema,  # realized_vol for inactivity gating
        "terminal": False,         # updated below if done
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

        # baseline success flag for TRR++/curriculum stats (Phase-B)
        success = bool(pnl_t1 > 0.0) if done else False
        comps["success"] = float(success)  # for TRR++/curriculum stats

        price_obs_next = _sanitize(self._get_obs())
        t = self._task_embed_for_obs()
        obs_next = _sanitize(np.concatenate([price_obs_next, t]).astype(np.float32))

        if done:
            ctx["terminal"] = True
            # Recompute reward with terminal flag? Not needed—our reward_fn handles terminal within same call.

        info = {
            "position": self.pos,
            "reward_comps": comps,
            "success": success,
            "task": {**self.current_task.to_dict(), "start_idx": self._episode_start_idx},
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
    
