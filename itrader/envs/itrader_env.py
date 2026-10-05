from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import pandas as pd

from itrader.rl_lib.tasks.samplers import TaskSampler00
from itrader.rl_lib.tasks.encoder import TaskEncoder, DEFAULT_T_EMBED_DIM
from itrader.rl_lib.tasks.rewards import RewardEngine
from itrader.rl_lib.evaluation import DataPartition, EpisodeEvidence
from itrader.rl_lib.evaluation.provenance import canonical_json_sha256
from itrader.models.observation_contract import validate_semantic_config


_ACTION_TO_POSITION = {0: -1, 1: 0, 2: 1}
_FEATURE_COLUMN_NAMES = {"OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"}
_START_IDX_MODES = {"random_episode", "stream_worker", "task_spec"}
_RET_NORM_RESET_MODES = {"episode", "stream"}
_OBSERVATION_TRANSFORMS = {"raw", "market_relative_v1"}
SOLVER_STATE_MODES = {"off", "masked", "position", "time", "both"}
_OBSERVATION_SCHEMA_KEYS = (
    "observation_schema_id",
    "base_obs_dim",
    "solver_state_dim",
    "agent_obs_dim",
)


def _sanitize(x: np.ndarray) -> np.ndarray:
    """Hard guard against NaN/Inf to keep PPO stable."""
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)


def _resolve_choice(value: Any, allowed: set[str], default: str) -> str:
    value = str(value).lower()
    return value if value in allowed else default


def resolve_solver_state_mode(value: Any) -> str:
    mode = str(value or "off").strip().lower()
    if mode not in SOLVER_STATE_MODES:
        choices = ", ".join(sorted(SOLVER_STATE_MODES))
        raise ValueError(f"solver_state_mode must be one of: {choices}")
    return mode


def solver_state_observation_contract(
    *,
    window_size: int,
    n_features: int,
    task_embed_dim: int,
    solver_state_mode: Any,
) -> dict[str, int | str]:
    mode = resolve_solver_state_mode(solver_state_mode)
    base_obs_dim = int(window_size) * int(n_features) + int(task_embed_dim)
    solver_state_dim = 0 if mode == "off" else 2
    return {
        "observation_schema_id": "r0_base" if mode == "off" else "r4_v1_min",
        "base_obs_dim": base_obs_dim,
        "solver_state_dim": solver_state_dim,
        "agent_obs_dim": base_obs_dim + solver_state_dim,
    }


def _validate_declared_observation_contract(
    cfg: dict[str, Any],
    contract: dict[str, int | str],
) -> None:
    for key in _OBSERVATION_SCHEMA_KEYS:
        if key in cfg and cfg[key] != contract[key]:
            raise ValueError(
                f"{key}={cfg[key]!r} does not match derived value "
                f"{contract[key]!r}"
            )


def _resolve_df_path(df_path: Any) -> Path:
    if df_path is None:
        candidate = Path(__file__).resolve().parents[1] / "data" / "ohlcv.parquet"
    else:
        candidate = Path(df_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parents[2] / candidate
    return candidate.resolve(strict=True)


def _resolve_feature_columns(df: pd.DataFrame, feature_cols: Any) -> list[str]:
    default_cols = [col for col in df.columns if col.upper() in _FEATURE_COLUMN_NAMES]
    columns = list(feature_cols or default_cols)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not in dataframe: {missing}")
    return columns


def _load_market_frame(
    path: Path,
    sessions: Any,
    symbols: Any,
    columns: Any,
) -> pd.DataFrame:
    filters = []
    normalized_sessions = [str(value) for value in (sessions or []) if str(value)]
    normalized_symbols = [str(value).upper() for value in (symbols or []) if str(value)]
    if normalized_sessions:
        filters.append(("session", "in", normalized_sessions))
    if normalized_symbols:
        filters.append(("symbol", "in", normalized_symbols))
    projected_columns = list(dict.fromkeys(columns or [])) or None
    frame = pd.read_parquet(path, filters=filters or None, columns=projected_columns)
    symbol_cols = [column for column in frame.columns if column.upper() == "SYMBOL"]
    timestamp_cols = [
        column for column in frame.columns if column.upper() in {"TS", "TIMESTAMP", "DATETIME"}
    ]
    sort_columns = symbol_cols[:1] + timestamp_cols[:1]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="stable")
    return frame.reset_index(drop=True)


def _projected_market_columns(cfg: dict[str, Any]) -> list[str] | None:
    boundary_columns = list(cfg.get("episode_boundary_cols", ()))
    if not (cfg.get("data_sessions") or cfg.get("data_symbols") or boundary_columns):
        return None
    columns = list(cfg.get("feature_cols") or ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])
    columns.extend(["symbol", "ts"])
    columns.extend(boundary_columns)
    if cfg.get("data_sessions"):
        columns.append("session")
    return list(dict.fromkeys(columns))


class ITraderEnv(gym.Env):
    """
    Long / flat / short trading environment that returns a *flattened* rolling
    OHLCV window (length = `window_size × n_features`).

    ── env_config keys ───────────────────────────────────────────────────────
    df_path      : str | Path   – parquet file to load (default: repo/data/…)
    window_size  : int          – timesteps per observation      (default 30)
    feature_cols : list[str]    – override default OHLCV columns
    """

    # --------------------------------------------------------------------- #
    def __init__(self, env_config: Optional[dict] = None):
        raw_cfg = env_config or {}
        self.cfg = dict(raw_cfg)
        cfg = self.cfg
        self.T = int(cfg.get("window_size", 30))

        # Worker-aware offsets for task sampler (safe defaults if not present)
        self.worker_index = int(getattr(raw_cfg, "worker_index", cfg.get("worker_index", 0)))
        self.vector_index = int(getattr(raw_cfg, "vector_index", cfg.get("vector_index", 0)))

        # If you want per-task control, you can move alpha into TaskSpec later.
        self.ret_norm_alpha = float(cfg.get("ret_norm_alpha", 0.01))
        self.evidence_cost_bps = float(cfg.get("evidence_cost_bps", 1.0))
        self.evidence_periods_per_year = float(cfg.get("evidence_periods_per_year", 0.0))
        self.data_split = str(cfg.get("data_split", "all"))
        self.capture_replay_trace = bool(cfg.get("capture_replay_trace", False))
        self._ret_ema = 1e-3  # typical per-step abs return; reset each episode
        self.start_idx_mode = _resolve_choice(
            cfg.get("start_idx_mode", "random_episode"),
            _START_IDX_MODES,
            "random_episode",
        )
        self.stream_jump_every_episodes = int(cfg.get("stream_jump_every_episodes", 0))
        self.stream_jump_on_wrap = bool(cfg.get("stream_jump_on_wrap", True))
        self.stream_stride_cfg = int(cfg.get("stream_stride", 0))
        self.ret_norm_reset_mode = _resolve_choice(
            cfg.get("ret_norm_reset", "episode"),
            _RET_NORM_RESET_MODES,
            "episode",
        )
        self.observation_transform = _resolve_choice(
            cfg.get("observation_transform", "raw"),
            _OBSERVATION_TRANSFORMS,
            "raw",
        )
        self.solver_state_mode = resolve_solver_state_mode(
            cfg.get("solver_state_mode", "off")
        )
        self._stream_pos: Optional[int] = None
        self._stream_horizon: Optional[int] = None
        self._stream_episode_count = 0
        self._reset_count = -1

        # ── load dataframe ------------------------------------------------
        self.df = _load_market_frame(
            _resolve_df_path(cfg.get("df_path")),
            cfg.get("data_sessions"),
            cfg.get("data_symbols"),
            _projected_market_columns(cfg),
        )
        for categorical_column in ("symbol", "session", *cfg.get("episode_boundary_cols", ())):
            if categorical_column in self.df.columns:
                self.df[categorical_column] = self.df[categorical_column].astype("category")
        self.cols = _resolve_feature_columns(self.df, cfg.get("feature_cols"))
        self.semantic_observation_contract = None
        if "semantic_observation_contract" in cfg:
            self.semantic_observation_contract = validate_semantic_config(
                cfg, cfg["semantic_observation_contract"], boundary="env",
                actual_feature_columns=self.cols,
            )
        close_cols = [c for c in self.df.columns if c.upper() == "CLOSE"]
        if not close_cols:
            raise ValueError("Dataframe does not contain a CLOSE column")
        self.close_col = close_cols[0]
        self._feature_values = self.df[self.cols].to_numpy(np.float32, copy=True)
        self._close_values = self.df[self.close_col].to_numpy(np.float64, copy=False)
        symbol_cols = [c for c in self.df.columns if c.upper() == "SYMBOL"]
        self.symbol_col = symbol_cols[0] if symbol_cols else None
        timestamp_cols = [c for c in self.df.columns if c.upper() in {"TS", "TIMESTAMP", "DATETIME"}]
        self.timestamp_col = timestamp_cols[0] if timestamp_cols else None
        self.data_partition = DataPartition(
            self.df,
            window_size=self.T,
            split=self.data_split,
            split_fractions=cfg.get("split_fractions", (0.7, 0.15, 0.15)),
            purge_rows=int(cfg.get("split_purge_rows", 0)),
            symbol_col=self.symbol_col,
            timestamp_col=self.timestamp_col,
            boundary_cols=cfg.get("episode_boundary_cols", ()),
            expected_bar_interval=cfg.get("expected_bar_interval"),
        )

        # --- Task sampling / embedding -------------------------------------
        ts_cfg = dict(cfg.get("task_sampler", {}))
        base_seed = int(ts_cfg.get("seed", 1234))
        ts_cfg["seed"] = int(base_seed + 1_000_003 * self.worker_index + 97 * self.vector_index)
        self._sampler_seed_offset = 1_000_003 * self.worker_index + 97 * self.vector_index
        self.task_sampler = TaskSampler00(ts_cfg)
        stream_seed = int(base_seed + 7 * self.worker_index + 13 * self.vector_index + 99)
        self._stream_seed_offset = 7 * self.worker_index + 13 * self.vector_index + 99
        self._stream_rng = np.random.default_rng(stream_seed)

        # ── Task sampling / embedding --------------------------------------------------------
        self.t_embed_dim = int(cfg.get("t_embed_dim", DEFAULT_T_EMBED_DIM))
        self.task_encoder_version = str(cfg.get("task_encoder_version", "legacy_v1"))
        self.reward_engine = RewardEngine
        self.current_task = None
        self.t_embed_mode = cfg.get("t_embed_mode", "concat")  # off|scale|concat
        self.t_scale = float(cfg.get("t_scale", 1.0))
        self.debug_resets = bool(cfg.get("debug_resets", False))
        self.debug_reset_every = int(cfg.get("debug_reset_every", 100))

        feat_dim = len(self.cols)
        self.price_obs_dim = self.T * feat_dim
        observation_contract = solver_state_observation_contract(
            window_size=self.T,
            n_features=feat_dim,
            task_embed_dim=self.t_embed_dim,
            solver_state_mode=self.solver_state_mode,
        )
        _validate_declared_observation_contract(cfg, observation_contract)
        self.observation_schema_id = str(
            observation_contract["observation_schema_id"]
        )
        self.base_obs_dim = int(observation_contract["base_obs_dim"])
        self.solver_state_dim = int(observation_contract["solver_state_dim"])
        self.agent_obs_dim = int(observation_contract["agent_obs_dim"])
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.agent_obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)  # 0=short, 1=flat, 2=long

        # rolling buffer holding last T rows
        self._buf: deque[np.ndarray] = deque(maxlen=self.T)

        # --- episode-level accumulators (used by callbacks) ----------------
        # MUST exist before the first _reset_state() call.
        self._act_counts = np.zeros(int(self.action_space.n), dtype=np.int64)
        self._rc_sums: Dict[str, float] = {}
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
        window = np.stack(self._buf, axis=0).astype(np.float64, copy=True)
        if self.observation_transform == "market_relative_v1":
            window = self._market_relative_window(window)
        return window.astype(np.float32, copy=False).reshape(-1)

    def _market_relative_window(self, window: np.ndarray) -> np.ndarray:
        """Create bounded, fitted-state-free OHLCV features for portable checkpoints."""
        if self.close_col not in self.cols:
            raise ValueError("market_relative_v1 requires CLOSE in feature_cols")

        close_ref = float(window[-1, self.cols.index(self.close_col)])
        price_scale = max(abs(close_ref), 1e-12)
        for index, column in enumerate(self.cols):
            if column.upper() in {"OPEN", "HIGH", "LOW", "CLOSE"}:
                window[:, index] = window[:, index] / price_scale - 1.0
                continue

            values = window[:, index]
            if column.upper() == "VOLUME":
                values = np.log1p(np.maximum(values, 0.0))
            mean = float(np.mean(values))
            std = float(np.std(values))
            window[:, index] = (values - mean) / max(std, 1e-8)

        return np.clip(_sanitize(window), -10.0, 10.0)

    def _build_obs(self) -> np.ndarray:
        price_obs = _sanitize(self._get_obs())
        task_obs = self._task_embed_for_obs()
        solver_state_obs = self._solver_state_for_obs()
        return _sanitize(
            np.concatenate([price_obs, task_obs, solver_state_obs]).astype(np.float32)
        )

    def _solver_state_for_obs(self) -> np.ndarray:
        if self.solver_state_mode == "off":
            return np.empty(0, dtype=np.float32)

        position = (
            float(self.pos)
            if self.solver_state_mode in {"position", "both"}
            else 0.0
        )
        remaining = 0.0
        if self.solver_state_mode in {"time", "both"}:
            remaining = max(
                0.0,
                (float(self.current_task.horizon) - float(self._t))
                / float(self.current_task.horizon),
            )
        return np.asarray([position, remaining], dtype=np.float32)

    def _task_embed_for_obs(self) -> np.ndarray:
        if self.t_embed_mode == "off":
            return np.zeros_like(self._t_embed)
        if self.t_embed_mode == "scale":
            return self._t_embed * self.t_scale
        return self._t_embed

    def _clamp_start_idx(self, start_idx: Optional[int]) -> int:
        if start_idx is None:
            return self.T
        min_idx = self.T
        max_idx = max(min_idx, len(self.df) - 2)
        return max(min_idx, min(int(start_idx), max_idx))

    def _reset_episode_stats(self) -> None:
        self._t = 0
        self._action_reward_trace: list[list[int | float]] = []
        self._episode_pnl = 0.0
        self._optimization_reward_sum = 0.0
        self._turnover_sum = 0.0
        self._turnover_steps: list[float] = []
        self._gross_exposure_sum = 0.0
        self._position_changes = 0
        self._episode_strategy_returns: list[float] = []
        self._max_equity = 0.0
        self._max_dd = 0.0
        self._act_counts[:] = 0
        self._rc_sums.clear()
        self._last_info = {}

    def _accumulate_reward_components(self, comps: Dict[str, float]) -> None:
        for key, value in comps.items():
            self._rc_sums[key] = self._rc_sums.get(key, 0.0) + float(value)

    def _reset_state(self, start_idx: Optional[int] = None, reset_ret_norm: bool = True):
        """
        Reset episode state. If start_idx is None, uses default (self.T).

        Args:
            start_idx: Starting index in dataframe. If None, defaults to self.T (after first window).
            reset_ret_norm: Whether to reset return normalization state.
        """
        self.pos  = 0          # -1 short | 0 flat | 1 long

        self.idx = self._clamp_start_idx(start_idx)
        self._episode_start_idx = int(self.idx)

        # Fill buffer with rows [idx - T, ..., idx - 1]
        self._buf = deque(
            (self._row(i) for i in range(self.idx - self.T, self.idx)),
            maxlen=self.T
        )
        self._reset_episode_stats()

        # reset EMA for this episode
        if reset_ret_norm:
            self._ret_ema = 1e-3

    def _start_idx_bounds(self, horizon: int) -> tuple[int, int]:
        ranges = self.data_partition.valid_ranges(horizon)
        return min(item.first for item in ranges), max(item.last for item in ranges)

    def _sample_random_start_idx(self, horizon: int, seed: int) -> int:
        return self.data_partition.sample(horizon, np.random.default_rng(seed))

    def _select_stream_start_idx(self, horizon: int) -> tuple[int, bool]:
        stride = int(self.stream_stride_cfg) if self.stream_stride_cfg != 0 else int(horizon)
        stride = max(1, stride)
        stream_changed = False

        if self._stream_pos is None or self._stream_horizon != int(horizon):
            self._stream_pos = self.data_partition.sample(horizon, self._stream_rng)
            self._stream_episode_count = 0
            stream_changed = True

        if (
            self.stream_jump_every_episodes > 0
            and self._stream_episode_count >= self.stream_jump_every_episodes
        ):
            self._stream_pos = self.data_partition.sample(horizon, self._stream_rng)
            self._stream_episode_count = 0
            stream_changed = True

        if not self.data_partition.contains(self._stream_pos, horizon):
            ranges = self.data_partition.valid_ranges(horizon)
            if self.stream_jump_on_wrap:
                self._stream_pos = self.data_partition.sample(horizon, self._stream_rng)
            else:
                self._stream_pos = ranges[0].first
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
        return self.data_partition.clamp(start_idx, horizon), True

    def _should_reset_ret_norm(self, stream_changed: bool) -> bool:
        if self.ret_norm_reset_mode == "stream" and self.start_idx_mode == "stream_worker":
            return stream_changed
        return True

    def _merge_task_sampler_overrides(self) -> None:
        overrides = self.cfg.get("task_sampler_overrides", {})
        if overrides:
            params = dict(self.current_task.reward_params)
            params.update(overrides)
            self.current_task.reward_params = self.reward_engine.normalize_params(
                self.current_task.reward_id,
                params,
            )

    def set_task_sampler_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply normalized reward-task overrides on subsequent resets."""
        if not isinstance(overrides, dict):
            raise TypeError("task sampler overrides must be a mapping")
        current = dict(self.cfg.get("task_sampler_overrides", {}))
        current.update(overrides)
        self.cfg["task_sampler_overrides"] = current

    def _log_reset(self, start_idx: int) -> None:
        self._reset_count += 1
        if not self.debug_resets:
            return
        if self._reset_count % max(1, self.debug_reset_every) != 0:
            return
        print(
            f"[Worker {self.worker_index}] Reset #{self._reset_count}: "
            f"start_idx={start_idx}, horizon={self.current_task.horizon}, df_len={len(self.df)}"
        )

    def _episode_done(self) -> bool:
        ep_horizon = int(self.current_task.horizon)
        return bool(self._t >= ep_horizon or self.idx >= len(self.df) - 1)

    def _reward_context(
        self,
        *,
        action: int,
        price_t: float,
        price_next: float,
        pnl_t: float,
        pnl_t1: float,
        dd_t: float,
        dd_t1: float,
        dd_inc: float,
        turnover_t: float,
        ret: float,
        ret_unit: float,
        terminal: bool,
    ) -> dict[str, Any]:
        return {
            "pnl_t": pnl_t,
            "pnl_t1": pnl_t1,
            "pnl_delta": pnl_t1 - pnl_t,
            "dd_t": dd_t,
            "dd_t1": dd_t1,
            "dd_inc": dd_inc,
            "max_dd": self._max_dd,
            "turnover_t": turnover_t,
            "price_t": price_t,
            "price_t1": price_next,
            "position_t": float(self.pos),
            "action_t": action,
            "ret": ret,
            "strategy_returns": tuple(self._episode_strategy_returns),
            "ret_unit": ret_unit,
            "ret_ema": self._ret_ema,
            "terminal": terminal,
        }

    def _episode_info(self, *, success: bool, comps: dict[str, Any], done: bool) -> dict[str, Any]:
        info = {
            "position": self.pos,
            "reward_comps": comps,
            "success": success,
            "task": {**self.current_task.to_dict(), "start_idx": self._episode_start_idx},
        }
        if done:
            liquidation_turnover = abs(float(self.pos))
            symbol = self.data_partition.symbol_for(
                self._episode_start_idx,
                int(self.current_task.horizon),
            )
            evidence = EpisodeEvidence.build(
                optimization_reward=self._optimization_reward_sum,
                gross_return=self._episode_pnl,
                turnover=self._turnover_sum,
                liquidation_turnover=liquidation_turnover,
                max_drawdown=self._max_dd,
                evidence_cost_bps=self.evidence_cost_bps,
                evidence_periods_per_year=self.evidence_periods_per_year,
                strategy_returns=list(self._episode_strategy_returns),
                turnover_steps=list(self._turnover_steps),
                gross_exposure_sum=self._gross_exposure_sum,
                position_changes=self._position_changes,
                task=self.current_task,
                symbol=symbol,
                split=self.data_split,
                start_idx=self._episode_start_idx,
                end_idx=self.idx,
                steps=self._t,
            )
            info["episode_evidence"] = evidence.to_dict()
            if self.capture_replay_trace:
                info["episode_evidence"]["action_reward_trace_sha256"] = (
                    canonical_json_sha256(self._action_reward_trace)
                )
            info["success"] = evidence.success
            info["ep_sums"] = {
                "steps": max(1, int(self._t)),
                "horizon": int(self.current_task.horizon),
                "act_counts": self._act_counts.copy(),
                "rc_sums": dict(self._rc_sums),
            }
        return info
    # ===================================================================== #
    # gymnasium.Env interface
    # --------------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            seed = int(seed)
            self.task_sampler.reseed(seed + self._sampler_seed_offset)
            self._stream_rng = np.random.default_rng(seed + self._stream_seed_offset)
            self._stream_pos = None
            self._stream_horizon = None
            self._stream_episode_count = 0
        # sample task for the episode first
        self.current_task = self.task_sampler.sample()
        horizon = int(self.current_task.horizon)
        start_idx, stream_changed = self._choose_start_idx(horizon)

        # Reset state with determined start_idx
        self._reset_state(
            start_idx=start_idx,
            reset_ret_norm=self._should_reset_ret_norm(stream_changed),
        )

        self._merge_task_sampler_overrides()
        # Create deterministic embedding from the normalized task contract.
        self._t_embed = TaskEncoder.embed(
            self.current_task,
            dim=self.t_embed_dim,
            version=self.task_encoder_version,
        )
        obs = self._build_obs()
        task_info = self.current_task.to_dict()
        task_info["start_idx"] = start_idx
        info = {"task": task_info}
        self._last_info = info
        self._log_reset(start_idx)
        return obs, info

    def step(self, action: int):
        # map discrete action to position target {-1,0,+1}
        action = int(action)
        target_pos = _ACTION_TO_POSITION[action]

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
        self._episode_strategy_returns.append(mtm)

        # drawdown (level and delta) BEFORE updating max_equity
        dd_t = max(0.0, self._max_equity - pnl_t)

        self._episode_pnl = pnl_t1
        self._max_equity  = max(self._max_equity, pnl_t1)

        dd_t1  = max(0.0, self._max_equity - pnl_t1)
        self._max_dd = max(self._max_dd, dd_t1)

        dd_inc = max(0.0, dd_t1 - dd_t)

        self._t += 1
        done = self._episode_done()
        trunc = False
        ctx = self._reward_context(
            action=action,
            price_t=price_t,
            price_next=price_next,
            pnl_t=pnl_t,
            pnl_t1=pnl_t1,
            dd_t=dd_t,
            dd_t1=dd_t1,
            dd_inc=dd_inc,
            turnover_t=turnover_t,
            ret=ret,
            ret_unit=ret_unit,
            terminal=done,
        )

        # compute reward via registry
        r, comps = self.reward_engine.compute(
            self.current_task.reward_id,
            self.current_task.reward_params,
            ctx,
        )
        if self.capture_replay_trace:
            self._action_reward_trace.append([action, float(r)])
        self._optimization_reward_sum += float(r)
        self._turnover_sum += float(turnover_t)
        self._turnover_steps.append(float(turnover_t))
        self._gross_exposure_sum += abs(float(self.pos))
        self._position_changes += int(float(self.pos) != prev_pos)

        # accumulate episode aggregates
        self._act_counts[action] += 1
        self._accumulate_reward_components(comps)

        # baseline success flag for TRR++/curriculum stats (Phase-B)
        if done:
            cost = (self._turnover_sum + abs(float(self.pos))) * self.evidence_cost_bps * 1e-4
            success = bool(pnl_t1 - cost > 0.0)
        else:
            success = False
        comps["success"] = float(success)  # for TRR++/curriculum stats

        obs_next = self._build_obs()
        info = self._episode_info(success=success, comps=comps, done=done)
        self._last_info = info
        return obs_next, float(r), done, trunc, info
