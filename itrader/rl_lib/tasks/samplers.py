from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .families import DEFAULT_BAR_RESOLUTION
from .rewards import RewardEngine
from .spec import TaskSpec


class TaskSampler00:
    """Samples TaskSpec for family ("00")."""

    def __init__(self, env_config: Dict[str, Any]):
        self.cfg = env_config
        self.seed = int(self.cfg.get("seed", 1234))
        self.rng = np.random.default_rng(self.seed)
        self._reward_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1]))
        self.window_size = int(self.cfg.get("window_size", 30))
        self.n_features = int(self.cfg.get("n_features", 5))
        self.t_embed_dim = int(self.cfg.get("t_embed_dim", 32))
        self.action_dim = int(self.cfg.get("action_dim", 3))
        self.bar_resolution = str(self.cfg.get("bar_resolution", DEFAULT_BAR_RESOLUTION))

        self.task_pool = None
        self.p_pool = float(self.cfg.get("p_pool", 0.7))
        self._load_task_pool(self.cfg.get("task_pool_path"))
        raw_fixed_tasks = self.cfg.get("fixed_tasks", [])
        if raw_fixed_tasks is None:
            raw_fixed_tasks = []
        if not isinstance(raw_fixed_tasks, list):
            raise ValueError("fixed_tasks must be a list of TaskSpec payloads")
        self._fixed_tasks = tuple(TaskSpec.from_dict(item) for item in raw_fixed_tasks)
        self._fixed_task_index = 0

    def reseed(self, seed: int) -> None:
        """Restart task and reward streams from one reproducible seed."""
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._reward_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1]))
        self._fixed_task_index = 0

    def _warn(self, message: str) -> None:
        print(f"Warning: {message}")

    def _load_task_pool(self, task_pool_path: Any) -> None:
        if not task_pool_path:
            return

        try:
            from .task_pool import TaskPool

            pool_path = Path(task_pool_path)
            if not pool_path.exists():
                self._warn(
                    f"task_pool_path {pool_path} does not exist, falling back to random sampling"
                )
                return

            self.task_pool = TaskPool(str(pool_path), read_only=True)
            print(f"TaskPool loaded from {pool_path} (size: {self.task_pool.size()})")
        except Exception as exc:
            self._warn(f"Failed to load TaskPool: {exc}, falling back to random sampling")

    def _task_shape(self) -> dict[str, int]:
        return {
            "obs_dim": self.window_size * self.n_features + self.t_embed_dim,
            "action_dim": self.action_dim,
        }

    def _sample_seed(self) -> int:
        return int(self.rng.integers(0, 1_000_000))

    def _pool_enabled(self) -> bool:
        return bool(self.task_pool and self.task_pool.size() > 0 and self.rng.random() < self.p_pool)

    def _sample_from_task_pool(self) -> Optional[TaskSpec]:
        if not self._pool_enabled():
            return None

        try:
            pool_tasks = self.task_pool.sample(batch_size=1, strategy="weighted")
        except Exception as exc:
            self._warn(f"TaskPool sampling failed: {exc}, falling back to random")
            return None

        if not pool_tasks:
            return None

        task_dict = dict(pool_tasks[0])
        task_dict.update(self._task_shape())
        task_dict["seed"] = self._sample_seed()
        task_dict.setdefault("family_id", "00")
        task_dict.setdefault("bar_resolution", self.bar_resolution)
        return TaskSpec.from_dict(task_dict)

    def _reward_param_overrides(self, reward_id: str) -> dict[str, Any]:
        per_reward = self.cfg.get("reward_params_by_id", {})
        per_reward_params = per_reward.get(reward_id, {}) if isinstance(per_reward, dict) else {}
        global_params = self.cfg.get("reward_params", {})
        overrides: dict[str, Any] = {}
        if isinstance(global_params, dict):
            overrides.update(global_params)
        if isinstance(per_reward_params, dict):
            overrides.update(per_reward_params)
        return overrides

    def _reward_params_for(self, reward_id: str) -> dict[str, Any]:
        params = RewardEngine.sample_params(reward_id, self._reward_rng)
        if "dd_w" in params and "dd_w_range" in self.cfg:
            bounds = tuple(float(value) for value in self.cfg["dd_w_range"])
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError("dd_w_range must be [minimum, maximum]")
            params["dd_w"] = float(self._reward_rng.uniform(bounds[0], bounds[1]))
        if "turnover_w" in params and "turnover_w_choices" in self.cfg:
            choices = list(self.cfg["turnover_w_choices"])
            if not choices:
                raise ValueError("turnover_w_choices must not be empty")
            params["turnover_w"] = float(self._reward_rng.choice(choices))
        if "dd_mode" in params and "dd_modes" in self.cfg:
            choices = list(self.cfg["dd_modes"])
            if not choices:
                raise ValueError("dd_modes must not be empty")
            params["dd_mode"] = str(self._reward_rng.choice(choices))
        for key in ("turnover_cap", "dd_cap"):
            if key in self.cfg:
                params[key] = self.cfg[key]
        params.update(self._reward_param_overrides(reward_id))
        return RewardEngine.normalize_params(reward_id, params)

    def _sample_random_task(self) -> TaskSpec:
        shape = self._task_shape()
        price_id = self.rng.choice(self.cfg.get("price_ids", ["SIM"])).item()
        horizon = int(self.rng.choice(self.cfg.get("horizons", [96, 128, 160, 200])))
        reward_id = str(self.rng.choice(self.cfg.get("reward_ids", ["pnl_linear"])))

        return TaskSpec(
            family_id="00",
            price_id=str(price_id),
            start_idx=None,
            horizon=horizon,
            reward_id=reward_id,
            reward_params=self._reward_params_for(reward_id),
            seed=self._sample_seed(),
            obs_dim=shape["obs_dim"],
            action_dim=shape["action_dim"],
            bar_resolution=self.bar_resolution,
        )

    def sample(self) -> TaskSpec:
        if self._fixed_tasks:
            task = self._fixed_tasks[self._fixed_task_index % len(self._fixed_tasks)]
            self._fixed_task_index += 1
            return TaskSpec.from_dict(task.to_dict())
        task = self._sample_from_task_pool()
        if task is not None:
            return task
        return self._sample_random_task()
