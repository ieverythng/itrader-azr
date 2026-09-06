from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence

from itrader.rl_lib.tasks.families import DEFAULT_BAR_RESOLUTION
from itrader.rl_lib.tasks.rewards import RewardEngine

from .base import BaseProposer


class RuleBasedProposer(BaseProposer):
    """Deterministic, dependency-light proposer fallback for local/dev flows."""

    backend_name = "rule_based"

    def __init__(
        self,
        *,
        seed: int = 42,
        window_size: int = 30,
        n_features: int = 5,
        t_embed_dim: int = 32,
        action_dim: int = 3,
        price_ids: Sequence[str] | None = None,
        horizons: Sequence[int] | None = None,
        reward_id: str = "pnl_linear",
        reward_params: Dict[str, Any] | None = None,
        bar_resolution: str = DEFAULT_BAR_RESOLUTION,
    ):
        seed = int(seed)
        self.rng = random.Random(seed)
        self._reward_rng = random.Random(seed + 1)
        self.window_size = int(window_size)
        self.n_features = int(n_features)
        self.t_embed_dim = int(t_embed_dim)
        self.action_dim = int(action_dim)
        self.price_ids = [str(x) for x in (price_ids or ["SIM"])]
        self.horizons = [int(x) for x in (horizons or [64, 96, 128, 160, 200, 256])]
        self.reward_id = str(reward_id)
        self.reward_params = dict(reward_params or {})
        self.bar_resolution = str(bar_resolution)

    @property
    def obs_dim(self) -> int:
        return self.window_size * self.n_features + self.t_embed_dim

    def _reward_params_for(self) -> Dict[str, Any]:
        params = RewardEngine.sample_params(self.reward_id, self._reward_rng)
        params.update(self.reward_params)
        return RewardEngine.normalize_params(self.reward_id, params)

    def _sample_task(self) -> Dict[str, Any]:
        return {
            "family_id": "00",
            "price_id": self.rng.choice(self.price_ids),
            "horizon": self.rng.choice(self.horizons),
            "reward_id": self.reward_id,
            "reward_params": self._reward_params_for(),
            "seed": self.rng.randint(0, 1_000_000),
            "obs_dim": int(self.obs_dim),
            "action_dim": self.action_dim,
            "start_idx": None,
            "bar_resolution": self.bar_resolution,
        }

    def propose_tasks(self, market_summary: str, n: int, **kwargs: Any) -> List[Dict[str, Any]]:
        del market_summary, kwargs
        count = max(0, int(n))
        return [self._sample_task() for _ in range(count)]
