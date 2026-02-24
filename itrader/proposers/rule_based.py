from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence

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
    ):
        self.rng = random.Random(int(seed))
        self.window_size = int(window_size)
        self.n_features = int(n_features)
        self.t_embed_dim = int(t_embed_dim)
        self.action_dim = int(action_dim)
        self.price_ids = [str(x) for x in (price_ids or ["SIM"])]
        self.horizons = [int(x) for x in (horizons or [64, 96, 128, 160, 200, 256])]
        self.reward_id = str(reward_id)

    def _sample_reward_params(self) -> Dict[str, Any]:
        return {
            "pnl_w": 1.0,
            "turnover_w": self.rng.choice([0.0, 0.0005, 0.001, 0.002]),
            "dd_mode": "delta",
            "dd_w": round(self.rng.uniform(0.0, 0.08), 6),
            "dd_T_weight": 0.0,
            "turnover_cap": 0.02,
            "dd_cap": 0.1,
            "reward_scale": 1.0,
            "normalize": True,
            "norm_mode": "all",
            "unit_clip": 100000000.0,
        }

    def _sample_task(self) -> Dict[str, Any]:
        obs_dim = self.window_size * self.n_features + self.t_embed_dim
        return {
            "family_id": "00",
            "price_id": self.rng.choice(self.price_ids),
            "horizon": self.rng.choice(self.horizons),
            "reward_id": self.reward_id,
            "reward_params": self._sample_reward_params(),
            "seed": self.rng.randint(0, 1_000_000),
            "obs_dim": int(obs_dim),
            "action_dim": self.action_dim,
            "start_idx": None,
        }

    def propose_tasks(self, market_summary: str, n: int, **kwargs: Any) -> List[Dict[str, Any]]:
        del market_summary, kwargs
        count = max(0, int(n))
        return [self._sample_task() for _ in range(count)]
