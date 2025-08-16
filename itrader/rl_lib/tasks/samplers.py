from __future__ import annotations
from typing import Dict, Any
import numpy as np
from .spec import TaskSpec

class TaskSampler00:
    """Samples TaskSpec for family ("00").

    env_config example:
    {
      "price_ids": ["AAPL", "BTCUSDT"],
      "horizon": [64, 128, 256],
      "reward_ids": ["pnl_sharpe"],
      "reward_params": {"pnl_w": 1.0, "dd_w": 0.1, "turnover_w": 0.0},
      "obs_dim": 32,
      "action_dim": 3,
      "seed": 1234
    }
    """
    def __init__(self, env_config: Dict[str, Any]):
        self.cfg = env_config
        self.rng = np.random.default_rng(self.cfg.get("seed", 1234))

    def sample(self) -> TaskSpec:
        price_id = self.rng.choice(self.cfg.get("price_ids", ["SIM"])).item()
        horizon_choices = self.cfg.get("horizon", [128])
        horizon = int(self.rng.choice(horizon_choices))
        reward_id = self.rng.choice(self.cfg.get("reward_ids", ["pnl_sharpe"]))
        reward_params = dict(self.cfg.get("reward_params", {"pnl_w": 1.0, "dd_w": 0.1, "turnover_w": 0.0}))
        obs_dim = int(self.cfg.get("obs_dim", 32))
        action_dim = int(self.cfg.get("action_dim", 3))
        seed = int(self.rng.integers(0, 1_000_000))
        return TaskSpec(
            family_id="00",
            price_id=str(price_id),
            horizon=horizon,
            reward_id=str(reward_id),
            reward_params=reward_params,
            seed=seed,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )