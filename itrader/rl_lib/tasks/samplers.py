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
      "reward_params": {"pnl_w": 1.0, "dd_w": 0.1, "turnover_w": 0.0},      "window_size": 30,
      "n_features": 5,
      "t_embed_dim": 32,
      "action_dim": 3,
      "seed": 1234
    }
    """
    def __init__(self, env_config: Dict[str, Any]):
        self.cfg = env_config
        self.seed = int(self.cfg.get("seed", 1234))
        self.rng = np.random.default_rng(self.seed)

    def sample(self) -> TaskSpec:
        price_id = self.rng.choice(self.cfg.get("price_ids", ["SIM"])).item()
        horizon = int(self.rng.choice(self.cfg.get("horizons", [96, 128, 160, 200])))

        # Small penalties to avoid drowning signal; later widened by PBT
        turnover_w = float(self.rng.choice(self.cfg.get("turnover_w_choices", [0.0, 0.0005, 0.001, 0.002, 0.004])))
        # dd_mode    = str(self.rng.choice(self.cfg.get("dd_modes", ["none", "delta"])))
        dd_w       = float(self.rng.uniform(*self.cfg.get("dd_w_range", (0.02, 0.08))))
        dd_T_weight = float(self.cfg.get("dd_T_weight", 0.0))
        # hold_cost_w = float(self.rng.choice(self.cfg.get("hold_cost_w_choices", [0.0, 0.001, 0.002])))

        reward_id = str(self.rng.choice(self.cfg.get("reward_ids", ["pnl_linear"])))
        reward_params = {
            "pnl_w": 1.0,
            "turnover_w": turnover_w,
            "dd_mode": 'delta',
            "dd_w": dd_w,
            "dd_T_weight": dd_T_weight,
            "turnover_cap": float(self.cfg.get("turnover_cap", 0.02)),
            "dd_cap": float(self.cfg.get("dd_cap", 0.05)),
            "reward_scale": 1.0,
            "normalize": True,
            "norm_mode": "all",
            # "hold_cost_w": hold_cost_w,
            "normalize_dd": True,     # also normalize dd_inc (keeps penalties consistent)
            "norm_clip": 5.0,  
        }

        window_size = int(self.cfg.get("window_size", 30))
        n_features  = int(self.cfg.get("n_features", 5))
        t_embed_dim = int(self.cfg.get("t_embed_dim", 32))
        obs_dim     = window_size * n_features + t_embed_dim
        action_dim  = int(self.cfg.get("action_dim", 3))
        seed        = int(self.rng.integers(0, 1_000_000))

        return TaskSpec(
            family_id="00",
            price_id=str(price_id),
            horizon=horizon,
            reward_id=reward_id,
            reward_params=reward_params,
            seed=seed,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )