from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

@dataclass
class TaskSpec:
    """Single source of truth for a training task configuration.

    Notes
    -----
    - family_id: logical task family (e.g., "00" for P‑S sprint family)
    - price_id: which price stream or generator to use
    - horizon: episode length in timesteps
    - reward_id: key into RewardEngine registry
    - reward_params: param dict passed to the reward function
    - t_embed: optional precomputed task embedding (list[float]). If None, encoder will derive one.
    - seed: RNG seed for reproducibility
    - obs_dim: total observation space dimension (price + t_embed)
    - action_dim: action space dimension (policy output size)
    - start_idx: optional starting index in dataframe. If None, env samples uniformly.
    """
    family_id: str
    price_id: str
    horizon: int
    reward_id: str
    reward_params: Dict[str, Any]
    seed: int
    obs_dim: int
    action_dim: int
    t_embed: Optional[List[float]] = None
    start_idx: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "TaskSpec":
        # Handle backward compatibility: start_idx may not be present in old task specs
        if "start_idx" not in cfg:
            cfg = {**cfg, "start_idx": None}
        return TaskSpec(**cfg)
