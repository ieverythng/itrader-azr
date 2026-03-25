from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


_TASKSPEC_ALIASES = {
    "task_family": "family_id",
    "horizon_bars": "horizon",
    "start_index": "start_idx",
}


def normalize_task_spec_payload(cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    """Normalize alias-heavy task payloads into the repo's canonical TaskSpec shape."""
    raw: Dict[str, Any] = dict(cfg or {})

    for alias, canonical in _TASKSPEC_ALIASES.items():
        if alias in raw and canonical not in raw:
            raw[canonical] = raw.pop(alias)

    reward_cfg = raw.pop("reward_config", None)
    if isinstance(reward_cfg, dict):
        if "reward_id" not in raw:
            raw["reward_id"] = reward_cfg.get("name")
        if "reward_params" not in raw:
            raw["reward_params"] = reward_cfg.get("params")

    raw.setdefault("start_idx", None)
    raw.setdefault("bar_resolution", None)
    raw.setdefault("regime_tags", [])
    raw.setdefault("difficulty_target", {})
    raw.setdefault("metadata", {})
    raw.setdefault("t_embed", None)
    return raw

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
    bar_resolution: Optional[str] = None
    regime_tags: List[str] = field(default_factory=list)
    difficulty_target: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for key in ("bar_resolution", "t_embed"):
            if d.get(key) is None:
                d.pop(key, None)
        for key in ("regime_tags", "difficulty_target", "metadata"):
            if not d.get(key):
                d.pop(key, None)
        return d

    @property
    def task_family(self) -> str:
        return self.family_id

    @property
    def horizon_bars(self) -> int:
        return self.horizon

    @property
    def reward_config(self) -> Dict[str, Any]:
        return {
            "name": self.reward_id,
            "params": dict(self.reward_params),
        }

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "TaskSpec":
        return TaskSpec(**normalize_task_spec_payload(cfg))
