from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class TaskFamilySpec:
    family_id: str
    name: str
    description: str
    action_dim: int = 3
    default_reward_id: str = "pnl_linear"
    default_bar_resolution: str = "1m"
    horizon_range: Tuple[int, int] = (64, 512)


_FAMILY_REGISTRY: Dict[str, TaskFamilySpec] = {
    "00": TaskFamilySpec(
        family_id="00",
        name="directional_predictor",
        description="Current proven short-horizon directional task family.",
    ),
    "01": TaskFamilySpec(
        family_id="01",
        name="breakout_expansion",
        description="Directional breakout/expansion framing over the existing 3-action control path.",
    ),
    "02": TaskFamilySpec(
        family_id="02",
        name="mean_reversion",
        description="Directional mean-reversion framing over the existing 3-action control path.",
    ),
}


def family_registry() -> Dict[str, TaskFamilySpec]:
    return dict(_FAMILY_REGISTRY)


def known_family_ids() -> Tuple[str, ...]:
    return tuple(sorted(_FAMILY_REGISTRY.keys()))


def get_family_spec(family_id: str) -> TaskFamilySpec | None:
    return _FAMILY_REGISTRY.get(str(family_id))


def is_known_family(family_id: str) -> bool:
    return str(family_id) in _FAMILY_REGISTRY
