from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

DEFAULT_BAR_RESOLUTION = "1m"


@dataclass(frozen=True)
class TaskFamilySpec:
    family_id: str
    name: str
    description: str
    action_dim: int = 3
    default_reward_id: str = "pnl_linear"
    default_bar_resolution: str = DEFAULT_BAR_RESOLUTION
    horizon_range: Tuple[int, int] = (64, 512)
    scope: str = ""
    extra_features: Tuple[str, ...] = ()
    first_target_learner: str = "solver_mlp_film"
    allowed_reward_ids: Tuple[str, ...] = ("pnl_linear",)
    enabled: bool = False
    bootstrap: bool = False


TASK_FAMILY_REGISTRY: Dict[str, TaskFamilySpec] = {
    "00": TaskFamilySpec(
        family_id="00",
        name="directional_predictor",
        description="Current proven short-horizon directional task family.",
        scope="Single-instrument intraday directional",
        extra_features=("rsi14", "sma20", "sin_tod", "cos_tod"),
        allowed_reward_ids=("pnl_linear", "pnl_sharpe", "sharpe_only", "composed_linear"),
        enabled=True,
        bootstrap=True,
    ),
    "01": TaskFamilySpec(
        family_id="01",
        name="breakout_expansion",
        description="Directional breakout/expansion framing over the existing 3-action control path.",
        scope="High-vol breakout",
        extra_features=("atr14", "bb_width"),
        allowed_reward_ids=("pnl_linear", "composed_linear"),
        bootstrap=True,
    ),
    "02": TaskFamilySpec(
        family_id="02",
        name="mean_reversion",
        description="Directional mean-reversion framing over the existing 3-action control path.",
        scope="Mean reversion / snapback",
        extra_features=("spread_zscore", "distance_from_sma"),
        allowed_reward_ids=("pnl_linear", "composed_linear"),
        bootstrap=True,
    ),
    "03": TaskFamilySpec(
        family_id="03",
        name="trade_management",
        description="Future hold-vs-exit task family for already-open positions.",
        scope="Trade management / hold-vs-exit",
        extra_features=("position_age", "unrealized_pnl", "drawdown_from_peak"),
    ),
    "04": TaskFamilySpec(
        family_id="04",
        name="abstain_unsuitable_regime",
        description="Future no-action family for unsuitable market regimes.",
        scope="Abstain / unsuitable regime",
        extra_features=("vol_regime", "liquidity_regime", "spread_bps"),
    ),
}

_FAMILY_REGISTRY = TASK_FAMILY_REGISTRY

ACTIVE_FAMILY_IDS: Tuple[str, ...] = tuple(
    family_id for family_id, spec in TASK_FAMILY_REGISTRY.items() if spec.enabled
)
if not ACTIVE_FAMILY_IDS:
    ACTIVE_FAMILY_IDS = ("00",)

PRIMARY_FAMILY_ID: str = ACTIVE_FAMILY_IDS[0]
PRIMARY_REWARD_ID: str = "pnl_linear"


def family_registry() -> Dict[str, TaskFamilySpec]:
    return dict(_FAMILY_REGISTRY)


def known_family_ids() -> Tuple[str, ...]:
    return tuple(
        sorted(family_id for family_id, spec in _FAMILY_REGISTRY.items() if spec.bootstrap)
    )


def get_family_spec(family_id: str) -> Optional[TaskFamilySpec]:
    return _FAMILY_REGISTRY.get(str(family_id))


def is_known_family(family_id: str) -> bool:
    return str(family_id) in _FAMILY_REGISTRY


def normalize_family_id(family_id: object, *, default: str = PRIMARY_FAMILY_ID) -> str:
    raw = str(family_id if family_id is not None else "").strip()
    if raw.isdigit():
        raw = f"{int(raw):02d}"
    if raw in TASK_FAMILY_REGISTRY:
        return raw
    return default


def enforce_active_family_id(family_id: object) -> str:
    normalized = normalize_family_id(family_id, default=PRIMARY_FAMILY_ID)
    if normalized in ACTIVE_FAMILY_IDS:
        return normalized
    return PRIMARY_FAMILY_ID


def enforce_active_reward_id(reward_id: object, *, family_id: object = PRIMARY_FAMILY_ID) -> str:
    family = enforce_active_family_id(family_id)
    candidate = str(reward_id if reward_id is not None else "").strip()
    allowed = TASK_FAMILY_REGISTRY[family].allowed_reward_ids
    if candidate in allowed:
        return candidate
    return TASK_FAMILY_REGISTRY[family].default_reward_id


def llm_constraints_text() -> str:
    return (
        'Bootstrap proposer runtime should emit family_id "00" and reward_id "pnl_linear" '
        "unless a caller explicitly requests another validated family."
    )
