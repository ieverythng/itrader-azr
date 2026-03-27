from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class CapabilityProfile:
    task_family: str
    bar_resolution: str
    regime_tags: List[str]
    horizon_bars: int
    estimated_p_solve: float
    expected_net_utility: float
    calibration_error: float
    sample_count: int
    last_updated: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "CapabilityProfile":
        raw = dict(cfg or {})
        return CapabilityProfile(
            task_family=str(raw.get("task_family", "00")),
            bar_resolution=str(raw.get("bar_resolution", "1m")),
            regime_tags=[str(tag) for tag in raw.get("regime_tags", [])],
            horizon_bars=int(raw.get("horizon_bars", 0)),
            estimated_p_solve=float(raw.get("estimated_p_solve", 0.0)),
            expected_net_utility=float(raw.get("expected_net_utility", 0.0)),
            calibration_error=float(raw.get("calibration_error", 0.0)),
            sample_count=int(raw.get("sample_count", 0)),
            last_updated=str(raw.get("last_updated", "")),
            metadata=dict(raw.get("metadata", {})),
        )
