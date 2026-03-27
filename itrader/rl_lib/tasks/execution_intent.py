from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ExecutionIntent:
    entry_logic: Dict[str, Any]
    exit_logic: Dict[str, Any]
    ttl_bars: int
    confidence_band: Dict[str, float]
    risk_envelope: Dict[str, float]
    abstain_allowed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "ExecutionIntent":
        raw = dict(cfg or {})
        return ExecutionIntent(
            entry_logic=dict(raw.get("entry_logic", {})),
            exit_logic=dict(raw.get("exit_logic", {})),
            ttl_bars=int(raw.get("ttl_bars", 1)),
            confidence_band=dict(raw.get("confidence_band", {})),
            risk_envelope=dict(raw.get("risk_envelope", {})),
            abstain_allowed=bool(raw.get("abstain_allowed", True)),
        )
