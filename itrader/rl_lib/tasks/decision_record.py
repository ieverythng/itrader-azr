from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .candidate_task import CandidateTask


@dataclass
class DecisionRecord:
    market_summary_ref: Optional[str]
    candidate_tasks: List[CandidateTask]
    selected_candidate_id: Optional[str]
    wait_selected: bool
    solver_scores: Dict[str, Any]
    risk_decisions: Dict[str, Any]
    executor_action: Dict[str, Any]
    realized_outcome: Dict[str, Any]
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_summary_ref": self.market_summary_ref,
            "candidate_tasks": [candidate.to_dict() for candidate in self.candidate_tasks],
            "selected_candidate_id": self.selected_candidate_id,
            "wait_selected": self.wait_selected,
            "solver_scores": dict(self.solver_scores),
            "risk_decisions": dict(self.risk_decisions),
            "executor_action": dict(self.executor_action),
            "realized_outcome": dict(self.realized_outcome),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "DecisionRecord":
        raw = dict(cfg or {})
        return DecisionRecord(
            market_summary_ref=raw.get("market_summary_ref"),
            candidate_tasks=[CandidateTask.from_dict(item) for item in raw.get("candidate_tasks", [])],
            selected_candidate_id=raw.get("selected_candidate_id"),
            wait_selected=bool(raw.get("wait_selected", False)),
            solver_scores=dict(raw.get("solver_scores", {})),
            risk_decisions=dict(raw.get("risk_decisions", {})),
            executor_action=dict(raw.get("executor_action", {})),
            realized_outcome=dict(raw.get("realized_outcome", {})),
            created_at=raw.get("created_at"),
            metadata=dict(raw.get("metadata", {})),
        )
