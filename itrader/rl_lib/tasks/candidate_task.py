from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from .execution_intent import ExecutionIntent
from .spec import TaskSpec


@dataclass
class CandidateTask:
    task_spec: TaskSpec
    execution_intent: Optional[ExecutionIntent] = None
    expected_difficulty: Optional[float] = None
    expected_utility: Optional[float] = None
    rationale: str = ""
    abstain_option: bool = False
    created_at: Optional[str] = None
    ttl_generated_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["task_spec"] = self.task_spec.to_dict()
        if self.execution_intent is not None:
            payload["execution_intent"] = self.execution_intent.to_dict()
        return payload

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "CandidateTask":
        raw = dict(cfg or {})
        execution_intent = raw.get("execution_intent")
        return CandidateTask(
            task_spec=TaskSpec.from_dict(raw.get("task_spec", {})),
            execution_intent=ExecutionIntent.from_dict(execution_intent) if execution_intent else None,
            expected_difficulty=raw.get("expected_difficulty"),
            expected_utility=raw.get("expected_utility"),
            rationale=str(raw.get("rationale", "")),
            abstain_option=bool(raw.get("abstain_option", False)),
            created_at=raw.get("created_at"),
            ttl_generated_at=raw.get("ttl_generated_at"),
            metadata=dict(raw.get("metadata", {})),
        )
