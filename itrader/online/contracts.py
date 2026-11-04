"""Bridges between the online stack and the repo's typed contracts.

The repo already defines TaskSpec, ExecutionIntent, CandidateTask, CapabilityProfile and
DecisionRecord (itrader/rl_lib/tasks/*). Nothing here redefines them. This module only adds:

* ONLINE_FAMILIES: the strategy families this stack trades, registered on the online side so
  the repo's TASK_FAMILY_REGISTRY (families 00-04, the research roadmap) is left untouched.
* Proposal: what a strategy emits each cycle (target weights plus a rationale), and the
  function that turns one into a CandidateTask the rest of the spec's loop understands.
* wait_candidate(): the spec's first-class WAIT, always present in the candidate set.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from itrader.rl_lib.tasks.candidate_task import CandidateTask
from itrader.rl_lib.tasks.execution_intent import ExecutionIntent
from itrader.rl_lib.tasks.spec import TaskSpec

# Family ids 10+ are reserved here for the online strategy library; 00-04 stay the research families.
ONLINE_FAMILIES: Dict[str, Dict[str, Any]] = {
    "10": {"name": "turtle_breakout", "bar": "1d", "scope": "Original 1983 Turtle rules"},
    "11": {"name": "trend_modern", "bar": "1d", "scope": "Multi-speed trend, continuous vol-scaled sizing"},
    "12": {"name": "xs_momentum", "bar": "1d", "scope": "Cross-sectional momentum rotation"},
    "13": {"name": "mean_reversion", "bar": "1h", "scope": "Short-term reversion inside the trend"},
    "14": {"name": "funding_carry", "bar": "1d", "scope": "Spot long vs perp short, collect funding"},
    "15": {"name": "pairs", "bar": "1h", "scope": "Co-moving pairs, trade the spread"},
    "16": {"name": "seasonality", "bar": "1h", "scope": "Time-of-day and turn-of-month effects"},
    "17": {"name": "prediction_sleeve", "bar": "5m", "scope": "Polymarket BTC rules and the reward farm (paper)"},
    "18": {"name": "experimental", "bar": "1d", "scope": "New ideas and the repo's RL solver, same bench as the rest"},
    "99": {"name": "wait", "bar": "-", "scope": "Abstain"},
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Proposal:
    """What a strategy proposes for the next bar: target weights per symbol (fraction of equity,
    signed, before portfolio scaling) and why. Weights of zero mean 'be flat'."""

    strategy: str
    family_id: str
    bar: str
    weights: Dict[str, float]
    expected_utility: float            # expected net return per bar in fraction units, after modelled costs
    confidence: float                  # 0..1, from the strategy's CapabilityProfile
    rationale: str = ""
    instrument: str = "perp"           # "perp" | "spot" | "spot+perp"
    ttl_bars: int = 1
    stop_loss_vol_units: float = 2.0   # exit if the move against us exceeds this many daily vols
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def gross(self) -> float:
        return float(sum(abs(w) for w in self.weights.values()))

    @property
    def candidate_id(self) -> str:
        key = json.dumps({"s": self.strategy, "w": {k: round(v, 6) for k, v in sorted(self.weights.items())}}, sort_keys=True)
        return f"{self.strategy}:{hashlib.sha1(key.encode()).hexdigest()[:10]}"

    def to_candidate(self, *, price_id: str = "online_panel", seed: int = 0) -> CandidateTask:
        spec = TaskSpec(
            family_id=self.family_id,
            price_id=price_id,
            horizon=int(self.ttl_bars),
            reward_id="pnl_linear",
            reward_params={"net": True},
            seed=int(seed),
            obs_dim=0,
            action_dim=len(self.weights),
            bar_resolution=self.bar,
            regime_tags=list(self.metadata.get("regime_tags", [])),
            metadata={"source": "online_strategy", "strategy": self.strategy, "candidate_id": self.candidate_id},
        )
        intent = ExecutionIntent(
            entry_logic={"type": "rebalance_to_target", "params": {"weights": dict(self.weights), "instrument": self.instrument}},
            exit_logic={"type": "ttl_or_stop", "params": {"ttl_bars": int(self.ttl_bars), "stop_loss_vol_units": self.stop_loss_vol_units}},
            ttl_bars=int(self.ttl_bars),
            confidence_band={"min": 0.0, "target": float(self.confidence)},
            risk_envelope={"gross": self.gross, "max_position_pct": max((abs(w) for w in self.weights.values()), default=0.0)},
            abstain_allowed=True,
        )
        return CandidateTask(
            task_spec=spec,
            execution_intent=intent,
            expected_difficulty=None,
            expected_utility=float(self.expected_utility),
            rationale=self.rationale,
            abstain_option=False,
            created_at=now_iso(),
            ttl_generated_at=now_iso(),
            metadata={"candidate_id": self.candidate_id, "strategy": self.strategy, "confidence": self.confidence, **self.metadata},
        )


def wait_candidate(reason: str = "no candidate cleared the gate") -> CandidateTask:
    spec = TaskSpec(family_id="99", price_id="online_panel", horizon=1, reward_id="pnl_linear", reward_params={}, seed=0,
                    obs_dim=0, action_dim=0, bar_resolution="-", metadata={"source": "online_stack", "candidate_id": "WAIT"})
    intent = ExecutionIntent(entry_logic={"type": "none"}, exit_logic={"type": "none"}, ttl_bars=1,
                             confidence_band={"min": 1.0, "target": 1.0}, risk_envelope={"gross": 0.0}, abstain_allowed=True)
    return CandidateTask(task_spec=spec, execution_intent=intent, expected_utility=0.0, rationale=reason,
                         abstain_option=True, created_at=now_iso(), ttl_generated_at=now_iso(), metadata={"candidate_id": "WAIT"})


def candidate_id(c: CandidateTask) -> str:
    return str(c.metadata.get("candidate_id") or c.task_spec.metadata.get("candidate_id") or "?")
