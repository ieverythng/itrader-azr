"""DecisionGate: deterministic filters, then WAIT if nothing clears them.

Inputs: the framer's proposals and candidates, the current book, the risk state. Every
threshold comes from settings and every check is written into the decision with the value it
saw, so a DecisionRecord shows exactly why a candidate passed or failed (spec 11.4).

A candidate passes if it is valid, its confidence (the bench's probability that the strategy
has an edge after costs) clears the floor, its expected utility per bar clears the floor, and
the utility it expects over its horizon exceeds a multiple of the cost of getting into the
position from where the book is now. Candidates that pass are approved as a set; the portfolio
layer decides how much capital each gets. If none pass, the gate selects WAIT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .contracts import Proposal
from .framer import Frame
from .settings import Settings, settings as default_settings


@dataclass
class GateDecision:
    approved: List[Proposal]
    selected_candidate_id: Optional[str]      # the top-ranked approved candidate, or "WAIT"
    wait: bool
    checks: Dict[str, List[Dict[str, Any]]]   # candidate_id -> list of checks
    scores: Dict[str, float]
    thresholds: Dict[str, float]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"approved": [p.candidate_id for p in self.approved], "selected_candidate_id": self.selected_candidate_id, "wait": self.wait,
                "checks": self.checks, "scores": self.scores, "thresholds": self.thresholds, "reason": self.reason}


class DecisionGate:
    def __init__(self, s: Optional[Settings] = None) -> None:
        self.s = s or default_settings()
        g = self.s.gate
        self.min_conf = float(g.min_confidence)
        self.min_eu = float(g.min_expected_utility_bps) / 1e4
        self.cost_mult = float(g.utility_cost_multiple)
        self.universe = set(self.s.data.universe)

    def _check(self, name: str, value: float, op: str, limit: float, detail: str = "") -> Dict[str, Any]:
        ok = {"ge": value >= limit, "gt": value > limit, "le": value <= limit, "eq": value == limit}[op]
        return {"check": name, "value": float(value) if np.isfinite(value) else None, "op": op, "limit": float(limit), "ok": bool(ok), "detail": detail}

    def decide(self, frame: Frame, current: Optional[pd.Series] = None, risk_locked: bool = False, lock_reason: str = "") -> GateDecision:
        checks: Dict[str, List[Dict[str, Any]]] = {}
        scores: Dict[str, float] = {}
        approved: List[Proposal] = []
        for p in frame.proposals:
            cid = p.candidate_id
            c: List[Dict[str, Any]] = []
            w = pd.Series(p.weights, dtype=float)
            valid = bool(len(w)) and bool(np.isfinite(w).all()) and set(w.index) <= self.universe and p.gross <= 1.0 + 1e-9
            c.append(self._check("valid", 1.0 if valid else 0.0, "eq", 1.0, "finite weights, known symbols, gross <= 1"))
            c.append(self._check("confidence", p.confidence, "ge", self.min_conf, "bench PSR: P(edge > 0 after costs)"))
            c.append(self._check("expected_utility_per_bar", p.expected_utility, "ge", self.min_eu, "bench mean net return per bar"))
            cost = float(p.metadata.get("cost_to_trade", 0.0))
            horizon_utility = p.expected_utility * max(1, p.ttl_bars)
            c.append(self._check("utility_vs_cost", horizon_utility, "ge", self.cost_mult * cost, f"utility over {p.ttl_bars} bars vs {self.cost_mult}x cost to trade"))
            c.append(self._check("risk_unlocked", 0.0 if risk_locked else 1.0, "eq", 1.0, lock_reason or "risk engine not locked"))
            checks[cid] = c
            score = p.expected_utility * p.confidence / (cost + 1e-6)
            scores[cid] = float(score)
            if all(x["ok"] for x in c):
                approved.append(p)
        approved.sort(key=lambda p: scores[p.candidate_id], reverse=True)
        thresholds = {"min_confidence": self.min_conf, "min_expected_utility_per_bar": self.min_eu, "utility_cost_multiple": self.cost_mult,
                      "settings_hash": self.s.hash}
        if not approved:
            reasons = []
            for cid, c in checks.items():
                failed = [x["check"] for x in c if not x["ok"]]
                if failed:
                    reasons.append(f"{cid}: {', '.join(failed)}")
            reason = "WAIT: " + ("; ".join(reasons) if reasons else "no candidates")
            return GateDecision([], "WAIT", True, checks, scores, thresholds, reason)
        return GateDecision(approved, approved[0].candidate_id, False, checks, scores, thresholds,
                            f"{len(approved)} of {len(frame.proposals)} candidates approved")
