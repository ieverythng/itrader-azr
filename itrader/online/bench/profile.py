"""CapabilityProfiles in the repo's format, and the promotion ladder.

A CapabilityProfile (itrader/rl_lib/tasks/capability_profile.py) is how the spec wants a
strategy's track record expressed so the DecisionGate and the allocator can weigh it:

* task_family          the ONLINE_FAMILIES id
* estimated_p_solve    here: the probabilistic Sharpe on the development period, i.e. the
                       probability the strategy has any edge at all after costs
* expected_net_utility mean net return per bar (fraction), the same unit as Proposal.expected_utility
* calibration_error    |development Sharpe - holdout Sharpe|; big means the dev numbers did not travel
* sample_count         bars with exposure in the development period
* metadata             the full metrics, the evidence meter, the promotion stage and its gate results

The promotion ladder has four rungs: bench -> paper -> small_live -> full. Every gate is a number
in settings.toml and every check is logged with the value it saw, so a promotion (or a refusal)
can be audited. The ladder is advisory: nothing here flips live on.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from itrader.rl_lib.tasks.capability_profile import CapabilityProfile

from ..strategies.base import Strategy

STAGES = ("bench", "paper", "small_live", "full")


def _check(name: str, value: float, op: str, limit: float) -> Dict[str, Any]:
    ok = {"ge": value >= limit, "le": value <= limit, "gt": value > limit, "lt": value < limit}[op]
    return {"check": name, "value": float(value) if value == value else None, "op": op, "limit": float(limit), "ok": bool(ok)}


def promotion_stage(wf: Dict[str, Any], gates, paper: Optional[Dict[str, Any]] = None, live: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Walk the ladder from the bottom; stop at the first rung whose gates fail.

    `wf` is a walk_forward() result; `paper` and `live` are optional dicts with the same keys the
    runner's statistics produce (days, psr, sharpe, evidence_ratio, slippage_ratio, dsr)."""
    dev, hold = wf["dev"], wf["holdout"]
    g1 = gates.bench_to_paper
    checks1 = [
        _check("dev_dsr", dev["dsr"], "ge", g1.dsr_min),
        _check("holdout_sharpe", hold["sharpe"], "ge", g1.holdout_sharpe_min),
        _check("holdout_return", hold["total_return"], "gt", g1.holdout_return_min),
        _check("max_dd", abs(dev["max_dd"]), "le", g1.max_dd_max),
        _check("folds_positive_share", wf["folds_positive_share"], "ge", g1.folds_positive_min),
        _check("cost_share", min(dev["cost_share"], 9.99), "le", g1.cost_share_max),
    ]
    rungs = [{"from": "bench", "to": "paper", "checks": checks1, "passed": all(c["ok"] for c in checks1)}]
    stage = "paper" if rungs[0]["passed"] else "bench"

    if stage == "paper":
        g2 = gates.paper_to_small_live
        p = paper or {}
        checks2 = [
            _check("paper_days", p.get("days", 0.0), "ge", g2.paper_days_min),
            _check("paper_psr", p.get("psr", 0.0), "ge", g2.psr_min),
            _check("tracking_error", abs(p.get("sharpe", 0.0) - hold["sharpe"]) if p else 9.99, "le", g2.tracking_error_max),
            _check("evidence_ratio", p.get("evidence_ratio", 0.0), "ge", g2.evidence_ratio_min),
        ]
        rungs.append({"from": "paper", "to": "small_live", "checks": checks2, "passed": all(c["ok"] for c in checks2)})
        if rungs[-1]["passed"]:
            stage = "small_live"
            g3 = gates.small_live_to_full
            lv = live or {}
            checks3 = [
                _check("live_days", lv.get("days", 0.0), "ge", g3.live_days_min),
                _check("slippage_ratio", lv.get("slippage_ratio", 9.99), "le", g3.slippage_ratio_max),
                _check("live_dsr", lv.get("dsr", 0.0), "ge", g3.dsr_min),
            ]
            rungs.append({"from": "small_live", "to": "full", "checks": checks3, "passed": all(c["ok"] for c in checks3)})
            if rungs[-1]["passed"]:
                stage = "full"
    return {"stage": stage, "rungs": rungs, "live_allowed_by_ladder": stage in ("small_live", "full"),
            "note": "advisory only: live execution additionally needs the env flag, keys and the confirmation file"}


def capability_profile(strategy: Strategy, wf: Dict[str, Any], stage: Optional[Dict[str, Any]] = None) -> CapabilityProfile:
    dev, hold = wf["dev"], wf["holdout"]
    meta = {
        "strategy": strategy.name,
        "instrument": strategy.instrument,
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in strategy.params.items()},
        "n_variants_tried": strategy.n_variants_tried,
        "dev": {k: v for k, v in dev.items() if not isinstance(v, dict)},
        "holdout": {k: v for k, v in hold.items() if not isinstance(v, dict)},
        "evidence": dev.get("evidence", {}),
        "capacity": dev.get("capacity", {}),
        "folds": wf["folds"],
        "folds_positive_share": wf["folds_positive_share"],
        "promotion": stage or {},
        "source": "itrader.online.bench",
    }
    return CapabilityProfile(
        task_family=str(strategy.family_id),
        bar_resolution=str(strategy.bar),
        regime_tags=["all"],
        horizon_bars=1,
        estimated_p_solve=float(dev["psr"]),
        expected_net_utility=float(dev["mean_net_per_bar"]),
        calibration_error=float(abs(dev["sharpe"] - hold["sharpe"])),
        sample_count=int(dev["bars_exposed"]),
        last_updated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        metadata=meta,
    )
