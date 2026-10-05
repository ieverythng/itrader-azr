"""Framework-neutral episode metric projection shared by Ray adapters."""
from __future__ import annotations

import math
from numbers import Real
from typing import Any, Dict


def safe_metric_token(value: Any) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value))
    return token.strip("_") or "unknown"


def record_episode_evidence(metrics, evidence: Dict[str, Any]) -> None:
    """Project stable episode evidence onto an object's ``custom_metrics`` map."""
    custom_metrics = metrics.custom_metrics
    scalar_map = {
        "optimization_reward": "optimization_reward",
        "gross_return": "economic_gross_return",
        "cost_return": "economic_cost_return",
        "net_return": "economic_net_return",
        "turnover": "economic_turnover",
        "liquidation_turnover": "economic_liquidation_turnover",
        "max_drawdown": "economic_max_drawdown",
        "gross_sharpe_per_sqrt_bar": "economic_gross_sharpe_per_sqrt_bar",
        "net_sharpe_per_sqrt_bar": "economic_net_sharpe_per_sqrt_bar",
        "annualized_net_sharpe": "economic_annualized_net_sharpe",
        "positive_net_bar_fraction": "economic_positive_net_bar_fraction",
        "gross_exposure_fraction": "economic_gross_exposure_fraction",
        "position_changes": "economic_position_changes",
        "success": "success",
    }
    for source, metric in scalar_map.items():
        if source in evidence and evidence[source] is not None:
            custom_metrics[metric] = float(evidence[source])

    net_return = evidence.get("net_return")
    if net_return is None:
        return
    net_return = float(net_return)
    family = safe_metric_token(evidence.get("family_id", "unknown"))
    horizon = safe_metric_token(evidence.get("horizon", "unknown"))
    custom_metrics[f"family_{family}_net_return"] = net_return
    custom_metrics[f"horizon_{horizon}_net_return"] = net_return
    for tag in evidence.get("regime_tags", []) or []:
        custom_metrics[f"regime_{safe_metric_token(tag)}_net_return"] = net_return


def normalized_action_counts(act_counts) -> list[float]:
    if act_counts is None:
        act_counts = []
    if isinstance(act_counts, dict):
        max_index = max(act_counts.keys(), default=-1)
        act_counts = [act_counts.get(index, 0) for index in range(max_index + 1)]
    return [float(value) for value in list(act_counts)]


def record_action_metrics(metrics, act_counts: list[float], steps: int) -> None:
    custom_metrics = metrics.custom_metrics
    for index, count in enumerate(act_counts):
        custom_metrics[f"act_{index}_frac"] = count / steps
    if len(act_counts) >= 3:
        custom_metrics["act_short_frac"] = act_counts[0] / steps
        custom_metrics["act_flat_frac"] = act_counts[1] / steps
        custom_metrics["act_long_frac"] = act_counts[2] / steps


def record_reward_component_metrics(metrics, rc_sums: Dict[str, Any], steps: int) -> None:
    for key in sorted(rc_sums):
        metrics.custom_metrics[f"rc_{key}"] = float(rc_sums[key]) / steps


def mean_from_totals(
    metrics: dict,
    name: str,
    *,
    required: bool = False,
) -> float | None:
    """Validate and reduce one native ``sum``/``count`` metric pair.

    Missing pairs are optional by default; partially present, non-numeric,
    boolean, non-finite, or non-positive-count pairs are invalid everywhere.
    """
    total_key = f"{name}__sum"
    count_key = f"{name}__count"
    total = metrics.get(total_key)
    count = metrics.get(count_key)
    if total is None and count is None:
        if required:
            raise KeyError(f"Native evaluation did not emit {name!r} totals")
        return None
    if (
        isinstance(total, bool)
        or not isinstance(total, Real)
        or isinstance(count, bool)
        or not isinstance(count, Real)
    ):
        raise ValueError(f"Native metric {name!r} requires numeric sum and count")
    total = float(total)
    count = float(count)
    if not math.isfinite(total) or not math.isfinite(count) or count <= 0.0:
        raise ValueError(f"Native metric {name!r} has invalid sum or count")
    mean = total / count
    if not math.isfinite(mean):
        raise ValueError(f"Native metric {name!r} has a non-finite mean")
    return mean


def pool_aggregate_means(pool) -> tuple[list[float], list[float]]:
    all_returns = []
    all_successes = []
    for stats in pool._stats.values():
        if stats["count"] <= 0:
            continue
        all_returns.append(stats["mean_return"])
        all_successes.append(stats["mean_success"])
    return all_returns, all_successes


__all__ = [
    "normalized_action_counts",
    "mean_from_totals",
    "pool_aggregate_means",
    "record_action_metrics",
    "record_episode_evidence",
    "record_reward_component_metrics",
    "safe_metric_token",
]
