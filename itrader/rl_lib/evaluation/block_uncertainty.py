"""Exploratory paired uncertainty for overlapping Phase A validation episodes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _matched_rows(policy: list[dict], control: list[dict]) -> list[tuple[dict, dict]]:
    def keyed(rows: list[dict]) -> dict[tuple, dict]:
        identities = [
            (
                row["symbol"], int(row["start_idx"]), int(row["end_idx"]),
                int(row["horizon"]), int(row["task_seed"]),
            )
            for row in rows
        ]
        if len(set(identities)) != len(rows):
            raise ValueError("Expected unique frozen episode identities")
        return dict(zip(identities, rows))

    left, right = keyed(policy), keyed(control)
    if not left or left.keys() != right.keys():
        raise ValueError("Policies must use identical frozen episode identities")
    return [(left[key], right[key]) for key in sorted(left)]


def paired_block_interval(
    policy: list[dict],
    control: list[dict],
    *,
    block_rows: int = 5000,
    draws: int = 10000,
    seed: int = 20260917,
) -> dict[str, Any]:
    """Resample chronological row blocks within each symbol, preserving pairs.

    This is a diagnostic on the *selection* panel, not an independent test or
    a selection-adjusted confidence interval for the winning checkpoint.
    """
    if block_rows <= 0 or draws <= 0:
        raise ValueError("block_rows and draws must be positive")
    groups: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    differences: list[float] = []
    for candidate, baseline in _matched_rows(policy, control):
        difference = float(candidate["net_return"]) - float(baseline["net_return"])
        if not np.isfinite(difference):
            raise ValueError("All paired returns must be finite")
        groups[str(candidate["symbol"])][int(candidate["start_idx"]) // block_rows].append(
            difference
        )
        differences.append(difference)
    block_count = sum(len(blocks) for blocks in groups.values())
    if block_count < 2:
        raise ValueError("At least two chronological blocks are needed")

    rng = np.random.default_rng(seed)
    replicates = np.empty(draws, dtype=float)
    per_symbol = [list(blocks.values()) for _, blocks in sorted(groups.items())]
    for draw in range(draws):
        sampled: list[float] = []
        for blocks in per_symbol:
            for index in rng.integers(len(blocks), size=len(blocks)):
                sampled.extend(blocks[int(index)])
        replicates[draw] = float(np.mean(sampled))
    low, high = np.quantile(replicates, (0.025, 0.975))
    return {
        "mean_delta": float(np.mean(differences)),
        "ci_low": float(low),
        "ci_high": float(high),
        "episode_count": len(differences),
        "effective_blocks": block_count,
        "symbols": len(groups),
        "block_rows": block_rows,
        "draws": draws,
        "seed": seed,
        "scope": "exploratory_selection_panel_not_selection_adjusted",
    }


def _paired_anchor_differences(
    policy: list[dict],
    control: list[dict],
    *,
    metric: str,
) -> list[list[float]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for candidate, baseline in _matched_rows(policy, control):
        difference = float(candidate[metric]) - float(baseline[metric])
        if not np.isfinite(difference):
            raise ValueError(f"All paired {metric} values must be finite")
        grouped[(str(candidate["symbol"]), int(candidate["start_idx"]))].append(
            difference
        )
    anchors = [grouped[key] for key in sorted(grouped)]
    if len(anchors) < 2:
        raise ValueError("At least two market anchors are needed")
    horizon_counts = {len(values) for values in anchors}
    if len(horizon_counts) != 1:
        raise ValueError("Every market anchor must carry the same crossed horizons")
    return anchors


def paired_anchor_interval(
    policy: list[dict],
    control: list[dict],
    *,
    metric: str,
    draws: int = 10_000,
    seed: int = 20260925,
) -> dict[str, Any]:
    """Bootstrap paired task deltas by market anchor, carrying all horizons."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    anchors = _paired_anchor_differences(policy, control, metric=metric)
    rng = np.random.default_rng(seed)
    replicates = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = [
            value
            for index in rng.integers(len(anchors), size=len(anchors))
            for value in anchors[int(index)]
        ]
        replicates[draw] = float(np.mean(sampled))
    observed = [value for anchor in anchors for value in anchor]
    low, high = np.quantile(replicates, (0.025, 0.975))
    return {
        "metric": metric,
        "mean_delta": float(np.mean(observed)),
        "ci_low": float(low),
        "ci_high": float(high),
        "episode_count": len(observed),
        "anchor_count": len(anchors),
        "tasks_per_anchor": len(anchors[0]),
        "draws": draws,
        "seed": seed,
        "scope": "paired_development_panel_anchor_bootstrap",
    }


def pooled_paired_anchor_interval(
    comparisons: dict[int, tuple[list[dict], list[dict]]],
    *,
    metric: str,
    draws: int = 10_000,
    seed: int = 20260925,
) -> dict[str, Any]:
    """Bootstrap training seeds, then crossed market anchors within each seed."""
    if draws <= 0 or not comparisons:
        raise ValueError("Expected positive draws and at least one training seed")
    seeds = sorted(comparisons)
    grouped = {
        training_seed: _paired_anchor_differences(*comparisons[training_seed], metric=metric)
        for training_seed in seeds
    }
    anchor_counts = {len(anchors) for anchors in grouped.values()}
    horizon_counts = {
        len(anchor)
        for anchors in grouped.values()
        for anchor in anchors
    }
    if len(anchor_counts) != 1 or len(horizon_counts) != 1:
        raise ValueError("Every seed must carry the same anchor-by-horizon panel")

    rng = np.random.default_rng(seed)
    replicates = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled: list[float] = []
        for seed_index in rng.integers(len(seeds), size=len(seeds)):
            anchors = grouped[seeds[int(seed_index)]]
            for anchor_index in rng.integers(len(anchors), size=len(anchors)):
                sampled.extend(anchors[int(anchor_index)])
        replicates[draw] = float(np.mean(sampled))
    observed = [
        value
        for training_seed in seeds
        for anchor in grouped[training_seed]
        for value in anchor
    ]
    low, high = np.quantile(replicates, (0.025, 0.975))
    return {
        "metric": metric,
        "mean_delta": float(np.mean(observed)),
        "ci_low": float(low),
        "ci_high": float(high),
        "episode_count": len(observed),
        "training_seed_count": len(seeds),
        "anchor_count_per_seed": next(iter(anchor_counts)),
        "tasks_per_anchor": next(iter(horizon_counts)),
        "draws": draws,
        "seed": seed,
        "scope": "seed_then_anchor_bootstrap_on_development_panel",
    }


def cost_sensitivity(episodes: list[dict], costs_bps: tuple[float, ...]) -> dict[str, float]:
    """Reprice arithmetic returns; other per-step metrics need a fresh replay."""
    costs = tuple(float(cost) for cost in costs_bps)
    if not episodes or not costs or any(
        not np.isfinite(cost) or cost < 0 for cost in costs
    ):
        raise ValueError("Expected episodes and finite nonnegative cost scenarios")
    economics = [
        (float(row["gross_return"]), float(row["turnover"]))
        for row in episodes
    ]
    if any(
        not np.isfinite(gross_return) or not np.isfinite(turnover)
        for gross_return, turnover in economics
    ):
        raise ValueError("Expected finite gross return and turnover")
    return {
        str(float(cost)): float(np.mean([
            gross_return - turnover * cost * 1e-4
            for gross_return, turnover in economics
        ]))
        for cost in costs
    }


def regime_conditioned_summary(
    policy: list[dict],
    long_control: list[dict],
    *,
    sideways_abs_return: float,
) -> dict[str, Any]:
    """Describe policy behavior by ex-post regimes on a matched selection panel.

    The long-only gross episode return supplies a transparent market-direction
    label. This is a diagnostic, not a promotion gate or a predictive regime
    detector.
    """
    if sideways_abs_return < 0 or not np.isfinite(sideways_abs_return):
        raise ValueError("sideways_abs_return must be finite and nonnegative")
    grouped: dict[str, list[tuple[dict, float]]] = defaultdict(list)
    for candidate, baseline in _matched_rows(policy, long_control):
        market_return = float(baseline["gross_return"])
        actions = np.asarray(candidate["action_fractions"], dtype=float)
        if not np.isfinite(market_return) or actions.shape != (3,) or not np.all(np.isfinite(actions)):
            raise ValueError("Regime inputs and three action fractions must be finite")
        if market_return > sideways_abs_return:
            regime = "bullish"
        elif market_return < -sideways_abs_return:
            regime = "bearish"
        else:
            regime = "sideways"
        grouped[regime].append((candidate, market_return))

    summaries: dict[str, dict[str, float | int]] = {}
    for regime in ("bullish", "bearish", "sideways"):
        rows = grouped.get(regime, [])
        if not rows:
            continue
        action_means = np.mean(
            [np.asarray(row["action_fractions"], dtype=float) for row, _ in rows],
            axis=0,
        )
        summaries[regime] = {
            "episode_count": len(rows),
            "market_gross_return_mean": float(np.mean([value for _, value in rows])),
            "policy_gross_return_mean": float(np.mean([row["gross_return"] for row, _ in rows])),
            "policy_net_return_mean": float(np.mean([row["net_return"] for row, _ in rows])),
            "short_action_fraction": float(action_means[0]),
            "flat_action_fraction": float(action_means[1]),
            "long_action_fraction": float(action_means[2]),
        }
    return {
        "sideways_abs_return": float(sideways_abs_return),
        "episode_count": sum(len(rows) for rows in grouped.values()),
        "regimes": summaries,
        "scope": "exploratory_ex_post_selection_panel",
    }
