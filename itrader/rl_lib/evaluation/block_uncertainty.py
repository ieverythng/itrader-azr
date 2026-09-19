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


def cost_sensitivity(episodes: list[dict], costs_bps: tuple[float, ...]) -> dict[str, float]:
    """Reprice arithmetic returns; other per-step metrics need a fresh replay."""
    if not episodes or any(cost < 0 for cost in costs_bps):
        raise ValueError("Expected episodes and nonnegative cost scenarios")
    return {
        str(float(cost)): float(np.mean([
            float(row["gross_return"]) - float(row["turnover"]) * cost * 1e-4
            for row in episodes
        ]))
        for cost in costs_bps
    }
