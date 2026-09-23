"""Combine replay receipts into cost, block, and regime diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from itrader.rl_lib.evaluation.block_uncertainty import (
    cost_sensitivity,
    paired_block_interval,
    regime_conditioned_summary,
)


def _summary(episodes: list[dict]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("Expected at least one episode")
    actions = np.asarray([row["action_fractions"] for row in episodes], dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 3 or not np.all(np.isfinite(actions)):
        raise ValueError("Expected three finite action fractions per episode")
    action_means = np.mean(actions, axis=0)
    return {
        "episode_count": len(episodes),
        "gross_return_mean": float(np.mean([row["gross_return"] for row in episodes])),
        "net_return_mean": float(np.mean([row["net_return"] for row in episodes])),
        "net_sharpe_per_sqrt_bar_mean": float(
            np.mean([row["net_sharpe_per_sqrt_bar"] for row in episodes])
        ) if all("net_sharpe_per_sqrt_bar" in row for row in episodes) else None,
        "turnover_mean": float(np.mean([row["turnover"] for row in episodes])),
        "short_action_fraction": float(action_means[0]),
        "flat_action_fraction": float(action_means[1]),
        "long_action_fraction": float(action_means[2]),
    }


def analyze_episode_evidence(
    candidate: list[dict],
    controls: dict[str, list[dict]],
    *,
    costs_bps: tuple[float, ...],
    sideways_abs_return: float,
    block_rows: int,
    block_draws: int,
    block_seed: int,
) -> dict[str, Any]:
    if "long_only" not in controls:
        raise ValueError("Regime analysis requires matched long_only episodes")
    return {
        "schema_version": 1,
        "scope": "exploratory_frozen_selection_panel_not_selection_adjusted",
        "candidate": _summary(candidate),
        "controls": {name: _summary(rows) for name, rows in sorted(controls.items())},
        "cost_sensitivity": cost_sensitivity(candidate, costs_bps),
        "paired_block_intervals": {
            name: paired_block_interval(
                candidate,
                rows,
                block_rows=block_rows,
                draws=block_draws,
                seed=block_seed,
            )
            for name, rows in sorted(controls.items())
        },
        "regime_behavior": regime_conditioned_summary(
            candidate,
            controls["long_only"],
            sideways_abs_return=sideways_abs_return,
        ),
    }


def _load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"No episode rows in {path}")
    return rows


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("controls must use NAME=PATH")
    return name, Path(raw_path)


def _write_plots(report: dict[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = report["regime_behavior"]["regimes"]
    labels = [name for name in ("bullish", "bearish", "sideways") if name in regimes]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for action, color in (("short", "#c44e52"), ("flat", "#8c8c8c"), ("long", "#55a868")):
        values = np.asarray([regimes[name][f"{action}_action_fraction"] for name in labels])
        ax.bar(x, values, bottom=bottom, label=action, color=color)
        bottom += values
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean action fraction")
    ax.set_title("Checkpoint behavior by ex-post validation regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "regime_action_fractions.png", dpi=220)
    plt.close(fig)

    width = 0.36
    market_returns = np.asarray(
        [regimes[name]["market_gross_return_mean"] for name in labels]
    ) * 100.0
    policy_returns = np.asarray(
        [regimes[name]["policy_net_return_mean"] for name in labels]
    ) * 100.0
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, market_returns, width, label="market gross", color="#4c72b0")
    ax.bar(x + width / 2, policy_returns, width, label="policy net at 1 bp", color="#55a868")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean episode return (%)")
    ax.set_title("Checkpoint return by ex-post validation regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "regime_returns.png", dpi=220)
    plt.close(fig)

    costs = report["cost_sensitivity"]
    cost_x = np.asarray([float(value) for value in costs])
    cost_y = np.asarray(list(costs.values())) * 100.0
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(cost_x, cost_y, marker="o", color="#4c72b0")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Round-trip evidence cost (bp per turnover unit)")
    ax.set_ylabel("Mean net return (%)")
    ax.set_title("Checkpoint cost sensitivity")
    fig.tight_layout()
    fig.savefig(out_dir / "cost_sensitivity.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-episodes", type=Path, required=True)
    parser.add_argument("--control", action="append", type=_named_path, default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--costs-bps", default="0.5,1,2")
    parser.add_argument("--sideways-abs-return", type=float, default=0.0002)
    parser.add_argument("--block-rows", type=int, default=5000)
    parser.add_argument("--block-draws", type=int, default=10000)
    parser.add_argument("--block-seed", type=int, default=20260919)
    args = parser.parse_args()

    controls = {name: _load_jsonl(path.resolve(strict=True)) for name, path in args.control}
    report = analyze_episode_evidence(
        _load_jsonl(args.candidate_episodes.resolve(strict=True)),
        controls,
        costs_bps=tuple(float(value) for value in args.costs_bps.split(",")),
        sideways_abs_return=args.sideways_abs_return,
        block_rows=args.block_rows,
        block_draws=args.block_draws,
        block_seed=args.block_seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoint_evidence.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_plots(report, args.out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
