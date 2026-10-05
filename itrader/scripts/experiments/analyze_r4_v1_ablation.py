"""Freeze the paired terminal analysis for the blocking R4-v1 4x3 ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from itrader.rl_lib.evaluation.block_uncertainty import (
    paired_anchor_interval,
    pooled_paired_anchor_interval,
)
from itrader.rl_lib.evaluation.provenance import (
    canonical_json_sha256, evaluation_contract_sha256, is_sha256,
)
from itrader.rl_lib.evaluation.replay_evidence import episode_payload_sha256, verify_receipt


TERMINAL_METRICS = {
    "net_return": "evaluation/net_return_mean",
    "sharpe": "evaluation/net_sharpe_per_sqrt_bar_mean",
    "gross_return": "evaluation/gross_return_mean",
    "turnover": "evaluation/custom_metrics/economic_turnover_mean",
    "position_changes": "evaluation/custom_metrics/economic_position_changes_mean",
    "liquidation": "evaluation/custom_metrics/economic_liquidation_turnover_mean",
    "entropy": "entropy",
    "explained_variance": "explained_variance",
    "short_fraction": "evaluation/custom_metrics/act_short_frac_mean",
    "flat_fraction": "evaluation/custom_metrics/act_flat_frac_mean",
    "long_fraction": "evaluation/custom_metrics/act_long_frac_mean",
}
TRAINING_SEEDS = (20260924, 20260925, 20260926)
STATE_ARMS = ("masked", "position", "time", "both")
CANDIDATE_STATE_ARMS = STATE_ARMS[1:]
RELATIVE_METRICS = ("net_return", "net_sharpe_per_sqrt_bar")


def matched_evidence_gate(cells: list[dict], controls: list[dict]) -> dict:
    """Missing identities are ineligible, not permission to compare anything."""
    errors = []
    protected = ("dataset_sha256", "evaluation_panel_sha256", "source_tree_sha256",
                 "evaluation_contract_sha256", "runtime_sha256")
    records = cells + controls
    for key in protected:
        values = [row.get("provenance", {}).get(key) for row in records]
        if not values or not all(is_sha256(value) for value in values) or len(set(values)) != 1:
            errors.append(f"missing_or_mismatched:{key}")
    for seed in {cell["training_seed"] for cell in cells}:
        values = [cell["provenance"].get("attribution_config_sha256")
                  for cell in cells if cell["training_seed"] == seed]
        if not all(is_sha256(value) for value in values) or len(set(values)) != 1:
            errors.append(f"undeclared_config_difference:seed={seed}")
    if not cells or not controls:
        errors.append("missing_candidates_or_controls")
    return {"passes": not errors, "reasons": errors}


def _attribution_config_hash(config: dict, dataset_hash: str) -> str:
    cfg = deepcopy(config)
    for key in ("source_provenance", "stable_metrics_root", "pbt_experiment_name"):
        cfg.pop(key, None)
    for env in (cfg.get("env_config", {}), cfg.get("evaluation_config", {}).get("env_config", {})):
        env["df_path"] = f"sha256:{dataset_hash}"
        env.pop("solver_state_mode", None)
        env.get("semantic_observation_contract", {}).pop("solver_state_mode", None)
    model_cfg = cfg.get("model", {}).get("custom_model_config", {})
    model_cfg.pop("solver_state_mode", None)
    model_cfg.get("semantic_observation_contract", {}).pop("solver_state_mode", None)
    return canonical_json_sha256(cfg)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cell_artifact(
    root: Path,
    *,
    training_seed: int,
    arm: str,
    suffix: str,
) -> Path:
    """Resolve one evidence artifact from its stable seed/arm cell directory."""
    cell_dir = root / f"seed{training_seed}_{arm}"
    matches = sorted(path for path in cell_dir.glob(f"*{suffix}") if path.is_file())
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {suffix} artifact for seed={training_seed}, "
            f"arm={arm}; found {len(matches)} in {cell_dir}"
        )
    return matches[0]


def _float(row: dict, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Required metric is not finite: {key}={row[key]}")
    return value


def evaluate_absolute_economic_gate(
    arm_cells: list[dict],
    *,
    strongest_control_net: float,
    strongest_control_sharpe: float,
    max_turnover: float,
    max_flat_fraction: float,
) -> dict[str, bool | float]:
    """Apply the declared absolute promotion conditions to one solver-state arm."""
    if not arm_cells:
        raise ValueError("Absolute economic gate requires at least one cell")
    maximum_turnover = float(max_turnover)
    maximum_flat = float(max_flat_fraction)
    if not math.isfinite(maximum_turnover) or maximum_turnover < 0.0:
        raise ValueError("max_turnover must be finite and nonnegative")
    if not math.isfinite(maximum_flat) or not 0.0 <= maximum_flat <= 1.0:
        raise ValueError("max_flat_fraction must be between 0 and 1")
    nets = [float(cell["metrics"]["net_return"]) for cell in arm_cells]
    sharpes = [float(cell["metrics"]["sharpe"]) for cell in arm_cells]
    conditions = {
        "all_seeds_positive_net": all(value > 0.0 for value in nets),
        "all_seeds_positive_sharpe": all(value > 0.0 for value in sharpes),
        "mean_net_beats_strongest_control": float(np.mean(nets))
        > float(strongest_control_net),
        "mean_sharpe_beats_strongest_control": float(np.mean(sharpes))
        > float(strongest_control_sharpe),
        "all_seeds_turnover_within_limit": all(
            float(cell["metrics"]["turnover"]) <= maximum_turnover
            for cell in arm_cells
        ),
        "all_seeds_below_near_flat_ceiling": all(
            float(cell["metrics"]["flat_fraction"]) < maximum_flat
            for cell in arm_cells
        ),
        "all_terminal_replays_exact": all(
            cell["replay"]["checkpoint_parity"] is True for cell in arm_cells
        ),
    }
    return {
        **conditions,
        "max_turnover": maximum_turnover,
        "max_flat_fraction": maximum_flat,
        "passes": all(conditions.values()),
    }


def _relative_factor_summary(
    arm: str,
    cells: dict[tuple[int, str], dict],
    episode_rows: dict[tuple[int, str], list[dict]],
) -> tuple[dict[str, dict], dict[str, dict], dict]:
    per_seed: dict[str, dict] = {}
    pooled: dict[str, dict] = {}
    for metric in RELATIVE_METRICS:
        comparisons = {}
        for training_seed in TRAINING_SEEDS:
            candidate = episode_rows[(training_seed, arm)]
            baseline = episode_rows[(training_seed, "masked")]
            comparisons[training_seed] = (candidate, baseline)
            per_seed.setdefault(str(training_seed), {})[metric] = (
                paired_anchor_interval(
                    candidate,
                    baseline,
                    metric=metric,
                    draws=10_000,
                    seed=training_seed,
                )
            )
        pooled[metric] = pooled_paired_anchor_interval(
            comparisons,
            metric=metric,
            draws=10_000,
            seed=30260943,
        )

    net_effects = [
        cells[(seed, arm)]["metrics"]["net_return"]
        - cells[(seed, "masked")]["metrics"]["net_return"]
        for seed in TRAINING_SEEDS
    ]
    sharpe_effects = [
        cells[(seed, arm)]["metrics"]["sharpe"]
        - cells[(seed, "masked")]["metrics"]["sharpe"]
        for seed in TRAINING_SEEDS
    ]
    gate = {
        "paired_terminal_net_deltas": net_effects,
        "paired_terminal_sharpe_deltas": sharpe_effects,
        "net_directionally_consistent_positive": all(
            value > 0 for value in net_effects
        ),
        "sharpe_directionally_consistent_positive": all(
            value > 0 for value in sharpe_effects
        ),
        "net_pooled_interval_excludes_zero_positive": (
            pooled["net_return"]["ci_low"] > 0
        ),
        "sharpe_pooled_interval_excludes_zero_positive": (
            pooled["net_sharpe_per_sqrt_bar"]["ci_low"] > 0
        ),
    }
    gate["passes_net"] = (
        gate["net_directionally_consistent_positive"]
        and gate["net_pooled_interval_excludes_zero_positive"]
    )
    gate["passes_sharpe"] = (
        gate["sharpe_directionally_consistent_positive"]
        and gate["sharpe_pooled_interval_excludes_zero_positive"]
    )
    return per_seed, pooled, gate


def _load_controls(evidence_root: Path) -> list[dict]:
    controls = []
    for result_path in sorted(
        (evidence_root / "controls").glob("*/training_fn_*/baseline_result.json")
    ):
        result = _json(result_path)
        controls.append(
            {
                "experiment": result_path.parents[1].name,
                "policy_control": result.get("policy_control"),
                "solver_state_mode": result.get("solver_state_mode", "off"),
                "solver_state_residual_init": result.get(
                    "solver_state_residual_init", "random_v1"
                ),
                "net_return": _float(result, "evaluation/net_return_mean"),
                "sharpe": _float(
                    result,
                    "evaluation/net_sharpe_per_sqrt_bar_mean",
                ),
                "turnover": _float(result, "evaluation/turnover_mean"),
                "provenance": result.get("provenance", {}),
            }
        )
    if not controls:
        raise ValueError("Expected at least one frozen policy-control result")
    return controls


def _terminal_replay_matches(replay_path, episodes_path, episodes, stable_dir, resolved, terminal):
    """Bind consumed evidence to the declared terminal checkpoint, not a claim flag."""
    replay = _json(replay_path)
    index_path = stable_dir / "checkpoint_index.jsonl"
    if replay.get("receipt_version") != 1 or not index_path.is_file():
        return False
    verify_receipt(replay_path)
    indexed = [row for row in _jsonl(index_path)
               if row.get("pbt_progress_steps") == int(terminal["pbt_progress_steps"])
               and row.get("pbt_progress_updates") == int(terminal["pbt_progress_updates"])]
    if len(indexed) != 1:
        return False
    from itrader.rl_lib.evaluation.policy_baselines import summarize_control_episodes

    summary = summarize_control_episodes(episodes)
    economic_columns = {
        "evaluation/net_return_mean": "evaluation/net_return_mean",
        "evaluation/net_sharpe_per_sqrt_bar_mean": "evaluation/net_sharpe_per_sqrt_bar_mean",
        "evaluation/gross_return_mean": "evaluation/gross_return_mean",
        "evaluation/custom_metrics/economic_turnover_mean": "evaluation/turnover_mean",
        **{f"evaluation/custom_metrics/act_{action}_frac_mean":
           f"evaluation/custom_metrics/act_{action}_frac_mean" for action in ("short", "flat", "long")},
    }
    if not all(math.isclose(_float(terminal, column), summary[metric], rel_tol=0, abs_tol=1e-12)
               for column, metric in economic_columns.items()):
        return False
    checkpoint = indexed[0]
    evaluation = resolved["config"]["evaluation_config"]
    expected = {
        "dataset_sha256": resolved["dataset_sha256"],
        "source_panel_sha256": resolved["evaluation_panel_sha256"],
        "evaluation_panel_sha256": resolved["evaluation_panel_sha256"],
        "checkpoint_config_sha256": resolved["config_hash"],
        "effective_env_config_sha256": canonical_json_sha256(evaluation["env_config"]),
        "observation_schema": resolved["observation_schema"],
        "evaluation_seed": evaluation.get("seed"),
        "runtime": resolved.get("runtime"),
        "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
        "progress_steps": checkpoint.get("pbt_progress_steps"),
        "restore_generation": checkpoint.get("pbt_restore_generation"),
        "explore": False,
    }
    contract = replay.get("evaluation_contract", {})
    return (
        replay.get("checkpoint_parity") is True
        and replay.get("producer", {}).get("operation") == "episode_replay"
        and replay.get("checkpoint") == checkpoint.get("stable_checkpoint")
        and checkpoint.get("config_hash") == resolved["config_hash"]
        and is_sha256(checkpoint.get("checkpoint_sha256"))
        and all(key in contract and contract[key] == value for key, value in expected.items())
        and contract.get("explore") is False
        and contract.get("replay_source", {}).get("source_tree_sha256")
        == resolved.get("source_tree_fingerprint", {}).get("source_tree_sha256")
        and any(record.get("file") == episodes_path.name for record in replay["artifacts"])
        and all(is_sha256(row.get("action_reward_trace_sha256")) for row in episodes)
        and episode_payload_sha256(episodes) == checkpoint.get("evaluation_payload_sha256")
    )


def analyze(
    stable_root: Path,
    evidence_root: Path,
    *,
    max_turnover: float = 20.0,
    max_flat_fraction: float = 0.98,
    expected_updates: int = 12,
) -> dict:
    replay_root = evidence_root / "terminal_replay"
    diagnostics_root = evidence_root / "terminal_state_diagnostics"
    cells: dict[tuple[int, str], dict] = {}
    episode_rows: dict[tuple[int, str], list[dict]] = {}

    for stable_dir in sorted(path for path in stable_root.iterdir() if path.is_dir()):
        resolved = _json(stable_dir / "resolved_config.json")
        config = resolved["config"]
        training_seed = int(config["seed"])
        arm = str(config["env_config"]["solver_state_mode"])
        key = (training_seed, arm)
        if key in cells:
            raise ValueError(f"Duplicate cell: {key}")

        reports = _csv_rows(stable_dir / "report_rows.csv")
        if (
            len(reports) != int(expected_updates)
            or int(reports[-1]["pbt_progress_updates"]) != int(expected_updates)
        ):
            raise ValueError(
                f"Cell does not have exactly {expected_updates} reports: {stable_dir}"
            )
        terminal = reports[-1]
        replay_path = _cell_artifact(
                replay_root,
                training_seed=training_seed,
                arm=arm,
                suffix=".json",
            )
        replay = _json(replay_path)
        diagnostics = _json(
            _cell_artifact(
                diagnostics_root,
                training_seed=training_seed,
                arm=arm,
                suffix=".state_diagnostics.json",
            )
        )
        episodes_path = _cell_artifact(
                replay_root,
                training_seed=training_seed,
                arm=arm,
                suffix=".episodes.jsonl",
            )
        episodes = _jsonl(episodes_path)
        if len(episodes) != 32:
            raise ValueError(f"Expected 32 terminal replay episodes: {stable_dir}")
        verified_replay = _terminal_replay_matches(
            replay_path, episodes_path, episodes, stable_dir, resolved, terminal)

        metrics = {name: _float(terminal, column) for name, column in TERMINAL_METRICS.items()}
        final_three = [
            {
                "update": int(row["pbt_progress_updates"]),
                "net_return": _float(row, "evaluation/net_return_mean"),
                "sharpe": _float(row, "evaluation/net_sharpe_per_sqrt_bar_mean"),
                "turnover": _float(row, "evaluation/custom_metrics/economic_turnover_mean"),
            }
            for row in reports[-3:]
        ]
        cells[key] = {
            "training_seed": training_seed,
            "arm": arm,
            "trial": stable_dir.name,
            "terminal_steps": int(terminal["pbt_progress_steps"]),
            "metrics": metrics,
            "final_three": final_three,
            "final_three_mean": {
                metric: float(np.mean([row[metric] for row in final_three]))
                for metric in ("net_return", "sharpe", "turnover")
            },
            "replay": {
                "checkpoint": replay["checkpoint"],
                "selected_score": replay["selected_score"],
                "replayed_score": replay["replayed_score"],
                "replay_delta": replay["replay_delta"],
                "checkpoint_parity": replay["checkpoint_parity"],
                "receipt_verified": verified_replay,
            },
            "diagnostics": diagnostics["diagnostics"],
            "counterfactual_state_sensitivity": diagnostics.get(
                "counterfactual_state_sensitivity"
            ),
            "provenance": {
                "observation_schema": resolved["observation_schema"],
                "dataset_sha256": resolved["dataset_sha256"],
                "evaluation_panel_sha256": resolved["evaluation_panel_sha256"],
                "source_tree_sha256": resolved["source_tree_fingerprint"]["source_tree_sha256"],
                "config_hash": resolved["config_hash"],
                "comparison_config_hash": resolved["comparison_config_hash"],
                "attribution_config_sha256": _attribution_config_hash(config, resolved["dataset_sha256"]),
                "evaluation_contract_sha256": evaluation_contract_sha256(
                    config["evaluation_config"]["env_config"], resolved["dataset_sha256"]),
                "runtime_sha256": canonical_json_sha256(resolved["runtime"]) if resolved.get("runtime") else None,
                "solver_state_residual_init": resolved["config"]
                .get("model", {})
                .get("custom_model_config", {})
                .get("solver_state_residual_init", "random_v1"),
            },
        }
        episode_rows[key] = episodes

    expected = {(seed, arm) for seed in TRAINING_SEEDS for arm in STATE_ARMS}
    if set(cells) != expected:
        raise ValueError(f"Cell matrix mismatch: {sorted(set(cells) ^ expected)}")

    provenance_fields = (
        "observation_schema",
        "dataset_sha256",
        "evaluation_panel_sha256",
        "source_tree_sha256",
        "solver_state_residual_init",
    )
    provenance_uniques = {
        field: sorted({json.dumps(cell["provenance"][field], sort_keys=True) for cell in cells.values()})
        for field in provenance_fields
    }

    per_seed_intervals: dict[str, dict] = {}
    pooled_intervals: dict[str, dict] = {}
    relative_gate: dict[str, dict] = {}
    for arm in CANDIDATE_STATE_ARMS:
        (
            per_seed_intervals[arm],
            pooled_intervals[arm],
            relative_gate[arm],
        ) = _relative_factor_summary(
            arm,
            cells,
            episode_rows,
        )

    controls = _load_controls(evidence_root)
    provenance_gate = matched_evidence_gate(list(cells.values()), controls)
    if not all(cell["replay"]["receipt_verified"] for cell in cells.values()):
        provenance_gate["passes"] = False
        provenance_gate["reasons"].append("unverified_or_mismatched_replay_receipt")
    strongest_control_net = max(control["net_return"] for control in controls)
    strongest_control_sharpe = max(control["sharpe"] for control in controls)

    arm_summaries = {}
    absolute_gate = {}
    for arm in STATE_ARMS:
        arm_cells = [cells[(seed, arm)] for seed in TRAINING_SEEDS]
        arm_summaries[arm] = {
            metric: float(np.mean([cell["metrics"][metric] for cell in arm_cells]))
            for metric in TERMINAL_METRICS
        }
        absolute_gate[arm] = evaluate_absolute_economic_gate(
            arm_cells,
            strongest_control_net=strongest_control_net,
            strongest_control_sharpe=strongest_control_sharpe,
            max_turnover=max_turnover,
            max_flat_fraction=max_flat_fraction,
        )

    return {
        "schema_version": 1,
        "expected_updates": int(expected_updates),
        "cell_count": len(cells),
        "provenance_unique_counts": {key: len(value) for key, value in provenance_uniques.items()},
        "provenance_values": provenance_uniques,
        "matched_evidence_gate": provenance_gate,
        "all_terminal_replays_exact": all(cell["replay"]["checkpoint_parity"] is True for cell in cells.values()),
        "cells": [cells[key] for key in sorted(cells)],
        "arm_terminal_means": arm_summaries,
        "per_seed_anchor_intervals": per_seed_intervals,
        "pooled_seed_anchor_intervals": pooled_intervals,
        "relative_factor_gate": relative_gate,
        "controls": controls,
        "strongest_control_net": strongest_control_net,
        "strongest_control_sharpe": strongest_control_sharpe,
        "absolute_economic_gate": absolute_gate,
        "promotion_passes": provenance_gate["passes"] and any(
            absolute_gate[arm]["passes"]
            and (relative_gate[arm]["passes_net"] or relative_gate[arm]["passes_sharpe"])
            for arm in CANDIDATE_STATE_ARMS
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--max-turnover", type=float, default=20.0)
    parser.add_argument("--max-flat-fraction", type=float, default=0.98)
    parser.add_argument("--expected-updates", type=int, default=12)
    args = parser.parse_args()
    evidence_root = args.evidence_root.resolve(strict=True)
    result = analyze(
        args.stable_root.resolve(strict=True),
        evidence_root,
        max_turnover=args.max_turnover,
        max_flat_fraction=args.max_flat_fraction,
        expected_updates=args.expected_updates,
    )
    output = evidence_root / "analysis.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "analysis": str(output),
        "promotion_passes": result["promotion_passes"],
        "arm_terminal_means": result["arm_terminal_means"],
        "relative_factor_gate": result["relative_factor_gate"],
        "absolute_economic_gate": result["absolute_economic_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
