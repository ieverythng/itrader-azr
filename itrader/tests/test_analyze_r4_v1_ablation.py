from __future__ import annotations

import pytest
from copy import deepcopy
import csv
import json

from itrader.scripts.experiments.analyze_r4_v1_ablation import (
    _cell_artifact,
    _load_controls,
    evaluate_absolute_economic_gate,
    matched_evidence_gate,
)


def _cell(*, net: float, sharpe: float, turnover: float, flat: float) -> dict:
    return {
        "metrics": {
            "net_return": net,
            "sharpe": sharpe,
            "turnover": turnover,
            "flat_fraction": flat,
        },
        "replay": {"checkpoint_parity": True},
    }


def test_r4_absolute_gate_rejects_near_flat_or_excessive_turnover():
    cells = [
        _cell(net=0.002, sharpe=0.02, turnover=4.0, flat=0.25),
        _cell(net=0.003, sharpe=0.03, turnover=21.0, flat=0.30),
        _cell(net=0.004, sharpe=0.04, turnover=3.0, flat=0.99),
    ]

    gate = evaluate_absolute_economic_gate(
        cells,
        strongest_control_net=0.001,
        strongest_control_sharpe=0.01,
        max_turnover=20.0,
        max_flat_fraction=0.98,
    )

    assert gate["all_seeds_turnover_within_limit"] is False
    assert gate["all_seeds_below_near_flat_ceiling"] is False
    assert gate["passes"] is False


def test_r4_absolute_gate_passes_only_when_every_declared_condition_passes():
    cells = [
        _cell(net=0.002, sharpe=0.02, turnover=4.0, flat=0.25),
        _cell(net=0.003, sharpe=0.03, turnover=5.0, flat=0.30),
        _cell(net=0.004, sharpe=0.04, turnover=3.0, flat=0.35),
    ]

    gate = evaluate_absolute_economic_gate(
        cells,
        strongest_control_net=0.001,
        strongest_control_sharpe=0.01,
        max_turnover=20.0,
        max_flat_fraction=0.98,
    )

    assert gate["all_seeds_turnover_within_limit"] is True
    assert gate["all_seeds_below_near_flat_ceiling"] is True
    assert gate["passes"] is True


def test_cell_artifact_resolves_the_seed_arm_directory(tmp_path):
    cell_dir = tmp_path / "seed20260925_both"
    cell_dir.mkdir()
    artifact = cell_dir / "trial.state_diagnostics.json"
    artifact.write_text("{}", encoding="utf-8")

    assert _cell_artifact(
        tmp_path,
        training_seed=20260925,
        arm="both",
        suffix=".state_diagnostics.json",
    ) == artifact


def test_cell_artifact_rejects_missing_or_ambiguous_evidence(tmp_path):
    cell_dir = tmp_path / "seed20260925_both"
    cell_dir.mkdir()

    try:
        _cell_artifact(
            tmp_path,
            training_seed=20260925,
            arm="both",
            suffix=".json",
        )
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("missing evidence must fail closed")

    (cell_dir / "one.json").write_text("{}", encoding="utf-8")
    (cell_dir / "two.json").write_text("{}", encoding="utf-8")
    try:
        _cell_artifact(
            tmp_path,
            training_seed=20260925,
            arm="both",
            suffix=".json",
        )
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("ambiguous evidence must fail closed")


def test_r4_analysis_requires_a_frozen_control_result(tmp_path):
    with pytest.raises(ValueError, match="at least one frozen policy-control"):
        _load_controls(tmp_path)


@pytest.mark.parametrize("receipt", ["false", "true", 1, 0, None, {}, []])
def test_replay_gate_requires_a_literal_boolean_true(receipt):
    cell = _cell(net=0.01, sharpe=0.1, turnover=1, flat=0.1)
    cell["replay"]["checkpoint_parity"] = receipt
    gate = evaluate_absolute_economic_gate(
        [cell], strongest_control_net=0, strongest_control_sharpe=0,
        max_turnover=20, max_flat_fraction=0.98,
    )
    assert gate["passes"] is False


def test_matched_evidence_requires_all_protected_identities_and_controls():
    provenance = {key: "a" * 64 for key in (
        "dataset_sha256", "evaluation_panel_sha256", "source_tree_sha256",
        "evaluation_contract_sha256", "attribution_config_sha256", "runtime_sha256",
    )}
    cells = [{"training_seed": 1, "provenance": deepcopy(provenance)} for _ in range(2)]
    controls = [{"provenance": deepcopy(provenance)}]
    assert matched_evidence_gate(cells, controls)["passes"] is True
    for field in provenance:
        bad = deepcopy(cells)
        bad[1]["provenance"][field] = "b" * 64
        assert matched_evidence_gate(bad, controls)["passes"] is False, field
    for field in ("dataset_sha256", "evaluation_panel_sha256", "source_tree_sha256", "evaluation_contract_sha256", "runtime_sha256"):
        for value in (None, "0" * 64, "b" * 64):
            bad = deepcopy(controls)
            bad[0]["provenance"][field] = value
            assert matched_evidence_gate(cells, bad)["passes"] is False, field


def test_artifact_analysis_cannot_promote_a_mismatched_dataset(tmp_path):
    """Exercise persisted 4x3 evidence through the actual analyzer/gate wiring."""
    from itrader.scripts.experiments.analyze_r4_v1_ablation import (
        analyze, TERMINAL_METRICS, TRAINING_SEEDS, STATE_ARMS,
    )
    from itrader.rl_lib.evaluation.provenance import canonical_json_sha256, evaluation_contract_sha256
    from itrader.rl_lib.evaluation.replay_evidence import ReplayInvocation, episode_payload_sha256

    stable = tmp_path / "stable"
    evidence = tmp_path / "evidence"
    env = {"df_path": "bars.parquet", "data_split": "validation"}
    runtime = {"ray": "2.58.0"}
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    for seed in TRAINING_SEEDS:
        for arm in STATE_ARMS:
            name = f"seed{seed}_{arm}"
            directory = stable / name
            config = {"seed": seed, "env_config": {**env, "solver_state_mode": arm},
                      "evaluation_config": {"env_config": dict(env)}}
            write(directory / "resolved_config.json", {
                "config": config, "runtime": runtime, "observation_schema": {"state": 2},
                "dataset_sha256": "a"*64, "evaluation_panel_sha256": "b"*64,
                "source_tree_fingerprint": {"source_tree_sha256": "c"*64},
                "config_hash": canonical_json_sha256(config), "comparison_config_hash": "d"*64,
            })
            row = {column: 0.01 if arm == "masked" else 0.02 for column in TERMINAL_METRICS.values()}
            row.update(pbt_progress_updates=1, pbt_progress_steps=4800)
            with (directory / "report_rows.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
            replay_report = {
                "checkpoint": name, "selected_score": 1, "replayed_score": 1,
                "replay_delta": 0, "checkpoint_parity": True,
            }
            invocation = ReplayInvocation(evidence / "terminal_replay" / name / "trial.json", operation="episode_replay")
            episodes = [{"symbol": "SIM", "start_idx": anchor * 1000,
                         "end_idx": anchor * 1000 + horizon, "horizon": horizon, "task_seed": anchor,
                         "action_reward_trace_sha256": "f" * 64,
                         "steps": horizon, "optimization_reward": 0.0,
                         "gross_return": row["evaluation/gross_return_mean"],
                         "turnover": row["evaluation/custom_metrics/economic_turnover_mean"],
                         "action_fractions": [row["evaluation/custom_metrics/act_short_frac_mean"],
                                              row["evaluation/custom_metrics/act_flat_frac_mean"],
                                              row["evaluation/custom_metrics/act_long_frac_mean"]],
                         "net_return": row["evaluation/net_return_mean"],
                         "net_sharpe_per_sqrt_bar": row["evaluation/net_sharpe_per_sqrt_bar_mean"]}
                        for anchor in range(8) for horizon in (96, 128, 160, 200)]
            (evidence / "terminal_replay" / name / "trial.episodes.jsonl").write_text(
                "".join(json.dumps(episode) + "\n" for episode in episodes))
            indexed = {"stable_checkpoint": name, "pbt_progress_steps": 4800,
                       "pbt_progress_updates": 1, "pbt_restore_generation": 0,
                       "checkpoint_sha256": "e"*64, "config_hash": canonical_json_sha256(config),
                       "evaluation_payload_sha256": episode_payload_sha256(episodes)}
            (directory / "checkpoint_index.jsonl").write_text(json.dumps(indexed) + "\n")
            invocation.complete(replay_report, contract={
                "dataset_sha256": "a"*64, "evaluation_panel_sha256": "b"*64,
                "checkpoint_config_sha256": canonical_json_sha256(config),
                "source_panel_sha256": "b"*64, "runtime": runtime,
                "replay_source": {"source_tree_sha256": "c"*64},
                "effective_env_config_sha256": canonical_json_sha256(env),
                "observation_schema": {"state": 2}, "evaluation_seed": None,
                "checkpoint_sha256": "e"*64, "progress_steps": 4800,
                "restore_generation": 0, "explore": False,
            }, artifacts=[evidence / "terminal_replay" / name / "trial.episodes.jsonl"])
            write(evidence / "terminal_state_diagnostics" / name / "trial.state_diagnostics.json", {"diagnostics": {}})
    control = evidence / "controls/flat/training_fn_flat/baseline_result.json"
    write(control, {"policy_control": "flat", "evaluation/net_return_mean": 0,
                    "evaluation/net_sharpe_per_sqrt_bar_mean": 0, "evaluation/turnover_mean": 0,
                    "provenance": {"dataset_sha256": "a"*64, "evaluation_panel_sha256": "b"*64,
                                   "source_tree_sha256": "c"*64,
                                   "runtime_sha256": canonical_json_sha256(runtime),
                                   "evaluation_contract_sha256": evaluation_contract_sha256(env, "a"*64)}})
    valid = analyze(stable, evidence, expected_updates=1)
    assert valid["matched_evidence_gate"]["passes"] is True
    assert valid["promotion_passes"] is True
    receipt_path = evidence / "terminal_replay" / f"seed{TRAINING_SEEDS[0]}_both/trial.json"
    original = json.loads(receipt_path.read_text())
    altered_receipt = deepcopy(original)
    altered_receipt["evaluation_contract"].update({
        "runtime": {"ray": "other"}, "progress_steps": 12,
        "replay_source": {"source_tree_sha256": "e"*64},
        "effective_env_config_sha256": "e"*64, "evaluation_seed": 99,
    })
    altered_receipt["evaluation_contract_sha256"] = canonical_json_sha256(altered_receipt["evaluation_contract"])
    write(receipt_path, altered_receipt)
    assert analyze(stable, evidence, expected_updates=1)["promotion_passes"] is False
    write(receipt_path, original)
    report_path = stable / f"seed{TRAINING_SEEDS[0]}_both/report_rows.csv"
    report_before = report_path.read_text()
    with report_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["evaluation/net_return_mean"] = "0.9"
    with report_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    assert analyze(stable, evidence, expected_updates=1)["promotion_passes"] is False
    report_path.write_text(report_before)
    path = stable / f"seed{TRAINING_SEEDS[0]}_both/resolved_config.json"
    altered = json.loads(path.read_text()); altered["dataset_sha256"] = "e"*64; write(path, altered)
    invalid = analyze(stable, evidence, expected_updates=1)
    assert invalid["promotion_passes"] is False
    assert "missing_or_mismatched:dataset_sha256" in invalid["matched_evidence_gate"]["reasons"]
