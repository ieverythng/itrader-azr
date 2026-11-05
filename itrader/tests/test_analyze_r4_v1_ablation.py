from __future__ import annotations

import pytest

from itrader.scripts.experiments.analyze_r4_v1_ablation import (
    _cell_artifact,
    _load_controls,
    evaluate_absolute_economic_gate,
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
