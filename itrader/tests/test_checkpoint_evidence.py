import pytest

from itrader.scripts.experiments.analyze_checkpoint_evidence import (
    analyze_episode_evidence,
)


def episode(start: int, gross: float, *, turnover: float, actions: list[float]) -> dict:
    return {
        "symbol": "MSFT",
        "start_idx": start,
        "end_idx": start + 96,
        "horizon": 96,
        "task_seed": start,
        "gross_return": gross,
        "turnover": turnover,
        "net_return": gross - turnover * 1e-4,
        "action_fractions": actions,
    }


def test_checkpoint_evidence_combines_cost_blocks_and_regime_behavior():
    candidate = [
        episode(100, 0.01, turnover=2.0, actions=[0.1, 0.2, 0.7]),
        episode(5100, 0.02, turnover=3.0, actions=[0.8, 0.1, 0.1]),
    ]
    long_only = [
        episode(100, 0.02, turnover=2.0, actions=[0.0, 0.0, 1.0]),
        episode(5100, -0.02, turnover=2.0, actions=[0.0, 0.0, 1.0]),
    ]
    flat = [
        episode(100, 0.0, turnover=0.0, actions=[0.0, 1.0, 0.0]),
        episode(5100, 0.0, turnover=0.0, actions=[0.0, 1.0, 0.0]),
    ]

    result = analyze_episode_evidence(
        candidate,
        {"long_only": long_only, "flat": flat},
        costs_bps=(0.5, 1.0, 2.0),
        sideways_abs_return=0.001,
        block_rows=5000,
        block_draws=200,
        block_seed=7,
    )

    assert result["candidate"]["episode_count"] == 2
    assert result["cost_sensitivity"]["2.0"] == pytest.approx(0.0145)
    assert result["regime_behavior"]["regimes"]["bullish"]["long_action_fraction"] == pytest.approx(0.7)
    assert result["regime_behavior"]["regimes"]["bearish"]["short_action_fraction"] == pytest.approx(0.8)
    assert set(result["paired_block_intervals"]) == {"flat", "long_only"}
