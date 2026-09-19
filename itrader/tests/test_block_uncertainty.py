import pytest

from itrader.rl_lib.evaluation.block_uncertainty import (
    paired_block_interval,
    cost_sensitivity,
)


def episode(start: int, gross: float, turnover: float = 0.0) -> dict:
    return {
        "symbol": "MSFT", "start_idx": start, "end_idx": start + 96,
        "horizon": 96, "task_seed": start, "gross_return": gross,
        "turnover": turnover, "net_return": gross - turnover * 1e-4,
    }


def test_paired_blocks_preserve_matched_episode_differences():
    policy = [episode(100, 0.01), episode(5100, -0.01)]
    control = [episode(100, 0.0), episode(5100, 0.0)]

    result = paired_block_interval(policy, control, block_rows=5000, draws=500, seed=17)

    assert result["mean_delta"] == pytest.approx(0.0)
    assert result["effective_blocks"] == 2
    assert result["ci_low"] < 0.0 < result["ci_high"]


def test_paired_blocks_reject_nonidentical_evaluation_starts():
    with pytest.raises(ValueError, match="identical frozen episode identities"):
        paired_block_interval([episode(100, 0.01)], [episode(200, 0.0)])


def test_cost_sensitivity_uses_gross_return_and_total_turnover():
    result = cost_sensitivity([episode(100, 0.01, turnover=2.0)], (0.5, 1.0, 2.0))

    assert result == {"0.5": pytest.approx(0.0099),
                      "1.0": pytest.approx(0.0098),
                      "2.0": pytest.approx(0.0096)}
