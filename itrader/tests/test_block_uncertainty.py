import pytest

from itrader.rl_lib.evaluation.block_uncertainty import (
    cost_sensitivity,
    paired_anchor_interval,
    paired_block_interval,
    pooled_paired_anchor_interval,
    regime_conditioned_summary,
)


def test_anchor_bootstrap_keeps_crossed_horizons_in_one_market_cluster():
    baseline = []
    candidate = []
    for start_idx, delta in ((100, 0.01), (200, -0.01)):
        for horizon in (96, 200):
            row = {
                "symbol": "SPY",
                "start_idx": start_idx,
                "end_idx": start_idx + horizon,
                "horizon": horizon,
                "task_seed": start_idx,
                "net_return": 0.0,
                "net_sharpe_per_sqrt_bar": 0.0,
            }
            baseline.append(row)
            candidate.append({
                **row,
                "net_return": delta,
                "net_sharpe_per_sqrt_bar": delta * 10,
            })

    interval = paired_anchor_interval(
        candidate,
        baseline,
        metric="net_return",
        draws=100,
        seed=7,
    )
    pooled = pooled_paired_anchor_interval(
        {1: (candidate, baseline), 2: (candidate, baseline)},
        metric="net_sharpe_per_sqrt_bar",
        draws=100,
        seed=7,
    )

    assert interval["anchor_count"] == 2
    assert interval["tasks_per_anchor"] == 2
    assert interval["mean_delta"] == pytest.approx(0.0)
    assert pooled["training_seed_count"] == 2
    assert pooled["anchor_count_per_seed"] == 2


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


@pytest.mark.parametrize("costs", [(), (float("nan"),), (-0.5,)])
def test_cost_sensitivity_rejects_invalid_cost_scenarios(costs):
    with pytest.raises(ValueError, match="finite nonnegative cost scenarios"):
        cost_sensitivity([episode(100, 0.01)], costs)


@pytest.mark.parametrize("field", ["gross_return", "turnover"])
def test_cost_sensitivity_rejects_nonfinite_episode_economics(field):
    row = episode(100, 0.01, turnover=2.0)
    row[field] = float("nan")

    with pytest.raises(ValueError, match="finite gross return and turnover"):
        cost_sensitivity([row], (1.0,))


def test_regime_summary_uses_matched_long_only_gross_return_without_penalizing_direction():
    long_control = [
        episode(100, 0.01),
        episode(5100, -0.01),
        episode(10100, 0.0002),
    ]
    policy = [
        {**episode(100, 0.004), "action_fractions": [0.1, 0.2, 0.7]},
        {**episode(5100, 0.006), "action_fractions": [0.8, 0.1, 0.1]},
        {**episode(10100, 0.001), "action_fractions": [0.2, 0.6, 0.2]},
    ]

    result = regime_conditioned_summary(
        policy,
        long_control,
        sideways_abs_return=0.001,
    )

    assert result["regimes"]["bullish"]["long_action_fraction"] == pytest.approx(0.7)
    assert result["regimes"]["bearish"]["short_action_fraction"] == pytest.approx(0.8)
    assert result["regimes"]["sideways"]["flat_action_fraction"] == pytest.approx(0.6)
    assert result["regimes"]["bearish"]["policy_net_return_mean"] == pytest.approx(0.006)
