import random

import pytest

from itrader.rl_lib.tasks.rewards import RewardEngine, register


def test_net_return_reward_matches_one_bp_economic_arithmetic():
    params = {"cost_bps": 1.0, "reward_scale": 1000.0}
    steps = [
        {"pnl_delta": 0.0020, "turnover_t": 1.0, "position_t": 1.0, "terminal": False},
        {"pnl_delta": -0.0005, "turnover_t": 0.0, "position_t": 1.0, "terminal": True},
    ]

    rewards = [RewardEngine.compute("net_return", params, ctx)[0] for ctx in steps]

    # Gross 15 bp - entry 1 bp - terminal liquidation 1 bp = net 13 bp.
    assert sum(rewards) / params["reward_scale"] == pytest.approx(0.0013)


def test_net_return_reward_reports_cost_components_without_double_counting():
    reward, components = RewardEngine.compute(
        "net_return",
        {"cost_bps": 2.0, "reward_scale": 100.0},
        {
            "pnl_delta": 0.003,
            "turnover_t": 2.0,
            "position_t": -1.0,
            "terminal": True,
        },
    )

    assert components.keys() == {"pnl_delta", "turnover_cost", "liquidation_cost"}
    assert reward == pytest.approx(sum(components.values()))
    assert components["turnover_cost"] < 0.0
    assert components["liquidation_cost"] < 0.0


def test_pnl_linear_reward_components_signs():
    reward, comps = RewardEngine.compute(
        "pnl_linear",
        {
            "pnl_w": 1.0,
            "turnover_w": 0.01,
            "dd_mode": "delta",
            "dd_w": 0.2,
            "normalize": False,
        },
        {
            "pnl_delta": 0.5,
            "turnover_t": 1.0,
            "dd_inc": 0.3,
            "ret_unit": 1.0,
            "terminal": False,
        },
    )
    assert reward == comps["pnl_delta"] + comps["turnover"] + comps["dd"]
    assert comps["pnl_delta"] > 0.0
    assert comps["turnover"] <= 0.0
    assert comps["dd"] <= 0.0


def test_terminal_drawdown_penalty_applies_only_on_terminal():
    params = {"dd_mode": "terminal", "dd_T_weight": 0.5, "normalize": False}
    base_ctx = {"pnl_delta": 0.0, "turnover_t": 0.0, "max_dd": 2.0}

    r_non_terminal, _ = RewardEngine.compute("pnl_linear", params, {**base_ctx, "terminal": False})
    r_terminal, comps_terminal = RewardEngine.compute("pnl_linear", params, {**base_ctx, "terminal": True})

    assert r_terminal < r_non_terminal
    assert comps_terminal["dd"] < 0.0


def test_reward_engine_preserves_arbitrary_component_names_for_non_phase_a_rewards():
    @register("test_reward_generic_components")
    def _test_reward(params: dict, ctx: dict) -> tuple[float, dict]:
        return params.get("score", 0.0), {
            "utility_gain": ctx.get("utility_gain", 0.0),
            "risk_cost": ctx.get("risk_cost", 0.0),
        }

    reward, comps = RewardEngine.compute(
        "test_reward_generic_components",
        {"score": 1.5},
        {"utility_gain": 0.3, "risk_cost": -0.1},
    )
    assert reward == 1.5
    assert comps == {"utility_gain": 0.3, "risk_cost": -0.1}


def test_reward_engine_samples_reward_specific_params():
    params = RewardEngine.sample_params("pnl_sharpe", random.Random(7))
    assert params["pnl_w"] == 1.0
    assert "dd_w" in params
    assert "turnover_w" in params
    assert params["dd_mode"] == "delta"
    assert params["sharpe_window"] >= params["sharpe_min_samples"]


def test_pnl_sharpe_adds_a_bounded_incremental_sharpe_component():
    reward, comps = RewardEngine.compute(
        "pnl_sharpe",
        {
            "pnl_w": 1.0,
            "dd_mode": "delta",
            "dd_w": 0.0,
            "turnover_w": 0.0,
            "normalize": False,
            "w_sharpe": 0.5,
            "sharpe_window": 8,
            "sharpe_min_samples": 4,
            "sharpe_target": 1.0,
        },
        {
            "pnl_delta": 0.004,
            "turnover_t": 0.0,
            "dd_inc": 0.0,
            "ret_unit": 1.0,
            "terminal": False,
            "strategy_returns": (0.01, -0.005, 0.008, -0.002, 0.006, 0.004),
        },
    )

    assert comps["sharpe_delta"] > 0.0
    assert abs(comps["sharpe_delta"]) <= 1.0
    assert reward == sum(comps.values())


def test_pnl_sharpe_is_zero_during_warmup():
    _, comps = RewardEngine.compute(
        "pnl_sharpe",
        {
            "normalize": False,
            "w_sharpe": 0.5,
            "sharpe_min_samples": 4,
        },
        {
            "pnl_delta": 0.0,
            "strategy_returns": (0.01, 0.02, 0.03),
        },
    )

    assert comps["sharpe_delta"] == 0.0


def test_sharpe_only_excludes_pnl_and_penalty_components():
    reward, comps = RewardEngine.compute(
        "sharpe_only",
        {
            "w_sharpe": 1.0,
            "sharpe_window": 8,
            "sharpe_min_samples": 4,
            "sharpe_target": 1.0,
        },
        {
            "pnl_delta": 100.0,
            "turnover_t": 100.0,
            "dd_inc": 100.0,
            "strategy_returns": (0.01, -0.005, 0.008, -0.002, 0.006, 0.004),
        },
    )

    assert comps.keys() == {"sharpe_delta"}
    assert reward == comps["sharpe_delta"]
    assert reward > 0.0


def test_sharpe_only_sampled_params_do_not_include_pnl_weights():
    params = RewardEngine.sample_params("sharpe_only", random.Random(9))

    assert "pnl_w" not in params
    assert "dd_w" not in params
    assert "turnover_w" not in params
    assert params["sharpe_window"] >= params["sharpe_min_samples"]


def test_sharpe_params_preserve_explicit_values_during_normalization():
    params = RewardEngine.normalize_params(
        "sharpe_only",
        {
            "w_sharpe": 0.75,
            "sharpe_window": 64,
            "sharpe_min_samples": 16,
            "sharpe_target": 2.0,
        },
    )

    assert params == {
        "w_sharpe": 0.75,
        "sharpe_window": 64,
        "sharpe_min_samples": 16,
        "sharpe_target": 2.0,
    }
