import random

from itrader.rl_lib.tasks.rewards import RewardEngine, register


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
    assert "dd_mode" not in params
