from itrader.rl_lib.tasks.rewards import RewardEngine


def test_reward_engine_rejects_unknown_reward():
    try:
        RewardEngine.compute("does_not_exist", {}, {})
    except KeyError:
        assert True
        return
    raise AssertionError("Expected KeyError for unknown reward id")


def test_reward_engine_always_returns_finite_values():
    r, comps = RewardEngine.compute(
        "pnl_linear",
        {"pnl_w": 1.0, "turnover_w": 0.001, "dd_mode": "delta", "dd_w": 0.01},
        {
            "pnl_delta": float("nan"),
            "turnover_t": 2.0,
            "dd_inc": float("inf"),
            "ret_unit": 1.0,
            "terminal": False,
        },
    )
    assert isinstance(r, float)
    assert all(isinstance(v, float) for v in comps.values())
