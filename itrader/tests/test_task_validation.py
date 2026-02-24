from itrader.rl_lib.tasks.validation import validate_and_clamp_task


def test_validation_populates_required_defaults():
    task = validate_and_clamp_task({}, {"window_size": 30, "n_features": 5, "t_embed_dim": 32, "action_dim": 3}, fallback_seed=111)
    assert task["family_id"] == "00"
    assert task["price_id"] == "SIM"
    assert task["reward_id"] == "pnl_linear"
    assert task["seed"] == 111
    assert task["obs_dim"] == 30 * 5 + 32
    assert task["action_dim"] == 3
    assert task["start_idx"] is None


def test_validation_clamps_out_of_range_values():
    raw = {
        "horizon": 99999,
        "start_idx": -10,
        "reward_params": {
            "turnover_w": 1.5,
            "dd_w": 5.0,
            "dd_mode": "invalid",
            "norm_mode": "invalid",
            "normalize": "true",
        },
    }
    task = validate_and_clamp_task(raw, {"window_size": 30, "n_features": 5, "t_embed_dim": 32, "action_dim": 3}, fallback_seed=123)
    assert task["horizon"] == 512
    assert task["start_idx"] is None
    assert task["reward_params"]["turnover_w"] == 0.01
    assert task["reward_params"]["dd_w"] == 0.2
    assert task["reward_params"]["dd_mode"] == "delta"
    assert task["reward_params"]["norm_mode"] == "all"
    assert task["reward_params"]["normalize"] is True
