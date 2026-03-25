from itrader.rl_lib.tasks.spec import TaskSpec
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
    assert "bar_resolution" not in task
    assert "regime_tags" not in task
    assert "difficulty_target" not in task
    assert "metadata" not in task

    spec = TaskSpec.from_dict(task)
    assert spec.bar_resolution is None
    assert spec.regime_tags == []
    assert spec.difficulty_target == {}
    assert spec.metadata == {}


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


def test_validation_accepts_gpt_style_aliases_and_normalizes_to_canonical_shape():
    raw = {
        "task_family": "01",
        "price_id": "AAPL",
        "horizon_bars": 12,
        "reward_config": {
            "name": "composed_linear",
            "params": {"pnl_w": 2.0, "turnover_w": 0.02},
        },
        "bar_resolution": "5m",
        "regime_tags": "trend",
        "difficulty_target": {
            "p_solve_min": 0.80,
            "p_solve_max": 0.55,
        },
        "metadata": {"source": "gpt54"},
        "t_embed": [1, "2.5", 3],
    }
    task = validate_and_clamp_task(
        raw,
        {
            "window_size": 30,
            "n_features": 5,
            "t_embed_dim": 32,
            "action_dim": 3,
            "min_horizon": 8,
            "max_horizon": 64,
        },
        fallback_seed=999,
    )
    assert task["family_id"] == "01"
    assert task["horizon"] == 12
    assert task["reward_id"] == "composed_linear"
    assert task["reward_params"]["pnl_w"] == 2.0
    assert task["reward_params"]["turnover_w"] == 0.01
    assert task["bar_resolution"] == "5m"
    assert task["regime_tags"] == ["trend"]
    assert task["difficulty_target"] == {"p_solve_min": 0.55, "p_solve_max": 0.8}
    assert task["metadata"] == {"source": "gpt54"}
    assert task["t_embed"] == [1.0, 2.5, 3.0]
