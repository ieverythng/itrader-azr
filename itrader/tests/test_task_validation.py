from itrader.rl_lib.tasks.spec import TaskSpec
from itrader.rl_lib.tasks.samplers import TaskSampler00
from itrader.rl_lib.tasks.validation import validate_and_clamp_task


def test_validation_populates_required_defaults():
    task = validate_and_clamp_task(
        {},
        {"window_size": 30, "n_features": 5, "t_embed_dim": 32, "action_dim": 3},
        fallback_seed=111,
    )
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
    task = validate_and_clamp_task(
        raw,
        {"window_size": 30, "n_features": 5, "t_embed_dim": 32, "action_dim": 3},
        fallback_seed=123,
    )
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


def test_validation_uses_env_bar_resolution_when_provided():
    task = validate_and_clamp_task(
        {"family_id": "00", "price_id": "SIM"},
        {
            "window_size": 30,
            "n_features": 5,
            "t_embed_dim": 32,
            "action_dim": 3,
            "bar_resolution": "1m",
        },
        fallback_seed=77,
    )
    assert task["bar_resolution"] == "1m"
    assert TaskSpec.from_dict(task).bar_resolution == "1m"


def test_validation_env_bar_resolution_overrides_model_output():
    task = validate_and_clamp_task(
        {"family_id": "00", "price_id": "SIM", "bar_resolution": "1h"},
        {
            "window_size": 30,
            "n_features": 5,
            "t_embed_dim": 32,
            "action_dim": 3,
            "bar_resolution": "1m",
        },
        fallback_seed=78,
    )
    assert task["bar_resolution"] == "1m"


def test_validation_preserves_unknown_reward_params_for_future_reward_families():
    task = validate_and_clamp_task(
        {
            "family_id": "02",
            "reward_id": "future_family_reward",
            "reward_params": {"alpha": 0.7, "beta": "keep-me"},
        },
        {
            "window_size": 30,
            "n_features": 5,
            "t_embed_dim": 32,
            "action_dim": 3,
        },
        fallback_seed=17,
    )
    assert task["reward_params"] == {"alpha": 0.7, "beta": "keep-me"}


def test_task_sampler_uses_reward_specific_params_for_sampled_reward():
    sampler = TaskSampler00(
        {
            "seed": 9,
            "window_size": 30,
            "n_features": 5,
            "t_embed_dim": 32,
            "action_dim": 3,
            "reward_ids": ["pnl_sharpe"],
        }
    )
    task = sampler.sample()
    assert task.reward_id == "pnl_sharpe"
    assert "dd_w" in task.reward_params
    assert "turnover_w" in task.reward_params
    assert task.reward_params["dd_mode"] == "delta"
    assert task.reward_params["sharpe_window"] >= task.reward_params["sharpe_min_samples"]


def test_reward_choice_does_not_change_sampled_task_seed_sequence():
    common = {
        "seed": 91,
        "window_size": 30,
        "n_features": 5,
        "t_embed_dim": 32,
        "action_dim": 3,
        "horizons": [96, 128, 160],
    }
    pnl_sampler = TaskSampler00({**common, "reward_ids": ["pnl_linear"]})
    sharpe_sampler = TaskSampler00({**common, "reward_ids": ["pnl_sharpe"]})

    pnl_tasks = [pnl_sampler.sample() for _ in range(5)]
    sharpe_tasks = [sharpe_sampler.sample() for _ in range(5)]

    assert [task.horizon for task in pnl_tasks] == [task.horizon for task in sharpe_tasks]
    assert [task.seed for task in pnl_tasks] == [task.seed for task in sharpe_tasks]


def test_validation_normalizes_numeric_family_ids():
    task = validate_and_clamp_task(
        {"family_id": 2, "reward_id": "pnl_linear"},
        {"window_size": 30, "n_features": 5, "t_embed_dim": 32, "action_dim": 3},
        fallback_seed=222,
    )
    assert task["family_id"] == "02"
