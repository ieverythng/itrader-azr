from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from itrader.rl_lib.evaluation.policy_baselines import (
    ControlPolicy,
    FrozenCheckpointPolicy,
    summarize_control_episodes,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    (("short", 0), ("flat", 1), ("long", 2)),
)
def test_fixed_control_policies_emit_the_expected_action(name: str, expected: int):
    policy = ControlPolicy(name=name, seed=123)

    assert policy.action(np.zeros(8, dtype=np.float32)) == expected


def test_random_control_is_reproducible_but_not_an_untrained_policy():
    left = ControlPolicy(name="random", seed=321)
    right = ControlPolicy(name="random", seed=321)

    left_actions = [left.action(np.zeros(4)) for _ in range(20)]
    right_actions = [right.action(np.zeros(4)) for _ in range(20)]

    assert left_actions == right_actions
    assert len(set(left_actions)) > 1


def test_frozen_checkpoint_policy_requests_deterministic_actions():
    class CheckpointAlgorithm:
        def compute_single_action(self, observation, *, explore):
            assert np.array_equal(observation, np.array([1.0, 2.0]))
            assert explore is False
            return 2

    policy = FrozenCheckpointPolicy(CheckpointAlgorithm())

    assert policy.action(np.array([1.0, 2.0])) == 2


def test_checkpoint_replay_uses_hashed_frozen_validation_panel(tmp_path):
    from itrader.rl_lib.evaluation.run_checkpoint_replay import load_validation_env

    dataset = tmp_path / "ohlcv.parquet"
    dataset.write_bytes(b"fixed dataset")
    resolved = tmp_path / "resolved_config.json"
    resolved.write_text(json.dumps({
        "dataset_sha256": hashlib.sha256(b"fixed dataset").hexdigest(),
        "config": {"evaluation_config": {"env_config": {
            "df_path": str(dataset), "data_split": "validation",
            "task_sampler": {"fixed_tasks": [{"seed": 10}]},
        }}},
    }), encoding="utf-8")

    config = load_validation_env(resolved)

    assert config["task_sampler"]["fixed_tasks"] == [{"seed": 10}]


def test_checkpoint_replay_refuses_a_final_test_panel(tmp_path):
    from itrader.rl_lib.evaluation.run_checkpoint_replay import load_validation_env

    resolved = tmp_path / "resolved_config.json"
    resolved.write_text(json.dumps({
        "config": {"evaluation_config": {"env_config": {
            "data_split": "test", "task_sampler": {"fixed_tasks": [{"seed": 10}]},
        }}},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="validation"):
        load_validation_env(resolved)


def test_control_summary_preserves_episode_rows_and_economic_means():
    episodes = [
        {
            "net_return": 0.01,
            "gross_return": 0.012,
            "net_sharpe_per_sqrt_bar": 0.1,
            "turnover": 2.0,
            "optimization_reward": 10.0,
            "steps": 100,
            "action_fractions": [0.0, 1.0, 0.0],
        },
        {
            "net_return": -0.005,
            "gross_return": -0.003,
            "net_sharpe_per_sqrt_bar": -0.05,
            "turnover": 1.0,
            "optimization_reward": -5.0,
            "steps": 200,
            "action_fractions": [0.0, 1.0, 0.0],
        },
    ]

    summary = summarize_control_episodes(episodes)

    assert summary["evaluation/net_return_mean"] == pytest.approx(0.0025)
    assert summary["evaluation/gross_return_mean"] == pytest.approx(0.0045)
    assert summary["evaluation/net_sharpe_per_sqrt_bar_mean"] == pytest.approx(0.025)
    assert summary["evaluation/turnover_mean"] == pytest.approx(1.5)
    assert summary["episode_len_mean"] == pytest.approx(150.0)
    assert summary["custom_metrics/act_flat_frac_mean"] == pytest.approx(1.0)
    assert summary["evaluation/custom_metrics/act_flat_frac_mean"] == pytest.approx(1.0)
