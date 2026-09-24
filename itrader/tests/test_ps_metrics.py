from types import SimpleNamespace

import numpy as np

from itrader.rl_lib.callbacks.ps_metrics import PSMetrics


def test_ps_metrics_emits_dynamic_reward_component_metrics():
    metrics = PSMetrics()
    env = SimpleNamespace(
        _last_info={
            "ep_sums": {
                "steps": 2,
                "horizon": 16,
                "act_counts": [1, 0, 1],
                "rc_sums": {"utility_gain": 0.6, "risk_cost": -0.2},
            },
            "reward_comps": {"success": 1.0},
            "success": True,
            "task": {"family_id": "00"},
        }
    )
    episode = SimpleNamespace(total_reward=1.25, custom_metrics={})
    base_env = SimpleNamespace(envs=[env])

    metrics.on_episode_end(worker=None, base_env=base_env, policies=None, episode=episode)

    assert episode.custom_metrics["rc_utility_gain"] == 0.3
    assert episode.custom_metrics["rc_risk_cost"] == -0.1
    assert episode.custom_metrics["optimization_reward"] == 1.25
    assert "episode_return" not in episode.custom_metrics
    assert episode.custom_metrics["act_0_frac"] == 0.5
    assert episode.custom_metrics["act_2_frac"] == 0.5


def test_ps_metrics_accepts_numpy_action_counts():
    counts = PSMetrics._normalized_action_counts(np.array([1, 2, 3]))
    assert counts == [1.0, 2.0, 3.0]


def test_ps_metrics_uses_episode_terminal_evidence_without_private_env_lookup():
    metrics = PSMetrics()
    evidence = {
        "schema_version": 1,
        "optimization_reward": 1.25,
        "gross_return": 0.003,
        "cost_return": 0.001,
        "net_return": 0.002,
        "turnover": 10.0,
        "liquidation_turnover": 1.0,
        "max_drawdown": 0.004,
        "success": True,
        "family_id": "00",
        "horizon": 16,
        "regime_tags": [],
        "split": "validation",
    }
    episode = SimpleNamespace(
        total_reward=1.25,
        custom_metrics={},
        last_info_for=lambda *_args, **_kwargs: {"episode_evidence": evidence},
    )

    metrics.on_episode_end(
        worker=None,
        base_env=SimpleNamespace(envs=[]),
        policies=None,
        episode=episode,
    )

    assert episode.custom_metrics["optimization_reward"] == 1.25
    assert episode.custom_metrics["economic_gross_return"] == 0.003
    assert episode.custom_metrics["economic_net_return"] == 0.002
    assert episode.custom_metrics["economic_turnover"] == 10.0
    assert episode.custom_metrics["family_00_net_return"] == 0.002
    assert episode.custom_metrics["horizon_16_net_return"] == 0.002
    assert "episode_return" not in episode.custom_metrics
    assert "episode_return_mean" not in episode.custom_metrics
