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
    assert episode.custom_metrics["episode_return"] == 1.25
    assert episode.custom_metrics["act_0_frac"] == 0.5
    assert episode.custom_metrics["act_2_frac"] == 0.5


def test_ps_metrics_accepts_numpy_action_counts():
    counts = PSMetrics._normalized_action_counts(np.array([1, 2, 3]))
    assert counts == [1.0, 2.0, 3.0]
