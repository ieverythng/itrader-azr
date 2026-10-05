import json
from types import SimpleNamespace

import pytest

from itrader.rl_lib.callbacks.ps_metrics_native import NativePSMetrics


def test_native_metrics_prefers_env_runner_population_over_nonempty_wrapper():
    metrics = NativePSMetrics()
    nested = {
        "economic_net_return__sum": -0.04196,
        "economic_net_return__count": 4,
        "success__sum": 1.0,
        "success__count": 4,
    }
    result = {
        "custom_metrics": {"success_mean": 0.9},
        "env_runners": {"custom_metrics": nested},
    }

    metrics.on_train_result(algorithm=None, result=result)

    assert result["custom_metrics"] is nested
    assert result["custom_metrics"]["economic_net_return__sum"] == -0.04196
    assert result["p_solve"] == 0.25


def test_native_metric_mean_reduction_is_shared_and_rejects_booleans():
    from itrader.rl_lib.callbacks.metrics_contract import mean_from_totals
    from itrader.rl_lib.train.train_solo_V2 import _numeric_custom_metric_items
    from itrader.tools.replay_ray_checkpoint import _metric_value

    totals = {"success__sum": 1.0, "success__count": 4.0}
    assert mean_from_totals(totals, "success", required=True) == 0.25
    assert dict(_numeric_custom_metric_items(totals))["success_mean"] == 0.25
    assert _metric_value(totals, "success") == 0.25
    result = {"env_runners": {"custom_metrics": totals}}
    NativePSMetrics().on_train_result(algorithm=None, result=result)
    assert result["p_solve"] == 0.25

    malformed = {"success__sum": True, "success__count": True}
    with pytest.raises(ValueError, match="numeric sum and count"):
        mean_from_totals(malformed, "success", required=True)
    with pytest.raises(ValueError, match="numeric sum and count"):
        _numeric_custom_metric_items(malformed)
    with pytest.raises(ValueError, match="numeric sum and count"):
        _metric_value(malformed, "success")
    with pytest.raises(ValueError, match="numeric sum and count"):
        NativePSMetrics().on_train_result(
            algorithm=None,
            result={"env_runners": {"custom_metrics": malformed}},
        )


def test_native_metrics_logs_terminal_evidence_and_replay_payload():
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
        "action_reward_trace_sha256": "a" * 64,
    }
    terminal_info = {
        "episode_evidence": evidence,
        "ep_sums": {
            "steps": 2,
            "horizon": 16,
            "act_counts": [1, 0, 1],
            "rc_sums": {"utility_gain": 0.6, "risk_cost": -0.2},
        },
        "reward_comps": {"success": 1.0},
    }

    class Episode:
        def get_infos(self):
            return [terminal_info]

        def get_rewards(self):
            return [0.5, 0.75]

    class MetricsLogger:
        def __init__(self):
            self.values = []

        def log_value(self, key, value, **kwargs):
            self.values.append((key, value, kwargs))

    logger = MetricsLogger()
    NativePSMetrics().on_episode_end(
        episode=Episode(), metrics_logger=logger
    )

    metrics = {
        key[-1]: (value, kwargs)
        for key, value, kwargs in logger.values
        if key[0] == "custom_metrics"
    }
    assert metrics["optimization_reward__sum"][0] == 1.25
    assert metrics["optimization_reward__count"][0] == 1.0
    assert metrics["economic_net_return__sum"][0] == 0.002
    assert metrics["economic_net_return__count"][0] == 1.0
    assert metrics["rc_utility_gain__sum"][0] == 0.3
    assert metrics["rc_risk_cost__sum"][0] == -0.1
    assert metrics["act_0_frac__sum"][0] == 0.5
    assert metrics["act_2_frac__sum"][0] == 0.5
    assert metrics["act_flat_frac__count"][0] == 1.0
    assert metrics["act_flat_frac__sum"][1]["reduce"] == "sum"
    assert metrics["act_flat_frac__sum"][1]["window"] is None

    payload_entry = next(
        (key, value, kwargs)
        for key, value, kwargs in logger.values
        if key == ("hist_stats", "replay_episode_payload")
    )
    payload = json.loads(payload_entry[1])
    assert payload["action_reward_trace_sha256"] == "a" * 64
    assert payload["action_fractions"] == [0.5, 0.0, 0.5]
    assert payload_entry[2]["reduce"] == "item_series"


def test_native_metrics_and_ray_logger_preserve_weighted_evaluation_means():
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

    from itrader.rl_lib.train.train_solo_V2 import _extract_report

    callback = NativePSMetrics()

    def log_episode(logger, *, net_return, flat_action):
        counts = [0, 2, 0] if flat_action else [1, 0, 1]
        terminal_info = {
            "episode_evidence": {
                "optimization_reward": net_return * 1000.0,
                "gross_return": net_return + 0.001,
                "cost_return": 0.001,
                "net_return": net_return,
                "net_sharpe_per_sqrt_bar": net_return,
                "success": float(net_return > 0.0),
                "family_id": "00",
                "horizon": 16,
                "regime_tags": [],
            },
            "ep_sums": {"steps": 2, "horizon": 16, "act_counts": counts},
        }

        class Episode:
            def get_infos(self):
                return [terminal_info]

            def get_rewards(self):
                return [net_return * 1000.0]

        callback.on_episode_end(episode=Episode(), metrics_logger=logger)

    first_runner = MetricsLogger()
    second_runner = MetricsLogger()
    log_episode(first_runner, net_return=-0.04, flat_action=True)
    log_episode(second_runner, net_return=0.02, flat_action=False)
    log_episode(second_runner, net_return=0.02, flat_action=False)

    aggregate = MetricsLogger()
    aggregate.aggregate([first_runner.stats, second_runner.stats])
    native_evaluation_metrics = aggregate.compile()["custom_metrics"]
    report = _extract_report(
        {
            "evaluation": {
                "custom_metrics": {
                    "economic_net_return_mean": 0.9,
                    "act_flat_frac_mean": 1.0,
                },
                "env_runners": {"custom_metrics": native_evaluation_metrics},
            }
        },
        default_iteration=1,
    )

    assert native_evaluation_metrics["economic_net_return__sum"] == pytest.approx(0.0)
    assert native_evaluation_metrics["economic_net_return__count"] == 3
    assert report["evaluation/net_return_mean"] == pytest.approx(0.0)
    assert report["evaluation/custom_metrics/act_flat_frac_mean"] == pytest.approx(1 / 3)
