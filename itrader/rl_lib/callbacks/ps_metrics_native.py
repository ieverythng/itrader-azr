"""Episode-evidence adapter for Ray's SingleAgentEpisode/MetricsLogger stack."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any

from ray.rllib.algorithms.callbacks import DefaultCallbacks

from itrader.rl_lib.callbacks.metrics_contract import (
    mean_from_totals,
    normalized_action_counts,
    record_action_metrics,
    record_episode_evidence,
    record_reward_component_metrics,
)


def _json_scalar(value):
    """Convert NumPy scalar evidence to its JSON-native Python equivalent."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class NativePSMetrics(DefaultCallbacks):
    """Keep repository metric names while emitting through MetricsLogger."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _terminal_info(episode) -> dict[str, Any]:
        getter = getattr(episode, "get_infos", None)
        if callable(getter):
            infos = getter()
            if infos:
                info = infos[-1]
                return info if isinstance(info, dict) else {}
        return {}

    @staticmethod
    def _log(metrics_logger, key, value, *, reduce="sum", window=None):
        if metrics_logger is not None:
            metrics_logger.log_value(
                key,
                value,
                reduce=reduce,
                window=window,
            )

    @classmethod
    def _log_episode_metrics(cls, metrics_logger, metrics: dict[str, float]) -> None:
        """Emit exact per-report totals and denominators for cross-runner means."""
        for key, raw_value in metrics.items():
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"Native episode metric {key!r} must be finite")
            cls._log(
                metrics_logger,
                ("custom_metrics", f"{key}__sum"),
                value,
            )
            cls._log(
                metrics_logger,
                ("custom_metrics", f"{key}__count"),
                1.0,
            )

    def on_episode_end(
        self,
        *,
        episode,
        metrics_logger=None,
        env=None,
        env_runner=None,
        **kwargs,
    ):
        info = self._terminal_info(episode)
        metrics = SimpleNamespace(custom_metrics={}, hist_data={})
        rewards = episode.get_rewards() if hasattr(episode, "get_rewards") else []
        total_reward = float(sum(rewards or []))
        metrics.custom_metrics["optimization_reward"] = total_reward

        evidence = info.get("episode_evidence")
        if isinstance(evidence, dict):
            record_episode_evidence(metrics, evidence)
            if "action_reward_trace_sha256" in evidence:
                sums = info.get("ep_sums", {})
                steps = max(1, int(sums.get("steps", len(rewards or []) or 1)))
                payload = {
                    **evidence,
                    "action_fractions": [
                        count / steps
                        for count in normalized_action_counts(
                            sums.get("act_counts", [0, 0, 0])
                        )
                    ],
                }
                metrics.hist_data["replay_episode_payload"] = [
                    json.dumps(
                        payload,
                        sort_keys=True,
                        allow_nan=False,
                        default=_json_scalar,
                    )
                ]

        sums = info.get("ep_sums")
        if isinstance(sums, dict):
            steps = max(1, int(sums.get("steps", 1)))
            record_action_metrics(
                metrics,
                normalized_action_counts(
                    sums.get("act_counts", [0, 0, 0])
                ),
                steps,
            )
            record_reward_component_metrics(
                metrics,
                sums.get("rc_sums", {}),
                steps,
            )
            metrics.custom_metrics["task_horizon"] = float(
                sums.get("horizon", 0.0)
            )
            metrics.custom_metrics["success"] = float(
                info.get("reward_comps", {}).get("success", 0.0)
            )

        self._log_episode_metrics(metrics_logger, metrics.custom_metrics)
        for key, values in metrics.hist_data.items():
            for value in values:
                self._log(
                    metrics_logger,
                    ("hist_stats", key),
                    value,
                    reduce="item_series",
                    window=128,
                )
        del env, env_runner, kwargs

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # Native EnvRunner metrics are nested one level deeper than the
        # compatibility result. They are authoritative whenever present; never
        # let a stale top-level population override or merge with them.
        env_runner_metrics = result.get("env_runners", {}).get("custom_metrics")
        current_metrics = result.get("custom_metrics")
        if isinstance(env_runner_metrics, dict) and env_runner_metrics:
            result["custom_metrics"] = env_runner_metrics
            current_metrics = env_runner_metrics
        p_solve = (
            mean_from_totals(current_metrics, "success")
            if isinstance(current_metrics, dict)
            else None
        )
        if p_solve is not None:
            result["p_solve"] = float(p_solve)
        del algorithm, kwargs
        return None


__all__ = ["NativePSMetrics"]
