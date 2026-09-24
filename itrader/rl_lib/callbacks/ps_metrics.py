from __future__ import annotations
from typing import Any, Dict, Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class PSMetrics(DefaultCallbacks):
    """
    Logs:
      - optimization reward, explicitly distinct from economic returns
      - action fractions per episode, both legacy family-00 names and generic act_{i}_frac
      - reward components per step avg as rc_<component>
      - success flag mean
      - task horizon mean
    Compatible with EnvRunnerV2 / EpisodeV2.
    """
    def __init__(self):
        super().__init__()
        self._cached_env = None
        self._warned = set()

    def _warn_once(self, key: str, msg: str):
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"[PSMetrics] {msg}")

    @staticmethod
    def _extract_env(base_env) -> Optional[Any]:
        """Best-effort to get the first underlying env instance."""
        # Vector env (common in RLlib)
        if hasattr(base_env, "get_sub_environments"):
            try:
                envs = base_env.get_sub_environments()
                if envs:
                    return envs[0]
            except Exception:
                pass
        # Older access paths
        if hasattr(base_env, "vector_env") and hasattr(base_env.vector_env, "envs"):
            envs = base_env.vector_env.envs
            if envs:
                return envs[0]
        if hasattr(base_env, "envs"):
            envs = base_env.envs
            if envs:
                return envs[0]
        return None

    def _resolve_env(self, base_env) -> Optional[Any]:
        env = self._cached_env
        if env is None or not hasattr(env, "_last_info"):
            env = self._extract_env(base_env)
            self._cached_env = env
        return env

    @staticmethod
    def _task_pool(env) -> Optional[Any]:
        sampler = getattr(env, "task_sampler", None)
        return getattr(sampler, "task_pool", None)

    @staticmethod
    def _episode_total_reward(episode) -> float:
        try:
            return float(episode.total_reward)
        except Exception:
            return 0.0

    @staticmethod
    def _last_info(env) -> Dict[str, Any]:
        info = getattr(env, "_last_info", None)
        return info if isinstance(info, dict) else {}

    @staticmethod
    def _episode_last_info(episode) -> Dict[str, Any]:
        getter = getattr(episode, "last_info_for", None)
        if not callable(getter):
            return {}
        for args in ((None,), ()):
            try:
                info = getter(*args)
            except (TypeError, KeyError):
                continue
            if isinstance(info, dict):
                return info
        return {}

    @staticmethod
    def _safe_metric_token(value: Any) -> str:
        token = "".join(ch if ch.isalnum() else "_" for ch in str(value))
        return token.strip("_") or "unknown"

    @classmethod
    def _record_episode_evidence(cls, episode, evidence: Dict[str, Any]) -> None:
        scalar_map = {
            "optimization_reward": "optimization_reward",
            "gross_return": "economic_gross_return",
            "cost_return": "economic_cost_return",
            "net_return": "economic_net_return",
            "turnover": "economic_turnover",
            "liquidation_turnover": "economic_liquidation_turnover",
            "max_drawdown": "economic_max_drawdown",
            "gross_sharpe_per_sqrt_bar": "economic_gross_sharpe_per_sqrt_bar",
            "net_sharpe_per_sqrt_bar": "economic_net_sharpe_per_sqrt_bar",
            "annualized_net_sharpe": "economic_annualized_net_sharpe",
            "positive_net_bar_fraction": "economic_positive_net_bar_fraction",
            "gross_exposure_fraction": "economic_gross_exposure_fraction",
            "position_changes": "economic_position_changes",
            "success": "success",
        }
        for source, metric in scalar_map.items():
            if source in evidence and evidence[source] is not None:
                episode.custom_metrics[metric] = float(evidence[source])

        net_return = evidence.get("net_return")
        if net_return is None:
            return
        net_return = float(net_return)
        family = cls._safe_metric_token(evidence.get("family_id", "unknown"))
        horizon = cls._safe_metric_token(evidence.get("horizon", "unknown"))
        episode.custom_metrics[f"family_{family}_net_return"] = net_return
        episode.custom_metrics[f"horizon_{horizon}_net_return"] = net_return
        for tag in evidence.get("regime_tags", []) or []:
            episode.custom_metrics[f"regime_{cls._safe_metric_token(tag)}_net_return"] = net_return

    @staticmethod
    def _normalized_action_counts(act_counts) -> list[float]:
        if act_counts is None:
            act_counts = []
        if isinstance(act_counts, dict):
            max_index = max(act_counts.keys(), default=-1)
            act_counts = [act_counts.get(i, 0) for i in range(max_index + 1)]
        return [float(value) for value in list(act_counts)]

    @staticmethod
    def _record_action_metrics(episode, act_counts: list[float], steps: int) -> None:
        for index, count in enumerate(act_counts):
            episode.custom_metrics[f"act_{index}_frac"] = count / steps
        if len(act_counts) >= 3:
            episode.custom_metrics["act_short_frac"] = act_counts[0] / steps
            episode.custom_metrics["act_flat_frac"] = act_counts[1] / steps
            episode.custom_metrics["act_long_frac"] = act_counts[2] / steps

    @staticmethod
    def _record_reward_component_metrics(episode, rc_sums: Dict[str, Any], steps: int) -> None:
        for key in sorted(rc_sums):
            value = rc_sums[key]
            episode.custom_metrics[f"rc_{key}"] = float(value) / steps

    @staticmethod
    def _pool_aggregate_means(pool) -> tuple[list[float], list[float]]:
        all_returns = []
        all_successes = []
        for stats in pool._stats.values():
            if stats["count"] <= 0:
                continue
            all_returns.append(stats["mean_return"])
            all_successes.append(stats["mean_success"])
        return all_returns, all_successes

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        env = self._resolve_env(base_env)

        # 1) Optimization objective (RLlib computes this). This is not return.
        total_r = self._episode_total_reward(episode)
        episode.custom_metrics["optimization_reward"] = total_r

        # 2) Prefer episode-correlated terminal info; private env state is only a
        # compatibility fallback for old EpisodeV2 fixtures.
        info = self._episode_last_info(episode)
        if not info and env is not None:
            info = self._last_info(env)
        evidence = info.get("episode_evidence")
        if isinstance(evidence, dict):
            self._record_episode_evidence(episode, evidence)

        # 3) Action fractions + reward components (from env.ep_sums)
        if isinstance(info.get("ep_sums"), dict):
            sums = info["ep_sums"]
            steps = max(1, int(sums.get("steps", 1)))

            # action fractions
            act_counts = self._normalized_action_counts(sums.get("act_counts", [0, 0, 0]))
            self._record_action_metrics(episode, act_counts, steps)

            # reward components (per-step average)
            rc_sums = sums.get("rc_sums", {})
            self._record_reward_component_metrics(episode, rc_sums, steps)

            # horizon for task-level plots
            episode.custom_metrics["task_horizon_mean"] = float(sums.get("horizon", 0.0))

            # 4) Success flag (also present in info)
            # success flag if carried in info
            succ = float(info.get("reward_comps", {}).get("success", 0.0))
            episode.custom_metrics["success_mean"] = succ

        # 5) TaskPool stats update (if task pool is available)
        task_dict = info.get("task")
        if task_dict and env is not None:
            try:
                pool = self._task_pool(env)
                if pool is not None:
                    from itrader.rl_lib.tasks.task_pool import task_uid

                    task_uid_val = task_uid(task_dict)
                    success = bool(info.get("success", False))
                    score = float(evidence.get("net_return", total_r)) if isinstance(evidence, dict) else total_r
                    pool.update_stats(task_uid_val, score, success)
            except Exception as exc:
                self._warn_once("task_pool_update", f"task pool update failed (continuing): {exc}")

        # 6) Task pool custom metrics (aggregate stats if available)
        if env is not None:
            try:
                pool = self._task_pool(env)
                if pool is not None and pool.size() > 0:
                    all_returns, all_successes = self._pool_aggregate_means(pool)
                    if all_returns:
                        episode.custom_metrics["task_pool_return_mean"] = float(
                            sum(all_returns) / len(all_returns)
                        )
                        episode.custom_metrics["task_pool_success_rate_mean"] = float(
                            sum(all_successes) / len(all_successes)
                        )
            except Exception as exc:
                self._warn_once("task_pool_metrics", f"task pool aggregate metrics failed (continuing): {exc}")

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # success_mean is aggregated by Ray over episodes in the last train() call.
        # Expose it at top level so Tune TB/CSV has a stable `p_solve` key
        # (recovered from stash@{1} wip-local-before-sync).
        cm = result.get("custom_metrics", {}) or {}
        psolve = cm.get("success_mean", None)
        if psolve is not None:
            try:
                result["p_solve"] = float(psolve)
            except (TypeError, ValueError):
                pass
