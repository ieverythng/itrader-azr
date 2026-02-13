from __future__ import annotations
from typing import Any, Dict, Optional
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class PSMetrics(DefaultCallbacks):
    """
    Logs:
      - episode_return_mean (our primary)
      - action fractions per episode: act_short/flat/long_frac
      - reward components per step avg: rc_pnl_delta_mean, rc_dd_mean, rc_turnover_mean
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

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        env = self._cached_env
        if env is None:
            env = self._extract_env(base_env)
            self._cached_env = env
        elif not hasattr(env, "_last_info"):
            env = self._extract_env(base_env)
            self._cached_env = env

        # 1) Primary scalar: episode total reward (RLlib computes this)
        try:
            total_r = float(episode.total_reward)
        except Exception:
            total_r = 0.0
        episode.custom_metrics["episode_return_mean"] = total_r

        # 2) Pull last info dict if available
        info: Dict[str, Any] = {}
        try:
            # EpisodeV2: last_info_for(policy_id=None) is removed; use our env stash.
            if env is not None and isinstance(getattr(env, "_last_info", None), dict):
                info = env._last_info  # set by env.step()
        except Exception:
            info = {}

        # 3) Action fractions + reward components (from env.ep_sums)
        if isinstance(info.get("ep_sums"), dict):
            sums = info["ep_sums"]
            steps = max(1, int(sums.get("steps", 1)))

            # action fractions
            act_counts = sums.get("act_counts", [0, 0, 0])
            if isinstance(act_counts, dict):  # if someone stored dict indices
                act_counts = [act_counts.get(i, 0) for i in range(3)]
            if len(act_counts) < 3:
                act_counts = list(act_counts) + [0] * (3 - len(act_counts))
            act_short, act_flat, act_long = (
                float(act_counts[0]),
                float(act_counts[1]),
                float(act_counts[2]),
            )
            episode.custom_metrics["act_short_frac"] = act_short / steps
            episode.custom_metrics["act_flat_frac"]  = act_flat  / steps
            episode.custom_metrics["act_long_frac"]  = act_long  / steps

            # reward components (per-step average)
            rc_sums = sums.get("rc_sums", {})
            episode.custom_metrics["rc_pnl_delta"] = float(rc_sums.get("pnl_delta", 0.0)) / steps
            episode.custom_metrics["rc_dd"]        = float(rc_sums.get("dd", 0.0))        / steps
            episode.custom_metrics["rc_turnover"]  = float(rc_sums.get("turnover", 0.0))  / steps

            # horizon for task-level plots
            episode.custom_metrics["task_horizon_mean"] = float(sums.get("horizon", 0.0))

            # 4) Success flag (also present in info)
            # success flag if carried in info
            succ = float(info.get("reward_comps", {}).get("success", 0.0))
            episode.custom_metrics["success_mean"] = succ
 
            # a custom "episode_return" based on raw pnl (scale-invariant)
            episode.custom_metrics["episode_return"] = rc_sums.get("pnl_delta", 0.0)
        
        # 5) TaskPool stats update (if task pool is available)
        task_dict = info.get("task")
        if task_dict and env is not None:
            try:
                # Try to access task_pool via task_sampler
                if hasattr(env, "task_sampler") and hasattr(env.task_sampler, "task_pool") and env.task_sampler.task_pool is not None:
                    from itrader.rl_lib.tasks.task_pool import task_uid
                    task_uid_val = task_uid(task_dict)
                    success = bool(info.get("success", False))
                    episode_return = float(total_r)
                    env.task_sampler.task_pool.update_stats(task_uid_val, episode_return, success)
            except Exception as e:
                self._warn_once("task_pool_update", f"task pool update failed (continuing): {e}")
        
        # 6) Task pool custom metrics (aggregate stats if available)
        if env is not None:
            try:
                if hasattr(env, "task_sampler") and hasattr(env.task_sampler, "task_pool") and env.task_sampler.task_pool is not None:
                    pool = env.task_sampler.task_pool
                    if pool.size() > 0:
                        # Compute aggregate stats from pool
                        all_returns = []
                        all_successes = []
                        for uid in pool._stats.keys():
                            stats = pool._stats[uid]
                            if stats["count"] > 0:
                                all_returns.append(stats["mean_return"])
                                all_successes.append(stats["mean_success"])
                        
                        if all_returns:
                            episode.custom_metrics["task_pool_return_mean"] = float(sum(all_returns) / len(all_returns))
                            episode.custom_metrics["task_pool_success_rate_mean"] = float(sum(all_successes) / len(all_successes))
            except Exception as e:
                self._warn_once("task_pool_metrics", f"task pool aggregate metrics failed (continuing): {e}")
        