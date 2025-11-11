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
        # Always set a default to avoid UnboundLocalError
        env = self._extract_env(base_env)

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
            act_short, act_flat, act_long = (float(act_counts[0]), float(act_counts[1]), float(act_counts[2]))
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
        