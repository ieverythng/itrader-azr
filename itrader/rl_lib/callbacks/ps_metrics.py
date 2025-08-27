from __future__ import annotations
from collections import Counter
from typing import Any, Dict
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class PSMetrics(DefaultCallbacks):
    """Logs episode diagnostics without relying on EpisodeV2.last_* helpers.

    Works with single‑agent discrete(3) envs.
    Aggregates from postprocessed trajectory batches:
      - actions → per‑episode action fractions
      - infos.reward_comps.{pnl_delta,dd,turnover} → means
      - infos.success (terminal) → success flag
      - infos.task → task metadata (e.g., horizon)
    """

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        # init accumulators
        episode.user_data["act_counts"] = np.zeros(3, dtype=np.int64)
        episode.user_data["act_total"] = 0
        episode.user_data["rc_sums"] = {"pnl_delta": 0.0, "dd": 0.0, "turnover": 0.0}
        episode.user_data["rc_count"] = 0
        episode.user_data["success_flag"] = 0.0
        episode.user_data["task"] = {}
        # try to capture task immediately from env (best effort)
        try:
            envs = base_env.get_sub_environments()
            if envs:
                env = envs[env_index or 0]
                if getattr(env, "current_task", None) is not None:
                    episode.user_data["task"] = dict(env.current_task.to_dict())
        except Exception:
            pass

    def on_postprocess_trajectory(self, *, worker, episode, agent_id, policy_id, policies,
                                  postprocessed_batch, original_batches, **kwargs):
        # Actions (assume Discrete(3))
        acts = postprocessed_batch.get("actions", None)
        if acts is not None:
            a = np.asarray(acts)
            if a.ndim > 1:
                a = np.squeeze(a, axis=-1)
            if not np.issubdtype(a.dtype, np.integer):
                a = np.rint(a).astype(np.int64)
            a = np.clip(a, 0, 2)
            counts = np.bincount(a, minlength=3)
            episode.user_data["act_counts"] += counts
            episode.user_data["act_total"] += a.size

        # Infos → reward components, success, task
        infos = postprocessed_batch.get("infos", None)
        if infos is not None:
            for info in infos:
                if not isinstance(info, dict):
                    continue
                rc = info.get("reward_comps")
                if isinstance(rc, dict):
                    for k in ("pnl_delta", "dd", "turnover"):
                        v = rc.get(k)
                        if v is not None and np.isfinite(v):
                            episode.user_data["rc_sums"][k] += float(v)
                    episode.user_data["rc_count"] += 1
                # success is usually only present at terminal step; last one wins
                if info.get("success"):
                    episode.user_data["success_flag"] = 1.0
                # keep first non‑empty task dict if we failed to read it on start
                if not episode.user_data["task"] and "task" in info and isinstance(info["task"], dict):
                    episode.user_data["task"] = dict(info["task"])

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # Action fractions
        total = max(int(episode.user_data.get("act_total", 0)), 1)
        counts = episode.user_data.get("act_counts", np.zeros(3, dtype=np.int64))
        for a in range(3):
            episode.custom_metrics[f"act_{a}_frac"] = float(counts[a]) / float(total)

        # Reward component means
        rc_sums = episode.user_data.get("rc_sums", {})
        n = max(int(episode.user_data.get("rc_count", 0)), 1)
        for k in ("pnl_delta", "dd", "turnover"):
            v = float(rc_sums.get(k, 0.0))
            episode.custom_metrics[f"rc_{k}_mean"] = v / float(n)

        # Success flag
        episode.custom_metrics["success"] = float(episode.user_data.get("success_flag", 0.0))

        # Task metadata (numeric only so TB filters won’t break)
        task = episode.user_data.get("task", {})
        if isinstance(task, dict):
            try:
                episode.custom_metrics["task_horizon"] = float(task.get("horizon", 0))
            except Exception:
                pass