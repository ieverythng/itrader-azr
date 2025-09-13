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

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        info = None
        # Try to get last info directly (works on many RLlib builds)
        try:
            info = episode.last_info_for()
        except Exception:
            info = None

        # Fallback: read the env's cached last info (single-agent)
        if info is None:
            try:
                env = base_env.get_unwrapped()[0]
                info = getattr(env, "_last_info", None)
            except Exception:
                info = None

        if not info:
            return

        # Defaults
        act0 = act1 = act2 = 0.0
        rc_pnl = rc_dd = rc_to = 0.0
        success = 0.0
        horizon = None

        if env is not None and isinstance(getattr(env, "_last_info", None), dict):
            info: Dict[str, Any] = env._last_info
            success = float(bool(info.get("success", False)))
            horizon = int(info.get("task", {}).get("horizon", 0))

            ep = info.get("ep_sums", {})
            steps = max(1, int(ep.get("steps", 1)))

            # Action fractions
            counts = ep.get("act_counts", [0, 0, 0])
            if isinstance(counts, (list, tuple)) and len(counts) >= 3:
                act0 = float(counts[0]) / steps
                act1 = float(counts[1]) / steps
                act2 = float(counts[2]) / steps

            # Reward component means
            rc = ep.get("rc_sums", {})
            rc_pnl = float(rc.get("pnl_delta", 0.0)) / steps
            rc_dd = float(rc.get("dd", 0.0)) / steps
            rc_to = float(rc.get("turnover", 0.0)) / steps

        # Write metrics
        episode.custom_metrics["success"] = success
        if horizon is not None:
            episode.custom_metrics["task_horizon"] = float(horizon)

        episode.custom_metrics["act_0_frac"] = act0
        episode.custom_metrics["act_1_frac"] = act1
        episode.custom_metrics["act_2_frac"] = act2

        episode.custom_metrics["rc_pnl_delta_mean"] = rc_pnl
        episode.custom_metrics["rc_dd_mean"] = rc_dd
        episode.custom_metrics["rc_turnover_mean"] = rc_to