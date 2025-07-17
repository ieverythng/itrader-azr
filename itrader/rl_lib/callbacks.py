# itrader/callbacks.py
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.columns import Columns
import numpy as np
import time


class StatsCallback(DefaultCallbacks):
    """
    Logs a few extra metrics per episode **and** wall-clock throughput.
    Accuracy is dummy here (long/flat/short vs 'gt_action' if you add one).
    """

    def on_episode_end(self, *, episode, **kwargs):
        # ---- standard return
        episode.custom_metrics["episode_return"] = np.sum(
            list(episode._agent_reward_history.values())
        )
        

        # # ---- example accuracy metric (skip if no ground-truth column)
        # if "gt_action" in episode.last_raw_obs:
        #     correct = (episode.last_info_for()[0]["position"]
        #                == episode.last_raw_obs["gt_action"])
        #     episode.custom_metrics["accuracy"] = float(correct)

    # ---- keep simple wall-clock throughput (iters / second)
    def on_train_result(self, *, result, **kwargs):
        iters = result["training_iteration"]
        elapsed = result["time_total_s"]
        result.setdefault("custom_metrics", {})
        result["custom_metrics"]["iters_per_sec"] = iters / max(elapsed, 1e-6)
