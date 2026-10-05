"""Native RLlib PPO optimizer-visit diagnostics.

This module intentionally depends only on PyTorch and generic mixins. Ray 2.58's
PPO package eagerly imports its legacy Policy API even when its RLModule/Learner
stack is selected, so the application composes these mixins with its pinned
native PPO classes at the adapter boundary instead of importing PPO here.
"""

from collections.abc import Mapping

import torch

CLIP_FRACTION_VERSION = "ppo_ratio_outside_v1"
CLIP_FRACTION_SCOPE = "all_optimizer_sample_visits_per_train_report"

# Stable RLlib 2.58 Columns keys, kept local so importing this native diagnostic
# layer does not initialize RLlib's legacy Policy package through core.columns.
_ACTION_DIST_INPUTS = "action_dist_inputs"
_ACTION_LOGP = "action_logp"
_ACTIONS = "actions"
_LOSS_MASK = "loss_mask"


def _count_clipped_values(ratio, clip_param, mask=None):
    """Return detached (clipped, valid) visit counts for one optimizer batch."""
    values = ratio.detach()
    if mask is not None:
        values = values[mask]
    if not torch.isfinite(values).all():
        raise FloatingPointError("clip_fraction: non-finite valid likelihood ratio")
    outside = (values < 1 - clip_param) | (values > 1 + clip_param)
    return int(outside.sum().item()), values.numel()


def native_clip_fraction_report(learner_reports):
    """Aggregate per-module/per-learner counts into one weighted train report."""
    clipped_count = sum(
        report["clip_fraction_clipped_count"]
        for learner_report in learner_reports
        for report in learner_report.values()
    )
    valid_count = sum(
        report["clip_fraction_valid_count"]
        for learner_report in learner_reports
        for report in learner_report.values()
    )
    if valid_count <= 0:
        raise RuntimeError("Native PPO produced no valid optimizer samples")
    return {
        "clip_fraction": clipped_count / valid_count,
        "clip_fraction_clipped_count": clipped_count,
        "clip_fraction_valid_count": valid_count,
        "clip_fraction_version": CLIP_FRACTION_VERSION,
        "clip_fraction_scope": CLIP_FRACTION_SCOPE,
    }


def _resolve_learner_result(call_result):
    if isinstance(call_result, Mapping):
        return call_result
    get_result = getattr(call_result, "get", None)
    return get_result() if callable(get_result) else call_result


class NativeClipFractionLearnerMixin:
    """Same-forward PPO clipping counts for a native PPO Torch Learner.

    Compose this before Ray's native ``PPOTorchLearner`` base. It observes the
    exact action-distribution logits already used by the loss, and counts each
    optimizer visit rather than unique sampled transitions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._clip_counts = {}

    def begin_clip_fraction(self):
        self._clip_counts = {}

    def consume_clip_fraction(self):
        reports = {
            module_id: {
                "clip_fraction_clipped_count": clipped,
                "clip_fraction_valid_count": valid,
            }
            for module_id, (clipped, valid) in self._clip_counts.items()
        }
        self._clip_counts = {}
        return reports

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        loss = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )
        module = self.module[module_id].unwrapped()
        current_dist = module.get_train_action_dist_cls().from_logits(
            fwd_out[_ACTION_DIST_INPUTS]
        )
        ratio = torch.exp(
            current_dist.logp(batch[_ACTIONS]) - batch[_ACTION_LOGP]
        ).detach()
        clipped, valid = _count_clipped_values(
            ratio, config.clip_param, batch.get(_LOSS_MASK)
        )
        previous = self._clip_counts.get(module_id, (0, 0))
        self._clip_counts[module_id] = (
            previous[0] + clipped,
            previous[1] + valid,
        )
        return loss


class NativeClipFractionAlgorithmMixin:
    """Publish weighted native Learner counts on each PPO training report.

    Compose this before Ray's PPO Algorithm class. The mixin has no stack
    selector or legacy fallback: it always brackets one native LearnerGroup
    train report with count reset/collection.
    """

    def train(self):
        learner_group = self.learner_group
        learner_group.foreach_learner(lambda learner: learner.begin_clip_fraction())
        result = super().train()
        learner_reports = [
            _resolve_learner_result(call_result)
            for call_result in learner_group.foreach_learner(
                lambda learner: learner.consume_clip_fraction()
            )
        ]
        result.setdefault("learners", {}).setdefault("default_policy", {}).update(
            native_clip_fraction_report(learner_reports)
        )
        return result
