"""Optimizer-visit clipping diagnostics for the pinned Ray 2.58 legacy PPO.

The loss below follows Ray 2.58.0's Apache-2.0 PPOTorchPolicy.loss, with only
one detached observation added after its likelihood ratio. Keep parity tests
against the installed original policy when changing the runtime or this adapter.
"""

from threading import Lock

import ray
import torch
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)

CLIP_FRACTION_VERSION = "ppo_ratio_outside_v1"
CLIP_FRACTION_SCOPE = "all_optimizer_sample_visits_per_train_report"


class ClipFractionAccumulator:
    """Thread-safe detached counts, explicitly scoped to one training report."""

    def __init__(self):
        self._lock = Lock()
        self._active = False
        self._clipped = 0
        self._valid = 0

    def begin(self):
        with self._lock:
            self._active = True
            self._clipped = self._valid = 0

    def record(self, ratio, clip_param, mask=None):
        if not self._active:
            return
        values = ratio.detach()
        if mask is not None:
            values = values[mask]
        if not torch.isfinite(values).all():
            raise FloatingPointError("clip_fraction: non-finite valid likelihood ratio")
        outside = (values < 1 - clip_param) | (values > 1 + clip_param)
        clipped, valid = int(outside.sum().item()), values.numel()
        with self._lock:
            self._clipped += clipped
            self._valid += valid

    def consume(self):
        with self._lock:
            clipped, valid = self._clipped, self._valid
            self._active = False
            self._clipped = self._valid = 0
        return {
            "clip_fraction": clipped / valid if valid else None,
            "clip_fraction_clipped_count": clipped,
            "clip_fraction_valid_count": valid,
            "clip_fraction_version": CLIP_FRACTION_VERSION,
            "clip_fraction_scope": CLIP_FRACTION_SCOPE,
        }


class DiagnosticPPOTorchPolicy(PPOTorchPolicy):
    """Ray 2.58 legacy PPO with same-forward ratio counting."""

    def __init__(self, observation_space, action_space, config):
        if ray.__version__ != "2.58.0":
            raise RuntimeError("PPO clipping diagnostics require Ray 2.58.0")
        self._clip_counts = ClipFractionAccumulator()
        super().__init__(observation_space, action_space, config)

    def begin_clip_fraction(self):
        self._clip_counts.begin()

    def consume_clip_fraction(self):
        return self._clip_counts.consume()

    def loss(self, model, dist_class, train_batch):
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        if state:
            batch_size = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // batch_size
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )
        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )
        self._clip_counts.record(logp_ratio, self.config["clip_param"], mask)

        if self.config["use_kl_loss"]:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)
        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )
        if self.config["use_kl_loss"]:
            total_loss += self.kl_coeff * mean_kl_loss

        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss
        return total_loss


class DiagnosticPPO(PPO):
    """Publish weighted diagnostics after all legacy optimizer work in train()."""

    @classmethod
    def get_default_policy_class(cls, config):
        if config["framework"] != "torch":
            raise ValueError("PPO clipping diagnostics require the Torch policy")
        return DiagnosticPPOTorchPolicy

    def train(self):
        if (
            self.config.enable_rl_module_and_learner
            and self.config.enable_env_runner_and_connector_v2
        ):
            return self._train_native_with_clip_diagnostics()
        if (
            self.config.enable_rl_module_and_learner
            or self.config.enable_env_runner_and_connector_v2
        ):
            raise ValueError("Ray API stack flags must be enabled or disabled together")
        policies = dict(self.env_runner.policy_map.items())
        for policy in policies.values():
            if not isinstance(policy, DiagnosticPPOTorchPolicy):
                raise TypeError(
                    "PPO clipping diagnostics require DiagnosticPPOTorchPolicy"
                )
            policy.begin_clip_fraction()
        try:
            result = super().train()
        finally:
            reports = {
                pid: policy.consume_clip_fraction() for pid, policy in policies.items()
            }
        learner = result.setdefault("info", {}).setdefault("learner", {})
        for pid, report in reports.items():
            learner.setdefault(pid, {}).setdefault("learner_stats", {}).update(report)
        return result

    def _train_native_with_clip_diagnostics(self):
        learner_group = self.learner_group
        learner_group.foreach_learner(
            lambda learner: learner.begin_clip_fraction()
        )
        result = super().train()
        learner_reports = [
            call_result.get() if hasattr(call_result, "get") else call_result
            for call_result in learner_group.foreach_learner(
                lambda learner: learner.consume_clip_fraction()
            )
        ]
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
        result.setdefault("learners", {}).setdefault("default_policy", {}).update(
            {
                "clip_fraction": clipped_count / valid_count,
                "clip_fraction_clipped_count": clipped_count,
                "clip_fraction_valid_count": valid_count,
                "clip_fraction_version": CLIP_FRACTION_VERSION,
                "clip_fraction_scope": CLIP_FRACTION_SCOPE,
            }
        )
        return result


class DiagnosticPPOTorchLearner(PPOTorchLearner):
    """Native PPO learner with same-forward, all-optimizer-visit clip counts."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._clip_counts = {}

    def begin_clip_fraction(self):
        self._clip_counts = {}

    def consume_clip_fraction(self):
        reports = {}
        for module_id, (clipped, valid) in self._clip_counts.items():
            reports[module_id] = {
                "clip_fraction_clipped_count": clipped,
                "clip_fraction_valid_count": valid,
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
            fwd_out[Columns.ACTION_DIST_INPUTS]
        )
        previous_dist = module.get_exploration_action_dist_cls().from_logits(
            batch[Columns.ACTION_DIST_INPUTS]
        )
        ratio = torch.exp(
            current_dist.logp(batch[Columns.ACTIONS])
            - batch[Columns.ACTION_LOGP]
        ).detach()
        mask = batch.get(Columns.LOSS_MASK)
        values = ratio if mask is None else ratio[mask]
        if not torch.isfinite(values).all():
            raise FloatingPointError("clip_fraction: non-finite valid likelihood ratio")
        clipped = int(
            (
                (values < 1 - config.clip_param)
                | (values > 1 + config.clip_param)
            ).sum().item()
        )
        previous = self._clip_counts.get(module_id, (0, 0))
        self._clip_counts[module_id] = (previous[0] + clipped, previous[1] + values.numel())
        return loss
