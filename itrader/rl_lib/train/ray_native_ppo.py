"""Pinned Ray 2.58 PPO base-class composition for the native API stack."""
from __future__ import annotations

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

from itrader.rl_lib.train.ppo_native_diagnostics import (
    NativeClipFractionAlgorithmMixin,
    NativeClipFractionLearnerMixin,
)


if ray.__version__ != "2.58.0":
    raise RuntimeError("The native PPO adapter is qualified for Ray 2.58.0 only")


class DiagnosticPPOTorchLearner(NativeClipFractionLearnerMixin, PPOTorchLearner):
    """Native PPO Learner with same-forward optimizer-visit diagnostics."""


class DiagnosticPPO(NativeClipFractionAlgorithmMixin, PPO):
    """Native PPO Algorithm with weighted learner clip diagnostics."""


__all__ = ["DiagnosticPPO", "DiagnosticPPOTorchLearner"]
