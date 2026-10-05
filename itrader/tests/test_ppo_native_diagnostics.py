"""Native PPO clip diagnostics stay free of legacy RLlib API imports."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from itrader.rl_lib.train.ppo_native_diagnostics import (
    CLIP_FRACTION_SCOPE,
    CLIP_FRACTION_VERSION,
    NativeClipFractionAlgorithmMixin,
    NativeClipFractionLearnerMixin,
    _count_clipped_values,
    native_clip_fraction_report,
)


def test_native_diagnostics_import_does_not_initialize_legacy_policy_api():
    project_root = Path(__file__).resolve().parents[2]
    script = (
        "import importlib, sys; "
        "importlib.import_module('itrader.rl_lib.train.ppo_native_diagnostics'); "
        "forbidden = ('ray.rllib.algorithms.ppo.ppo_torch_policy', "
        "'ray.rllib.policy.policy', 'ray.rllib.policy'); "
        "loaded = [name for name in forbidden if name in sys.modules]; "
        "assert not loaded, f'legacy RLlib modules loaded: {loaded}'"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_clip_count_uses_strict_bounds_mask_and_finite_valid_values():
    ratio = torch.tensor([0.7, 0.8, 1.0, 1.2, 1.3], dtype=torch.float64)
    mask = torch.tensor([True, True, False, True, True])

    assert _count_clipped_values(ratio, 0.2, mask) == (2, 4)
    assert _count_clipped_values(
        torch.tensor([1.1, float("nan")]),
        0.2,
        torch.tensor([True, False]),
    ) == (0, 1)
    with pytest.raises(FloatingPointError, match="non-finite valid"):
        _count_clipped_values(
            torch.tensor([1.0, float("nan")]),
            0.2,
            torch.tensor([True, True]),
        )


class _ActionDistribution:
    def __init__(self, logp):
        self._logp = logp

    @classmethod
    def from_logits(cls, logits):
        return cls(logits)

    def logp(self, actions):
        del actions
        return self._logp


class _Module:
    def unwrapped(self):
        return self

    def get_train_action_dist_cls(self):
        return _ActionDistribution


class _LearnerBase:
    def __init__(self, **kwargs):
        del kwargs
        self.module = {"default_policy": _Module()}

    def compute_loss_for_module(self, **kwargs):
        del kwargs
        return torch.tensor(7.0)


class _NativeLearner(NativeClipFractionLearnerMixin, _LearnerBase):
    pass


def test_learner_mixin_counts_same_forward_optimizer_visits_and_resets():
    learner = _NativeLearner()
    learner.begin_clip_fraction()
    batch = {
        "actions": torch.zeros(5, dtype=torch.int64),
        "action_logp": torch.zeros(5, dtype=torch.float64),
        "loss_mask": torch.tensor([True, True, False, True, True]),
    }
    fwd_out = {
        "action_dist_inputs": torch.tensor(
            [0.7, 0.8, 1.0, 1.2, 1.3], dtype=torch.float64
        ).log()
    }
    config = SimpleNamespace(clip_param=0.2)

    assert (
        learner.compute_loss_for_module(
            module_id="default_policy", config=config, batch=batch, fwd_out=fwd_out
        ).item()
        == 7.0
    )
    learner.compute_loss_for_module(
        module_id="default_policy", config=config, batch=batch, fwd_out=fwd_out
    )

    assert learner.consume_clip_fraction() == {
        "default_policy": {
            "clip_fraction_clipped_count": 4,
            "clip_fraction_valid_count": 8,
        }
    }
    assert learner.consume_clip_fraction() == {}


class _FakeLearner:
    def __init__(self, counts):
        self._counts = counts

    def begin_clip_fraction(self):
        self._report = {}

    def consume_clip_fraction(self):
        return self._report


class _FakeLearnerGroup:
    def __init__(self, learners):
        self.learners = learners

    def foreach_learner(self, fn):
        return [fn(learner) for learner in self.learners]


class _PPOBase:
    def train(self):
        for learner in self.learner_group.learners:
            learner._report = learner._counts
        return {"learners": {"default_policy": {"loss": 1.0}}}


class _NativeDiagnosticPPO(NativeClipFractionAlgorithmMixin, _PPOBase):
    def __init__(self, learner_group):
        self.learner_group = learner_group

    @property
    def config(self):
        raise AssertionError("native diagnostics must not select by stack flags")


def test_algorithm_mixin_publishes_weighted_native_report_without_stack_branch():
    learner_group = _FakeLearnerGroup(
        [
            _FakeLearner(
                {
                    "module_a": {
                        "clip_fraction_clipped_count": 1,
                        "clip_fraction_valid_count": 2,
                    }
                }
            ),
            _FakeLearner(
                {
                    "module_b": {
                        "clip_fraction_clipped_count": 4,
                        "clip_fraction_valid_count": 18,
                    }
                }
            ),
        ]
    )
    algorithm = _NativeDiagnosticPPO(learner_group)

    report = algorithm.train()["learners"]["default_policy"]

    assert report == {
        "loss": 1.0,
        "clip_fraction": 5 / 20,
        "clip_fraction_clipped_count": 5,
        "clip_fraction_valid_count": 20,
        "clip_fraction_version": CLIP_FRACTION_VERSION,
        "clip_fraction_scope": CLIP_FRACTION_SCOPE,
    }


def test_native_clip_fraction_report_rejects_empty_optimizer_visits():
    with pytest.raises(RuntimeError, match="no valid optimizer samples"):
        native_clip_fraction_report([{}])
