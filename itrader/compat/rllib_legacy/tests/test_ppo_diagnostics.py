"""Exact diagnostic examples plus real legacy PPO numerical/learner parity."""

import gymnasium as gym
import numpy as np
import pytest
import ray
import torch
from gymnasium.spaces import Box, Discrete
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from itrader.compat.rllib_legacy.models.solver_mlp_film import SolverMLPFiLM
from itrader.compat.rllib_legacy.train.ppo_diagnostics import ClipFractionAccumulator

pytestmark = pytest.mark.skipif(
    ray.__version__ != "2.58.0",
    reason="diagnostic adapter pins the forward Ray runtime",
)


def test_report_counts_optimizer_visits_with_strict_bounds_and_padding():
    counts = ClipFractionAccumulator()
    counts.record(torch.tensor([2.0]), 0.5)  # Initialization is not training.
    assert counts.consume()["clip_fraction"] is None
    counts.begin()
    for _ in range(2):
        counts.record(torch.tensor([0.25, 0.5, 1.0, 1.5, 2.0]), 0.5)
        counts.record(
            torch.tensor([2.0, float("nan"), 0.0]),
            0.5,
            torch.tensor([True, False, False]),
        )
    report = counts.consume()
    assert report["clip_fraction"] == 0.5
    assert report["clip_fraction_clipped_count"] == 6
    assert report["clip_fraction_valid_count"] == 12
    assert counts.consume()["clip_fraction"] is None
    counts.begin()
    counts.record(torch.tensor([0.5, 1.0, 1.5]), 0.5)
    assert counts.consume()["clip_fraction"] == 0.0


def _config():
    ModelCatalog.register_custom_model("clip_test_film", SolverMLPFiLM)
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .to_dict()
    )
    config.update({"framework": "torch", "num_gpus": 0, "lr": 0.001})
    config["model"].update(
        {
            "custom_model": "clip_test_film",
            "custom_model_config": {
                "t_embed_dim": 4,
                "fcnet_hiddens": [8],
                "t_proj": 4,
                "conditioning": "film",
                "t_keep_p": 0.5,
            },
        }
    )
    return config


def _policy(cls):
    return cls(Box(-10, 10, (16,), np.float32), Discrete(3), _config())


def _batch(policy):
    obs = np.linspace(-1, 1, 8 * 16, dtype=np.float32).reshape(8, 16)
    actions, _, extras = policy.compute_actions(obs)
    return SampleBatch(
        {
            "obs": obs,
            "actions": actions,
            "action_dist_inputs": extras["action_dist_inputs"],
            "action_logp": extras["action_logp"],
            "advantages": np.array([1, -1] * 4, np.float32),
            "value_targets": np.linspace(-1, 1, 8, dtype=np.float32),
        }
    )


def test_real_policy_task_dropout_loss_gradients_rng_actions_and_update_match():
    from itrader.compat.rllib_legacy.train.ppo_diagnostics import DiagnosticPPOTorchPolicy

    original = _policy(PPOTorchPolicy)
    instrumented = _policy(DiagnosticPPOTorchPolicy)
    instrumented.set_weights(original.get_weights())
    batch = _batch(original)
    assert instrumented.consume_clip_fraction()["clip_fraction"] is None
    instrumented.begin_clip_fraction()
    outcomes = []
    for policy in (original, instrumented):
        policy.model.train()
        policy.model.zero_grad()
        tensor_batch = SampleBatch({k: torch.as_tensor(v) for k, v in batch.items()})
        tensor_batch.set_training(True)
        torch.manual_seed(928)
        loss = policy.loss(policy.model, policy.dist_class, tensor_batch)
        loss.backward()
        outcomes.append(
            (
                loss.detach().clone(),
                [p.grad.clone() for p in policy.model.parameters()],
                torch.random.get_rng_state().clone(),
            )
        )
    for before, after in zip(outcomes[0], outcomes[1]):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert instrumented.consume_clip_fraction()["clip_fraction_valid_count"] == 8

    instrumented.begin_clip_fraction()
    outcomes = []
    for policy in (original, instrumented):
        torch.manual_seed(137)
        result = policy.learn_on_batch(batch.copy())
        actions = policy.compute_actions(batch["obs"])[0]
        outcomes.append(
            (
                result["learner_stats"]["total_loss"],
                actions,
                torch.random.get_rng_state().clone(),
                policy.get_weights(),
            )
        )
    assert outcomes[0][0] == outcomes[1][0]
    np.testing.assert_array_equal(outcomes[0][1], outcomes[1][1])
    torch.testing.assert_close(outcomes[0][2], outcomes[1][2], rtol=0, atol=0)
    for name in outcomes[0][3]:
        np.testing.assert_array_equal(outcomes[0][3][name], outcomes[1][3][name])
    assert instrumented.consume_clip_fraction()["clip_fraction_valid_count"] == 8


class TinyClipEnv(gym.Env):
    observation_space = Box(-10, 10, (16,), np.float32)
    action_space = Discrete(3)

    def __init__(self, config=None):
        super().__init__()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.linspace(-1, 1, 16, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return (
            np.linspace(-1, 1, 16, dtype=np.float32),
            float(action == 1),
            self.steps == 4,
            False,
            {},
        )


@pytest.fixture(scope="module")
def local_ray():
    owned = not ray.is_initialized()
    if owned:
        ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
    yield
    if owned:
        ray.shutdown()


@pytest.mark.parametrize(
    "simple_optimizer,fake_gpus", [(True, 0), (False, 0), (False, 2)]
)
def test_real_legacy_learner_exports_per_report_counts(
    simple_optimizer,
    fake_gpus,
    local_ray,
    tmp_path,
):
    from itrader.compat.rllib_legacy.train.ppo_diagnostics import DiagnosticPPO

    config = _config()
    config.update(
        {
            "env": TinyClipEnv,
            "num_env_runners": 0,
            "num_envs_per_env_runner": 1,
            "rollout_fragment_length": 12,
            "train_batch_size": 12,
            "minibatch_size": 4,
            "num_epochs": 2,
            "simple_optimizer": simple_optimizer,
            "num_cpus_for_main_process": 1,
            "min_time_s_per_iteration": 0,
            "seed": 37,
            "num_gpus": fake_gpus,
            "_fake_gpus": bool(fake_gpus),
        }
    )
    algo = DiagnosticPPO(config=config)
    try:
        for _ in range(2):
            result = algo.train()
            stats = result["info"]["learner"]["default_policy"]["learner_stats"]
            assert stats["clip_fraction_valid_count"] == 24
            assert 0 <= stats["clip_fraction"] <= 1
            assert stats["clip_fraction"] == stats["clip_fraction_clipped_count"] / 24
            assert algo.get_policy().consume_clip_fraction()["clip_fraction"] is None
        if not simple_optimizer and not fake_gpus:
            saved = algo.save(str(tmp_path / "checkpoint"))
            restored = DiagnosticPPO.from_checkpoint(saved.checkpoint.path)
            try:
                for name, value in algo.get_policy().get_weights().items():
                    np.testing.assert_array_equal(
                        value, restored.get_policy().get_weights()[name]
                    )
                result = restored.train()
                stats = result["info"]["learner"]["default_policy"]["learner_stats"]
                assert stats["clip_fraction_valid_count"] == 24
            finally:
                restored.stop()
    finally:
        algo.stop()


@pytest.mark.parametrize("bad_ratio", [float("nan"), float("inf")])
def test_nonfinite_valid_ratio_is_a_diagnostic_failure(bad_ratio):
    counts = ClipFractionAccumulator()
    counts.begin()
    with pytest.raises(FloatingPointError, match="non-finite"):
        counts.record(torch.tensor([1.0, bad_ratio]), 0.5)


class ConstantModel:
    """Controlled model output at the policy loss seam, with real Categorical."""

    def __init__(self, size, recurrent):
        self.size = size
        self.recurrent = recurrent
        self.tower_stats = {}

    def __call__(self, batch):
        return torch.zeros(self.size, 3, dtype=torch.float64), (
            [0] if self.recurrent else []
        )

    def is_time_major(self):
        return False

    def value_function(self):
        return torch.zeros(self.size, dtype=torch.float64)


@pytest.mark.parametrize("advantage_sign", [1, -1])
@pytest.mark.parametrize(
    "seq_lens,expected_valid,expected_clipped",
    [
        (None, 6, 2),
        ([3, 1], 4, 2),
        ([0, 0], 0, 0),
    ],
)
def test_loss_counts_same_ratio_with_advantage_sign_and_padding(
    advantage_sign,
    seq_lens,
    expected_valid,
    expected_clipped,
):
    from itrader.compat.rllib_legacy.train.ppo_diagnostics import DiagnosticPPOTorchPolicy

    policy = _policy(DiagnosticPPOTorchPolicy)
    policy.config["clip_param"] = 0.5
    ratios = torch.tensor([0.25, 0.5, 1.0, 2.0, 1.5, 1.0], dtype=torch.float64)
    batch = SampleBatch(
        {
            "actions": torch.zeros(6, dtype=torch.int64),
            "action_dist_inputs": torch.zeros(6, 3, dtype=torch.float64),
            "action_logp": -torch.log(torch.tensor(3.0, dtype=torch.float64))
            - ratios.log(),
            "advantages": advantage_sign * torch.ones(6, dtype=torch.float64),
            "value_targets": torch.zeros(6, dtype=torch.float64),
        }
    )
    if seq_lens is not None:
        batch["seq_lens"] = torch.tensor(seq_lens)
        batch["action_logp"][4:] = float("nan")
    policy.begin_clip_fraction()
    model = ConstantModel(6, seq_lens is not None)
    policy.loss(model, policy.dist_class, batch)
    report = policy.consume_clip_fraction()
    assert report["clip_fraction_valid_count"] == expected_valid
    assert report["clip_fraction_clipped_count"] == expected_clipped
    assert report["clip_fraction"] == (
        expected_clipped / expected_valid if expected_valid else None
    )
