"""Real legacy PPO restores must preserve donor state and apply target settings."""

from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest
import ray
import torch
from gymnasium.spaces import Box, Discrete
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.sample_batch import SampleBatch

from itrader.compat.rllib_legacy.train.ppo_diagnostics import DiagnosticPPO
from itrader.compat.rllib_legacy.train.ppo_restore import reconcile_ppo_after_restore

pytestmark = pytest.mark.skipif(ray.__version__ != "2.58.0", reason="Ray 2.58 adapter")


class RestoreEnv(gym.Env):
    observation_space = Box(-1, 1, (4,), np.float32)
    action_space = Discrete(2)

    def __init__(self, config=None):
        super().__init__()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(4, np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(4, np.float32), float(action), self.steps == 4, False, {}


def _config(donor, scheduled):
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .to_dict()
    )
    config.update(
        {
            "env": RestoreEnv,
            "framework": "torch",
            "num_gpus": 0,
            "num_env_runners": 0,
            "rollout_fragment_length": 12,
            "train_batch_size": 12,
            "minibatch_size": 6 if donor else 4,
            "num_epochs": 1 if donor else 3,
            "simple_optimizer": False,
            "min_time_s_per_iteration": 0,
            "seed": 43,
            "log_level": "ERROR",
            "lr": 0.001 if donor else 0.003,
            "gamma": 0.91 if donor else 0.99,
            "lambda": 0.82 if donor else 0.96,
            "clip_param": 0.12 if donor else 0.3,
            "vf_loss_coeff": 0.5 if donor else 1.2,
            "vf_clip_param": 5.0 if donor else 10.0,
            "entropy_coeff": 0.01 if donor else 0.04,
            "evaluation_interval": 1,
            "evaluation_duration": 1,
            "evaluation_num_env_runners": 0,
            "evaluation_config": {"explore": False},
        }
    )
    config["model"]["fcnet_hiddens"] = [4]
    if scheduled:
        config["lr_schedule"] = (
            [[0, 0.001], [100, 0.0001]] if donor else [[0, 0.003], [100, 0.002]]
        )
        config["entropy_coeff_schedule"] = (
            [[0, 0.01], [100, 0.0]] if donor else [[0, 0.04], [100, 0.02]]
        )
    return config


@pytest.fixture(scope="module")
def local_ray():
    owned = not ray.is_initialized()
    if owned:
        ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
    yield
    if owned:
        ray.shutdown()


@pytest.mark.parametrize(
    "scheduled,expected_lr,expected_entropy,remote_workers",
    [
        (False, 0.003, 0.04, 0),
        (True, 0.0026, 0.032, 0),
        (False, 0.003, 0.04, 1),
    ],
)
def test_restore_applies_target_settings_without_replacing_donor_state(
    scheduled,
    expected_lr,
    expected_entropy,
    remote_workers,
    local_ray,
    tmp_path,
):
    target_config = _config(False, scheduled)
    donor_config = _config(True, scheduled)
    for config in (donor_config, target_config):
        config["num_env_runners"] = remote_workers
        config["num_cpus_per_env_runner"] = 0.25
        if remote_workers:
            # Workers can import Gym's environment without importing pytest's
            # transient test-module namespace.
            config["env"] = "CartPole-v1"
    donor = DiagnosticPPO(config=donor_config)
    target = DiagnosticPPO(config=target_config)
    try:
        donor.train()
        saved = donor.save(str(tmp_path / "donor"))
        target.restore(saved.checkpoint.path)
        policy = target.get_policy()
        # The raw Ray restore bug predates this repair: target algorithm settings
        # coexist with donor policy settings and donor Adam parameter groups.
        assert target.config["gamma"] == 0.99
        assert policy.config["gamma"] == 0.91
        before = deepcopy(policy.get_state())
        counters = deepcopy(target._counters)
        receipt = reconcile_ppo_after_restore(target, target_config, progress_steps=40)
        after = policy.get_state()
        torch.testing.assert_close(before["weights"], after["weights"], rtol=0, atol=0)
        torch.testing.assert_close(
            before["_optimizer_variables"][0]["state"],
            after["_optimizer_variables"][0]["state"],
            rtol=0,
            atol=0,
        )
        assert before["current_kl_coeff"] == after["current_kl_coeff"]
        assert before["global_timestep"] == after["global_timestep"]
        assert target._counters == counters
        assert policy.cur_lr == pytest.approx(expected_lr)
        assert policy.entropy_coeff == pytest.approx(expected_entropy)
        assert receipt["progress_steps"] == 40
        assert receipt["training_policies"]
        assert len(receipt["training_policies"]) == 1 + remote_workers
        assert receipt["evaluation_policies"]
        for record in receipt["training_policies"] + receipt["evaluation_policies"]:
            assert record["learning_config"] == receipt["target_learning_config"]
            assert record["optimizer_lrs"] == pytest.approx([expected_lr])
        assert all(
            record["explore"] is False for record in receipt["evaluation_policies"]
        )

        trajectory = SampleBatch(
            {
                "obs": np.zeros((2, 4), np.float32),
                "new_obs": np.zeros((2, 4), np.float32),
                "actions": np.zeros(2, np.int64),
                "rewards": np.ones(2, np.float32),
                "vf_preds": np.zeros(2, np.float32),
                "terminateds": np.array([False, True]),
                "truncateds": np.array([False, False]),
            }
        )
        processed = policy.postprocess_trajectory(trajectory)
        np.testing.assert_allclose(processed["advantages"], [1.9504, 1.0], atol=1e-6)

        step_lrs = []
        hook = policy._optimizers[0].register_step_pre_hook(
            lambda optimizer, args, kwargs: step_lrs.append(
                optimizer.param_groups[0]["lr"]
            )
        )
        try:
            result = target.train()
        finally:
            hook.remove()
        assert step_lrs == pytest.approx([expected_lr] * 9)
        stats = result["info"]["learner"]["default_policy"]["learner_stats"]
        assert stats["cur_lr"] == pytest.approx(expected_lr)
        assert stats["clip_fraction_valid_count"] == 36
    finally:
        target.stop()
        donor.stop()


def test_restore_reconciliation_rejects_unknown_runtime(monkeypatch):
    monkeypatch.setattr(ray, "__version__", "2.59.0")
    with pytest.raises(RuntimeError, match="requires Ray 2.58.0"):
        reconcile_ppo_after_restore(None, {}, 0)


@pytest.mark.parametrize(
    "restore_kind", ["manual_restore", "tune_self_resume", "pbt_exploit"]
)
@pytest.mark.skip(
    reason="The former legacy training_fn is retained in source history, not the active trainer tree"
)
def test_training_entry_persists_reconciled_settings_before_first_report(
    restore_kind,
    local_ray,
    tmp_path,
    tiny_ohlcv_path,
    monkeypatch,
):
    import json
    from types import SimpleNamespace

    from itrader.models.registry import register_solver_models
    from itrader.rl_lib.evaluation.provenance import checkpoint_sha256
    from itrader.rl_lib.train import train_solo_V2
    from itrader.rl_lib.train.pbt_contract import PBTProgressClock

    register_solver_models()
    donor_config, target_config = _config(True, True), _config(False, True)
    for config in (donor_config, target_config):
        config["model"]["custom_model"] = "solver_mlp_film"
        config["model"]["custom_model_config"] = {
            "t_embed_dim": 2,
            "fcnet_hiddens": [4],
            "t_proj": 2,
            "t_keep_p": 1.0,
        }
        config["env_config"] = {"df_path": str(tiny_ohlcv_path)}
        config["evaluation_config"]["env_config"] = {
            "df_path": str(tiny_ohlcv_path),
            "task_sampler": {"fixed_tasks": []},
        }
    donor = DiagnosticPPO(config=donor_config)
    try:
        donor.train()
        checkpoint = donor.save(str(tmp_path / "donor")).checkpoint
    finally:
        donor.stop()

    trial_dir = tmp_path / "trial"
    stable_root = tmp_path / "stable"
    stable_trial = stable_root / trial_dir.name
    target_config.update(
        {
            "stop_iters": 1,
            "pbt_target_steps": 52,
            "stable_metrics_root": str(stable_root),
            "checkpoint_interval": 0,
            "auto_plot_on_end": False,
        }
    )
    if restore_kind == "manual_restore":
        target_config.update(
            restore_ckpt=checkpoint.path,
            restore_progress_steps=40,
            restore_progress_updates=3,
        )
        incoming = None
    else:
        incoming = checkpoint
        PBTProgressClock(target_steps=52, completed_steps=40, updates=3).save(
            stable_trial / "pbt_progress.json"
        )
    if restore_kind == "pbt_exploit":
        target_config["pbt_restore_generation"] = 1
        target_config["pbt_restore_provenance"] = {
            "exploit_id": "probe-exploit",
            "target_trial_id": "target",
            "donor_checkpoint_sha256": checkpoint_sha256(checkpoint.path),
            "donor_checkpoint_path": checkpoint.path,
            "donor_progress_steps": 12,
        }
    context = SimpleNamespace(
        get_trial_dir=lambda: str(trial_dir),
        get_trial_id=lambda: "target",
        get_trial_name=lambda: "restore-probe",
    )
    monkeypatch.setattr(train_solo_V2.tune, "get_context", lambda: context)
    monkeypatch.setattr(train_solo_V2.tune, "get_checkpoint", lambda: incoming)
    reports = []

    def capture_report(metrics, checkpoint=None):
        receipt_paths = list((stable_trial / "restore_receipts").glob("*.json"))
        assert len(receipt_paths) == 1
        receipt = json.loads(receipt_paths[0].read_text(encoding="utf-8"))
        assert receipt["receipt_kind"] == restore_kind
        assert receipt["target_clock_at_restore"] == 40
        measured = receipt["learning_settings_reconciliation"]
        assert measured["progress_steps"] == 40
        assert measured["schedule_application_scope"] == "first_update_after_restore"
        assert measured["subsequent_schedule_clock"] == "rllib_native_unchanged"
        assert measured["training_policies"][0]["optimizer_lrs"] == pytest.approx(
            [0.0026]
        )
        assert metrics["learning_rate"] == pytest.approx(0.0026)
        assert metrics["pbt_progress_steps"] == 52
        reports.append(metrics)

    monkeypatch.setattr(train_solo_V2.tune, "report", capture_report)
    train_solo_V2.training_fn(target_config)
    assert len(reports) == 1
