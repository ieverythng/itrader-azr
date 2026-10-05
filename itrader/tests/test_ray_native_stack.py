from __future__ import annotations

import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ray")
pytest.importorskip("torch")


def test_active_builder_rejects_legacy_stack_selection(tmp_path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config
    from itrader.rl_lib.train.ray_native_config import build_native_ppo_config
    from itrader.tests.test_train_config import _args

    args = _args(seed=20261005)
    args.window = 60
    config = build_base_config(args, tmp_path / "synthetic.parquet", False)

    assert config["enable_rl_module_and_learner"] is True
    assert config["enable_env_runner_and_connector_v2"] is True
    assert "rllib_api_stack" not in config

    with pytest.raises(ValueError, match="[Ll]egacy API-stack selection is retired"):
        build_native_ppo_config({**config, "rllib_api_stack": "legacy"})
    with pytest.raises(ValueError, match="num_gpus"):
        build_native_ppo_config({**config, "num_gpus": 1})


@pytest.mark.parametrize("solver_state_dim", [0, 2])
def test_native_module_reproduces_legacy_mlp_film_logits_and_values(solver_state_dim):
    import torch
    from gymnasium.spaces import Box, Discrete
    from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    from itrader.models.solver_mlp_film_core import SolverMLPFiLMTensorCore
    from itrader.models.ray_native import SolverMLPFiLMTorchRLModule

    torch.manual_seed(19)
    obs_width = 5 * 4 + 32 + solver_state_dim
    obs_space = Box(-np.inf, np.inf, shape=(obs_width,), dtype=np.float32)
    action_space = Discrete(3)
    model_config = {
        "fcnet_hiddens": [32, 16],
        "custom_model_config": {
            "t_embed_dim": 32,
            "fcnet_hiddens": [32, 16],
            "conditioning": "film",
            "t_keep_p": 1.0,
            "t_proj": 12,
            "solver_state_dim": solver_state_dim,
            "solver_state_width": 8,
            "solver_state_residual_init": "zero_v1",
        },
    }
    class CoreHarness(torch.nn.Module, SolverMLPFiLMTensorCore):
        def __init__(self):
            super().__init__()
            self._setup_solver_mlp_film(
                obs_space,
                action_space.n,
                model_config["custom_model_config"],
            )

    core = CoreHarness()
    module = RLModuleSpec(
        module_class=SolverMLPFiLMTorchRLModule,
        observation_space=obs_space,
        action_space=action_space,
        model_config=deepcopy(model_config["custom_model_config"]),
        catalog_class=PPOCatalog,
    ).build()
    module.load_state_dict(core.state_dict(), strict=True)
    assert list(core.state_dict()) == list(module.state_dict())

    observations = torch.randn(7, obs_width)
    outputs = module.forward_inference({Columns.OBS: observations})
    core_logits, core_values, _ = core._solver_forward(
        observations, is_training=False
    )

    torch.testing.assert_close(outputs[Columns.ACTION_DIST_INPUTS], core_logits)
    torch.testing.assert_close(outputs[Columns.VF_PREDS], core_values)
    train_outputs = module._forward_for_phase(
        {Columns.OBS: observations},
        training=True,
    )
    torch.testing.assert_close(
        module.compute_values(
            {Columns.OBS: observations},
            train_outputs[Columns.EMBEDDINGS],
        ),
        core_values,
    )
    assert torch.isfinite(outputs[Columns.ACTION_DIST_INPUTS]).all()
    assert torch.isfinite(outputs[Columns.VF_PREDS]).all()

    core.zero_grad(set_to_none=True)
    (core_logits.square().mean() + core_values.square().mean()).backward()
    gradients = [parameter.grad for parameter in core.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_native_ppo_builder_preserves_the_corrected_optimizer_recipe(tmp_path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config
    from itrader.rl_lib.train.ray_native_config import build_native_ppo_config
    from itrader.tests.test_train_config import _args

    args = _args(seed=20261005)
    args.window = 60
    args.train_batch_size = 4800
    args.sgd_minibatch_size = 128
    args.num_sgd_iter = 30
    args.lr = 7.151674865214668e-6
    args.gamma = 0.99
    args.gae_lambda = 0.9
    args.clip_param = 0.1
    args.vf_clip_param = 20.0
    args.vf_loss_coeff = 2.0
    args.entropy_coeff_schedule = "0:0.005,96000:0.002,240000:0.0005"
    args.solver_state_mode = "both"
    args.solver_state_residual_init = "zero_v1"
    args.observation_transform = "market_relative_v1"
    args.observation_filter = "NoFilter"
    args.representation_profile = "r0_r1_v1"
    config = build_base_config(args, tmp_path / "synthetic.parquet", False)

    native = build_native_ppo_config(config)

    assert native.enable_rl_module_and_learner is True
    assert native.enable_env_runner_and_connector_v2 is True
    assert native.framework_str == "torch"
    assert native.train_batch_size_per_learner == 4800
    assert native.minibatch_size == 128
    assert native.num_epochs == 30
    assert native.lr == pytest.approx(7.151674865214668e-6)
    assert native.gamma == pytest.approx(0.99)
    assert native.lambda_ == pytest.approx(0.9)
    assert native.clip_param == pytest.approx(0.1)
    assert native.vf_clip_param == pytest.approx(20.0)
    assert native.vf_loss_coeff == pytest.approx(2.0)
    assert native.entropy_coeff == [
        [0, 0.005],
        [96000, 0.002],
        [240000, 0.0005],
    ]
    assert native.rl_module_spec.module_class.__name__ == "SolverMLPFiLMTorchRLModule"
    assert native.rl_module_spec.model_config["solver_state_dim"] == 2
    assert native.rl_module_spec.model_config["solver_state_residual_init"] == "zero_v1"


def test_native_deterministic_inference_rejects_multiple_observations():
    from types import SimpleNamespace

    from itrader.rl_lib.evaluation.ray_native_inference import (
        deterministic_action_details,
    )

    class Module:
        def forward_inference(self, _batch):
            raise AssertionError("multirow input must fail before module inference")

    algorithm = SimpleNamespace(
        config=SimpleNamespace(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        ),
        get_module=lambda: Module(),
    )

    with pytest.raises(ValueError, match="exactly one observation"):
        deterministic_action_details(algorithm, np.zeros((2, 332), dtype=np.float32))


@pytest.mark.parametrize("solver_state_mode", ["off", "both"])
def test_native_ppo_checkpoint_restore_replays_deterministically(
    tmp_path, tiny_ohlcv_path, solver_state_mode
):
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm

    from itrader.envs.itrader_env import ITraderEnv
    from itrader.rl_lib.evaluation.ray_native_inference import deterministic_action
    from itrader.rl_lib.train.ray_native_config import build_native_ppo_config
    from itrader.rl_lib.train.ray_native_ppo import DiagnosticPPO
    from itrader.rl_lib.train.train_solo_V2 import (
        build_base_config,
        make_env,
    )
    from itrader.tests.test_train_config import _args

    args = _args(seed=20261005)
    args.cpu_only = True
    args.workers = 0
    args.window = 60
    args.train_batch_size = 96
    args.sgd_minibatch_size = 32
    args.num_sgd_iter = 1
    args.lr = 7.151674865214668e-6
    args.gamma = 0.99
    args.gae_lambda = 0.9
    args.clip_param = 0.1
    args.vf_loss_coeff = 2.0
    args.vf_clip_param = 20.0
    args.entropy_coeff_schedule = "0:0.005,96000:0.002,240000:0.0005"
    args.evaluation_interval = 0
    args.evaluation_num_episodes = 1
    args.task_horizons = "24"
    args.task_reward_ids = "pnl_linear"
    args.solver_state_mode = solver_state_mode
    args.solver_state_residual_init = "zero_v1"
    args.representation_profile = "r0_r1_v1"
    args.observation_transform = "market_relative_v1"
    args.observation_filter = "NoFilter"
    args.auto_plot_on_end = False
    app_config = build_base_config(args, tiny_ohlcv_path, False)
    app_config["num_gpus_per_learner"] = 0
    app_config["num_gpus_per_env_runner"] = 0
    app_config["evaluation_num_env_runners"] = 0
    if solver_state_mode == "off":
        app_config["evaluation_interval"] = 1
        app_config["checkpoint_consistent_evaluation"] = True
    ray.init(
        include_dashboard=False,
        num_cpus=2,
        _temp_dir=str(tmp_path / "ray"),
    )
    from ray import tune

    tune.register_env("ITraderEnv", make_env)
    algorithm = None
    restored = None
    env = None
    try:
        algorithm = DiagnosticPPO(config=build_native_ppo_config(app_config))
        if solver_state_mode == "off":
            from itrader.rl_lib.train.train_solo_V2 import (
                train_with_checkpoint_consistent_evaluation,
            )

            result = train_with_checkpoint_consistent_evaluation(algorithm)
            assert "evaluation" in result
        else:
            result = algorithm.train()
        assert result["training_iteration"] == 1
        learner_stats = result["learners"]["default_policy"]
        assert np.isfinite(learner_stats["total_loss"])
        assert np.isfinite(learner_stats["default_optimizer_learning_rate"])
        assert learner_stats["default_optimizer_learning_rate"] == pytest.approx(
            app_config["lr"]
        )
        assert algorithm.config.entropy_coeff == app_config["entropy_coeff"]
        assert learner_stats["curr_entropy_coeff"] == pytest.approx(0.004997)
        assert algorithm.config.gamma == pytest.approx(0.99)
        assert algorithm.config.lambda_ == pytest.approx(0.9)
        assert algorithm.config.clip_param == pytest.approx(0.1)
        assert algorithm.config.vf_clip_param == pytest.approx(20.0)
        assert algorithm.config.vf_loss_coeff == pytest.approx(2.0)
        assert algorithm.config.minibatch_size == 32
        assert algorithm.config.num_epochs == 1
        assert learner_stats["clip_fraction_version"] == "ppo_ratio_outside_v1"
        assert learner_stats["clip_fraction_scope"] == (
            "all_optimizer_sample_visits_per_train_report"
        )
        assert learner_stats["clip_fraction_valid_count"] > 0
        assert 0.0 <= learner_stats["clip_fraction"] <= 1.0
        assert set(result["custom_metrics"]) >= {
            "optimization_reward__sum",
            "optimization_reward__count",
            "act_1_frac__sum",
            "act_1_frac__count",
        }
        action_fraction_sum = result["custom_metrics"]["act_1_frac__sum"]
        action_fraction_count = result["custom_metrics"]["act_1_frac__count"]
        assert np.isfinite(action_fraction_sum)
        assert action_fraction_count > 0
        assert 0.0 <= action_fraction_sum <= action_fraction_count
        assert learner_stats["num_module_steps_trained"] > 0

        env = ITraderEnv(app_config["env_config"])
        observation, _ = env.reset(seed=901)
        observations = []
        for _ in range(6):
            observations.append(observation.copy())
            action = deterministic_action(algorithm, observation)
            observation, _, terminated, truncated, _ = env.step(action)
            assert not terminated and not truncated
        expected_actions = [deterministic_action(algorithm, obs) for obs in observations]

        checkpoint_path = algorithm.save_to_path(str(tmp_path / "checkpoint"))
        restored = Algorithm.from_checkpoint(str(checkpoint_path))
        from itrader.models.registry import validate_native_restored_model

        validate_native_restored_model(restored, app_config)
        actual_actions = [
            deterministic_action(restored, obs) for obs in observations
        ]
        assert actual_actions == expected_actions
        restored.stop()
        restored = None

        # Model a PBT/manual donor restore into a target run with changed
        # hyperparameters. Weight and Adam-moment state must survive while the
        # target optimizer recipe and schedule clock are re-applied.
        target_config = deepcopy(app_config)
        target_config.update(
            {
                "lr": 1.1e-5,
                "lambda_": 0.95,
                "clip_param": 0.2,
                "vf_loss_coeff": 1.0,
                "vf_clip_param": 10.0,
                "entropy_coeff": [[0, 0.004], [100, 0.002]],
                "minibatch_size": 16,
                "num_epochs": 2,
            }
        )
        algorithm.stop()
        algorithm = None
        donor_restored = DiagnosticPPO(config=build_native_ppo_config(target_config))
        donor_restored.restore(str(checkpoint_path))
        validate_native_restored_model(donor_restored, target_config)
        from itrader.rl_lib.train.ppo_native_restore import (
            reconcile_native_ppo_after_restore,
        )

        receipt = reconcile_native_ppo_after_restore(
            donor_restored,
            target_config,
            progress_steps=123,
        )
        assert receipt["optimizer_state_preserved"] is True
        assert donor_restored.config.lambda_ == pytest.approx(0.95)
        assert donor_restored.config.clip_param == pytest.approx(0.2)
        assert donor_restored.config.num_epochs == 2
        assert donor_restored.config.minibatch_size == 16
        assert donor_restored.config.lr == pytest.approx(1.1e-5)
        assert receipt["training_learners"][0]["optimizer_learning_rates"] == [
            pytest.approx(1.1e-5)
        ]
        resumed_actions = [
            deterministic_action(donor_restored, obs) for obs in observations
        ]
        assert resumed_actions == expected_actions
        donor_restored.stop()
        donor_restored = None
    finally:
        if env is not None:
            env.close()
        if "donor_restored" in locals() and donor_restored is not None:
            donor_restored.stop()
        if restored is not None:
            restored.stop()
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()


def test_native_training_cli_completes_and_persists_a_restorable_checkpoint(
    tmp_path, tiny_ohlcv_path
):
    from itrader.rl_lib.api_stack import NATIVE_API_STACK_ID

    results_root = tmp_path / "ray-results"
    experiment_name = "native-cli-e2e"
    repository_root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-m",
        "itrader.rl_lib.train.train_solo_V2",
        "--iters",
        "1",
        "--window",
        "60",
        "--workers",
        "0",
        "--ray_num_cpus",
        "2",
        "--cpu_only",
        "--train_batch_size",
        "96",
        "--sgd_minibatch_size",
        "16",
        "--num_sgd_iter",
        "1",
        "--lr",
        "7.151674865214668e-6",
        "--lambda",
        "0.9",
        "--clip_param",
        "0.1",
        "--vf_clip_param",
        "20",
        "--vf_loss_coeff",
        "2",
        "--entropy_coeff_schedule",
        "0:0.005,96000:0.002,240000:0.0005",
        "--pbt_interval",
        "1",
        "--checkpoint_interval",
        "1",
        "--evaluation_num_episodes",
        "1",
        "--split_purge_rows",
        "0",
        "--task_horizons",
        "24",
        "--task_reward_ids",
        "pnl_linear",
        "--representation_profile",
        "r0_r1_v1",
        "--observation_transform",
        "market_relative_v1",
        "--solver_state_mode",
        "both",
        "--solver_state_residual_init",
        "zero_v1",
        "--no-auto_plot_on_end",
        "--experiment_name",
        experiment_name,
        "--results_root",
        str(results_root),
        "--data_path",
        str(tiny_ohlcv_path),
    ]
    completed = subprocess.run(
        command,
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, (
        "native training CLI failed\n"
        f"stdout:\n{completed.stdout[-12000:]}\n"
        f"stderr:\n{completed.stderr[-12000:]}"
    )

    stable_run_dir = results_root / "_stable_runs" / experiment_name
    manifests = list(stable_run_dir.glob("run_manifest_*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["api_stack_id"] == NATIVE_API_STACK_ID
    assert manifest["runtime"]["ray"] == "2.58.0"

    status_path = stable_run_dir / f"run_status_{manifest['run_id']}.json"
    run_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert run_status["status"] == "completed"
    assert run_status["trial_count"] == 1
    last_result = run_status["trials"][0]["last_result"]
    assert last_result["training_iteration"] == 1
    assert last_result["pbt_progress_steps"] >= 96

    index_paths = list((results_root / "_stable_metrics").rglob("checkpoint_index.jsonl"))
    assert len(index_paths) == 1
    checkpoint_rows = [
        json.loads(line)
        for line in index_paths[0].read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(checkpoint_rows) == 1
    checkpoint_record = checkpoint_rows[0]
    assert checkpoint_record["api_stack_id"] == manifest["api_stack_id"]
    assert checkpoint_record["reported_to_tune"] is True
    checkpoint_path = Path(checkpoint_record["stable_checkpoint"])
    assert checkpoint_path.is_dir()
    assert (checkpoint_path / "rllib_checkpoint.json").is_file()
    assert (
        checkpoint_path
        / "learner_group"
        / "learner"
        / "rl_module"
        / "default_policy"
        / "module_state.pkl"
    ).is_file()

    stable_trial_dir = index_paths[0].parent
    trial_status = json.loads(
        (stable_trial_dir / "trial_status.json").read_text(encoding="utf-8")
    )
    assert trial_status["status"] == "completed"
    assert trial_status["api_stack_id"] == NATIVE_API_STACK_ID
    metric_rows = [
        json.loads(line)
        for line in (stable_trial_dir / "metrics_full.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(metric_rows) == 1
    assert metric_rows[0]["training_iteration"] == 1
    assert metric_rows[0]["pbt_progress_steps"] >= 96
    report_rows = list(
        csv.DictReader(
            (stable_trial_dir / "report_rows.csv")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    )
    assert len(report_rows) == 1
    report_row = report_rows[0]
    assert report_row["training_iteration"] == "1"
    assert report_row["pbt_progress_steps"] == str(metric_rows[0]["pbt_progress_steps"])
    assert "evaluation/net_return_mean" in report_row
