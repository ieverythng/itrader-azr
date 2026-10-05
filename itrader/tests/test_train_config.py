from argparse import Namespace
import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pydantic")
pytest.importorskip("ray")

from itrader.rl_lib.api_stack import LEGACY_API_STACK_ID, NATIVE_API_STACK_ID


def _args(seed: int = 1234):
    return Namespace(
        iters=5,
        window=30,
        workers=2,
        pbt=False,
        cpu_pbt=False,
        cpu_only=False,
        pop=2,
        timeout_hours=1,
        restore_ckpt="",
        best_params_json="",
        results_root="",
        start_idx_mode="random_episode",
        stream_jump_every_episodes=0,
        stream_jump_on_wrap=True,
        stream_stride=0,
        ret_norm_reset="episode",
        task_pool_path="",
        p_pool=0.7,
        data_path="itrader/data/ohlcv.parquet",
        seed=seed,
        pbt_interval=3,
        checkpoint_interval=3,
        evaluation_interval=1,
        evaluation_num_episodes=4,
        evaluation_task_panel_path="",
        evidence_cost_bps=1.0,
        evidence_periods_per_year=0.0,
        split_fractions="0.7,0.15,0.15",
        split_purge_rows=8,
    )


def test_init_ray_starts_without_optional_dashboard(monkeypatch):
    from itrader.rl_lib.train import train_solo_V2

    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(train_solo_V2.ray, "init", fake_init)

    train_solo_V2.init_ray(use_pbt_mode=True)

    assert calls == [{"include_dashboard": False}]


def test_init_ray_can_cap_declared_cpus_for_memory_safe_pbt(monkeypatch):
    from itrader.rl_lib.train import train_solo_V2

    calls = []
    monkeypatch.setattr(train_solo_V2.ray, "init", lambda **kwargs: calls.append(kwargs))

    train_solo_V2.init_ray(use_pbt_mode=True, num_cpus=4)

    assert calls == [{"num_cpus": 4, "include_dashboard": False}]


def test_cpu_resources_respect_a_single_requested_rollout_worker():
    from itrader.rl_lib.train.train_solo_V2 import _resources_per_trial

    args = _args()
    args.cpu_only = True
    args.workers = 1
    config = {}

    resources = _resources_per_trial(args, config)

    assert config["num_env_runners"] == 1
    assert config["num_gpus_per_learner"] == 0
    assert "num_gpus" not in config
    assert resources.bundles == [{"CPU": 1.0}, {"CPU": 1.0}]


def test_cpu_resources_allow_memory_safe_local_worker_topology():
    from itrader.rl_lib.train.train_solo_V2 import _resources_per_trial

    args = _args()
    args.cpu_only = True
    args.workers = 0
    config = {}

    resources = _resources_per_trial(args, config)

    assert config["num_env_runners"] == 0
    assert config["num_gpus_per_learner"] == 0
    assert "num_gpus" not in config
    assert resources.bundles == [{"CPU": 1.0}]


def test_gpu_resources_configure_the_native_learner_device():
    from itrader.rl_lib.train.train_solo_V2 import _resources_per_trial

    args = _args()
    args.cpu_only = False
    args.cpu_pbt = False
    args.workers = 0
    config = {}

    resources = _resources_per_trial(args, config)

    assert config["num_gpus_per_learner"] == 1
    assert "num_gpus" not in config
    assert resources.bundles == [{"GPU": 1.0, "CPU": 1.0}]


def test_gpu_resource_selection_reaches_native_learner_config(tmp_path: Path):
    from itrader.rl_lib.train.ray_native_config import build_native_ppo_config
    from itrader.rl_lib.train.train_solo_V2 import (
        _resources_per_trial,
        build_base_config,
    )

    args = _args()
    args.cpu_only = False
    args.cpu_pbt = False
    args.workers = 0
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")
    config = build_base_config(args, data_path, use_pbt_mode=False)
    _resources_per_trial(args, config)

    native = build_native_ppo_config(config)

    assert native.num_gpus_per_learner == 1


def test_resources_reject_negative_rollout_worker_count():
    from itrader.rl_lib.train.train_solo_V2 import _resources_per_trial

    args = _args()
    args.cpu_only = True
    args.workers = -1

    with pytest.raises(ValueError, match="workers must be non-negative"):
        _resources_per_trial(args, {})


@pytest.mark.parametrize("filename", [
    "phase_a_corrected_ray258_r0_r4v1_fixed_smoke_20261005.json",
    "phase_a_corrected_ray258_r0_r4v1_fixed_2x2_cost_20261005.json",
])
def test_historical_ray258_legacy_recipes_are_preserved_but_not_active(filename):
    from itrader.scripts.experiments.run_ablation import validate_active_loadout

    root = Path(__file__).resolve().parents[2]
    loadout = json.loads(
        (root / "itrader/compat/rllib_legacy/configs" / filename).read_text()
    )
    assert loadout["common_args"]["rllib_api_stack"] == "legacy"
    assert loadout["common_args"]["train_batch_size"] == 4800
    assert loadout["common_args"]["sgd_minibatch_size"] == 128
    with pytest.raises(ValueError, match="compatibility-only"):
        validate_active_loadout(loadout["common_args"], loadout["variants"])


def test_fixed_recipe_entropy_schedule_reaches_rllib_without_changing_default(tmp_path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args()
    default = build_base_config(args, tmp_path / "market.parquet", False)
    assert default["entropy_coeff"] == [[0, 0.02], [200000, 0.01], [800000, 0.003]]
    args.entropy_coeff_schedule = "0:0.005,96000:0.002,240000:0.0005"
    selected = build_base_config(args, tmp_path / "market.parquet", False)
    assert selected["entropy_coeff"] == [[0, 0.005], [96000, 0.002], [240000, 0.0005]]


@pytest.mark.parametrize("schedule", ["1:0.005", "0:-0.1", "0:nan", "0:inf", "0:0.1,0:0.2", "0:0.1,3.5:0.0"])
def test_fixed_recipe_rejects_invalid_entropy_schedule(tmp_path, schedule):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args()
    args.entropy_coeff_schedule = schedule
    with pytest.raises(ValueError, match="entropy_coeff_schedule"):
        build_base_config(args, tmp_path / "market.parquet", False)


def test_build_base_config_sets_seed_in_all_required_places(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=2026)
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=False)
    assert cfg["seed"] == 2026
    assert cfg["env_config"]["task_sampler"]["seed"] == 2026
    assert cfg["env_config"]["task_sampler"]["bar_resolution"] == "1m"


def test_build_base_config_wires_explicit_solver_state_without_previous_action(
    tmp_path: Path,
):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=2026)
    args.solver_state_mode = "both"
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=False)

    assert cfg["env_config"]["solver_state_mode"] == "both"
    assert cfg["evaluation_config"]["env_config"]["solver_state_mode"] == "both"
    assert cfg["env_config"]["base_obs_dim"] == 30 * 5 + 32
    assert cfg["env_config"]["solver_state_dim"] == 2
    assert cfg["env_config"]["agent_obs_dim"] == 30 * 5 + 32 + 2
    assert cfg["env_config"]["observation_schema_id"] == "r4_v1_min"
    assert cfg["evaluation_config"]["env_config"]["agent_obs_dim"] == 184
    assert cfg["model"]["custom_model_config"]["solver_state_dim"] == 2
    assert "previous_action" not in json.dumps(cfg, default=str)


def test_build_base_config_wires_zero_initialized_state_residual_profile(
    tmp_path: Path,
):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=2026)
    args.solver_state_mode = "both"
    args.solver_state_residual_init = "zero_v1"
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=False)

    assert (
        cfg["model"]["custom_model_config"]["solver_state_residual_init"]
        == "zero_v1"
    )


def test_parameter_matched_solver_state_modes_share_one_model_shape(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")
    model_configs = []

    for mode in ("masked", "position", "time", "both"):
        args = _args(seed=2026)
        args.solver_state_mode = mode
        cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=False)

        assert cfg["env_config"]["solver_state_mode"] == mode
        assert cfg["evaluation_config"]["env_config"]["solver_state_mode"] == mode
        model_configs.append(cfg["model"])

    assert all(model_config == model_configs[0] for model_config in model_configs)
    assert model_configs[0]["custom_model_config"]["solver_state_dim"] == 2
    assert model_configs[0]["custom_model_config"]["solver_state_width"] == 8


def test_build_base_config_keeps_solver_state_disabled_by_default(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(
        args=_args(seed=2026),
        data_path=data_path,
        use_pbt_mode=False,
    )

    assert cfg["env_config"]["solver_state_mode"] == "off"
    assert cfg["env_config"]["base_obs_dim"] == 30 * 5 + 32
    assert cfg["env_config"]["solver_state_dim"] == 0
    assert cfg["env_config"]["agent_obs_dim"] == 30 * 5 + 32
    assert cfg["model"]["custom_model_config"]["solver_state_dim"] == 0


def test_build_base_config_selects_rth_snapshot_partitions(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=2026)
    args.data_sessions = "regular"
    args.data_symbols = "AAPL,MSFT"
    args.episode_boundary_cols = "session_id"
    args.expected_bar_interval = "1min"
    data_path = tmp_path / "bars"
    data_path.mkdir()

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=False)

    assert cfg["env_config"]["data_sessions"] == ["regular"]
    assert cfg["env_config"]["data_symbols"] == ["AAPL", "MSFT"]
    assert cfg["env_config"]["episode_boundary_cols"] == ["session_id"]
    assert cfg["env_config"]["expected_bar_interval"] == "1min"
    assert cfg["evaluation_config"]["env_config"]["data_sessions"] == ["regular"]


def test_dataset_fingerprint_supports_partitioned_parquet_directories(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import _dataset_sha256

    dataset = tmp_path / "bars"
    (dataset / "session=regular" / "symbol=AAPL").mkdir(parents=True)
    (dataset / "session=regular" / "symbol=AAPL" / "part.parquet").write_bytes(b"one")
    first = _dataset_sha256(dataset)
    (dataset / "session=regular" / "symbol=AAPL" / "part.parquet").write_bytes(b"two")

    assert first != _dataset_sha256(dataset)


def test_build_base_config_can_hold_evaluation_seed_constant_during_recovery(
    tmp_path: Path,
):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=627927)
    args.evaluation_seed = 30_260_938
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=False)

    assert cfg["seed"] == 627927
    assert cfg["env_config"]["task_sampler"]["seed"] == 627927
    assert cfg["evaluation_config"]["seed"] == 30_260_938
    assert cfg["evaluation_config"]["env_config"]["task_sampler"]["seed"] == 30_260_938


def test_build_base_config_pbt_disables_prometheus(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=99)
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=True)
    assert cfg["metrics_sinks"] == []
    assert cfg["prometheus_export_port"] is None


def test_build_base_config_disables_actor_local_plotter_when_requested(tmp_path: Path):
    from itrader.rl_lib.train import train_solo_V2

    args = _args(seed=99)
    args.auto_plot_on_end = False
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = train_solo_V2.build_base_config(
        args=args,
        data_path=data_path,
        use_pbt_mode=False,
    )

    assert cfg["callbacks"] is train_solo_V2.NativeMetricsOnlyCallback


def test_build_base_config_wires_task_pool_when_path_is_set(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=777)
    args.task_pool_path = "itrader/data/task_pool_azr.jsonl"
    args.p_pool = 0.9
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=False)
    ts = cfg["env_config"]["task_sampler"]
    assert ts["task_pool_path"] == "itrader/data/task_pool_azr.jsonl"
    assert ts["p_pool"] == pytest.approx(0.9)


def test_build_base_config_accepts_a_single_horizon_ablation_override(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=777)
    args.task_horizons = "128"
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=False)
    assert cfg["env_config"]["task_sampler"]["horizons"] == [128]


def test_build_base_config_accepts_a_reward_ablation_override(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=778)
    args.task_reward_ids = "pnl_sharpe"
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=False)
    assert cfg["env_config"]["task_sampler"]["reward_ids"] == ["pnl_sharpe"]


def test_net_return_objective_defaults_to_the_evidence_cost_assumption(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=779)
    args.task_reward_ids = "net_return"
    args.evidence_cost_bps = 1.0
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=False)

    reward_params = cfg["env_config"]["task_sampler"]["reward_params_by_id"]["net_return"]
    assert reward_params == {"cost_bps": 1.0, "reward_scale": 1000.0}


def test_cost_curriculum_uses_trial_owned_environment_steps(tmp_path: Path):
    from itrader.rl_lib.train.pbt_contract import scheduled_value
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=1789)
    args.task_reward_ids = "net_return"
    args.evidence_cost_bps = 1.0
    args.optimization_cost_schedule = "0:0,96000:0.25,192000:0.5,288000:1"
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=True)

    schedule = cfg["optimization_cost_schedule"]
    training_params = cfg["env_config"]["task_sampler"]["reward_params_by_id"]["net_return"]
    evaluation_params = cfg["evaluation_config"]["env_config"]["task_sampler"]
    evaluation_params = evaluation_params["reward_params_by_id"]["net_return"]
    assert training_params["cost_bps"] == 0.0
    assert evaluation_params["cost_bps"] == 0.0
    assert scheduled_value(schedule, 95_999) == 0.0
    assert scheduled_value(schedule, 96_000) == 0.25
    assert scheduled_value(schedule, 400_000) == 1.0


@pytest.mark.parametrize(
    "value",
    ["96000:0.25", "0:0,0:1", "0:-0.5", "0:0,earlier:1"],
)
def test_cost_curriculum_rejects_ambiguous_or_invalid_steps(tmp_path: Path, value: str):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=1790)
    args.task_reward_ids = "net_return"
    args.optimization_cost_schedule = value
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    with pytest.raises(ValueError, match="optimization_cost_schedule"):
        build_base_config(args=args, data_path=data_path, use_pbt_mode=True)


def test_fixed_evaluation_panel_balances_configured_horizons(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=780)
    args.evaluation_num_episodes = 10
    args.task_horizons = "96,128,160,200"
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=True)
    tasks = cfg["evaluation_config"]["env_config"]["task_sampler"]["fixed_tasks"]
    counts = {horizon: sum(task["horizon"] == horizon for task in tasks) for horizon in (96, 128, 160, 200)}

    assert max(counts.values()) - min(counts.values()) <= 1


def test_predeclared_evaluation_panel_uses_exact_task_spec_starts(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=781)
    args.evaluation_num_episodes = 2
    task = {
        "family_id": "00",
        "price_id": "SIM",
        "horizon": 96,
        "reward_id": "net_return",
        "reward_params": {"cost_bps": 1.0, "reward_scale": 1000.0},
        "seed": 101,
        "obs_dim": 332,
        "action_dim": 3,
        "start_idx": 1000,
        "bar_resolution": "1m",
    }
    panel = tmp_path / "crossed_panel.json"
    panel.write_text(
        json.dumps({"schema_version": 1, "tasks": [task, {**task, "horizon": 128}]}),
        encoding="utf-8",
    )
    args.evaluation_task_panel_path = str(panel)
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=True)
    evaluation_env = cfg["evaluation_config"]["env_config"]

    assert evaluation_env["start_idx_mode"] == "task_spec"
    assert evaluation_env["task_sampler"]["fixed_tasks"] == [
        task,
        {**task, "horizon": 128},
    ]


def test_predeclared_evaluation_panel_rejects_missing_start_indices(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=782)
    args.evaluation_num_episodes = 1
    panel = tmp_path / "invalid_panel.json"
    panel.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "family_id": "00",
                        "price_id": "SIM",
                        "horizon": 96,
                        "reward_id": "net_return",
                        "reward_params": {"cost_bps": 1.0, "reward_scale": 1000.0},
                        "seed": 101,
                        "obs_dim": 332,
                        "action_dim": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    args.evaluation_task_panel_path = str(panel)
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    with pytest.raises(ValueError, match="start_idx"):
        build_base_config(args=args, data_path=data_path, use_pbt_mode=True)


def test_alignment_ablation_has_a_matched_fixed_horizon_pair():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "configs"
        / "episode_regime_alignment_ablation.json"
    )
    variants = {
        variant["id"]: variant["args"]
        for variant in json.loads(config_path.read_text(encoding="utf-8"))["variants"]
    }

    random_args = variants["v3_random_horizon_128"]
    stream_args = variants["v4_stream_horizon_128"]
    assert random_args["task_horizons"] == stream_args["task_horizons"] == "128"
    assert random_args["start_idx_mode"] == "random_episode"
    assert stream_args["start_idx_mode"] == "stream_worker"


def test_build_base_config_defaults_new_ppo_args_when_missing(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=123)
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=False)
    assert cfg["clip_param"] == 0.2
    assert cfg["lambda_"] == 0.95
    assert cfg["vf_loss_coeff"] == 1.0
    assert cfg["train_batch_size_per_learner"] == 4800
    from itrader.rl_lib.train.ray_native_config import build_native_ppo_config
    effective = build_native_ppo_config(cfg)
    assert effective.minibatch_size == 128
    assert effective.num_epochs == 30
    assert effective.enable_rl_module_and_learner is True
    assert effective.enable_env_runner_and_connector_v2 is True
    assert effective.num_gpus_per_learner == 0
    assert cfg["gamma"] == 0.99
    assert cfg["vf_clip_param"] == 10.0
    assert cfg["lr"] == 5e-5
    assert cfg["auto_plot_on_end"] is True


def test_build_base_config_respects_explicit_ppo_args(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=321)
    args.clip_param = 0.15
    args.gae_lambda = 0.9
    args.vf_loss_coeff = 0.7
    args.train_batch_size = 8192
    args.sgd_minibatch_size = 256
    args.num_sgd_iter = 12
    args.gamma = 0.985
    args.vf_clip_param = 12.5
    args.lr = 3e-4
    args.auto_plot_on_end = False

    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=False)
    assert cfg["clip_param"] == pytest.approx(0.15)
    assert cfg["lambda_"] == pytest.approx(0.9)
    assert cfg["vf_loss_coeff"] == pytest.approx(0.7)
    assert cfg["train_batch_size_per_learner"] == 8192
    from itrader.rl_lib.train.ray_native_config import build_native_ppo_config
    effective = build_native_ppo_config(cfg)
    assert effective.minibatch_size == 256
    assert effective.num_epochs == 12
    assert cfg["gamma"] == pytest.approx(0.985)
    assert cfg["vf_clip_param"] == pytest.approx(12.5)
    assert cfg["lr"] == pytest.approx(3e-4)
    assert cfg["auto_plot_on_end"] is False


def test_build_base_config_rejects_retired_api_stack_selector(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args()
    args.rllib_api_stack = "legacy"

    with pytest.raises(ValueError, match="rllib_api_stack is retired"):
        build_base_config(args, tmp_path / "market.parquet", False)


def test_config_hash_handles_non_json_values():
    from itrader.rl_lib.train.train_solo_V2 import _config_hash

    class FakeCallback:
        pass

    digest = _config_hash({"callbacks": FakeCallback, "nested": {"x": 1}})
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_extract_report_preserves_dynamic_metric_keys():
    from itrader.rl_lib.train.train_solo_V2 import _extract_report

    report = _extract_report(
        {
            "training_iteration": 4,
            "time_this_iter_s": 2.0,
            "custom_metrics": {
                "act_0_frac__sum": 0.4,
                "act_0_frac__count": 2,
                "act_3_frac__sum": 0.1,
                "act_3_frac__count": 1,
                "rc_utility_gain__sum": 0.8,
                "rc_utility_gain__count": 2,
                "success__sum": 0.75,
                "success__count": 1,
                "task_horizon__sum": 192.0,
                "task_horizon__count": 2,
            },
            "learners": {
                "default_policy": {
                    "policy_loss": 0.1,
                    "vf_loss": 0.2,
                    "entropy": 0.3,
                }
            },
        },
        default_iteration=4,
    )
    assert report["custom_metrics/act_0_frac_mean"] == pytest.approx(0.2)
    assert report["custom_metrics/act_3_frac_mean"] == pytest.approx(0.1)
    assert report["custom_metrics/rc_utility_gain_mean"] == pytest.approx(0.4)
    assert report["custom_metrics/success_mean"] == pytest.approx(0.75)
    assert report["custom_metrics/task_horizon_mean"] == pytest.approx(96.0)


def test_build_base_config_separates_training_and_validation_data(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=44)
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=True)

    assert cfg["checkpoint_interval"] == 3
    assert cfg["env_config"]["data_split"] == "train"
    assert cfg["env_config"]["task_encoder_version"] == "semantic_v2"
    assert cfg["evaluation_interval"] == 1
    assert cfg["metrics_num_episodes_for_smoothing"] == args.evaluation_num_episodes
    assert cfg["evaluation_config"]["env_config"]["data_split"] == "validation"
    assert cfg["evaluation_config"]["explore"] is False
    assert cfg["checkpoint_consistent_evaluation"] is True
    fixed_tasks = cfg["evaluation_config"]["env_config"]["task_sampler"]["fixed_tasks"]
    assert len(fixed_tasks) == args.evaluation_num_episodes
    assert all(task["start_idx"] is None for task in fixed_tasks)
    assert len({task["seed"] for task in fixed_tasks}) == len(fixed_tasks)


def test_native_stack_uses_explicit_no_filter_for_semantic_observations(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args()
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")
    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=True)

    assert cfg["observation_filter"] == "NoFilter"
    assert cfg["enable_rl_module_and_learner"] is True
    assert cfg["enable_env_runner_and_connector_v2"] is True


def test_native_stack_rejects_legacy_mean_std_filter(tmp_path: Path):
    from itrader.rl_lib.train.ray_native_config import build_native_ppo_config
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args()
    args.observation_filter = "MeanStdFilter"
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")
    config = build_base_config(args=args, data_path=data_path, use_pbt_mode=False)

    with pytest.raises(ValueError, match="native stack requires observation_filter=NoFilter"):
        build_native_ppo_config(config)


def test_checkpoint_consistent_train_result_replaces_pre_sync_evaluation():
    from itrader.rl_lib.train.train_solo_V2 import (
        train_with_checkpoint_consistent_evaluation,
    )

    calls = []

    class FakeEvaluationGroup:
        def sync_weights(self, **kwargs):
            calls.append(("sync", kwargs))

    class FakeAlgorithm:
        env_runner_group = object()
        eval_env_runner_group = FakeEvaluationGroup()
        learner_group = "learner-group"

        def train(self):
            calls.append(("train", {}))
            return {"training_iteration": 6, "evaluation": {"score": "pre-sync"}}

        def evaluate(self):
            calls.append(("evaluate", {}))
            return {"evaluation": {"score": "post-sync"}}

    result = train_with_checkpoint_consistent_evaluation(FakeAlgorithm())

    assert result["evaluation"] == {"score": "post-sync"}
    assert calls == [
        ("train", {}),
        ("sync", {"from_worker_or_learner_group": "learner-group", "inference_only": True}),
        ("evaluate", {}),
    ]


def test_checkpoint_consistent_evaluation_supports_env_runner_groups():
    from itrader.rl_lib.train.train_solo_V2 import (
        train_with_checkpoint_consistent_evaluation,
    )

    calls = []

    class FakeTrainingGroup:

        pass

    class FakeEvaluationGroup:

        def sync_weights(self, **kwargs):
            calls.append(("sync", kwargs))

    class FakeAlgorithm:
        env_runner_group = FakeTrainingGroup()
        eval_env_runner_group = FakeEvaluationGroup()
        learner_group = "learner-runner-group"

        def train(self):
            return {"training_iteration": 6, "evaluation": {"score": "pre-sync"}}

        def evaluate(self):
            calls.append(("evaluate", {}))
            return {"evaluation": {"score": "post-sync"}}

    result = train_with_checkpoint_consistent_evaluation(FakeAlgorithm())

    assert result["evaluation"] == {"score": "post-sync"}
    assert calls == [
        ("sync", {"from_worker_or_learner_group": "learner-runner-group", "inference_only": True}),
        ("evaluate", {}),
    ]


def test_training_reward_overrides_use_the_group_environment_seam():
    from itrader.rl_lib.train.train_solo_V2 import _set_training_reward_overrides
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.vector import SyncVectorEnv
    from gymnasium.wrappers.vector import DictInfoToList

    calls = []

    class FakeEnvironment(gym.Env):
        observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        action_space = spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            del action
            return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

        def set_task_sampler_overrides(self, overrides):
            calls.append(dict(overrides))

    class FakeTrainingGroup:
        def foreach_env_runner(self, fn):
            vector_env = DictInfoToList(
                SyncVectorEnv([FakeEnvironment, FakeEnvironment])
            )
            env_runner = type("EnvRunner", (), {"env": vector_env})()
            return [fn(env_runner)]

    class FakeAlgorithm:
        env_runner_group = FakeTrainingGroup()
        eval_env_runner_group = None

    _set_training_reward_overrides(FakeAlgorithm(), {"cost_bps": 0.25})

    assert calls == [{"cost_bps": 0.25}, {"cost_bps": 0.25}]


def test_checkpoint_save_result_resolves_a_copyable_directory(tmp_path: Path):
    from ray.train import Checkpoint

    from itrader.rl_lib.train.train_solo_V2 import _checkpoint_from_save_result

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "state.bin").write_bytes(b"state")
    checkpoint = Checkpoint.from_directory(str(checkpoint_dir))

    class SaveResult:
        pass

    saved = SaveResult()
    saved.checkpoint = checkpoint

    converted, copy_source = _checkpoint_from_save_result(saved)

    assert converted is not None
    assert Path(copy_source).is_dir()
    assert (Path(copy_source) / "state.bin").read_bytes() == b"state"


def test_checkpoint_snapshot_is_copied_outside_tune_ownership(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import _copy_stable_checkpoint

    tune_checkpoint = tmp_path / "tune" / "checkpoint"
    tune_checkpoint.mkdir(parents=True)
    (tune_checkpoint / "algorithm_state.pkl").write_bytes(b"state")
    (tune_checkpoint / "rllib_checkpoint.json").write_text(
        '{"type": "Algorithm"}', encoding="utf-8"
    )
    policy_dir = tune_checkpoint / "policies" / "default_policy"
    policy_dir.mkdir(parents=True)
    (policy_dir / "policy_state.pkl").write_bytes(b"policy")

    stable_checkpoint = _copy_stable_checkpoint(
        tune_checkpoint,
        stable_trial_dir=tmp_path / "stable",
        training_iteration=12,
        api_stack_id=LEGACY_API_STACK_ID,
    )

    assert stable_checkpoint.parent == tmp_path / "stable" / "checkpoints"
    assert (stable_checkpoint / "algorithm_state.pkl").read_bytes() == b"state"
    assert (stable_checkpoint / "rllib_checkpoint.json").stat().st_size > 0


def test_pbt_restore_checkpoint_must_match_selected_donor_digest(tmp_path: Path):
    from itrader.rl_lib.evaluation.provenance import checkpoint_sha256
    from itrader.rl_lib.train.train_solo_V2 import _validate_pbt_restore_checkpoint

    checkpoint = tmp_path / "restore"
    policy_dir = checkpoint / "policies" / "default_policy"
    policy_dir.mkdir(parents=True)
    (checkpoint / "algorithm_state.pkl").write_bytes(b"weights")
    (checkpoint / "rllib_checkpoint.json").write_text("{}\n", encoding="utf-8")
    (policy_dir / "policy_state.pkl").write_bytes(b"policy")
    provenance = {
        "exploit_id": "exploit-1",
        "api_stack_id": LEGACY_API_STACK_ID,
        "donor_checkpoint_sha256": checkpoint_sha256(
            checkpoint, api_stack_id=LEGACY_API_STACK_ID
        ),
    }

    receipt = _validate_pbt_restore_checkpoint(checkpoint, provenance)
    assert receipt["digest_match"] is True
    assert receipt["observed_checkpoint_sha256"] == provenance["donor_checkpoint_sha256"]

    provenance["donor_checkpoint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="does not match selected donor"):
        _validate_pbt_restore_checkpoint(checkpoint, provenance)


def test_pbt_lineage_accepts_an_indexed_tune_self_resume(tmp_path: Path):
    from itrader.rl_lib.evaluation.provenance import checkpoint_sha256
    from itrader.rl_lib.train.train_solo_V2 import _validate_tune_restore_checkpoint

    donor = tmp_path / "donor"
    donor_policy = donor / "policies" / "default_policy"
    donor_policy.mkdir(parents=True)
    (donor / "algorithm_state.pkl").write_bytes(b"donor-weights")
    (donor / "rllib_checkpoint.json").write_text("{}\n", encoding="utf-8")
    (donor_policy / "policy_state.pkl").write_bytes(b"donor-policy")

    resumed = tmp_path / "resumed"
    resumed_policy = resumed / "policies" / "default_policy"
    resumed_policy.mkdir(parents=True)
    (resumed / "algorithm_state.pkl").write_bytes(b"advanced-weights")
    (resumed / "rllib_checkpoint.json").write_text("{}\n", encoding="utf-8")
    (resumed_policy / "policy_state.pkl").write_bytes(b"advanced-policy")

    exploit_id = "exploit-1"
    target_trial_id = "trial-00003"
    donor_digest = checkpoint_sha256(donor, api_stack_id=LEGACY_API_STACK_ID)
    resumed_digest = checkpoint_sha256(resumed, api_stack_id=LEGACY_API_STACK_ID)
    stable_trial_dir = tmp_path / "stable-target"
    receipts_dir = stable_trial_dir / "restore_receipts"
    receipts_dir.mkdir(parents=True)
    (receipts_dir / f"{exploit_id}.json").write_text(
        json.dumps(
            {
                "receipt_kind": "pbt_exploit",
                "api_stack_id": LEGACY_API_STACK_ID,
                "exploit_id": exploit_id,
                "target_trial_id": target_trial_id,
                "observed_checkpoint_sha256": donor_digest,
                "expected_checkpoint_sha256": donor_digest,
                "digest_match": True,
            }
        ),
        encoding="utf-8",
    )
    (stable_trial_dir / "checkpoint_index.jsonl").write_text(
        json.dumps(
            {
                "trial_id": target_trial_id,
                "api_stack_id": LEGACY_API_STACK_ID,
                "checkpoint_sha256": resumed_digest,
                "pbt_restore_generation": 1,
                "pbt_parent_exploit_id": exploit_id,
                "reported_to_tune": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = _validate_tune_restore_checkpoint(
        resumed,
        {
            "exploit_id": exploit_id,
            "target_trial_id": target_trial_id,
                "donor_checkpoint_sha256": donor_digest,
                "api_stack_id": LEGACY_API_STACK_ID,
        },
        stable_trial_dir=stable_trial_dir,
        target_trial_id=target_trial_id,
        restore_generation=1,
        api_stack_id=LEGACY_API_STACK_ID,
    )

    assert receipt["receipt_kind"] == "tune_self_resume"
    assert receipt["observed_checkpoint_sha256"] == resumed_digest
    assert receipt["lineage_verified"] is True

    (stable_trial_dir / "checkpoint_index.jsonl").write_text(
        json.dumps(
            {
                "trial_id": target_trial_id,
                "api_stack_id": LEGACY_API_STACK_ID,
                "checkpoint_sha256": resumed_digest,
                "pbt_restore_generation": 1,
                "pbt_parent_exploit_id": exploit_id,
                "reported_to_tune": False,
                "post_sync_evaluation": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not match selected donor"):
        _validate_tune_restore_checkpoint(
            resumed,
            {
                "exploit_id": exploit_id,
                "target_trial_id": target_trial_id,
                "donor_checkpoint_sha256": donor_digest,
                "api_stack_id": LEGACY_API_STACK_ID,
            },
            stable_trial_dir=stable_trial_dir,
            target_trial_id=target_trial_id,
            restore_generation=1,
            api_stack_id=LEGACY_API_STACK_ID,
        )


def test_pbt_lineage_accepts_an_exact_post_sync_sidecar_resume(tmp_path: Path):
    """Capacity resumes may use the exact durable sidecar between Tune cadences."""
    from itrader.rl_lib.evaluation.provenance import checkpoint_sha256
    from itrader.rl_lib.train.train_solo_V2 import _validate_tune_restore_checkpoint

    donor = tmp_path / "donor"
    donor_policy = donor / "policies" / "default_policy"
    donor_policy.mkdir(parents=True)
    (donor / "algorithm_state.pkl").write_bytes(b"donor-weights")
    (donor / "rllib_checkpoint.json").write_text("{}\n", encoding="utf-8")
    (donor_policy / "policy_state.pkl").write_bytes(b"donor-policy")

    resumed = tmp_path / "resumed-sidecar"
    resumed_policy = resumed / "policies" / "default_policy"
    resumed_policy.mkdir(parents=True)
    (resumed / "algorithm_state.pkl").write_bytes(b"advanced-weights")
    (resumed / "rllib_checkpoint.json").write_text("{}\n", encoding="utf-8")
    (resumed_policy / "policy_state.pkl").write_bytes(b"advanced-policy")

    exploit_id = "exploit-1"
    target_trial_id = "trial-00003"
    donor_digest = checkpoint_sha256(donor, api_stack_id=LEGACY_API_STACK_ID)
    resumed_digest = checkpoint_sha256(resumed, api_stack_id=LEGACY_API_STACK_ID)
    stable_trial_dir = tmp_path / "stable-target"
    receipts_dir = stable_trial_dir / "restore_receipts"
    receipts_dir.mkdir(parents=True)
    (receipts_dir / f"{exploit_id}.json").write_text(
        json.dumps(
            {
                "receipt_kind": "pbt_exploit",
                "api_stack_id": LEGACY_API_STACK_ID,
                "exploit_id": exploit_id,
                "target_trial_id": target_trial_id,
                "observed_checkpoint_sha256": donor_digest,
                "expected_checkpoint_sha256": donor_digest,
                "digest_match": True,
            }
        ),
        encoding="utf-8",
    )
    (stable_trial_dir / "checkpoint_index.jsonl").write_text(
        json.dumps(
            {
                "trial_id": target_trial_id,
                "api_stack_id": LEGACY_API_STACK_ID,
                "checkpoint_sha256": resumed_digest,
                "pbt_restore_generation": 1,
                "pbt_parent_exploit_id": exploit_id,
                "reported_to_tune": False,
                "post_sync_evaluation": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = _validate_tune_restore_checkpoint(
        resumed,
        {
            "exploit_id": exploit_id,
            "target_trial_id": target_trial_id,
            "donor_checkpoint_sha256": donor_digest,
            "api_stack_id": LEGACY_API_STACK_ID,
        },
        stable_trial_dir=stable_trial_dir,
        target_trial_id=target_trial_id,
        restore_generation=1,
        api_stack_id=LEGACY_API_STACK_ID,
    )

    assert receipt["receipt_kind"] == "tune_self_resume"
    assert receipt["observed_checkpoint_sha256"] == resumed_digest
    assert receipt["lineage_verified"] is True


def test_pbt_lineage_rejects_an_unindexed_tune_self_resume(tmp_path: Path):
    from itrader.rl_lib.evaluation.provenance import checkpoint_sha256
    from itrader.rl_lib.train.train_solo_V2 import _validate_tune_restore_checkpoint

    checkpoint = tmp_path / "unknown-resume"
    policy_dir = checkpoint / "policies" / "default_policy"
    policy_dir.mkdir(parents=True)
    (checkpoint / "algorithm_state.pkl").write_bytes(b"unknown-weights")
    (checkpoint / "rllib_checkpoint.json").write_text("{}\n", encoding="utf-8")
    (policy_dir / "policy_state.pkl").write_bytes(b"unknown-policy")

    exploit_id = "exploit-1"
    target_trial_id = "trial-00003"
    donor_digest = "a" * 64
    stable_trial_dir = tmp_path / "stable-target"
    receipts_dir = stable_trial_dir / "restore_receipts"
    receipts_dir.mkdir(parents=True)
    (receipts_dir / f"{exploit_id}.json").write_text(
        json.dumps(
            {
                "receipt_kind": "pbt_exploit",
                "api_stack_id": LEGACY_API_STACK_ID,
                "exploit_id": exploit_id,
                "target_trial_id": target_trial_id,
                "observed_checkpoint_sha256": donor_digest,
                "expected_checkpoint_sha256": donor_digest,
                "digest_match": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not match selected donor"):
        _validate_tune_restore_checkpoint(
            checkpoint,
            {
                "exploit_id": exploit_id,
                "target_trial_id": target_trial_id,
            "donor_checkpoint_sha256": donor_digest,
            "api_stack_id": LEGACY_API_STACK_ID,
            },
            stable_trial_dir=stable_trial_dir,
            target_trial_id=target_trial_id,
            restore_generation=1,
            api_stack_id=LEGACY_API_STACK_ID,
        )

    assert checkpoint_sha256(
        checkpoint, api_stack_id=LEGACY_API_STACK_ID
    ) != donor_digest


def test_checkpoint_index_is_durable_before_session_report_can_pause(
    tmp_path: Path, monkeypatch
):
    from itrader.rl_lib.train import train_solo_V2

    index_path = tmp_path / "checkpoint_index.jsonl"
    record = {
        "training_iteration": 10,
        "pbt_progress_steps": 48_000,
        "stable_checkpoint": "stable/checkpoint-10",
    }
    observed = {}

    def pausing_report(report, *, checkpoint):
        observed["rows"] = index_path.read_text(encoding="utf-8").splitlines()
        observed["report"] = report
        observed["checkpoint"] = checkpoint

    monkeypatch.setattr(train_solo_V2, "_report_to_tune", pausing_report)

    train_solo_V2._report_checkpoint_with_index(
        index_path=index_path,
        record=record,
        report={"training_iteration": 10},
        checkpoint="checkpoint-object",
    )

    assert [json.loads(row) for row in observed["rows"]] == [record]
    assert observed["report"] == {"training_iteration": 10}
    assert observed["checkpoint"] == "checkpoint-object"


def test_native_training_fn_reports_a_durable_checkpoint_before_tune_report(
    monkeypatch, tmp_path: Path
):
    from itrader.rl_lib.train import (
        ray_native_config,
        ray_native_ppo,
        train_solo_V2,
    )

    data_path = tmp_path / "prices.parquet"
    data_path.write_bytes(b"synthetic native trainer fixture")
    args = _args(seed=90210)
    args.data_path = str(data_path)
    args.train_batch_size = 4
    args.iters = 1
    args.pbt_interval = 1
    args.checkpoint_interval = 1
    args.evaluation_num_episodes = 1
    args.auto_plot_on_end = False
    config = train_solo_V2.build_base_config(args, data_path, use_pbt_mode=False)
    config["stable_metrics_root"] = str(tmp_path / "stable")

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    monkeypatch.setattr(
        train_solo_V2,
        "_trial_context_values",
        lambda: (str(trial_dir), "trial-1", "native-trial"),
    )
    monkeypatch.setattr(train_solo_V2, "_assert_frozen_run_provenance", lambda _config: {})
    monkeypatch.setattr(train_solo_V2.tune, "get_checkpoint", lambda: None)
    native_config = object()
    monkeypatch.setattr(
        ray_native_config,
        "build_native_ppo_config",
        lambda _config: native_config,
    )
    report_calls = []

    class FakeNativeAlgorithm:
        def __init__(self, *, config):
            assert config is native_config
            self.iteration = 0

        def train(self):
            self.iteration += 1
            return {
                "training_iteration": self.iteration,
                "env_runners": {
                    "num_env_steps_sampled": 4,
                    "episode_len_mean": 4,
                    "custom_metrics": {
                        "optimization_reward__sum": 0.001,
                        "optimization_reward__count": 1,
                        "economic_gross_return__sum": 0.0012,
                        "economic_gross_return__count": 1,
                        "economic_net_return__sum": 0.001,
                        "economic_net_return__count": 1,
                        "economic_net_sharpe_per_sqrt_bar__sum": 0.1,
                        "economic_net_sharpe_per_sqrt_bar__count": 1,
                        "act_flat_frac__sum": 0.25,
                        "act_flat_frac__count": 1,
                    },
                },
                "evaluation": {
                    "env_runners": {
                        "custom_metrics": {
                            "optimization_reward__sum": 0.001,
                            "optimization_reward__count": 1,
                            "economic_gross_return__sum": 0.0012,
                            "economic_gross_return__count": 1,
                            "economic_net_return__sum": 0.001,
                            "economic_net_return__count": 1,
                            "economic_net_sharpe_per_sqrt_bar__sum": 0.1,
                            "economic_net_sharpe_per_sqrt_bar__count": 1,
                            "act_flat_frac__sum": 0.25,
                            "act_flat_frac__count": 1,
                        },
                    }
                },
                "learners": {
                    "default_policy": {
                        "policy_loss": 0.01,
                        "vf_loss": 0.02,
                        "entropy": 0.03,
                        "mean_kl_loss": 0.004,
                        "vf_explained_var": 0.5,
                        "total_loss": 0.06,
                    }
                },
            }

        def save(self, path):
            checkpoint_dir = Path(path)
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "algorithm_state.pkl").write_bytes(b"algorithm")
            (checkpoint_dir / "rllib_checkpoint.json").write_text("{}", encoding="utf-8")
            module_dir = (
                checkpoint_dir
                / "learner_group"
                / "learner"
                / "rl_module"
                / "default_policy"
            )
            module_dir.mkdir(parents=True)
            (module_dir / "module_state.pkl").write_bytes(b"native module")
            return str(checkpoint_dir)

        def stop(self):
            pass

    monkeypatch.setattr(ray_native_ppo, "DiagnosticPPO", FakeNativeAlgorithm)

    def report_to_tune(report, *, checkpoint=None):
        index_path = Path(config["stable_metrics_root"]) / "trial" / "checkpoint_index.jsonl"
        indexed = json.loads(index_path.read_text(encoding="utf-8").splitlines()[-1])
        assert indexed["reported_to_tune"] is True
        assert indexed["api_stack_id"] == NATIVE_API_STACK_ID
        assert indexed["checkpoint_sha256"] == train_solo_V2.checkpoint_sha256(
            indexed["stable_checkpoint"], api_stack_id=NATIVE_API_STACK_ID
        )
        assert report["training_iteration"] == indexed["training_iteration"]
        assert checkpoint is not None
        report_calls.append((report, checkpoint))

    monkeypatch.setattr(train_solo_V2, "_report_to_tune", report_to_tune)

    train_solo_V2.training_fn(config)

    assert len(report_calls) == 1
    report, checkpoint = report_calls[0]
    assert report["pbt_progress_steps"] == 4
    assert report["training/optimization_reward_mean"] == pytest.approx(0.001)
    assert report["evaluation/economic_selection_score"] == pytest.approx(0.001)
    assert report["evaluation/economic_search_score"] == pytest.approx(1.001)
    assert checkpoint.path
    status_path = Path(config["stable_metrics_root"]) / "trial" / "trial_status.json"
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "completed"


def test_best_checkpoint_index_only_advances_on_a_better_held_out_score(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import _update_best_checkpoint_record

    index_path = tmp_path / "best_checkpoint.json"
    first = _update_best_checkpoint_record(
        index_path,
        metric="evaluation/net_return_mean",
        report={"training_iteration": 10, "pbt_progress_steps": 48_000},
        stable_checkpoint=tmp_path / "checkpoint-10",
        score=0.001,
        economically_eligible=True,
    )
    worse = _update_best_checkpoint_record(
        index_path,
        metric="evaluation/net_return_mean",
        report={"training_iteration": 20, "pbt_progress_steps": 96_000},
        stable_checkpoint=tmp_path / "checkpoint-20",
        score=-0.001,
        economically_eligible=False,
    )

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert first is True
    assert worse is False
    assert payload["score"] == pytest.approx(0.001)
    assert payload["training_iteration"] == 10
    assert payload["pbt_progress_steps"] == 48_000


def test_promotion_candidate_is_retained_between_pbt_boundaries():
    from itrader.rl_lib.train.pbt_contract import checkpoint_retention_decision

    decision = checkpoint_retention_decision(
        cadence_due=False,
        terminal_due=False,
        selection_score=0.00099,
        current_best_score=-0.001,
    )

    assert decision.retain_stable is True
    assert decision.report_to_tune is False


def test_synchronous_pbt_retains_every_report_without_reporting_every_checkpoint():
    from itrader.rl_lib.train.pbt_contract import checkpoint_retention_decision

    decision = checkpoint_retention_decision(
        cadence_due=False,
        terminal_due=False,
        selection_score=-0.001,
        current_best_score=0.001,
        synchronous_pbt=True,
    )

    assert decision.retain_stable is True
    assert decision.report_to_tune is False


def test_pbt_contract_rejects_misaligned_checkpoint_cadence():
    from itrader.rl_lib.train.train_solo_V2 import _pbt_intervals

    args = _args()
    args.checkpoint_interval = 10
    args.pbt_interval = 3

    with pytest.raises(ValueError, match="checkpoint_interval"):
        _pbt_intervals(args)


def test_native_pbt_scheduler_is_the_single_objective_owner():
    from itrader.rl_lib.train.train_solo_V2 import _tune_objective_kwargs

    assert _tune_objective_kwargs(use_pbt_mode=True) == {}
    assert _tune_objective_kwargs(use_pbt_mode=False) == {
        "metric": "evaluation/net_return_mean",
        "mode": "max",
    }

    args = _args()
    args.pbt_promotion_eligible = True
    assert _tune_objective_kwargs(use_pbt_mode=False, args=args) == {
        "metric": "evaluation/economic_selection_score",
        "mode": "max",
    }


def test_tune_storage_uses_the_runtime_supported_keyword(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import _tune_storage_kwargs

    assert _tune_storage_kwargs(tmp_path) == {"storage_path": str(tmp_path)}


def test_trial_output_path_requires_the_pinned_ray_trial_property():
    from itrader.rl_lib.train.train_solo_V2 import _trial_output_path

    class CurrentTrial:
        local_path = "current"

        @property
        def logdir(self):
            raise AssertionError("deprecated property must not be touched")

    assert _trial_output_path(CurrentTrial()) == "current"
    with pytest.raises(TypeError, match="must expose local_path"):
        _trial_output_path(object())


def test_trial_context_uses_current_tune_context(monkeypatch):
    from itrader.rl_lib.train import train_solo_V2

    class Context:
        def get_trial_dir(self):
            return "current-dir"

        def get_trial_id(self):
            return "current-id"

        def get_trial_name(self):
            return "current-name"

    monkeypatch.setattr(
        train_solo_V2.tune,
        "get_context",
        lambda: Context(),
        raising=False,
    )
    assert train_solo_V2._trial_context_values() == (
        "current-dir",
        "current-id",
        "current-name",
    )


def test_incoming_checkpoint_uses_ray_tune_function_api(monkeypatch):
    from ray import train as ray_train

    from itrader.rl_lib.train import train_solo_V2

    expected = object()
    monkeypatch.setattr(
        train_solo_V2.tune,
        "get_checkpoint",
        lambda: expected,
    )
    monkeypatch.setattr(
        ray_train,
        "get_checkpoint",
        lambda: (_ for _ in ()).throw(
            AssertionError("Ray Train checkpoint API used in Tune function")
        ),
    )

    assert train_solo_V2._incoming_tune_checkpoint() is expected


def test_report_uses_ray_tune_function_api(monkeypatch):
    from ray import train as ray_train

    from itrader.rl_lib.train import train_solo_V2

    calls = []
    monkeypatch.setattr(
        train_solo_V2.tune,
        "report",
        lambda metrics, checkpoint=None: calls.append((metrics, checkpoint)),
    )
    monkeypatch.setattr(
        ray_train,
        "report",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Ray Train report API used in Tune function")
        ),
    )
    checkpoint = object()

    train_solo_V2._report_to_tune({"score": 1.0}, checkpoint=checkpoint)

    assert calls == [({"score": 1.0}, checkpoint)]


def test_auto_plotter_resolves_only_the_current_tune_trial_context(
    monkeypatch, tmp_path: Path
):
    from ray import tune

    from itrader.rl_lib.callbacks.auto_plotter import AutoPlotterCallback

    trial_dir = tmp_path / "current-trial"
    trial_dir.mkdir()
    plot_script = tmp_path / "plot.py"
    plot_script.write_text("", encoding="utf-8")

    class Context:
        def get_trial_dir(self):
            return str(trial_dir)

    monkeypatch.setattr(tune, "get_context", lambda: Context())
    callback = AutoPlotterCallback(plot_script_path=str(plot_script))

    assert callback._trial_dir() == trial_dir


def test_pbt_scheduler_uses_monotonic_environment_step_clock():
    from itrader.rl_lib.train.train_solo_V2 import _build_pbt_scheduler

    args = _args()
    args.train_batch_size = 4800
    args.pbt_interval = 10
    args.pbt_burn_in = 20
    args.checkpoint_interval = 10

    scheduler = _build_pbt_scheduler(args)

    assert scheduler._time_attr == "pbt_progress_steps"
    assert scheduler._perturbation_interval == 48_000

    args.pbt_promotion_eligible = True
    scheduler = _build_pbt_scheduler(args)
    assert scheduler._metric == "evaluation/economic_selection_score"
    assert scheduler._burn_in_period == 96_000

    args.pbt_promotion_eligible = False
    args.pbt_selection_mode = "nonflat_net"
    scheduler = _build_pbt_scheduler(args)
    assert scheduler._metric == "evaluation/economic_search_score"
    assert scheduler._require_positive_donor is False


def test_pbt_learning_rate_domain_can_be_narrowed_per_study():
    from itrader.rl_lib.train.train_solo_V2 import _pbt_mutations

    args = _args()
    args.pbt_lr_min = 5e-6
    args.pbt_lr_max = 6e-5

    learning_rate = _pbt_mutations(args)["lr"]

    assert learning_rate.lower == pytest.approx(5e-6)
    assert learning_rate.upper == pytest.approx(6e-5)


def test_pbt_learning_rate_domain_rejects_invalid_bounds():
    from itrader.rl_lib.train.train_solo_V2 import _pbt_mutations

    args = _args()
    args.pbt_lr_min = 6e-5
    args.pbt_lr_max = 5e-6

    with pytest.raises(ValueError, match="pbt_lr_min"):
        _pbt_mutations(args)


def test_synchronous_pbt_orders_donor_checkpoint_before_exploit():
    from itrader.rl_lib.train.train_solo_V2 import (
        _build_pbt_scheduler,
        build_base_config,
    )

    args = _args()
    args.train_batch_size = 4800
    args.pbt_interval = 10
    args.pbt_burn_in = 20
    args.checkpoint_interval = 10
    args.pbt_synch = True

    scheduler = _build_pbt_scheduler(args)

    assert scheduler._synch is True
    config = build_base_config(
        args=args,
        data_path=Path("itrader/data/ohlcv.parquet"),
        use_pbt_mode=True,
    )
    assert config["pbt_synch"] is True


def test_pbt_explore_contract_clamps_invalid_mutations():
    from itrader.rl_lib.train.pbt_contract import constrain_trial_config

    config = {
        "gamma": 1.188,
        "lambda_": 1.7,
        "lr": 1.0,
        "clip_param": 9.0,
        "minibatch_size": 307,
        "num_epochs": 19,
        "vf_clip_param": 999.0,
        "vf_loss_coeff": -4.0,
        "train_batch_size_per_learner": 5000,
    }

    constrained = constrain_trial_config(config)

    assert constrained is config
    assert 0.95 <= config["gamma"] <= 0.999
    assert 0.8 <= config["lambda_"] <= 1.0
    assert config["lr"] <= 0.005
    assert config["clip_param"] in {0.1, 0.2, 0.3}
    assert config["minibatch_size"] in {128, 256, 512}
    assert config["num_epochs"] in {10, 20, 30}
    assert config["vf_clip_param"] in {5.0, 10.0, 20.0}
    assert config["vf_loss_coeff"] in {0.5, 1.0, 2.0}
    assert config["train_batch_size_per_learner"] == 5000
    assert config["train_batch_size_per_learner"] >= 2 * config["minibatch_size"]


def test_pbt_explore_contract_respects_study_learning_rate_bounds():
    from itrader.rl_lib.train.pbt_contract import constrain_trial_config

    config = {
        "lr": 1e-3,
        "pbt_lr_min": 5e-6,
        "pbt_lr_max": 6e-5,
    }

    constrain_trial_config(config)

    assert config["lr"] == pytest.approx(6e-5)


def test_pbt_population_keeps_equal_exposure_and_matched_evaluation_seed():
    from itrader.rl_lib.train.train_solo_V2 import (
        PBT_MUTATIONS,
        _configure_pbt_population,
    )

    args = _args(seed=20260912)
    config = {
        "seed": args.seed,
        "train_batch_size_per_learner": 1024,
        "env_config": {"task_sampler": {"seed": args.seed}},
        "evaluation_config": {
            "env_config": {
                "task_sampler": {
                    "seed": args.seed + 10_000_019,
                    "fixed_tasks": [{"seed": 11}, {"seed": 22}],
                }
            }
        },
    }

    _configure_pbt_population(config, args)

    assert "train_batch_size_per_learner" not in PBT_MUTATIONS
    assert config["train_batch_size_per_learner"] == 1024
    assert not isinstance(config["seed"], int)
    assert not isinstance(config["env_config"]["task_sampler"]["seed"], int)
    assert config["evaluation_config"]["env_config"]["task_sampler"]["seed"] == 30_260_931
    assert config["evaluation_config"]["env_config"]["task_sampler"]["fixed_tasks"] == [
        {"seed": 11},
        {"seed": 22},
    ]


def test_pbt_population_searches_bounded_entropy_pressure_schedules():
    from itrader.rl_lib.train.train_solo_V2 import (
        PBT_MUTATIONS,
        _configure_pbt_population,
    )

    schedules = PBT_MUTATIONS["entropy_coeff"]

    assert schedules == [
        [[0, 0.0], [96_000, 0.0], [240_000, 0.0]],
        [[0, 0.001], [96_000, 0.0005], [240_000, 0.0]],
        [[0, 0.005], [96_000, 0.002], [240_000, 0.0005]],
        [[0, 0.01], [96_000, 0.005], [240_000, 0.001]],
    ]
    assert all(
        steps == sorted(steps)
        and steps[0] == 0
        and all(coefficient >= 0.0 for coefficient in coefficients)
        for steps, coefficients in (
            (
                [point[0] for point in schedule],
                [point[1] for point in schedule],
            )
            for schedule in schedules
        )
    )

    args = _args(seed=20260924)
    config = {
        "entropy_coeff": [[0, 0.5]],
        "seed": args.seed,
        "env_config": {"task_sampler": {"seed": args.seed}},
        "evaluation_config": {
            "env_config": {"task_sampler": {"seed": 30_260_943}}
        },
    }

    _configure_pbt_population(config, args)

    assert "entropy_coeff" not in config


def test_pbt_learning_rate_search_is_bounded_around_observed_stable_roots():
    from itrader.rl_lib.train.train_solo_V2 import PBT_MUTATIONS

    learning_rate = PBT_MUTATIONS["lr"]

    assert learning_rate.lower == pytest.approx(5e-6)
    assert learning_rate.upper == pytest.approx(3e-4)


def test_pbt_population_seed_can_change_without_changing_evaluation_panel():
    from itrader.rl_lib.train.train_solo_V2 import (
        _configure_pbt_population,
        _pbt_population_seed,
    )

    args = _args(seed=20260912)
    args.pbt_population_seed = 90210
    config = {
        "seed": args.seed,
        "env_config": {"task_sampler": {"seed": args.seed}},
        "evaluation_config": {
            "env_config": {
                "task_sampler": {
                    "seed": args.seed + 10_000_019,
                    "fixed_tasks": [{"seed": 11}, {"seed": 22}],
                }
            }
        },
    }

    _configure_pbt_population(config, args)

    assert _pbt_population_seed(args) == 90210
    assert config["seed"].lower == 90210
    assert config["seed"].upper == 1_090_210
    assert config["evaluation_config"]["env_config"]["task_sampler"]["seed"] == 30_260_931
    assert config["evaluation_config"]["env_config"]["task_sampler"]["fixed_tasks"] == [
        {"seed": 11},
        {"seed": 22},
    ]


def test_pbt_population_seed_reproduces_mutation_sequence():
    from itrader.rl_lib.train.train_solo_V2 import _build_pbt_scheduler

    args = _args(seed=20260919)
    args.pbt_population_seed = 271828
    donor_config = {
        "lr": 0.0036320658878667626,
        "entropy_coeff": [
            [0, 0.01],
            [96_000, 0.005],
            [240_000, 0.001],
        ],
        "clip_param": 0.1,
        "lambda_": 0.9,
        "gamma": 0.99,
        "minibatch_size": 512,
        "num_epochs": 30,
        "vf_clip_param": 20.0,
        "vf_loss_coeff": 1.0,
        "train_batch_size_per_learner": 4800,
    }

    def mutation_sequence():
        scheduler = _build_pbt_scheduler(args)
        target = Namespace(config={})
        donor = Namespace(config=donor_config)
        return [scheduler._get_new_config(target, donor)[0] for _ in range(4)]

    assert mutation_sequence() == mutation_sequence()


def test_restored_pbt_worker_runs_only_remaining_iterations():
    from itrader.rl_lib.train.train_solo_V2 import _remaining_training_steps

    assert list(_remaining_training_steps(current_iteration=0, stop_iteration=12)) == list(range(12))
    assert list(_remaining_training_steps(current_iteration=6, stop_iteration=12)) == list(range(6))
    assert list(_remaining_training_steps(current_iteration=12, stop_iteration=12)) == []


def test_progress_clock_recovers_target_exposure_across_donor_restore(tmp_path: Path):
    from itrader.rl_lib.train.pbt_contract import PBTProgressClock

    state_path = tmp_path / "pbt_progress.json"
    clock = PBTProgressClock.load(state_path, target_steps=960_000)
    clock.advance({"env_runners": {"num_env_steps_sampled": 4_832}})
    clock.save(state_path)

    restored = PBTProgressClock.load(state_path, target_steps=960_000)

    assert restored.completed_steps == 4_832
    assert restored.remaining


def test_progress_clock_uses_native_sampled_steps_and_ignores_legacy_tree():
    from itrader.rl_lib.train.pbt_contract import PBTProgressClock

    clock = PBTProgressClock(target_steps=9_600)
    sampled = clock.advance(
        {
            "num_env_steps_sampled_this_iter": 99_999,
            "sampler_results": {"num_env_steps_sampled_this_iter": 99_999},
            "env_runners": {"num_env_steps_sampled": 4_832},
        },
    )

    assert sampled == 4_832
    assert clock.completed_steps == 4_832


def test_progress_clock_rejects_missing_native_sampled_steps():
    from itrader.rl_lib.train.pbt_contract import PBTProgressClock

    clock = PBTProgressClock(target_steps=9_600)

    with pytest.raises(ValueError, match="env_runners.num_env_steps_sampled"):
        clock.advance({"training_iteration": 1})


def test_progress_clock_fails_closed_when_persisted_state_is_corrupt(tmp_path: Path):
    from itrader.rl_lib.train.pbt_contract import PBTProgressClock

    state_path = tmp_path / "pbt_progress.json"
    state_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid PBT progress state"):
        PBTProgressClock.load(state_path, target_steps=960_000)


def test_extract_report_exposes_native_ppo_learner_diagnostics():
    from itrader.rl_lib.train.train_solo_V2 import _extract_report

    report = _extract_report(
        {
            "training_iteration": 3,
            "env_runners": {
                "custom_metrics": {
                    "optimization_reward__sum": 0.1,
                    "optimization_reward__count": 1,
                }
            },
            "learners": {
                "default_policy": {
                    "policy_loss": 0.01,
                    "vf_loss": 0.02,
                    "entropy": 0.03,
                    "mean_kl_loss": 0.004,
                    "vf_explained_var": 0.5,
                    "total_loss": 0.06,
                    "gradients_default_optimizer_global_norm": 1.25,
                    "default_optimizer_learning_rate": 5e-5,
                    "curr_kl_coeff": 0.2,
                }
            },
        },
        default_iteration=3,
    )

    assert report["approx_kl"] == pytest.approx(0.004)
    assert report["explained_variance"] == pytest.approx(0.5)
    assert report["total_loss"] == pytest.approx(0.06)
    assert report["grad_norm"] == pytest.approx(1.25)
    assert report["learning_rate"] == pytest.approx(5e-5)
    assert report["kl_coeff"] == pytest.approx(0.2)


def test_run_manifest_is_written_outside_tune_owned_experiment_dir(tmp_path: Path, monkeypatch):
    from itrader.rl_lib.train import train_solo_V2

    data_path = tmp_path / "prices.parquet"
    data_path.write_bytes(b"dataset")
    args = _args(seed=77)
    args.experiment_name = "manifest_contract_probe"
    args.pbt_population_seed = 90210
    monkeypatch.setattr(train_solo_V2.Path, "home", staticmethod(lambda: tmp_path))

    manifest, path = train_solo_V2.write_run_manifest(
        base_config={
            "api_stack_id": NATIVE_API_STACK_ID,
            "seed": 77,
            "env_config": {"df_path": str(data_path)},
            "evaluation_config": {
                "env_config": {
                    "df_path": str(data_path),
                    "task_sampler": {"fixed_tasks": [{"seed": 11}]},
                }
            },
        },
        args=args,
        data_path=data_path,
    )

    assert path.parent == tmp_path / "ray_results" / "_stable_runs" / args.experiment_name
    assert path.stat().st_size > 0
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == manifest["run_id"]
    assert manifest["api_stack_id"] == NATIVE_API_STACK_ID
    assert manifest["pbt_population_seed"] == 90210
    assert len(manifest["comparison_config_hash"]) == 64
    assert len(manifest["evaluation_panel_sha256"]) == 64
    assert manifest["runtime"]["ray"] == train_solo_V2.ray.__version__
    assert manifest["runtime"]["python"].startswith("3.11")
    assert manifest["runtime"]["platform"]


def test_run_manifest_records_observation_schema_dimensions(tmp_path: Path, monkeypatch):
    from itrader.rl_lib.train import train_solo_V2

    data_path = tmp_path / "prices.parquet"
    data_path.write_bytes(b"dataset")
    args = _args(seed=79)
    args.experiment_name = "manifest_observation_schema_probe"
    args.solver_state_mode = "both"
    monkeypatch.setattr(train_solo_V2.Path, "home", staticmethod(lambda: tmp_path))
    config = train_solo_V2.build_base_config(
        args=args,
        data_path=data_path,
        use_pbt_mode=False,
    )

    manifest, _ = train_solo_V2.write_run_manifest(
        base_config=config,
        args=args,
        data_path=data_path,
    )

    assert manifest["observation_schema"] == {
        "observation_schema_id": "r4_v1_min",
        "base_obs_dim": 182,
        "solver_state_dim": 2,
        "agent_obs_dim": 184,
    }


def test_run_manifest_records_resolved_pbt_learning_rate_bounds(tmp_path: Path, monkeypatch):
    from itrader.rl_lib.train import train_solo_V2

    data_path = tmp_path / "prices.parquet"
    data_path.write_bytes(b"dataset")
    args = _args(seed=78)
    args.cpu_pbt = True
    args.experiment_name = "manifest_mutation_probe"
    args.pbt_lr_min = 7e-6
    args.pbt_lr_max = 6e-5
    monkeypatch.setattr(train_solo_V2.Path, "home", staticmethod(lambda: tmp_path))

    manifest, _ = train_solo_V2.write_run_manifest(
        base_config={
            "api_stack_id": NATIVE_API_STACK_ID,
            "seed": 78,
            "env_config": {"df_path": str(data_path)},
            "evaluation_config": {
                "env_config": {
                    "df_path": str(data_path),
                    "task_sampler": {"fixed_tasks": [{"seed": 12}]},
                }
            },
        },
        args=args,
        data_path=data_path,
    )

    assert manifest["pbt_mutations"]["lr"] == {
        "distribution": "loguniform",
        "lower": pytest.approx(7e-6),
        "upper": pytest.approx(6e-5),
    }
    assert manifest["pbt_mutations"]["gamma"] == [0.98, 0.99, 0.995]


def test_comparison_config_hash_ignores_branch_paths_and_source_fingerprint():
    from itrader.rl_lib.train.train_solo_V2 import _comparison_config_hash

    left = {
        "source_provenance": {"git_commit": "left"},
        "stable_metrics_root": "C:/left/results",
        "pbt_experiment_name": "left-run",
        "env_config": {"df_path": "C:/left/bars", "window_size": 60},
        "evaluation_config": {
            "seed": 30260943,
            "env_config": {"df_path": "C:/left/bars"},
        },
    }
    right = copy.deepcopy(left)
    right["source_provenance"] = {"git_commit": "right"}
    right["stable_metrics_root"] = "C:/right/results"
    right["pbt_experiment_name"] = "right-run"
    right["env_config"]["df_path"] = "C:/right/bars"
    right["evaluation_config"]["env_config"]["df_path"] = "C:/right/bars"

    assert _comparison_config_hash(left, "dataset-hash") == _comparison_config_hash(
        right,
        "dataset-hash",
    )
    right["evaluation_config"]["seed"] += 1
    assert _comparison_config_hash(left, "dataset-hash") != _comparison_config_hash(
        right,
        "dataset-hash",
    )


def test_explicit_results_root_routes_stable_metrics_and_manifest(tmp_path: Path):
    from itrader.rl_lib.train import train_solo_V2

    data_path = tmp_path / "prices.parquet"
    data_path.write_bytes(b"dataset")
    args = _args(seed=78)
    args.experiment_name = "workspace_root_probe"
    args.results_root = str(tmp_path / "workspace-results")

    config = train_solo_V2.build_base_config(
        args=args,
        data_path=data_path,
        use_pbt_mode=True,
    )
    manifest, path = train_solo_V2.write_run_manifest(
        base_config=config,
        args=args,
        data_path=data_path,
    )

    expected_root = (tmp_path / "workspace-results").resolve()
    assert config["stable_metrics_root"] == str(expected_root / "_stable_metrics")
    assert config["pbt_experiment_name"] == args.experiment_name
    assert path.parent == expected_root / "_stable_runs" / args.experiment_name
    assert manifest["results_root"] == str(expected_root)


def test_restore_rewinds_superseded_metric_rows(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import _rewind_iteration_artifacts

    report_path = tmp_path / "report_rows.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["training_iteration", "score"])
        writer.writeheader()
        writer.writerows(
            {"training_iteration": iteration, "score": iteration / 10}
            for iteration in range(1, 7)
        )
    metrics_path = tmp_path / "metrics_full.jsonl"
    metrics_path.write_text(
        "".join(
            json.dumps({"training_iteration": iteration}) + "\n"
            for iteration in range(1, 7)
        ),
        encoding="utf-8",
    )
    checkpoints_path = tmp_path / "checkpoint_index.jsonl"
    checkpoints_path.write_text(
        json.dumps({"training_iteration": 3}) + "\n"
        + json.dumps({"training_iteration": 6}) + "\n",
        encoding="utf-8",
    )

    _rewind_iteration_artifacts(
        restored_iteration=3,
        report_path=report_path,
        jsonl_paths=[metrics_path, checkpoints_path],
    )

    assert [int(row["training_iteration"]) for row in csv.DictReader(report_path.open())] == [1, 2, 3]
    assert [json.loads(line)["training_iteration"] for line in metrics_path.read_text().splitlines()] == [1, 2, 3]
    assert [json.loads(line)["training_iteration"] for line in checkpoints_path.read_text().splitlines()] == [3]


def test_resume_recovery_rewinds_progress_to_last_reported_checkpoint(tmp_path: Path):
    from itrader.rl_lib.train.pbt_contract import PBTProgressClock
    from itrader.rl_lib.train.train_solo_V2 import _rewind_resume_progress

    stable_trial_dir = tmp_path / "stable"
    stable_trial_dir.mkdir()
    (stable_trial_dir / "checkpoint_index.jsonl").write_text(
        json.dumps(
            {
                "training_iteration": 61,
                "pbt_progress_steps": 361_272,
                "reported_to_tune": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (stable_trial_dir / "metrics_full.jsonl").write_text(
        json.dumps(
            {
                "training_iteration": 61,
                "pbt_progress_steps": 361_272,
                "pbt_progress_updates": 73,
            }
        )
        + "\n"
        + json.dumps(
            {
                "training_iteration": 62,
                "pbt_progress_steps": 366_192,
                "pbt_progress_updates": 74,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = stable_trial_dir / "report_rows.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["training_iteration"])
        writer.writeheader()
        writer.writerows(
            [{"training_iteration": 61}, {"training_iteration": 62}]
        )
    progress = PBTProgressClock(
        target_steps=384_000,
        completed_steps=366_192,
        updates=74,
    )

    receipt = _rewind_resume_progress(
        progress=progress,
        restored_iteration=61,
        stable_trial_dir=stable_trial_dir,
        local_metrics_path=tmp_path / "local_metrics.jsonl",
    )

    assert receipt["rewound"] is True
    assert progress.completed_steps == 361_272
    assert progress.updates == 73
    assert [
        json.loads(line)["training_iteration"]
        for line in (stable_trial_dir / "metrics_full.jsonl").read_text().splitlines()
    ] == [61]
    assert [
        int(row["training_iteration"])
        for row in csv.DictReader(report_path.open())
    ] == [61]


def test_explicit_restore_progress_seeds_a_new_recovery_trial():
    from itrader.rl_lib.train.pbt_contract import PBTProgressClock
    from itrader.rl_lib.train.train_solo_V2 import _seed_restored_progress

    progress = PBTProgressClock(target_steps=384_000)

    seeded = _seed_restored_progress(
        progress,
        restore_source="stable/checkpoint-61",
        progress_state_exists=False,
        configured_steps=361_272,
        configured_updates=73,
    )

    assert seeded is True
    assert progress.completed_steps == 361_272
    assert progress.updates == 73


def test_explicit_restore_progress_never_overwrites_an_existing_clock():
    from itrader.rl_lib.train.pbt_contract import PBTProgressClock
    from itrader.rl_lib.train.train_solo_V2 import _seed_restored_progress

    progress = PBTProgressClock(
        target_steps=384_000,
        completed_steps=120_000,
        updates=25,
    )

    seeded = _seed_restored_progress(
        progress,
        restore_source="stable/checkpoint-20",
        progress_state_exists=True,
        configured_steps=96_000,
        configured_updates=20,
    )

    assert seeded is False
    assert progress.completed_steps == 120_000
    assert progress.updates == 25


def test_persisted_metrics_include_report_only_selection_fields():
    from itrader.rl_lib.train.train_solo_V2 import _persisted_scalar_metrics

    result = {
        "training_iteration": 7,
        "evaluation": {"custom_metrics": {"economic_net_return_mean": 0.001}},
    }
    report = {
        "training_iteration": 7,
        "pbt_progress_steps": 33_600,
        "training/optimization_cost_bps": 0.25,
        "evaluation/economic_selection_score": 0.0008,
    }

    persisted = _persisted_scalar_metrics(result, report)

    assert persisted["pbt_progress_steps"] == 33_600
    assert persisted["training/optimization_cost_bps"] == pytest.approx(0.25)
    assert persisted["evaluation/economic_selection_score"] == pytest.approx(0.0008)


def test_reported_training_cost_uses_static_optimization_cost_without_schedule():
    from itrader.rl_lib.train.train_solo_V2 import _training_optimization_cost_bps

    assert _training_optimization_cost_bps(
        {
            "optimization_cost_bps": 1.0,
            "optimization_cost_schedule": (),
        },
        completed_steps=48_000,
    ) == pytest.approx(1.0)

    assert _training_optimization_cost_bps(
        {
            "optimization_cost_schedule": (),
            "env_config": {
                "task_sampler": {
                    "reward_params_by_id": {
                        "net_return": {"cost_bps": 0.5, "reward_scale": 1000.0}
                    }
                }
            },
        },
        completed_steps=96_000,
    ) == pytest.approx(0.5)


def test_checkpoint_selection_scope_does_not_call_ineligible_improvement_eligible():
    from itrader.rl_lib.train.train_solo_V2 import _checkpoint_selection_scope

    assert _checkpoint_selection_scope(
        report_to_tune=False,
        selection_score=-0.999,
    ) == "ineligible_evaluation_improvement_for_forensics"
    assert _checkpoint_selection_scope(
        report_to_tune=False,
        selection_score=0.001,
    ) == "eligible_evaluation_improvement"
    assert _checkpoint_selection_scope(
        report_to_tune=True,
        selection_score=-0.999,
    ) == "reported_checkpoint_boundary"


def test_best_checkpoint_receipt_declares_economic_eligibility(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import _update_best_checkpoint_record

    record = tmp_path / "best_checkpoint.json"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    assert _update_best_checkpoint_record(
        record,
        metric="evaluation/economic_selection_score",
        report={"training_iteration": 16, "pbt_progress_steps": 77_800},
        stable_checkpoint=checkpoint,
        score=-0.999,
        economically_eligible=False,
        selection_scope="ineligible_evaluation_improvement_for_forensics",
    )

    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["economically_eligible"] is False


def test_best_checkpoint_economic_eligibility_is_independent_of_rank_score(
    tmp_path: Path,
):
    from itrader.rl_lib.train.train_solo_V2 import _update_best_checkpoint_record

    record = tmp_path / "best_checkpoint.json"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    assert _update_best_checkpoint_record(
        record,
        metric="evaluation/economic_search_score",
        report={"training_iteration": 16, "pbt_progress_steps": 77_800},
        stable_checkpoint=checkpoint,
        score=0.25,
        economically_eligible=False,
        selection_scope="reported_checkpoint_boundary",
    )

    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["score"] == pytest.approx(0.25)
    assert payload["economically_eligible"] is False


def test_source_tree_fingerprint_covers_untracked_python_sources(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import _source_tree_fingerprint

    source_root = tmp_path / "itrader"
    source_root.mkdir()
    (source_root / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    nested = source_root / "new_module"
    nested.mkdir()
    (nested / "untracked.py").write_text("VALUE = 2\n", encoding="utf-8")

    before = _source_tree_fingerprint(tmp_path)
    (nested / "untracked.py").write_text("VALUE = 3\n", encoding="utf-8")
    after = _source_tree_fingerprint(tmp_path)

    assert before["algorithm"] == "sha256(sorted relative-path TAB file-sha256 LF)"
    assert before["python_file_count"] == 2
    assert before["source_tree_sha256"] != after["source_tree_sha256"]


def test_actor_start_fails_closed_when_frozen_run_source_changes(tmp_path: Path):
    from itrader.rl_lib.evaluation.provenance import (
        canonical_json_sha256,
        dataset_sha256,
        source_tree_fingerprint,
    )
    from itrader.rl_lib.train.train_solo_V2 import _assert_frozen_run_provenance

    source_root = tmp_path / "itrader"
    source_root.mkdir()
    source = source_root / "solver.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    dataset = tmp_path / "prices.parquet"
    dataset.write_bytes(b"dataset")
    fixed_tasks = [{"seed": 11, "horizon": 96}]
    config = {
        "provenance_enforced": True,
        "source_provenance": source_tree_fingerprint(tmp_path),
        "dataset_sha256": dataset_sha256(dataset),
        "evaluation_panel_sha256": canonical_json_sha256(fixed_tasks),
        "env_config": {"df_path": str(dataset)},
        "evaluation_config": {
            "env_config": {"task_sampler": {"fixed_tasks": fixed_tasks}}
        },
    }

    _assert_frozen_run_provenance(config, repo_root=tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source tree changed"):
        _assert_frozen_run_provenance(config, repo_root=tmp_path)


def test_frozen_provenance_config_is_accepted_by_pinned_ray(tmp_path: Path):
    from ray.rllib.algorithms.ppo import PPOConfig

    from itrader.rl_lib.train.train_solo_V2 import (
        _freeze_run_provenance,
        build_base_config,
    )

    data_path = tmp_path / "prices.parquet"
    data_path.write_bytes(b"dataset")
    config = build_base_config(
        args=_args(seed=20260924),
        data_path=data_path,
        use_pbt_mode=True,
    )
    _freeze_run_provenance(config, data_path)

    PPOConfig().update_from_dict(config)

    assert config["frozen_provenance"]["evaluation_panel_sha256"]


def test_extract_report_exposes_held_out_economic_metrics():
    from itrader.rl_lib.train.train_solo_V2 import _extract_report

    result = {
        "training_iteration": 2,
        "env_runners": {
            "custom_metrics": {
                "optimization_reward__sum": 0.8,
                "optimization_reward__count": 1,
                "economic_net_return__sum": -0.002,
                "economic_net_return__count": 1,
            }
        },
        "evaluation": {
            "env_runners": {
                "custom_metrics": {
                    "optimization_reward__sum": 0.4,
                    "optimization_reward__count": 1,
                    "economic_gross_return__sum": 0.003,
                    "economic_gross_return__count": 1,
                    "economic_net_return__sum": 0.001,
                    "economic_net_return__count": 1,
                    "economic_net_sharpe_per_sqrt_bar__sum": 0.05,
                    "economic_net_sharpe_per_sqrt_bar__count": 1,
                }
            }
        },
        "learners": {
            "default_policy": {
                "policy_loss": 0.1,
                "vf_loss": 0.2,
                "entropy": 0.3,
            }
        },
    }

    report = _extract_report(result, default_iteration=2)

    assert report["training/optimization_reward_mean"] == pytest.approx(0.8)
    assert report["training/net_return_mean"] == pytest.approx(-0.002)
    assert report["evaluation/gross_return_mean"] == pytest.approx(0.003)
    assert report["evaluation/net_return_mean"] == pytest.approx(0.001)
    assert report["evaluation/net_sharpe_per_sqrt_bar_mean"] == pytest.approx(0.05)
    assert "custom_metrics/episode_return_mean" not in report


def test_extract_report_supports_env_runner_metric_nesting():
    from itrader.rl_lib.train.train_solo_V2 import _extract_report

    result = {
        "training_iteration": 3,
        "env_runners": {
            "episode_len_mean": 128.0,
            "custom_metrics": {
                "optimization_reward__sum": 2.5,
                "optimization_reward__count": 1,
                "economic_net_return__sum": 0.01,
                "economic_net_return__count": 1,
            },
        },
        "evaluation": {
            "env_runners": {
                "custom_metrics": {
                    "economic_gross_return__sum": 0.04,
                    "economic_gross_return__count": 1,
                    "economic_net_return__sum": 0.03,
                    "economic_net_return__count": 1,
                    "economic_net_sharpe_per_sqrt_bar__sum": 0.02,
                    "economic_net_sharpe_per_sqrt_bar__count": 1,
                }
            }
        },
    }

    report = _extract_report(result, default_iteration=1)

    assert report["training/optimization_reward_mean"] == pytest.approx(2.5)
    assert report["training/net_return_mean"] == pytest.approx(0.01)
    assert report["evaluation/gross_return_mean"] == pytest.approx(0.04)
    assert report["evaluation/net_return_mean"] == pytest.approx(0.03)
    assert report["evaluation/net_sharpe_per_sqrt_bar_mean"] == pytest.approx(0.02)
    assert report["episode_len_mean"] == pytest.approx(128.0)


def test_extract_report_uses_native_metrics_when_root_metrics_are_empty():
    from itrader.rl_lib.train.train_solo_V2 import _extract_report

    report = _extract_report(
        {
            "training_iteration": 10,
            "custom_metrics": {},
            "env_runners": {
                "custom_metrics": {
                    "optimization_reward__sum": -10.4917,
                    "optimization_reward__count": 1,
                    "economic_net_return__sum": -0.0104917,
                    "economic_net_return__count": 1,
                    "economic_net_sharpe_per_sqrt_bar__sum": -0.1,
                    "economic_net_sharpe_per_sqrt_bar__count": 1,
                }
            },
            "evaluation": {
                "custom_metrics": {},
                "env_runners": {
                    "custom_metrics": {
                        "economic_net_return__sum": -0.02,
                        "economic_net_return__count": 1,
                        "economic_net_sharpe_per_sqrt_bar__sum": -0.2,
                        "economic_net_sharpe_per_sqrt_bar__count": 1,
                    }
                },
            },
            "learners": {"default_policy": {}},
        },
        default_iteration=10,
    )

    assert report["training/optimization_reward_mean"] == pytest.approx(-10.4917)
    assert report["training/net_return_mean"] == pytest.approx(-0.0104917)
    assert report["evaluation/net_return_mean"] == pytest.approx(-0.02)
    assert math.isnan(report["evaluation/gross_return_mean"])


def test_extract_report_keeps_missing_economic_metrics_unknown():
    from itrader.rl_lib.train.train_solo_V2 import _extract_report

    report = _extract_report(
        {"training_iteration": 1, "custom_metrics": {}, "evaluation": {}},
        default_iteration=1,
    )

    assert math.isnan(report["training/net_return_mean"])
    assert math.isnan(report["evaluation/net_return_mean"])


def test_extract_report_does_not_treat_episode_return_as_task_reward():
    from itrader.rl_lib.train.train_solo_V2 import _extract_report

    report = _extract_report(
        {
            "training_iteration": 4,
            "env_runners": {
                "episode_return_mean": 1.25,
                "episode_len_mean": 96.0,
            },
        },
        default_iteration=1,
    )

    assert math.isnan(report["training/optimization_reward_mean"])
    assert report["episode_len_mean"] == pytest.approx(96.0)


def test_extract_report_ignores_legacy_result_tree_shapes():
    from itrader.rl_lib.train.train_solo_V2 import _extract_report

    report = _extract_report(
        {
            "training_iteration": 4,
            "episode_reward_mean": 1.25,
            "sampler_results": {
                "episode_len_mean": 96.0,
                "custom_metrics": {"optimization_reward_mean": 1.25},
            },
            "info": {
                "learner": {
                    "default_policy": {
                        "learner_stats": {"policy_loss": 0.7}
                    }
                }
            },
        },
        default_iteration=1,
    )

    assert math.isnan(report["training/optimization_reward_mean"])
    assert math.isnan(report["episode_len_mean"])
    assert math.isnan(report["policy_loss"])


def test_pbt_selection_score_makes_economic_eligibility_lexicographic():
    from itrader.rl_lib.train.pbt_contract import economic_selection_score

    assert economic_selection_score(
        net_return=0.001,
        net_sharpe=0.02,
        flat_action_fraction=0.5,
    ) == pytest.approx(0.001)
    assert economic_selection_score(
        net_return=0.003,
        net_sharpe=-0.01,
        flat_action_fraction=0.5,
    ) < -0.9
    assert economic_selection_score(
        net_return=-0.001,
        net_sharpe=0.02,
        flat_action_fraction=0.5,
    ) < -0.9
    assert economic_selection_score(
        net_return=0.003,
        net_sharpe=0.02,
        flat_action_fraction=0.99,
    ) < -0.9
    # Directional behavior is not penalized; only Sharpe and all-flat
    # eligibility affect the donor tier.
    assert economic_selection_score(
        net_return=0.002,
        net_sharpe=0.02,
        flat_action_fraction=0.01,
    ) == pytest.approx(0.002)


def test_pbt_selection_score_rejects_invalid_thresholds_and_measurements():
    from itrader.rl_lib.train.pbt_contract import economic_selection_score

    with pytest.raises(ValueError, match="between 0 and 1"):
        economic_selection_score(
            net_return=0.002,
            net_sharpe=0.02,
            flat_action_fraction=0.5,
            max_flat_action_fraction=1.1,
        )
    assert economic_selection_score(
        net_return=0.002,
        net_sharpe=0.02,
        flat_action_fraction=-0.1,
    ) == pytest.approx(-1.1)


def test_pbt_search_score_excludes_flat_but_ranks_nonflat_by_net_return():
    from itrader.rl_lib.train.pbt_contract import economic_search_score

    weak_nonflat = economic_search_score(
        net_return=-0.02,
        flat_action_fraction=0.5,
        max_flat_action_fraction=0.98,
    )
    stronger_nonflat = economic_search_score(
        net_return=-0.01,
        flat_action_fraction=0.0,
        max_flat_action_fraction=0.98,
    )
    flat = economic_search_score(
        net_return=0.0,
        flat_action_fraction=1.0,
        max_flat_action_fraction=0.98,
    )

    assert stronger_nonflat > weak_nonflat > flat


def test_append_report_row_expands_csv_header_for_new_metrics(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import _append_report_row

    report_path = tmp_path / "report_rows.csv"
    _append_report_row(report_path, {"training_iteration": 1, "policy_loss": 0.1})
    _append_report_row(
        report_path,
        {
            "training_iteration": 2,
            "policy_loss": 0.2,
            "custom_metrics/rc_utility_gain_mean": 0.4,
        },
    )

    rows = report_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == "training_iteration,policy_loss,custom_metrics/rc_utility_gain_mean"
    assert rows[1] == "1,0.1,"
    assert rows[2] == "2,0.2,0.4"


def test_atomic_json_evidence_is_complete_and_leaves_no_partial_file(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import _atomic_write_json

    target = tmp_path / "run_status.json"
    _atomic_write_json(target, {"status": "running", "iteration": 1})
    _atomic_write_json(target, {"status": "completed", "iteration": 3})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "status": "completed",
        "iteration": 3,
    }
    assert not target.with_suffix(".json.tmp").exists()


def test_run_completion_rejects_tune_terminated_trials_below_target_steps():
    from itrader.rl_lib.train.train_solo_V2 import _run_completion_status

    status, incomplete = _run_completion_status(
        [
            {
                "trial_id": "member00",
                "status": "TERMINATED",
                "last_result": {"pbt_progress_steps": 4_944, "done": False},
            },
            {
                "trial_id": "member01",
                "status": "TERMINATED",
                "last_result": {"pbt_progress_steps": 4_848, "done": False},
            },
        ],
        target_steps=57_600,
    )

    assert status == "incomplete"
    assert incomplete == ["member00", "member01"]


def test_run_completion_accepts_trials_that_reach_the_sampled_step_target():
    from itrader.rl_lib.train.train_solo_V2 import _run_completion_status

    status, incomplete = _run_completion_status(
        [
            {
                "trial_id": "member00",
                "status": "TERMINATED",
                "last_result": {"pbt_progress_steps": 57_720, "done": True},
            }
        ],
        target_steps=57_600,
    )

    assert status == "completed"
    assert incomplete == []


def test_cli_metric_columns_include_generic_action_slots():
    from itrader.rl_lib.train.train_solo_V2 import _cli_metric_columns

    columns = _cli_metric_columns(4)
    assert "custom_metrics/act_0_frac_mean" in columns
    assert "custom_metrics/act_3_frac_mean" in columns
    assert "custom_metrics/act_long_frac_mean" in columns
