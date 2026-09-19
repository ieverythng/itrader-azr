from argparse import Namespace
import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("ray")


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


def test_build_base_config_sets_seed_in_all_required_places(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=2026)
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=False)
    assert cfg["seed"] == 2026
    assert cfg["env_config"]["task_sampler"]["seed"] == 2026
    assert cfg["env_config"]["task_sampler"]["bar_resolution"] == "1m"


def test_build_base_config_pbt_disables_prometheus(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import build_base_config

    args = _args(seed=99)
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")

    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=True)
    assert cfg["metrics_sinks"] == []
    assert cfg["prometheus_export_port"] is None


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
    assert cfg["lambda"] == 0.95
    assert cfg["vf_loss_coeff"] == 1.0
    assert cfg["train_batch_size"] == 4800
    assert cfg["sgd_minibatch_size"] == 128
    assert cfg["num_sgd_iter"] == 30
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
    assert cfg["lambda"] == pytest.approx(0.9)
    assert cfg["vf_loss_coeff"] == pytest.approx(0.7)
    assert cfg["train_batch_size"] == 8192
    assert cfg["sgd_minibatch_size"] == 256
    assert cfg["num_sgd_iter"] == 12
    assert cfg["gamma"] == pytest.approx(0.985)
    assert cfg["vf_clip_param"] == pytest.approx(12.5)
    assert cfg["lr"] == pytest.approx(3e-4)
    assert cfg["auto_plot_on_end"] is False


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
            "episode_reward_mean": 1.25,
            "custom_metrics": {
                "act_0_frac_mean": 0.2,
                "act_3_frac_mean": 0.1,
                "rc_utility_gain_mean": 0.4,
                "success_mean": 0.75,
                "task_horizon_mean": 96.0,
            },
            "info": {
                "learner": {
                    "default_policy": {
                        "learner_stats": {
                            "policy_loss": 0.1,
                            "vf_loss": 0.2,
                            "entropy": 0.3,
                        }
                    }
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


def test_pbt_observation_contract_rejects_stateful_rllib_filter(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import (
        _validate_pbt_observation_contract,
        build_base_config,
    )

    args = _args()
    data_path = tmp_path / "fake.parquet"
    data_path.write_bytes(b"dummy")
    cfg = build_base_config(args=args, data_path=data_path, use_pbt_mode=True)

    with pytest.raises(ValueError, match="NoFilter"):
        _validate_pbt_observation_contract(cfg, use_pbt_mode=True)

    cfg["observation_filter"] = "NoFilter"
    cfg["env_config"]["observation_transform"] = "market_relative_v1"
    cfg["evaluation_config"]["env_config"]["observation_transform"] = "market_relative_v1"
    _validate_pbt_observation_contract(cfg, use_pbt_mode=True)


def test_checkpoint_consistent_train_result_replaces_pre_sync_evaluation():
    from itrader.rl_lib.train.train_solo_V2 import (
        train_with_checkpoint_consistent_evaluation,
    )

    calls = []

    class FakeWorkers:
        def local_worker(self):
            return "learner-worker"

    class FakeEvaluationWorkers:
        def foreach_policy(self, fn):
            policy = type("Policy", (), {"config": {"explore": True}})()
            fn(policy, "default_policy")
            calls.append(("deterministic", {"explore": policy.config["explore"]}))

        def sync_weights(self, **kwargs):
            calls.append(("sync", kwargs))

    class FakeAlgorithm:
        workers = FakeWorkers()
        evaluation_workers = FakeEvaluationWorkers()

        def train(self):
            calls.append(("train", {}))
            return {"training_iteration": 6, "evaluation": {"score": "pre-sync"}}

        def evaluate(self):
            calls.append(("evaluate", {}))
            # Ray 2.9 Algorithm.evaluate() returns the evaluation payload itself.
            return {"score": "post-sync"}

    result = train_with_checkpoint_consistent_evaluation(FakeAlgorithm())

    assert result["evaluation"] == {"score": "post-sync"}
    assert calls == [
        ("train", {}),
        ("deterministic", {"explore": False}),
        ("sync", {"from_worker_or_learner_group": "learner-worker"}),
        ("evaluate", {}),
    ]


def test_checkpoint_snapshot_is_copied_outside_tune_ownership(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import _copy_stable_checkpoint

    tune_checkpoint = tmp_path / "tune" / "checkpoint"
    tune_checkpoint.mkdir(parents=True)
    (tune_checkpoint / "algorithm_state.pkl").write_bytes(b"state")
    (tune_checkpoint / "rllib_checkpoint.json").write_text(
        '{"type": "Algorithm"}', encoding="utf-8"
    )

    stable_checkpoint = _copy_stable_checkpoint(
        tune_checkpoint,
        stable_trial_dir=tmp_path / "stable",
        training_iteration=12,
    )

    assert stable_checkpoint.parent == tmp_path / "stable" / "checkpoints"
    assert (stable_checkpoint / "algorithm_state.pkl").read_bytes() == b"state"
    assert (stable_checkpoint / "rllib_checkpoint.json").stat().st_size > 0


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

    monkeypatch.setattr(train_solo_V2.session, "report", pausing_report)

    train_solo_V2._report_checkpoint_with_index(
        index_path=index_path,
        record=record,
        report={"training_iteration": 10},
        checkpoint="checkpoint-object",
    )

    assert [json.loads(row) for row in observed["rows"]] == [record]
    assert observed["report"] == {"training_iteration": 10}
    assert observed["checkpoint"] == "checkpoint-object"


def test_best_checkpoint_index_only_advances_on_a_better_held_out_score(tmp_path: Path):
    from itrader.rl_lib.train.train_solo_V2 import _update_best_checkpoint_record

    index_path = tmp_path / "best_checkpoint.json"
    first = _update_best_checkpoint_record(
        index_path,
        metric="evaluation/net_return_mean",
        report={"training_iteration": 10, "pbt_progress_steps": 48_000},
        stable_checkpoint=tmp_path / "checkpoint-10",
        score=0.001,
    )
    worse = _update_best_checkpoint_record(
        index_path,
        metric="evaluation/net_return_mean",
        report={"training_iteration": 20, "pbt_progress_steps": 96_000},
        stable_checkpoint=tmp_path / "checkpoint-20",
        score=-0.001,
    )

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert first is True
    assert worse is False
    assert payload["score"] == pytest.approx(0.001)
    assert payload["training_iteration"] == 10
    assert payload["pbt_progress_steps"] == 48_000


def test_pbt_contract_rejects_misaligned_checkpoint_cadence():
    from itrader.rl_lib.train.train_solo_V2 import _pbt_intervals

    args = _args()
    args.checkpoint_interval = 10
    args.pbt_interval = 3

    with pytest.raises(ValueError, match="checkpoint_interval"):
        _pbt_intervals(args)


def test_ray_29_pbt_scheduler_is_the_single_objective_owner():
    from itrader.rl_lib.train.train_solo_V2 import _tune_objective_kwargs

    assert _tune_objective_kwargs(use_pbt_mode=True) == {}
    assert _tune_objective_kwargs(use_pbt_mode=False) == {
        "metric": "evaluation/net_return_mean",
        "mode": "max",
    }


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
    assert scheduler._burn_in_period == 96_000


def test_synchronous_pbt_orders_donor_checkpoint_before_exploit():
    from itrader.rl_lib.train.train_solo_V2 import _build_pbt_scheduler

    args = _args()
    args.train_batch_size = 4800
    args.pbt_interval = 10
    args.pbt_burn_in = 20
    args.checkpoint_interval = 10
    args.pbt_synch = True

    scheduler = _build_pbt_scheduler(args)

    assert scheduler._synch is True


def test_pbt_explore_contract_clamps_invalid_mutations():
    from itrader.rl_lib.train.pbt_contract import constrain_trial_config

    config = {
        "gamma": 1.188,
        "lambda": 1.7,
        "lr": 1.0,
        "clip_param": 9.0,
        "sgd_minibatch_size": 307,
        "num_sgd_iter": 19,
        "vf_clip_param": 999.0,
        "vf_loss_coeff": -4.0,
        "train_batch_size": 5000,
    }

    constrained = constrain_trial_config(config)

    assert constrained is config
    assert 0.95 <= config["gamma"] <= 0.999
    assert 0.8 <= config["lambda"] <= 1.0
    assert config["lr"] <= 0.005
    assert config["clip_param"] in {0.1, 0.2, 0.3}
    assert config["sgd_minibatch_size"] in {128, 256, 512}
    assert config["num_sgd_iter"] in {10, 20, 30}
    assert config["vf_clip_param"] in {5.0, 10.0, 20.0}
    assert config["vf_loss_coeff"] in {0.5, 1.0, 2.0}
    assert config["train_batch_size"] == 5000
    assert config["train_batch_size"] >= 2 * config["sgd_minibatch_size"]


def test_pbt_population_keeps_equal_exposure_and_matched_evaluation_seed():
    from itrader.rl_lib.train.train_solo_V2 import (
        PBT_MUTATIONS,
        _configure_pbt_population,
    )

    args = _args(seed=20260912)
    config = {
        "seed": args.seed,
        "train_batch_size": 1024,
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

    assert "train_batch_size" not in PBT_MUTATIONS
    assert config["train_batch_size"] == 1024
    assert not isinstance(config["seed"], int)
    assert not isinstance(config["env_config"]["task_sampler"]["seed"], int)
    assert config["evaluation_config"]["env_config"]["task_sampler"]["seed"] == 30_260_931
    assert config["evaluation_config"]["env_config"]["task_sampler"]["fixed_tasks"] == [
        {"seed": 11},
        {"seed": 22},
    ]


def test_restored_pbt_worker_runs_only_remaining_iterations():
    from itrader.rl_lib.train.train_solo_V2 import _remaining_training_steps

    assert list(_remaining_training_steps(current_iteration=0, stop_iteration=12)) == list(range(12))
    assert list(_remaining_training_steps(current_iteration=6, stop_iteration=12)) == list(range(6))
    assert list(_remaining_training_steps(current_iteration=12, stop_iteration=12)) == []


def test_progress_clock_recovers_target_exposure_across_donor_restore(tmp_path: Path):
    from itrader.rl_lib.train.pbt_contract import PBTProgressClock

    state_path = tmp_path / "pbt_progress.json"
    clock = PBTProgressClock.load(state_path, target_steps=960_000)
    clock.advance({"num_env_steps_sampled_this_iter": 4_832}, fallback_steps=4_800)
    clock.save(state_path)

    restored = PBTProgressClock.load(state_path, target_steps=960_000)

    assert restored.completed_steps == 4_832
    assert restored.remaining


def test_progress_clock_fails_closed_when_persisted_state_is_corrupt(tmp_path: Path):
    from itrader.rl_lib.train.pbt_contract import PBTProgressClock

    state_path = tmp_path / "pbt_progress.json"
    state_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid PBT progress state"):
        PBTProgressClock.load(state_path, target_steps=960_000)


def test_extract_report_exposes_ray_29_ppo_diagnostics():
    from itrader.rl_lib.train.train_solo_V2 import _extract_report

    report = _extract_report(
        {
            "training_iteration": 3,
            "episode_reward_mean": 0.1,
            "info": {
                "learner": {
                    "default_policy": {
                        "learner_stats": {
                            "policy_loss": 0.01,
                            "vf_loss": 0.02,
                            "entropy": 0.03,
                            "kl": 0.004,
                            "vf_explained_var": 0.5,
                            "total_loss": 0.06,
                            "grad_gnorm": 1.25,
                            "cur_lr": 5e-5,
                            "cur_kl_coeff": 0.2,
                        }
                    }
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
    monkeypatch.setattr(train_solo_V2.Path, "home", staticmethod(lambda: tmp_path))

    manifest, path = train_solo_V2.write_run_manifest(
        base_config={"seed": 77},
        args=args,
        data_path=data_path,
    )

    assert path.parent == tmp_path / "ray_results" / "_stable_runs" / args.experiment_name
    assert path.stat().st_size > 0
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == manifest["run_id"]


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


def test_extract_report_exposes_held_out_economic_metrics():
    from itrader.rl_lib.train.train_solo_V2 import _extract_report

    result = {
        "training_iteration": 2,
        "episode_reward_mean": 0.8,
        "custom_metrics": {
            "optimization_reward_mean": 0.8,
            "economic_net_return_mean": -0.002,
        },
        "evaluation": {
            "custom_metrics": {
                "optimization_reward_mean": 0.4,
                "economic_gross_return_mean": 0.003,
                "economic_net_return_mean": 0.001,
                "economic_net_sharpe_per_sqrt_bar_mean": 0.05,
            }
        },
        "info": {
            "learner": {
                "default_policy": {
                    "learner_stats": {"policy_loss": 0.1, "vf_loss": 0.2, "entropy": 0.3}
                }
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


def test_cli_metric_columns_include_generic_action_slots():
    from itrader.rl_lib.train.train_solo_V2 import _cli_metric_columns

    columns = _cli_metric_columns(4)
    assert "custom_metrics/act_0_frac_mean" in columns
    assert "custom_metrics/act_3_frac_mean" in columns
    assert "custom_metrics/act_long_frac_mean" in columns
