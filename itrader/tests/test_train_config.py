from argparse import Namespace
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
        pop=2,
        timeout_hours=1,
        restore_ckpt="",
        best_params_json="",
        start_idx_mode="random_episode",
        stream_jump_every_episodes=0,
        stream_jump_on_wrap=True,
        stream_stride=0,
        ret_norm_reset="episode",
        task_pool_path="",
        p_pool=0.7,
        data_path="itrader/data/ohlcv.parquet",
        seed=seed,
    )


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


def test_cli_metric_columns_include_generic_action_slots():
    from itrader.rl_lib.train.train_solo_V2 import _cli_metric_columns

    columns = _cli_metric_columns(4)
    assert "custom_metrics/act_0_frac_mean" in columns
    assert "custom_metrics/act_3_frac_mean" in columns
    assert "custom_metrics/act_long_frac_mean" in columns
