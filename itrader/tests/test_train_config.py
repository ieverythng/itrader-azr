from argparse import Namespace
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
