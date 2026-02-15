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
