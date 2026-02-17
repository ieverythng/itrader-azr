import numpy as np

from itrader.envs.itrader_env import ITraderEnv


def _make_env(df_path: str, start_idx_mode: str = "random_episode", seed: int = 1234):
    return ITraderEnv(
        {
            "df_path": df_path,
            "window_size": 30,
            "start_idx_mode": start_idx_mode,
            "ret_norm_reset": "episode",
            "task_sampler": {
                "seed": seed,
                "window_size": 30,
                "n_features": 5,
                "t_embed_dim": 32,
                "action_dim": 3,
                "horizons": [64],
                "reward_ids": ["pnl_linear"],
            },
        }
    )


def test_env_reset_and_step_outputs_are_finite(tiny_ohlcv_path):
    env = _make_env(str(tiny_ohlcv_path))
    obs, info = env.reset()
    assert obs.shape == (30 * 5 + 32,)
    assert np.isfinite(obs).all()
    assert "task" in info

    next_obs, reward, done, trunc, step_info = env.step(2)
    assert next_obs.shape == obs.shape
    assert np.isfinite(next_obs).all()
    assert np.isfinite(reward)
    assert isinstance(done, bool)
    assert isinstance(trunc, bool)
    assert "reward_comps" in step_info


def test_env_start_idx_respects_bounds(tiny_ohlcv_path):
    env = _make_env(str(tiny_ohlcv_path), start_idx_mode="stream_worker")
    obs, info = env.reset()
    assert obs is not None
    start_idx = int(info["task"]["start_idx"])
    horizon = int(info["task"]["horizon"])
    min_idx = env.T
    max_idx = max(min_idx, len(env.df) - horizon - 2)
    assert min_idx <= start_idx <= max_idx


def test_env_seeded_reset_is_reproducible_across_instances(tiny_ohlcv_path):
    cfg_df = str(tiny_ohlcv_path)
    env_a = _make_env(cfg_df, seed=2026)
    env_b = _make_env(cfg_df, seed=2026)
    _, info_a = env_a.reset()
    _, info_b = env_b.reset()
    assert info_a["task"]["start_idx"] == info_b["task"]["start_idx"]
    assert info_a["task"]["horizon"] == info_b["task"]["horizon"]
