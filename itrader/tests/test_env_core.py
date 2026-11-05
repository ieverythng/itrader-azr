import numpy as np
import pytest

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


def test_market_relative_observation_is_bounded_and_checkpoint_free(tiny_ohlcv_path):
    env = ITraderEnv(
        {
            "df_path": str(tiny_ohlcv_path),
            "window_size": 30,
            "observation_transform": "market_relative_v1",
            "task_sampler": {
                "seed": 2026,
                "window_size": 30,
                "n_features": 5,
                "t_embed_dim": 32,
                "action_dim": 3,
                "horizons": [64],
                "reward_ids": ["pnl_linear"],
            },
        }
    )

    obs, _ = env.reset()
    market = obs[: 30 * 5].reshape(30, 5)

    assert np.isfinite(market).all()
    assert np.max(np.abs(market)) <= 10.0
    assert market[-1, env.cols.index(env.close_col)] == pytest.approx(0.0)
    volume = market[:, env.cols.index("VOLUME")]
    assert float(volume.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(volume.std()) == pytest.approx(1.0, abs=1e-5)


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


def test_env_applies_terminal_drawdown_penalty_on_final_step(tiny_ohlcv_path):
    env = ITraderEnv(
        {
            "df_path": str(tiny_ohlcv_path),
            "window_size": 30,
            "task_sampler_overrides": {
                "dd_mode": "terminal",
                "dd_T_weight": 0.5,
                "normalize": False,
            },
            "task_sampler": {
                "seed": 11,
                "window_size": 30,
                "n_features": 5,
                "t_embed_dim": 32,
                "action_dim": 3,
                "horizons": [1],
                "reward_ids": ["pnl_linear"],
            },
        }
    )
    env.reset()

    _, reward, done, _, info = env.step(0)

    assert done is True
    assert np.isfinite(reward)
    assert info["reward_comps"]["dd"] < 0.0


def test_env_supplies_episode_returns_to_sharpe_reward(tiny_ohlcv_path):
    env = ITraderEnv(
        {
            "df_path": str(tiny_ohlcv_path),
            "window_size": 30,
            "task_sampler": {
                "seed": 11,
                "window_size": 30,
                "n_features": 5,
                "t_embed_dim": 32,
                "action_dim": 3,
                "horizons": [8],
                "reward_ids": ["pnl_sharpe"],
                "reward_params_by_id": {
                    "pnl_sharpe": {
                        "normalize": False,
                        "sharpe_min_samples": 4,
                        "sharpe_window": 8,
                    }
                },
            },
        }
    )
    env.reset()

    info = {}
    sharpe_deltas = []
    for action in (2, 0, 2, 0, 2, 0, 2, 0):
        _, reward, done, _, info = env.step(action)
        assert np.isfinite(reward)
        sharpe_deltas.append(info["reward_comps"]["sharpe_delta"])

    assert done is True
    assert np.isfinite(sharpe_deltas).all()
    assert any(delta != 0.0 for delta in sharpe_deltas)


def test_terminal_info_separates_optimization_reward_from_economic_evidence(tiny_ohlcv_path):
    env = ITraderEnv(
        {
            "df_path": str(tiny_ohlcv_path),
            "window_size": 30,
            "evidence_cost_bps": 1.0,
            "task_sampler_overrides": {
                "normalize": False,
                "dd_w": 0.0,
                "turnover_w": 0.0,
            },
            "task_sampler": {
                "seed": 11,
                "window_size": 30,
                "n_features": 5,
                "t_embed_dim": 32,
                "action_dim": 3,
                "horizons": [2],
                "reward_ids": ["pnl_linear"],
            },
        }
    )
    _, reset_info = env.reset(seed=77)
    start = int(reset_info["task"]["start_idx"])
    closes = env.df[env.close_col].to_numpy()
    expected_gross = (closes[start] - closes[start - 1]) / closes[start - 1]

    env.step(2)
    _, _, done, _, info = env.step(1)

    assert done is True
    evidence = info["episode_evidence"]
    assert evidence["schema_version"] == 1
    assert evidence["optimization_reward"] == pytest.approx(expected_gross)
    assert evidence["gross_return"] == pytest.approx(expected_gross)
    assert evidence["turnover"] == pytest.approx(2.0)
    assert evidence["liquidation_turnover"] == pytest.approx(0.0)
    assert evidence["cost_return"] == pytest.approx(0.0002)
    assert evidence["net_return"] == pytest.approx(expected_gross - 0.0002)
    assert evidence["success"] is bool(expected_gross > 0.0002)
    expected_net_steps = [expected_gross - 0.0001, -0.0001]
    expected_net_sharpe = np.mean(expected_net_steps) / np.std(expected_net_steps, ddof=1)
    assert evidence["gross_sharpe_per_sqrt_bar"] == pytest.approx(2.0 ** -0.5)
    assert evidence["net_sharpe_per_sqrt_bar"] == pytest.approx(expected_net_sharpe)
    assert evidence["annualized_net_sharpe"] is None
    assert evidence["positive_net_bar_fraction"] == pytest.approx(0.5)
    assert evidence["gross_exposure_fraction"] == pytest.approx(0.5)
    assert evidence["position_changes"] == 2
    assert evidence["split"] == "all"


def test_env_never_crosses_symbol_or_chronological_split(multi_symbol_ohlcv_path):
    env = ITraderEnv(
        {
            "df_path": str(multi_symbol_ohlcv_path),
            "window_size": 30,
            "data_split": "validation",
            "split_fractions": [0.6, 0.2, 0.2],
            "split_purge_rows": 4,
            "task_sampler": {
                "seed": 5,
                "window_size": 30,
                "n_features": 5,
                "t_embed_dim": 32,
                "action_dim": 3,
                "horizons": [16],
                "reward_ids": ["pnl_linear"],
            },
        }
    )

    source = env.df
    for seed in range(24):
        _, reset_info = env.reset(seed=seed)
        start = int(reset_info["task"]["start_idx"])
        horizon = int(reset_info["task"]["horizon"])
        info = {}
        for _ in range(horizon):
            _, _, done, _, info = env.step(1)
        assert done is True
        symbols = source.iloc[start : start + horizon]["symbol"].unique().tolist()
        assert symbols == [info["episode_evidence"]["symbol"]]
        assert info["episode_evidence"]["split"] == "validation"
        local = start % 240
        assert 148 <= local
        assert local + horizon <= 188


def test_reset_seed_replays_task_and_start_on_same_env(tiny_ohlcv_path):
    env = _make_env(str(tiny_ohlcv_path), seed=2026)
    _, first = env.reset(seed=91)
    env.reset()
    _, replay = env.reset(seed=91)

    assert replay["task"] == first["task"]


def test_fixed_evaluation_panel_is_identical_across_external_env_seeds(tiny_ohlcv_path):
    fixed_tasks = [
        {
            "family_id": "00",
            "price_id": "SIM",
            "horizon": 16,
            "reward_id": "pnl_linear",
            "reward_params": {"pnl_w": 1.0, "turnover_w": 0.001},
            "seed": task_seed,
            "obs_dim": 182,
            "action_dim": 3,
            "bar_resolution": "1m",
        }
        for task_seed in (101, 202)
    ]
    config = {
        "df_path": str(tiny_ohlcv_path),
        "window_size": 30,
        "start_idx_mode": "random_episode",
        "task_sampler": {
            "seed": 7,
            "window_size": 30,
            "n_features": 5,
            "t_embed_dim": 32,
            "action_dim": 3,
            "fixed_tasks": fixed_tasks,
        },
    }
    env_a = ITraderEnv(config)
    env_b = ITraderEnv(config)

    sequence_a = [env_a.reset(seed=111)[1]["task"], env_a.reset()[1]["task"]]
    sequence_b = [env_b.reset(seed=999)[1]["task"], env_b.reset()[1]["task"]]

    assert sequence_a == sequence_b
    assert [item["seed"] for item in sequence_a] == [101, 202]
    assert [item["start_idx"] for item in sequence_a] == [
        item["start_idx"] for item in sequence_b
    ]


def test_task_sampler_overrides_are_normalized_before_reset_info(tiny_ohlcv_path):
    env = ITraderEnv(
        {
            "df_path": str(tiny_ohlcv_path),
            "window_size": 30,
            "task_sampler_overrides": {"turnover_w": 99.0, "dd_mode": "not-a-mode"},
            "task_sampler": {
                "seed": 5,
                "window_size": 30,
                "n_features": 5,
                "t_embed_dim": 32,
                "action_dim": 3,
                "horizons": [16],
                "reward_ids": ["pnl_linear"],
            },
        }
    )

    _, info = env.reset(seed=2)

    assert info["task"]["reward_params"]["turnover_w"] == pytest.approx(0.01)
    assert info["task"]["reward_params"]["dd_mode"] == "delta"
