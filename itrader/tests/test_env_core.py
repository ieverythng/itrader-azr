from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itrader.envs.itrader_env import ITraderEnv


def _make_env(
    df_path: str,
    start_idx_mode: str = "random_episode",
    seed: int = 1234,
    solver_state_mode: str = "off",
):
    return ITraderEnv(
        {
            "df_path": df_path,
            "window_size": 30,
            "start_idx_mode": start_idx_mode,
            "ret_norm_reset": "episode",
            "solver_state_mode": solver_state_mode,
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


def test_solver_state_reset_observation_has_explicit_tail_and_legacy_prefix(
    tiny_ohlcv_path,
):
    legacy = _make_env(str(tiny_ohlcv_path), seed=2026)
    stateful = _make_env(
        str(tiny_ohlcv_path),
        seed=2026,
        solver_state_mode="both",
    )

    legacy_obs, legacy_info = legacy.reset(seed=91)
    stateful_obs, stateful_info = stateful.reset(seed=91)

    assert stateful_info["task"] == legacy_info["task"]
    assert stateful.observation_space.shape == (30 * 5 + 32 + 2,)
    assert stateful.base_obs_dim == 30 * 5 + 32
    assert stateful.solver_state_dim == 2
    assert stateful.agent_obs_dim == 30 * 5 + 32 + 2
    assert stateful.observation_schema_id == "r4_v1_min"
    assert stateful.current_task.obs_dim == stateful.base_obs_dim
    np.testing.assert_array_equal(stateful_obs[:-2], legacy_obs)
    np.testing.assert_array_equal(stateful_obs[-2:], np.array([0.0, 1.0]))


def test_solver_state_tracks_selected_position_progress_and_stream_reset(
    tiny_ohlcv_path,
):
    env = ITraderEnv(
        {
            "df_path": str(tiny_ohlcv_path),
            "window_size": 30,
            "start_idx_mode": "stream_worker",
            "solver_state_mode": "both",
            "task_sampler": {
                "seed": 11,
                "window_size": 30,
                "n_features": 5,
                "t_embed_dim": 32,
                "action_dim": 3,
                "horizons": [2],
                "reward_ids": ["net_return"],
                "reward_params_by_id": {
                    "net_return": {"cost_bps": 1.0, "reward_scale": 1000.0}
                },
            },
        }
    )

    reset_obs, _ = env.reset(seed=4)
    first_obs, _, first_done, _, _ = env.step(2)
    terminal_obs, _, terminal_done, _, _ = env.step(0)
    next_episode_obs, _ = env.reset()

    np.testing.assert_array_equal(reset_obs[-2:], np.array([0.0, 1.0]))
    np.testing.assert_array_equal(first_obs[-2:], np.array([1.0, 0.5]))
    assert first_done is False
    np.testing.assert_array_equal(terminal_obs[-2:], np.array([-1.0, 0.0]))
    assert terminal_done is True
    np.testing.assert_array_equal(next_episode_obs[-2:], np.array([0.0, 1.0]))


def test_solver_state_rejects_previous_action_feature(tiny_ohlcv_path):
    with pytest.raises(ValueError, match="solver_state_mode"):
        _make_env(
            str(tiny_ohlcv_path),
            solver_state_mode="previous_action",
        )


def test_solver_state_rejects_mismatched_declared_observation_dimensions(
    tiny_ohlcv_path,
):
    with pytest.raises(ValueError, match="agent_obs_dim"):
        ITraderEnv(
            {
                "df_path": str(tiny_ohlcv_path),
                "window_size": 30,
                "solver_state_mode": "both",
                "base_obs_dim": 182,
                "solver_state_dim": 2,
                "agent_obs_dim": 183,
                "observation_schema_id": "r4_v1_min",
            }
        )


def test_solver_state_modes_use_the_same_two_dimensional_masked_layout(
    tiny_ohlcv_path,
):
    expected = {
        "masked": [0.0, 0.0],
        "position": [1.0, 0.0],
        "time": [0.0, 0.5],
        "both": [1.0, 0.5],
    }

    for mode, state in expected.items():
        env = ITraderEnv(
            {
                "df_path": str(tiny_ohlcv_path),
                "window_size": 30,
                "solver_state_mode": mode,
                "task_sampler": {
                    "seed": 11,
                    "window_size": 30,
                    "n_features": 5,
                    "t_embed_dim": 32,
                    "action_dim": 3,
                    "horizons": [2],
                    "reward_ids": ["net_return"],
                    "reward_params_by_id": {
                        "net_return": {"cost_bps": 1.0, "reward_scale": 1000.0}
                    },
                },
            }
        )
        env.reset(seed=4)
        next_obs, *_ = env.step(2)

        assert env.solver_state_dim == 2
        np.testing.assert_array_equal(next_obs[-2:], np.asarray(state))


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


def test_training_can_advance_net_return_cost_without_rebuilding_env(tiny_ohlcv_path):
    env = ITraderEnv(
        {
            "df_path": str(tiny_ohlcv_path),
            "window_size": 30,
            "task_sampler": {
                "seed": 5,
                "window_size": 30,
                "n_features": 5,
                "t_embed_dim": 32,
                "action_dim": 3,
                "horizons": [16],
                "reward_ids": ["net_return"],
                "reward_params_by_id": {
                    "net_return": {"cost_bps": 0.0, "reward_scale": 1000.0}
                },
            },
        }
    )

    _, first = env.reset(seed=2)
    env.set_task_sampler_overrides({"cost_bps": 0.5})
    _, second = env.reset()

    assert first["task"]["reward_params"]["cost_bps"] == 0.0
    assert second["task"]["reward_params"]["cost_bps"] == 0.5


def test_env_filters_partitioned_dataset_and_keeps_episode_inside_session(tmp_path: Path):
    rows = []
    for session, offset in (("pre_market", 0), ("regular", 100), ("after_hours", 200)):
        for symbol in ("AAPL", "MSFT"):
            for day_index, day in enumerate(("2024-01-02", "2024-01-03")):
                close = np.linspace(100 + offset + day_index, 101 + offset + day_index, 40)
                rows.append(
                    pd.DataFrame(
                        {
                            "symbol": symbol,
                            "ts": pd.date_range(
                                f"{day} 14:30:00+00:00", periods=40, freq="min"
                            ),
                            "session": session,
                            "session_id": f"{symbol}:{day}:{session}",
                            "year": 2024,
                            "OPEN": close - 0.05,
                            "HIGH": close + 0.15,
                            "LOW": close - 0.15,
                            "CLOSE": close,
                            "VOLUME": 1000.0,
                        }
                    )
                )
    root = tmp_path / "bars"
    pd.concat(rows, ignore_index=True).to_parquet(
        root,
        partition_cols=["session", "symbol", "year"],
        index=False,
    )
    env = ITraderEnv(
        {
            "df_path": str(root),
            "data_sessions": ["regular"],
            "data_symbols": ["AAPL"],
            "episode_boundary_cols": ["session_id"],
            "expected_bar_interval": "1min",
            "window_size": 5,
            "task_sampler": {
                "seed": 11,
                "window_size": 5,
                "n_features": 5,
                "t_embed_dim": 32,
                "action_dim": 3,
                "horizons": [8],
                "reward_ids": ["pnl_linear"],
            },
        }
    )

    _, info = env.reset(seed=8)
    start = info["task"]["start_idx"]
    episode = env.df.iloc[start - env.T : start + info["task"]["horizon"]]

    assert set(env.df["session"]) == {"regular"}
    assert set(env.df["symbol"]) == {"AAPL"}
    assert episode["session_id"].nunique() == 1
