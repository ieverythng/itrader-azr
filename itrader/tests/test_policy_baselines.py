from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from itrader.rl_lib.api_stack import LEGACY_API_STACK_ID
from itrader.rl_lib.evaluation.policy_baselines import (
    ControlPolicy,
    FrozenCheckpointPolicy,
    counterfactual_state_details,
    remaining_time_bin,
    summarize_counterfactual_state_steps,
    summarize_state_conditioned_steps,
    summarize_control_episodes,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    (("short", 0), ("flat", 1), ("long", 2)),
)
def test_fixed_control_policies_emit_the_expected_action(name: str, expected: int):
    policy = ControlPolicy(name=name, seed=123)

    assert policy.action(np.zeros(8, dtype=np.float32)) == expected


def test_random_control_is_reproducible_but_not_an_untrained_policy():
    left = ControlPolicy(name="random", seed=321)
    right = ControlPolicy(name="random", seed=321)

    left_actions = [left.action(np.zeros(4)) for _ in range(20)]
    right_actions = [right.action(np.zeros(4)) for _ in range(20)]

    assert left_actions == right_actions
    assert len(set(left_actions)) > 1


def test_frozen_checkpoint_policy_requests_deterministic_actions():
    import torch
    from ray.rllib.core.columns import Columns

    class NativeModule:
        def forward_inference(self, inputs):
            torch.testing.assert_close(
                inputs[Columns.OBS], torch.tensor([[1.0, 2.0]])
            )
            return {
                Columns.ACTION_DIST_INPUTS: torch.tensor([[0.0, 1.0, 2.0]]),
                Columns.VF_PREDS: torch.tensor([0.75]),
            }

    class CheckpointAlgorithm:
        config = SimpleNamespace(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )

        def get_module(self):
            return NativeModule()

    policy = FrozenCheckpointPolicy(CheckpointAlgorithm())

    assert policy.action(np.array([1.0, 2.0])) == 2


def test_frozen_checkpoint_policy_exposes_probabilities_and_value_prediction():
    import torch
    from ray.rllib.core.columns import Columns

    class NativeModule:
        def forward_inference(self, inputs):
            torch.testing.assert_close(
                inputs[Columns.OBS], torch.tensor([[1.0, 2.0]])
            )
            return {
                Columns.ACTION_DIST_INPUTS: torch.tensor([[0.0, 1.0, 2.0]]),
                Columns.VF_PREDS: torch.tensor([0.75]),
            }

    class CheckpointAlgorithm:
        config = SimpleNamespace(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )

        def get_module(self):
            return NativeModule()

    details = FrozenCheckpointPolicy(CheckpointAlgorithm()).action_details(
        np.array([1.0, 2.0])
    )

    assert details["action"] == 2
    assert sum(details["action_probabilities"]) == pytest.approx(1.0)
    assert details["action_probabilities"][2] > details["action_probabilities"][1]
    assert details["value_prediction"] == pytest.approx(0.75)
    assert details["policy_entropy"] > 0.0


@pytest.mark.parametrize(
    ("remaining", "expected"),
    ((1.0, "q4_early"), (0.75, "q3"), (0.5, "q2"), (0.25, "q1_terminal"), (0.0, "q1_terminal")),
)
def test_remaining_time_bins_are_fixed_quarters(remaining: float, expected: str):
    assert remaining_time_bin(remaining) == expected


def test_state_conditioned_summary_reports_switching_and_action_probabilities():
    rows = [
        {
            "start_idx": 101,
            "prior_position": -1.0,
            "remaining_time_bin": "q4_early",
            "action": 0,
            "switched": False,
            "action_logits": [2.0, 1.0, 0.0],
            "action_probabilities": [0.8, 0.1, 0.1],
            "value_prediction": 0.5,
            "policy_entropy": 0.6,
        },
        {
            "start_idx": 101,
            "prior_position": -1.0,
            "remaining_time_bin": "q1_terminal",
            "action": 2,
            "switched": True,
            "action_logits": [0.0, 0.5, 1.0],
            "action_probabilities": [0.2, 0.3, 0.5],
            "value_prediction": -0.5,
            "policy_entropy": 1.0,
        },
    ]

    summary = summarize_state_conditioned_steps(rows)

    assert summary["step_count"] == 2
    assert summary["overall"]["switching_fraction"] == pytest.approx(0.5)
    assert summary["overall"]["action_fractions"] == pytest.approx([0.5, 0.0, 0.5])
    assert summary["overall"]["mean_action_probabilities"] == pytest.approx([0.5, 0.2, 0.3])
    assert summary["overall"]["top_two_logit_margin_median"] == pytest.approx(0.75)
    assert summary["overall"]["top_two_probability_margin_median"] == pytest.approx(0.45)
    assert summary["by_start_idx"]["101"]["step_count"] == 2
    assert summary["by_start_idx"]["101"]["action_fractions"] == pytest.approx(
        [0.5, 0.0, 0.5]
    )
    assert summary["by_prior_position"]["short"]["step_count"] == 2
    assert summary["by_remaining_time_bin"]["q4_early"]["action_fractions"] == pytest.approx([1.0, 0.0, 0.0])


def test_counterfactual_state_details_hold_market_and_task_fixed():
    class TailSensitivePolicy:
        def action_details(self, observation):
            position, remaining = np.asarray(observation, dtype=np.float64)[-2:]
            logits = np.array([position, 0.0, remaining], dtype=np.float64)
            return {
                "action": int(np.argmax(logits)),
                "action_logits": logits.tolist(),
                "action_probabilities": [0.2, 0.3, 0.5],
                "value_prediction": float(position + 2.0 * remaining),
                "policy_entropy": 1.0,
            }

    observation = np.array([10.0, 20.0, -1.0, 0.75], dtype=np.float32)

    details = counterfactual_state_details(
        TailSensitivePolicy(),
        observation,
        prior_position=-1.0,
        remaining_fraction=0.75,
    )

    assert details["observed"]["action_logits"] == pytest.approx([-1.0, 0.0, 0.75])
    assert details["masked"]["action_logits"] == pytest.approx([0.0, 0.0, 0.0])
    assert details["position_only"]["value_prediction"] == pytest.approx(-1.0)
    assert details["time_only"]["value_prediction"] == pytest.approx(1.5)
    assert details["position_flipped"]["value_prediction"] == pytest.approx(2.5)
    assert details["time_mirrored"]["value_prediction"] == pytest.approx(-0.5)
    assert observation.tolist() == [10.0, 20.0, -1.0, 0.75]


def test_counterfactual_summary_reports_logit_value_and_action_sensitivity():
    rows = [
        {
            "counterfactuals": {
                "observed": {
                    "action": 2,
                    "action_logits": [0.0, 0.0, 2.0],
                    "value_prediction": 1.0,
                },
                "masked": {
                    "action": 1,
                    "action_logits": [1.0, 1.0, 0.0],
                    "value_prediction": 0.25,
                },
            }
        },
        {
            "counterfactuals": {
                "observed": {
                    "action": 1,
                    "action_logits": [0.0, 1.0, 0.0],
                    "value_prediction": -0.5,
                },
                "masked": {
                    "action": 1,
                    "action_logits": [0.0, 1.0, 0.0],
                    "value_prediction": -0.25,
                },
            }
        },
    ]

    summary = summarize_counterfactual_state_steps(rows)

    assert summary["masked_vs_observed"]["argmax_change_fraction"] == pytest.approx(0.5)
    assert summary["masked_vs_observed"]["mean_abs_value_delta"] == pytest.approx(0.5)
    assert summary["masked_vs_observed"]["mean_logit_l2_delta"] == pytest.approx(
        np.sqrt(6.0) / 2.0
    )


def test_checkpoint_replay_uses_hashed_frozen_validation_panel(tmp_path):
    from itrader.rl_lib.evaluation.run_checkpoint_replay import load_validation_env

    dataset = tmp_path / "ohlcv.parquet"
    dataset.write_bytes(b"fixed dataset")
    resolved = tmp_path / "resolved_config.json"
    resolved.write_text(json.dumps({
        "dataset_sha256": hashlib.sha256(b"fixed dataset").hexdigest(),
        "config": {"evaluation_config": {"env_config": {
            "df_path": str(dataset), "data_split": "validation",
            "task_sampler": {"fixed_tasks": [{"seed": 10}]},
        }}},
    }), encoding="utf-8")

    config = load_validation_env(resolved)

    assert config["task_sampler"]["fixed_tasks"] == [{"seed": 10}]


def _fixed_task(*, start_idx: int, horizon: int = 96, seed: int = 10) -> dict:
    return {
        "family_id": "00",
        "price_id": "SIM",
        "horizon": horizon,
        "reward_id": "net_return",
        "reward_params": {"cost_bps": 1.0, "reward_scale": 1000.0},
        "seed": seed,
        "obs_dim": 332,
        "action_dim": 3,
        "start_idx": start_idx,
        "bar_resolution": "1m",
    }


def test_checkpoint_replay_can_use_a_non_overlapping_diagnostic_panel(tmp_path):
    from itrader.rl_lib.evaluation.provenance import canonical_json_sha256
    from itrader.rl_lib.evaluation.run_checkpoint_replay import (
        load_validation_context,
    )

    dataset = tmp_path / "ohlcv.parquet"
    dataset.write_bytes(b"fixed dataset")
    original_tasks = [_fixed_task(start_idx=1_000)]
    diagnostic_tasks = [_fixed_task(start_idx=2_000, seed=20)]
    resolved = tmp_path / "resolved_config.json"
    resolved.write_text(
        json.dumps(
            {
                "dataset_sha256": hashlib.sha256(b"fixed dataset").hexdigest(),
                "config": {
                    "evaluation_config": {
                        "env_config": {
                            "df_path": str(dataset),
                            "window_size": 60,
                            "data_split": "validation",
                            "task_sampler": {"fixed_tasks": original_tasks},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    panel = tmp_path / "fresh-panel.json"
    panel.write_text(json.dumps({"tasks": diagnostic_tasks}), encoding="utf-8")

    context = load_validation_context(
        resolved,
        evaluation_task_panel_path=panel,
    )

    assert context.diagnostic_only is True
    assert context.env_config["task_sampler"]["fixed_tasks"] == diagnostic_tasks
    assert context.source_panel_sha256 == canonical_json_sha256(original_tasks)
    assert context.evaluation_panel_sha256 == canonical_json_sha256(diagnostic_tasks)


def test_checkpoint_replay_refuses_an_overlapping_diagnostic_panel(tmp_path):
    from itrader.rl_lib.evaluation.run_checkpoint_replay import (
        load_validation_context,
    )

    dataset = tmp_path / "ohlcv.parquet"
    dataset.write_bytes(b"fixed dataset")
    original_tasks = [_fixed_task(start_idx=1_000, horizon=200)]
    resolved = tmp_path / "resolved_config.json"
    resolved.write_text(
        json.dumps(
            {
                "dataset_sha256": hashlib.sha256(b"fixed dataset").hexdigest(),
                "config": {
                    "evaluation_config": {
                        "env_config": {
                            "df_path": str(dataset),
                            "window_size": 60,
                            "data_split": "validation",
                            "task_sampler": {"fixed_tasks": original_tasks},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    panel = tmp_path / "overlap-panel.json"
    panel.write_text(
        json.dumps({"tasks": [_fixed_task(start_idx=1_250, seed=20)]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlaps the source selection panel"):
        load_validation_context(resolved, evaluation_task_panel_path=panel)


def test_checkpoint_replay_hashes_partitioned_validation_dataset(tmp_path):
    from itrader.rl_lib.evaluation.provenance import dataset_sha256
    from itrader.rl_lib.evaluation.run_checkpoint_replay import load_validation_env

    dataset = tmp_path / "bars"
    partition = dataset / "session=regular" / "symbol=AAPL"
    partition.mkdir(parents=True)
    (partition / "part-00000.parquet").write_bytes(b"fixed partition")
    resolved = tmp_path / "resolved_config.json"
    resolved.write_text(
        json.dumps(
            {
                "dataset_sha256": dataset_sha256(dataset),
                "config": {
                    "evaluation_config": {
                        "env_config": {
                            "df_path": str(dataset),
                            "data_split": "validation",
                            "task_sampler": {"fixed_tasks": [{"seed": 10}]},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_validation_env(resolved)

    assert config["df_path"] == str(dataset)


def test_checkpoint_fingerprint_requires_and_hashes_rllib_authority_files(tmp_path):
    from itrader.rl_lib.evaluation.provenance import checkpoint_sha256

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "algorithm_state.pkl").write_bytes(b"weights")
    with pytest.raises(ValueError, match="rllib_checkpoint.json"):
        checkpoint_sha256(checkpoint, api_stack_id=LEGACY_API_STACK_ID)

    (checkpoint / "rllib_checkpoint.json").write_text("{}\n", encoding="utf-8")
    policy = checkpoint / "policies" / "default_policy"
    policy.mkdir(parents=True)
    (policy / "policy_state.pkl").write_bytes(b"policy")
    before = checkpoint_sha256(checkpoint, api_stack_id=LEGACY_API_STACK_ID)
    (checkpoint / "algorithm_state.pkl").write_bytes(b"different weights")

    assert len(before) == 64
    assert checkpoint_sha256(checkpoint, api_stack_id=LEGACY_API_STACK_ID) != before


def test_checkpoint_replay_refuses_a_final_test_panel(tmp_path):
    from itrader.rl_lib.evaluation.run_checkpoint_replay import load_validation_env

    resolved = tmp_path / "resolved_config.json"
    resolved.write_text(json.dumps({
        "config": {"evaluation_config": {"env_config": {
            "data_split": "test", "task_sampler": {"fixed_tasks": [{"seed": 10}]},
        }}},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="validation"):
        load_validation_env(resolved)


def test_checkpoint_replay_recomputes_promotion_eligible_selection_score():
    from itrader.rl_lib.evaluation.run_checkpoint_replay import _replay_metric_value

    summary = {
        "evaluation/net_return_mean": 0.002,
        "evaluation/net_sharpe_per_sqrt_bar_mean": 0.03,
        "evaluation/custom_metrics/act_flat_frac_mean": 0.25,
    }
    config = {
        "pbt_selection_min_sharpe": 0.0,
        "pbt_selection_max_flat_action_fraction": 0.98,
    }

    assert _replay_metric_value(
        "evaluation/economic_selection_score", summary, config
    ) == pytest.approx(0.002)

    assert _replay_metric_value(
        "evaluation/economic_search_score", summary, config
    ) == pytest.approx(1.002)


def test_control_summary_preserves_episode_rows_and_economic_means():
    episodes = [
        {
            "net_return": 0.01,
            "gross_return": 0.012,
            "net_sharpe_per_sqrt_bar": 0.1,
            "turnover": 2.0,
            "optimization_reward": 10.0,
            "steps": 100,
            "action_fractions": [0.0, 1.0, 0.0],
        },
        {
            "net_return": -0.005,
            "gross_return": -0.003,
            "net_sharpe_per_sqrt_bar": -0.05,
            "turnover": 1.0,
            "optimization_reward": -5.0,
            "steps": 200,
            "action_fractions": [0.0, 1.0, 0.0],
        },
    ]

    summary = summarize_control_episodes(episodes)

    assert summary["evaluation/net_return_mean"] == pytest.approx(0.0025)
    assert summary["evaluation/gross_return_mean"] == pytest.approx(0.0045)
    assert summary["evaluation/net_sharpe_per_sqrt_bar_mean"] == pytest.approx(0.025)
    assert summary["evaluation/turnover_mean"] == pytest.approx(1.5)
    assert summary["episode_len_mean"] == pytest.approx(150.0)
    assert summary["custom_metrics/act_flat_frac_mean"] == pytest.approx(1.0)
    assert summary["evaluation/custom_metrics/act_flat_frac_mean"] == pytest.approx(1.0)


def test_control_artifacts_record_matched_dataset_panel_and_config_hashes(tmp_path):
    from itrader.rl_lib.evaluation.provenance import canonical_json_sha256
    from itrader.rl_lib.evaluation.run_policy_baselines import _write_artifacts

    dataset = tmp_path / "bars"
    partition = dataset / "session=regular" / "symbol=AAPL"
    partition.mkdir(parents=True)
    (partition / "part-00000.parquet").write_bytes(b"fixed partition")
    fixed_tasks = [{"seed": 11, "horizon": 96}]
    env_config = {
        "df_path": str(dataset),
        "window_size": 60,
        "data_split": "validation",
        "task_sampler": {"fixed_tasks": fixed_tasks},
    }
    args = SimpleNamespace(
        results_root=str(tmp_path / "results"),
        experiment_name="control_probe",
        policy_control="long",
        data_path=str(dataset),
        seed=77,
        evaluation_seed=30260943,
        evaluation_num_episodes=1,
        evidence_cost_bps=1.0,
        task_reward_ids="net_return",
        task_horizons="96",
        data_sessions="regular",
        data_symbols="AAPL",
        episode_boundary_cols="session_id",
        expected_bar_interval="1min",
        solver_state_mode="off",
        solver_state_residual_init="random_v1",
    )
    report = {
        "evaluation/gross_return_mean": 0.001,
        "evaluation/net_return_mean": 0.0008,
        "evaluation/net_sharpe_per_sqrt_bar_mean": 0.01,
        "evaluation/turnover_mean": 2.0,
    }

    _write_artifacts(
        args,
        report,
        [],
        env_config,
        canonical_json_sha256(fixed_tasks),
    )

    resolved = json.loads(
        (
            tmp_path
            / "results"
            / "_stable_metrics"
            / "control_probe"
            / "training_fn_baseline_long_00000"
            / "resolved_config.json"
        ).read_text(encoding="utf-8")
    )
    assert resolved["evaluation_seed"] == 30260943
    assert len(resolved["dataset_sha256"]) == 64
    assert len(resolved["evaluation_panel_sha256"]) == 64
    assert len(resolved["comparison_config_hash"]) == 64


def test_stateful_control_artifacts_are_arm_matched_and_namespaced(tmp_path):
    from itrader.rl_lib.evaluation.provenance import canonical_json_sha256
    from itrader.rl_lib.evaluation.run_policy_baselines import _write_artifacts

    dataset = tmp_path / "bars"
    partition = dataset / "session=regular" / "symbol=AAPL"
    partition.mkdir(parents=True)
    (partition / "part-00000.parquet").write_bytes(b"fixed partition")
    fixed_tasks = [{"seed": 11, "horizon": 96}]
    env_config = {
        "df_path": str(dataset),
        "window_size": 60,
        "solver_state_mode": "both",
        "data_split": "validation",
        "task_sampler": {"fixed_tasks": fixed_tasks},
    }
    args = SimpleNamespace(
        results_root=str(tmp_path / "results"),
        experiment_name="stateful_control_probe",
        policy_control="untrained",
        data_path=str(dataset),
        seed=77,
        evaluation_seed=30260943,
        evaluation_num_episodes=1,
        evidence_cost_bps=1.0,
        task_reward_ids="net_return",
        task_horizons="96",
        data_sessions="regular",
        data_symbols="AAPL",
        episode_boundary_cols="session_id",
        expected_bar_interval="1min",
        evaluation_task_panel_path="panel.json",
        solver_state_mode="both",
        solver_state_residual_init="zero_v1",
        window=60,
    )
    report = {
        "evaluation/gross_return_mean": 0.001,
        "evaluation/net_return_mean": 0.0008,
        "evaluation/net_sharpe_per_sqrt_bar_mean": 0.01,
        "evaluation/turnover_mean": 2.0,
    }

    _write_artifacts(
        args,
        report,
        [],
        env_config,
        canonical_json_sha256(fixed_tasks),
    )

    stable = (
        tmp_path
        / "results"
        / "_stable_metrics"
        / "stateful_control_probe"
        / "training_fn_baseline_untrained_both_zero_v1_00000"
        / "resolved_config.json"
    )
    resolved = json.loads(stable.read_text(encoding="utf-8"))
    assert resolved["solver_state_mode"] == "both"
    assert resolved["solver_state_residual_init"] == "zero_v1"
    assert resolved["model_config"]["custom_model_config"]["solver_state_dim"] == 2


def test_checkpoint_replay_can_select_the_terminal_checkpoint(tmp_path):
    from itrader.rl_lib.evaluation.run_checkpoint_replay import (
        load_checkpoint_selection,
    )

    (tmp_path / "best_checkpoint.json").write_text(
        json.dumps({
            "metric": "evaluation/net_return_mean",
            "score": 0.01,
            "pbt_progress_updates": 3,
            "stable_checkpoint": "best",
        }),
        encoding="utf-8",
    )
    rows = [
        {
            "pbt_progress_updates": 3,
            "selection_metric": "evaluation/net_return_mean",
            "selection_score": 0.01,
            "stable_checkpoint": "best",
        },
        {
            "pbt_progress_updates": 12,
            "selection_metric": "evaluation/net_return_mean",
            "selection_score": -0.02,
            "stable_checkpoint": "terminal",
        },
    ]
    (tmp_path / "checkpoint_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    terminal = load_checkpoint_selection(tmp_path, "terminal")

    assert terminal["pbt_progress_updates"] == 12
    assert terminal["metric"] == "evaluation/net_return_mean"
    assert terminal["score"] == -0.02
    assert terminal["stable_checkpoint"] == "terminal"


def test_checkpoint_replay_can_select_one_exact_progress_checkpoint(tmp_path):
    from itrader.rl_lib.evaluation.run_checkpoint_replay import (
        load_checkpoint_selection,
    )

    rows = [
        {
            "pbt_progress_steps": 48_120,
            "pbt_progress_updates": 10,
            "selection_metric": "evaluation/economic_search_score",
            "selection_score": 0.998,
            "stable_checkpoint": "stage-zero",
        },
        {
            "pbt_progress_steps": 96_240,
            "pbt_progress_updates": 20,
            "selection_metric": "evaluation/economic_search_score",
            "selection_score": 1.001,
            "stable_checkpoint": "stage-quarter-bp",
        },
    ]
    (tmp_path / "checkpoint_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    selected = load_checkpoint_selection(
        tmp_path,
        "best",
        progress_steps=96_240,
    )

    assert selected["pbt_progress_steps"] == 96_240
    assert selected["metric"] == "evaluation/economic_search_score"
    assert selected["score"] == 1.001
    assert selected["stable_checkpoint"] == "stage-quarter-bp"
    assert selected["selection_scope"] == "indexed_progress_checkpoint"
