from itrader.rl_lib.tasks.candidate_task import CandidateTask
from itrader.rl_lib.tasks.capability_profile import CapabilityProfile
from itrader.rl_lib.tasks.decision_record import DecisionRecord
import numpy as np

from itrader.rl_lib.tasks.encoder import TaskEncoder
from itrader.rl_lib.tasks.execution_intent import ExecutionIntent
from itrader.rl_lib.tasks.families import get_family_spec, known_family_ids
from itrader.rl_lib.tasks.samplers import TaskSampler00
from itrader.rl_lib.tasks.spec import TaskSpec
from itrader.rl_lib.tasks.validation import (
    validate_and_clamp_execution_intent,
    validate_candidate_task,
    validate_capability_profile,
    validate_decision_record,
)


def test_family_registry_exposes_bootstrap_families():
    assert known_family_ids() == ("00", "01", "02")
    assert get_family_spec("01").action_dim == 3
    assert get_family_spec("02").default_reward_id == "pnl_linear"


def test_task_spec_from_dict_supports_aliases_and_serializes_canonical_fields():
    spec = TaskSpec.from_dict(
        {
            "task_family": "02",
            "price_id": "BTCUSDT",
            "horizon_bars": 96,
            "reward_config": {"name": "pnl_sharpe", "params": {"pnl_w": 1.0}},
            "seed": 7,
            "obs_dim": 182,
            "action_dim": 3,
        }
    )
    payload = spec.to_dict()
    assert spec.family_id == "02"
    assert spec.task_family == "02"
    assert spec.horizon_bars == 96
    assert spec.reward_config == {"name": "pnl_sharpe", "params": {"pnl_w": 1.0}}
    assert payload["family_id"] == "02"
    assert "task_family" not in payload
    assert payload["reward_id"] == "pnl_sharpe"


def test_task_spec_from_dict_drops_duplicate_alias_fields():
    spec = TaskSpec.from_dict(
        {
            "task_family": "01",
            "family_id": "00",
            "price_id": "SIM",
            "horizon_bars": 24,
            "horizon": 96,
            "reward_id": "pnl_linear",
            "reward_params": {},
            "seed": 1,
            "obs_dim": 182,
            "action_dim": 3,
        }
    )
    assert spec.family_id == "00"
    assert spec.horizon == 96


def test_semantic_task_encoding_excludes_sampling_identity_but_includes_reward_contract():
    base = {
        "family_id": "00",
        "price_id": "SIM",
        "horizon": 96,
        "reward_id": "pnl_linear",
        "reward_params": {"pnl_w": 1.0, "dd_w": 0.02},
        "seed": 7,
        "start_idx": 100,
        "obs_dim": 182,
        "action_dim": 3,
        "bar_resolution": "1m",
        "regime_tags": ["trend"],
    }
    first = TaskSpec.from_dict(base)
    different_sample = TaskSpec.from_dict({**base, "seed": 999, "start_idx": 400})
    different_reward = TaskSpec.from_dict(
        {**base, "reward_params": {"pnl_w": 1.0, "dd_w": 0.04}}
    )

    first_vec = TaskEncoder.embed(first, version="semantic_v2")
    sample_vec = TaskEncoder.embed(different_sample, version="semantic_v2")
    reward_vec = TaskEncoder.embed(different_reward, version="semantic_v2")

    assert np.array_equal(first_vec, sample_vec)
    assert not np.array_equal(first_vec, reward_vec)


def test_fixed_task_panel_cycles_exact_tasks_and_survives_external_reseed():
    fixed_tasks = [
        {
            "family_id": "00",
            "price_id": "SIM",
            "horizon": horizon,
            "reward_id": "pnl_linear",
            "reward_params": {"pnl_w": 1.0, "turnover_w": 0.001},
            "seed": seed,
            "obs_dim": 182,
            "action_dim": 3,
            "bar_resolution": "1m",
        }
        for horizon, seed in ((64, 101), (96, 202))
    ]
    sampler = TaskSampler00(
        {
            "seed": 7,
            "window_size": 30,
            "n_features": 5,
            "t_embed_dim": 32,
            "action_dim": 3,
            "fixed_tasks": fixed_tasks,
        }
    )

    expected = [TaskSpec.from_dict(task).to_dict() for task in fixed_tasks]
    first_cycle = [sampler.sample().to_dict() for _ in range(2)]
    assert first_cycle == expected
    assert [sampler.sample().to_dict() for _ in range(2)] == expected

    sampler.reseed(999_999)
    assert [sampler.sample().to_dict() for _ in range(2)] == expected


def test_execution_intent_validation_clamps_bounds():
    intent = validate_and_clamp_execution_intent(
        {
            "entry_logic": {"type": "breakout_confirm", "params": {"side": "long"}},
            "exit_logic": {"type": "ttl_or_stop_take", "params": {"ttl_bars": 10}},
            "ttl_bars": 0,
            "confidence_band": {"min": 0.7, "target": 0.4},
            "risk_envelope": {"max_position_pct": 2.0, "max_spread_bps": -1, "max_slippage_bps": 25},
            "abstain_allowed": "true",
        }
    )
    typed = ExecutionIntent.from_dict(intent)
    assert typed.ttl_bars == 1
    assert typed.confidence_band == {"min": 0.7, "target": 0.7}
    assert typed.risk_envelope["max_position_pct"] == 1.0
    assert typed.risk_envelope["max_spread_bps"] == 0.0
    assert typed.risk_envelope["max_slippage_bps"] == 25.0
    assert typed.abstain_allowed is True


def test_candidate_task_validation_normalizes_nested_contracts():
    candidate = validate_candidate_task(
        {
            "task_spec": {
                "task_family": "01",
                "price_id": "AAPL",
                "horizon_bars": 24,
                "reward_config": {"name": "pnl_linear", "params": {"pnl_w": 1.0}},
            },
            "execution_intent": {
                "entry_logic": {"type": "breakout_confirm", "params": {"side": "long"}},
                "exit_logic": {"type": "ttl", "params": {}},
                "ttl_bars": 24,
                "confidence_band": {"min": 0.62, "target": 0.74},
                "risk_envelope": {"max_position_pct": 0.05, "max_spread_bps": 8},
            },
            "expected_difficulty": 1.2,
            "expected_utility": "0.21",
            "rationale": "Compression before likely breakout.",
        },
        {"window_size": 30, "n_features": 5, "t_embed_dim": 32, "action_dim": 3, "min_horizon": 8},
        fallback_seed=5,
    )
    typed = CandidateTask.from_dict(candidate)
    assert typed.task_spec.family_id == "01"
    assert typed.execution_intent.entry_logic["params"]["side"] == "long"
    assert typed.expected_difficulty == 1.0
    assert typed.expected_utility == 0.21


def test_capability_profile_validation_accepts_family_and_horizon_aliases():
    profile = validate_capability_profile(
        {
            "family_id": "01",
            "bar_resolution": "5m",
            "regime_tags": ["trend", "high_vol"],
            "horizon": 24,
            "estimated_p_solve": 1.5,
            "expected_net_utility": "0.08",
            "calibration_error": -0.5,
            "sample_count": -3,
            "last_updated": "2026-03-22T00:00:00Z",
        }
    )
    typed = CapabilityProfile.from_dict(profile)
    assert typed.task_family == "01"
    assert typed.horizon_bars == 24
    assert typed.estimated_p_solve == 1.0
    assert typed.calibration_error == 0.0
    assert typed.sample_count == 0


def test_decision_record_validation_round_trips_nested_candidates():
    record = validate_decision_record(
        {
            "market_summary_ref": "summary-123",
            "candidate_tasks": [
                {
                    "task_spec": {
                        "family_id": "00",
                        "price_id": "SIM",
                        "horizon": 96,
                        "reward_id": "pnl_linear",
                        "reward_params": {"pnl_w": 1.0},
                    },
                    "abstain_option": True,
                }
            ],
            "selected_candidate_id": None,
            "wait_selected": True,
            "solver_scores": {"candidate_0": {"confidence": 0.3}},
            "risk_decisions": {"candidate_0": "blocked"},
            "executor_action": {"mode": "shadow", "action": "wait"},
            "realized_outcome": {"pnl": 0.0},
        },
        {"window_size": 30, "n_features": 5, "t_embed_dim": 32, "action_dim": 3},
        fallback_seed=3,
    )
    typed = DecisionRecord.from_dict(record)
    assert typed.wait_selected is True
    assert typed.candidate_tasks[0].task_spec.family_id == "00"
    assert typed.executor_action["action"] == "wait"
