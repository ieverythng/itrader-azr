"""Integration probes for the opt-in fixed observation/lifecycle contract.

The data producer is a temporary Parquet file with feature and row tags. Tests
consume public environment observations and model outputs; no market/task
assembly is mocked. The strict contract is separate from historical restores.
"""

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from gymnasium.spaces import Box, Discrete

from itrader.envs.itrader_env import ITraderEnv
from itrader.models.observation_contract import semantic_observation_contract
from itrader.models.solver_mlp_film_core import SolverMLPFiLMTensorCore
from itrader.rl_lib.evaluation.provenance import canonical_json_sha256


class _CoreSolverModel(torch.nn.Module, SolverMLPFiLMTensorCore):
    """Framework-neutral harness for observation-contract integration tests."""

    def __init__(self, observation_space, num_outputs, model_config):
        super().__init__()
        self._setup_solver_mlp_film(observation_space, num_outputs, model_config)
        self._value_out = None

    def get_initial_state(self):
        return []

    def forward(self, input_dict, state=None, seq_lens=None):
        del seq_lens
        state = [] if state is None else state
        if state:
            raise ValueError("stateless solver does not accept carried state")
        is_training = input_dict.get("is_training", self.training)
        logits, self._value_out, _ = self._solver_forward(
            input_dict["obs"], is_training=bool(is_training)
        )
        return logits, state

    def value_function(self):
        return self._value_out


@pytest.fixture
def tagged_market(tmp_path):
    # At index 60 the visible closes are 100..159; the next (unseen) close is 160.
    rows = np.arange(100, dtype=np.float64)
    frame = pd.DataFrame({
        "OPEN": 99 + rows,
        "HIGH": 102 + rows,
        "LOW": 97 + rows,
        "CLOSE": 100 + rows,
        "VOLUME": np.where(rows % 2 == 0, 1.0, 3.0),
    })
    path = tmp_path / "tagged_ohlcv.parquet"
    frame.to_parquet(path, index=False)
    return path


def _env_config(path, mode="off"):
    contract = semantic_observation_contract(mode)
    return {
        **copy.deepcopy(contract),
        "semantic_observation_contract": contract,
        "df_path": str(path),
        "start_idx_mode": "task_spec",
        "task_sampler": {
            "window_size": 60, "n_features": 5, "t_embed_dim": 32,
            "horizons": [4], "reward_ids": ["net_return"], "seed": 31,
            "fixed_tasks": [{
                "family_id": "00", "price_id": "SIM", "start_idx": 60,
                "horizon": 4, "reward_id": "net_return", "reward_params": {},
                "seed": 31, "obs_dim": 332, "action_dim": 3,
            }],
        },
    }


def _model(config, obs_width=None):
    width = config["agent_obs_dim"] if obs_width is None else obs_width
    return _CoreSolverModel(
        Box(-np.inf, np.inf, shape=(width,), dtype=np.float32),
        3,
        config,
    )


def test_strict_model_rejects_state_tail_masquerading_as_market(tagged_market):
    cfg = _env_config(tagged_market)
    with pytest.raises(ValueError, match="observation.*shape|observation.*width"):
        _model(cfg, obs_width=334)


def test_parquet_column_order_is_validated_at_actual_environment_boundary(tagged_market):
    frame = pd.read_parquet(tagged_market)
    frame[["CLOSE", "HIGH", "LOW", "OPEN", "VOLUME"]].to_parquet(
        tagged_market, index=False,
    )
    cfg = _env_config(tagged_market)
    cfg.pop("feature_cols")
    with pytest.raises(ValueError, match="actual feature_cols"):
        ITraderEnv(cfg)


def test_tagged_parquet_preprocessing_preserves_chronology_and_stateless_calls(tagged_market):
    cfg = _env_config(tagged_market)
    env = ITraderEnv(cfg)
    obs_a, info = env.reset(seed=5)
    assert info["task"]["start_idx"] == 60
    market = obs_a[:300].reshape(60, 5)
    # Independent source tags recover row 0..59 and OHLC offsets using the
    # last visible close, 159. Alternating volumes 1/3 become exactly -1/+1.
    np.testing.assert_allclose(
        (market[:, :4] + 1.0) * 159,
        np.arange(60)[:, None] + [99, 102, 97, 100], atol=2e-5,
    )
    np.testing.assert_allclose(market[:, 4], [-1, 1] * 30, atol=1e-6)
    assert obs_a.shape == (332,)
    obs_b, *_ = env.step(2)
    np.testing.assert_allclose(
        (obs_b[:300].reshape(60, 5)[:, :4] + 1.0) * 160,
        np.arange(1, 61)[:, None] + [99, 102, 97, 100], atol=2e-5,
    )
    model = _model(cfg).eval()
    assert model.get_initial_state() == []

    def evaluate(rows):
        logits, state = model.forward({"obs": torch.tensor(rows)}, [], None)
        assert state == []
        return logits.detach().clone(), model.value_function().detach().clone()

    a_logits, a_value = evaluate(obs_a[None])
    evaluate(obs_b[None])
    repeat_logits, repeat_value = evaluate(obs_a[None])
    torch.testing.assert_close(a_logits, repeat_logits, rtol=0, atol=0)
    torch.testing.assert_close(a_value, repeat_value, rtol=0, atol=0)
    ab_logits, ab_value = evaluate(np.stack([obs_a, obs_b]))
    ba_logits, ba_value = evaluate(np.stack([obs_b, obs_a]))
    torch.testing.assert_close(ab_logits, ba_logits.flip(0))
    torch.testing.assert_close(ab_value, ba_value.flip(0))
    torch.testing.assert_close(a_logits[0], ab_logits[0], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(a_value[0], ab_value[0], rtol=1e-5, atol=1e-7)
    with pytest.raises(ValueError, match="carried state"):
        model.forward({"obs": torch.tensor(obs_a[None])}, [torch.zeros(1)], None)


def test_preprocessed_state_pairs_preserve_initial_policy_and_zero_state_neutrality(tagged_market):
    configs = {mode: _env_config(tagged_market, mode) for mode in ("off", "masked", "both")}
    observations = {}
    models = {}
    for mode, cfg in configs.items():
        cfg["solver_state_residual_init"] = "zero_v1"
        env = ITraderEnv(cfg)
        env.reset(seed=5)
        observations[mode], *_ = env.step(2)
        torch.manual_seed(71)
        models[mode] = _model(cfg).eval()
    np.testing.assert_array_equal(observations["masked"][-2:], [0, 0])
    np.testing.assert_array_equal(observations["both"][-2:], [1, 0.75])
    for mode in ("masked", "both"):
        np.testing.assert_array_equal(observations[mode][:-2], observations["off"])
        for name, weight in models["off"].state_dict().items():
            torch.testing.assert_close(weight, models[mode].state_dict()[name], rtol=0, atol=0)

    outputs = {}
    for mode, model in models.items():
        logits, _ = model.forward({"obs": torch.tensor(observations[mode][None])}, [], None)
        outputs[mode] = (logits, model.value_function().clone())
    for mode in ("masked", "both"):
        for expected, actual in zip(outputs["off"], outputs[mode]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Zero state remains neutral after the heads learn nonzero residuals.
    with torch.no_grad():
        models["masked"].state_policy_residual.weight.fill_(0.5)
        models["masked"].state_value_residual.weight.fill_(0.5)
    masked_logits, _ = models["masked"].forward(
        {"obs": torch.tensor(observations["masked"][None])}, [], None,
    )
    torch.testing.assert_close(masked_logits, outputs["off"][0], rtol=0, atol=0)
    torch.testing.assert_close(models["masked"].value_function(), outputs["off"][1], rtol=0, atol=0)

    # At zero_v1 initialization only the residual output heads can learn first.
    (outputs["both"][0].sum() + outputs["both"][1].sum()).backward()
    state_model = models["both"]
    assert torch.count_nonzero(state_model.state_encoder[0].weight.grad) == 0
    assert torch.count_nonzero(state_model.state_policy_residual.weight.grad) > 0
    assert torch.count_nonzero(state_model.state_value_residual.weight.grad) > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in state_model.parameters())
    stale_cfg = {**configs["both"], "solver_state_residual_init": "random_v1"}
    with pytest.raises(ValueError, match="zero_v1"):
        _model(stale_cfg)


def test_terminal_trace_digest_is_bound_to_returned_actions_and_rewards(tagged_market):
    env = ITraderEnv({**_env_config(tagged_market), "capture_replay_trace": True})

    def replay(actions):
        env.reset(seed=5)
        trace = []
        for action in actions:
            _, reward, done, _, info = env.step(action)
            trace.append([action, reward])
        assert done
        digest = info["episode_evidence"]["action_reward_trace_sha256"]
        assert digest == canonical_json_sha256(trace)
        return digest

    original = replay([2, 1, 0, 1])
    assert original == replay([2, 1, 0, 1])
    assert original != replay([0, 1, 2, 1])
    legacy = ITraderEnv(_env_config(tagged_market))
    legacy.reset(seed=5)
    for action in [2, 1, 0, 1]:
        *_, info = legacy.step(action)
    assert "action_reward_trace_sha256" not in info["episode_evidence"]


@pytest.mark.parametrize("key,value", [
    ("feature_cols", ["CLOSE", "HIGH", "LOW", "OPEN", "VOLUME"]),
    ("window_size", 30),
    ("window_end", "idx"),
    ("window_order", "newest_to_oldest"),
    ("t_embed_dim", 30),
    ("task_encoder_version", "legacy_v1"),
    ("t_embed_mode", "off"),
    ("observation_transform", "raw"),
    ("observation_filter", "MeanStdFilter"),
    ("observation_schema_id", "r4_v1_min"),
    ("solver_state_dim", 2),
])
def test_conflicting_semantics_fail_at_env_and_model_boundaries(tagged_market, key, value):
    cfg = _env_config(tagged_market)
    cfg[key] = value
    with pytest.raises(ValueError, match=key):
        ITraderEnv(cfg)
    with pytest.raises(ValueError, match=key):
        _model(cfg)
    # Matching the effective config to a corrupted declaration cannot relabel it.
    cfg["semantic_observation_contract"][key] = value
    with pytest.raises(ValueError, match=key):
        ITraderEnv(cfg)
    with pytest.raises(ValueError, match=key):
        _model(cfg)


def test_actual_forward_width_cannot_override_declared_observation_contract(tagged_market):
    model = _model(_env_config(tagged_market)).eval()
    with pytest.raises(ValueError, match="observation batch shape"):
        model.forward({"obs": torch.zeros(2, 334)}, [], None)
