import numpy as np
import torch
import pytest
from gymnasium.spaces import Box, Discrete

pytest.importorskip("pydantic")


from itrader.models.solver_mlp_film_core import SolverMLPFiLMTensorCore


class _CoreSolverModel(torch.nn.Module, SolverMLPFiLMTensorCore):
    """Tiny harness that exercises the framework-neutral solver math."""

    def __init__(self, obs_space, num_outputs, model_config):
        super().__init__()
        self._setup_solver_mlp_film(obs_space, num_outputs, model_config)
        self._value_out = None

    def forward(self, input_dict, state=None, seq_lens=None):
        del seq_lens
        is_training = input_dict.get("is_training", self.training)
        if isinstance(is_training, torch.Tensor):
            is_training = bool(is_training.reshape(-1)[0].item())
        logits, self._value_out, _ = self._solver_forward(
            input_dict["obs"], is_training=bool(is_training)
        )
        return logits, [] if state is None else state

    def value_function(self):
        return self._value_out


def _make_model(
    obs_dim: int,
    t_dim: int,
    *,
    solver_state_dim: int = 0,
    solver_state_residual_init: str = "random_v1",
    conditioning: str = "film",
):
    obs_space = Box(
        low=-1e3,
        high=1e3,
        shape=(obs_dim + t_dim + solver_state_dim,),
        dtype=np.float32,
    )
    model_config = {
        "custom_model_config": {
            "t_embed_dim": t_dim,
            "solver_state_dim": solver_state_dim,
            "solver_state_residual_init": solver_state_residual_init,
            "conditioning": conditioning,
            "fcnet_hiddens": [64, 64],
        }
    }
    return _CoreSolverModel(
        obs_space,
        3,
        model_config["custom_model_config"],
    )


def test_forward_no_nan_inf_in_logits():
    obs_dim = 100
    t_dim = 32
    model = _make_model(obs_dim=obs_dim, t_dim=t_dim)

    x = torch.randn(8, obs_dim + t_dim)
    x[0, 0] = float("nan")
    x[1, 1] = float("inf")
    x[2, 2] = float("-inf")

    logits, _ = model.forward({"obs": x}, [], None)
    assert logits.shape == (8, 3)
    assert torch.isfinite(logits).all()


def test_value_function_shape_and_finiteness():
    obs_dim = 120
    t_dim = 32
    model = _make_model(obs_dim=obs_dim, t_dim=t_dim)

    x = torch.randn(4, obs_dim + t_dim)
    model.forward({"obs": x}, [], None)
    values = model.value_function()
    assert values.shape == (4,)
    assert torch.isfinite(values).all()


def test_task_dropout_uses_rllib_inference_flag_not_module_mode():
    model = _make_model(obs_dim=12, t_dim=4)
    model.train()
    model.t_keep_p = 0.0
    model.dropout = torch.nn.Dropout(p=1.0)
    observation = torch.randn(3, 16)

    inference_logits, _ = model.forward(
        {"obs": observation, "is_training": False},
        [],
        None,
    )
    training_logits, _ = model.forward(
        {"obs": observation, "is_training": True},
        [],
        None,
    )

    assert not torch.equal(inference_logits, training_logits)


def test_solver_state_reaches_actor_and_value_without_task_film():
    torch.manual_seed(7)
    market_dim = 12
    task_dim = 4
    model = _make_model(
        obs_dim=market_dim,
        t_dim=task_dim,
        solver_state_dim=2,
        conditioning="off",
    )
    market = torch.full((1, market_dim), 0.5)
    task = torch.full((1, task_dim), 0.25)
    reset_state = torch.tensor([[0.0, 1.0]])
    carried_state = torch.tensor([[1.0, 0.5]])

    reset_logits, _ = model.forward(
        {"obs": torch.cat([market, task, reset_state], dim=-1)},
        [],
        None,
    )
    reset_value = model.value_function().clone()
    carried_logits, _ = model.forward(
        {"obs": torch.cat([market, task, carried_state], dim=-1)},
        [],
        None,
    )
    carried_value = model.value_function().clone()

    assert not torch.equal(reset_logits, carried_logits)
    assert not torch.equal(reset_value, carried_value)


def test_disabled_solver_state_preserves_legacy_model_shapes_and_outputs():
    obs_space = Box(low=-1e3, high=1e3, shape=(16,), dtype=np.float32)
    legacy_config = {
        "custom_model_config": {
            "t_embed_dim": 4,
            "conditioning": "film",
            "fcnet_hiddens": [64, 64],
        }
    }
    explicit_disabled_config = {
        "custom_model_config": {
            **legacy_config["custom_model_config"],
            "solver_state_dim": 0,
        }
    }
    torch.manual_seed(19)
    legacy = _CoreSolverModel(obs_space, 3, legacy_config["custom_model_config"])
    torch.manual_seed(19)
    explicit_disabled = _CoreSolverModel(
        obs_space, 3, explicit_disabled_config["custom_model_config"]
    )
    observation = torch.randn(2, 16)

    legacy_logits, _ = legacy.forward(
        {"obs": observation, "is_training": False}, [], None
    )
    explicit_logits, _ = explicit_disabled.forward(
        {"obs": observation, "is_training": False}, [], None
    )

    assert {
        key: tuple(value.shape) for key, value in legacy.state_dict().items()
    } == {
        key: tuple(value.shape)
        for key, value in explicit_disabled.state_dict().items()
    }
    torch.testing.assert_close(legacy_logits, explicit_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        legacy.value_function(),
        explicit_disabled.value_function(),
        rtol=0.0,
        atol=0.0,
    )


def test_masked_state_residual_exactly_matches_legacy_base_path():
    legacy_obs_space = Box(low=-1e3, high=1e3, shape=(16,), dtype=np.float32)
    state_obs_space = Box(low=-1e3, high=1e3, shape=(18,), dtype=np.float32)
    base_config = {
        "custom_model_config": {
            "t_embed_dim": 4,
            "conditioning": "film",
            "fcnet_hiddens": [64, 64],
            "t_keep_p": 0.7,
        }
    }
    state_config = {
        "custom_model_config": {
            **base_config["custom_model_config"],
            "solver_state_dim": 2,
            "solver_state_width": 8,
        }
    }
    torch.manual_seed(31)
    legacy = _CoreSolverModel(
        legacy_obs_space, 3, base_config["custom_model_config"]
    )
    torch.manual_seed(31)
    masked = _CoreSolverModel(
        state_obs_space, 3, state_config["custom_model_config"]
    )
    base_observation = torch.randn(5, 16)
    masked_observation = torch.cat(
        [base_observation, torch.zeros(5, 2)],
        dim=-1,
    )

    legacy_logits, _ = legacy.forward(
        {"obs": base_observation, "is_training": False}, [], None
    )
    legacy_value = legacy.value_function().clone()
    masked_logits, _ = masked.forward(
        {"obs": masked_observation, "is_training": False}, [], None
    )
    masked_value = masked.value_function().clone()

    common_state = legacy.state_dict()
    for key, value in common_state.items():
        torch.testing.assert_close(value, masked.state_dict()[key], rtol=0.0, atol=0.0)
    torch.testing.assert_close(legacy_logits, masked_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(legacy_value, masked_value, rtol=0.0, atol=0.0)


def test_solver_state_residual_can_represent_symmetric_position_cost():
    market_dim = 12
    task_dim = 4
    model = _make_model(
        obs_dim=market_dim,
        t_dim=task_dim,
        solver_state_dim=2,
        conditioning="off",
    )
    with torch.no_grad():
        model.state_encoder[0].weight.zero_()
        model.state_encoder[0].weight[0, 0] = 1.0
        model.state_encoder[0].weight[1, 0] = -1.0
        model.state_policy_residual.weight.zero_()
        model.state_policy_residual.weight[0, :2] = 1.0
        model.state_value_residual.weight.zero_()
        model.state_value_residual.weight[0, :2] = 1.0

    base = torch.zeros(3, market_dim + task_dim)
    observations = torch.cat(
        [
            base,
            torch.tensor(
                [
                    [-1.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                ]
            ),
        ],
        dim=-1,
    )
    logits, _ = model.forward(
        {"obs": observations, "is_training": False}, [], None
    )
    values = model.value_function()

    torch.testing.assert_close(logits[0, 0], logits[2, 0])
    torch.testing.assert_close(values[0], values[2])
    torch.testing.assert_close(logits[0, 0] - logits[1, 0], torch.tensor(1.0))
    torch.testing.assert_close(values[0] - values[1], torch.tensor(1.0))


def test_zero_initialized_solver_state_residual_preserves_update_zero_policy_and_value():
    market_dim = 12
    task_dim = 4
    legacy_obs_space = Box(
        low=-np.inf,
        high=np.inf,
        shape=(market_dim + task_dim,),
        dtype=np.float32,
    )
    state_obs_space = Box(
        low=-np.inf,
        high=np.inf,
        shape=(market_dim + task_dim + 2,),
        dtype=np.float32,
    )
    base_config = {
        "custom_model_config": {
            "t_embed_dim": task_dim,
            "solver_state_dim": 0,
            "fcnet_hiddens": [8, 8],
            "conditioning": "film",
            "t_keep_p": 1.0,
            "t_proj": 4,
        }
    }
    state_config = {
        "custom_model_config": {
            **base_config["custom_model_config"],
            "solver_state_dim": 2,
            "solver_state_width": 8,
            "solver_state_residual_init": "zero_v1",
        }
    }
    torch.manual_seed(91)
    legacy = _CoreSolverModel(
        legacy_obs_space, 3, base_config["custom_model_config"]
    )
    torch.manual_seed(91)
    stateful = _CoreSolverModel(
        state_obs_space, 3, state_config["custom_model_config"]
    )
    base_observation = torch.randn(5, market_dim + task_dim)
    nonzero_state = torch.tensor(
        [[-1.0, 1.0], [0.0, 0.75], [1.0, 0.5], [-1.0, 0.25], [1.0, 0.01]]
    )

    legacy_logits, _ = legacy.forward(
        {"obs": base_observation, "is_training": False}, [], None
    )
    legacy_value = legacy.value_function().clone()
    stateful_logits, _ = stateful.forward(
        {
            "obs": torch.cat([base_observation, nonzero_state], dim=-1),
            "is_training": False,
        },
        [],
        None,
    )
    stateful_value = stateful.value_function().clone()

    torch.testing.assert_close(legacy_logits, stateful_logits, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(legacy_value, stateful_value, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(
        torch.argmax(legacy_logits, dim=-1),
        torch.argmax(stateful_logits, dim=-1),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(stateful.state_policy_residual.weight) == 0
    assert torch.count_nonzero(stateful.state_value_residual.weight) == 0


def test_solver_state_residual_initialization_rejects_unknown_profile():
    with pytest.raises(ValueError, match="solver_state_residual_init"):
        _make_model(
            obs_dim=18,
            t_dim=4,
            solver_state_dim=2,
            solver_state_residual_init="mystery",
        )
