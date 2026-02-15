import numpy as np
import torch
import pytest
from gymnasium.spaces import Box, Discrete

pytest.importorskip("pydantic")
pytest.importorskip("ray")


def _make_model(obs_dim: int, t_dim: int):
    from itrader.models.solver_mlp_film import SolverMLPFiLM

    obs_space = Box(low=-1e3, high=1e3, shape=(obs_dim + t_dim,), dtype=np.float32)
    model_config = {
        "custom_model_config": {
            "t_embed_dim": t_dim,
            "conditioning": "film",
            "fcnet_hiddens": [64, 64],
        }
    }
    return SolverMLPFiLM(obs_space, Discrete(3), 3, model_config, "solver_mlp_film_test")


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
