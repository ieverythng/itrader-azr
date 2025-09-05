import torch
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.spaces.simplex import Simplex
from gymnasium.spaces import Box, Discrete

from itrader.models.solver_mlp_film import SolverMLPFiLM

# register if not already
ModelCatalog.register_custom_model("solver_mlp_film_v1", SolverMLPFiLM)

def test_forward_no_nan():
    obs_dim = 100
    t_dim = 32
    space = Box(low=-1e3, high=1e3, shape=(obs_dim+t_dim,), dtype=np.float32)
    model = SolverMLPFiLM(space, Discrete(3), 3, {"t_embed_dim": t_dim, "conditioning":"film"}, "m")
    x = torch.randn(5, obs_dim + t_dim)
    x[0,0] = float("nan"); x[1,1] = float("inf"); x[2,2] = float("-inf")
    logits, _ = model.forward({"obs": x}, [], None)
    assert torch.isfinite(logits).all()