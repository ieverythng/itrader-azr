# itrader/models/solver_mlp_v1.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

class SolverMLPModel(TorchModelV2, nn.Module):
    """
    MLP policy/value model.
    Expects flat observation (price_window_flat + t_embed).
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        flat_dim = int(np.prod(obs_space.shape))
        hidden   = int(model_config.get("fcnet_hiddens", [256, 256])[0])

        self.trunk = nn.Sequential(
            nn.Linear(flat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, num_outputs)
        self.vf = nn.Linear(hidden, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        h = self.trunk(x)
        logits = self.pi(h)
        self._value_out = self.vf(h).squeeze(-1)
        return logits, state

    def value_function(self):
        assert self._value_out is not None
        return self._value_out

# Register
ModelCatalog.register_custom_model("solver_mlp", SolverMLPModel)
