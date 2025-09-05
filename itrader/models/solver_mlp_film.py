from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class SolverMLPFiLM(TorchModelV2, nn.Module):
    """MLP policy/value with FiLM conditioning on task embedding.

    Observation layout: [price_window_flat, t_embed].
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        flat_dim = int(np.prod(obs_space.shape))
        t_dim = int(model_config.get("t_embed_dim", 32))
        hidden = int(model_config.get("fcnet_hiddens", [256, 256])[0])
        self.conditioning = model_config.get("conditioning", "film")  # default to FiLM
        self.t_keep_p = float(model_config.get("t_keep_p", 0.9))
        self.price_dim = flat_dim - t_dim
        assert self.price_dim > 0, "obs must be [price, t_embed]"

        self.trunk = nn.Sequential(
            nn.Linear(self.price_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        proj = int(model_config.get("t_proj", 64))
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, proj), nn.ReLU(),
            nn.Linear(proj, 2*hidden),  # gamma, beta
        )
        self.pi = nn.Linear(hidden, num_outputs)
        self.vf = nn.Linear(hidden, 1)
        self._value_out = None
        self.dropout = nn.Dropout(p=1.0 - self.t_keep_p)

    def forward(self, input_dict, state, seq_lens):
        x = torch.nan_to_num(input_dict["obs"].float())
        price = x[..., :self.price_dim]
        t = x[..., self.price_dim:]
        if self.training and self.t_keep_p < 1.0:
            t = self.dropout(t)
        h = self.trunk(price)
        if self.conditioning == "film":
            gb = self.t_mlp(t)
            gamma, beta = torch.chunk(gb, 2, dim=-1)
            gamma = torch.tanh(gamma)  # [-1, 1] clamp for stability
            h = gamma * h + beta
        logits = self.pi(h)
        self._value_out = self.vf(h).squeeze(-1)
        return logits, state

    def value_function(self):
        return self._value_out