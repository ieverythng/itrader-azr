from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class SolverMLPFiLM(TorchModelV2, nn.Module):
    """MLP policy/value with FiLM conditioning on task embedding.

    Observation layout: [price_window_flat, t_embed].
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", model_config)
        flat_dim = int(np.prod(obs_space.shape))
        t_dim = int(cfg.get("t_embed_dim", 32))
        fcnet_hiddens = cfg.get("fcnet_hiddens", [256, 256]) or [256, 256]
        hidden_sizes = [int(h) for h in fcnet_hiddens]
        if not hidden_sizes:
            hidden_sizes = [256, 256]
        self.conditioning = cfg.get("conditioning", "film")  # default to FiLM
        self.t_keep_p = float(cfg.get("t_keep_p", 0.9))
        self.price_dim = flat_dim - t_dim
        if self.price_dim <= 0:
            raise ValueError("obs must be laid out as [price_window_flat, t_embed]")

        trunk_layers = []
        in_dim = self.price_dim
        for h in hidden_sizes:
            trunk_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)
        hidden = hidden_sizes[-1]
        proj = int(cfg.get("t_proj", 64))
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, proj), nn.ReLU(),
            nn.Linear(proj, 2*hidden),  # gamma, beta
        )
        self.pi = nn.Linear(hidden, num_outputs)
        self.vf = nn.Linear(hidden, 1)
        self._value_out = None
        self.dropout = nn.Dropout(p=1.0 - self.t_keep_p)

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        price = obs[..., :self.price_dim]
        task_embed = obs[..., self.price_dim:]
        return price, task_embed

    def _condition_hidden(self, hidden: torch.Tensor, task_embed: torch.Tensor) -> torch.Tensor:
        if self.conditioning != "film":
            return hidden
        gamma_beta = self.t_mlp(task_embed)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = torch.tanh(gamma)  # [-1, 1] clamp for stability
        return gamma * hidden + beta

    def forward(self, input_dict, state, seq_lens):
        x = torch.nan_to_num(input_dict["obs"].float())
        price, task_embed = self._split_obs(x)
        if self.training and self.t_keep_p < 1.0:
            task_embed = self.dropout(task_embed)
        hidden = self.trunk(price)
        hidden = self._condition_hidden(hidden, task_embed)
        logits = self.pi(hidden)
        self._value_out = self.vf(hidden).squeeze(-1)
        return logits, state

    def value_function(self):
        return self._value_out
