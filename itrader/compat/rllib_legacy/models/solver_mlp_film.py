from __future__ import annotations

import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from itrader.models.solver_mlp_film_core import (
    SolverMLPFiLMMixin,
    SolverMLPFiLMTensorCore,
)


class SolverMLPFiLM(SolverMLPFiLMTensorCore, TorchModelV2, nn.Module):
    """Legacy RLlib ModelV2 adapter for the shared PyTorch MLP-FiLM core.

    Observation layout: [price_window_flat, t_embed, solver_state].

    ``solver_state`` is an optional fixed two-value block. It feeds narrow
    actor/value residuals, leaving the market trunk and task FiLM path intact.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", model_config)
        self._setup_solver_mlp_film(obs_space, num_outputs, cfg)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        if self.semantic_observation_contract is not None:
            if not isinstance(state, (list, tuple)) or len(state) != 0:
                raise ValueError("stateless solver does not accept carried state")
            obs = input_dict["obs"]
            expected_width = self.semantic_observation_contract["agent_obs_dim"]
            if obs.ndim != 2 or obs.shape[-1] != expected_width:
                raise ValueError(
                    f"semantic observation batch shape must be [batch, {expected_width}]"
                )
        is_training = getattr(input_dict, "is_training", None)
        if is_training is None:
            is_training = input_dict.get("is_training", self.training)
        if isinstance(is_training, torch.Tensor):
            is_training = bool(is_training.reshape(-1)[0].item())
        logits, self._value_out, _ = self._solver_forward(
            input_dict["obs"], is_training=bool(is_training)
        )
        return logits, [] if self.semantic_observation_contract is not None else state

    def value_function(self):
        return self._value_out


__all__ = [
    "SolverMLPFiLM",
    "SolverMLPFiLMMixin",
    "SolverMLPFiLMTensorCore",
]
