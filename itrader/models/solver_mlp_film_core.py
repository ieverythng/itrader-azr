"""RLlib-neutral PyTorch tensor core for the R0/R4 MLP-FiLM solver.

The core is a mixin for an owning :class:`torch.nn.Module`. It installs its
layers directly on that owner so historical parameter names, ordering, and
checkpoint state-dict keys remain unchanged. This module deliberately has no
Ray/RLlib imports.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from itrader.models.observation_contract import validate_semantic_config


class SolverMLPFiLMTensorCore:
    """Shared PyTorch-only model construction and forward math.

    Call ``_setup_solver_mlp_film`` after initializing the owning ``nn.Module``.
    The host must expose the resulting layers as its own attributes; this keeps
    the legacy checkpoint parameter paths byte-for-byte compatible.
    """

    def _setup_solver_mlp_film(self, obs_space, num_outputs, cfg):
        self.semantic_observation_contract = None
        if "semantic_observation_contract" in cfg:
            self.semantic_observation_contract = validate_semantic_config(
                cfg,
                cfg["semantic_observation_contract"],
                boundary="model",
                actual_obs_shape=obs_space.shape,
            )
            if self.semantic_observation_contract["solver_state_dim"]:
                if cfg.get("solver_state_residual_init") != "zero_v1":
                    raise ValueError(
                        "semantic solver state requires zero_v1 initialization"
                    )
        flat_dim = int(np.prod(obs_space.shape))
        t_dim = int(cfg.get("t_embed_dim", 32))
        solver_state_dim = int(cfg.get("solver_state_dim", 0))
        if solver_state_dim not in {0, 2}:
            raise ValueError("solver_state_dim must be 0 or 2")
        fcnet_hiddens = cfg.get("fcnet_hiddens", [256, 256]) or [256, 256]
        hidden_sizes = [int(h) for h in fcnet_hiddens]
        if not hidden_sizes:
            hidden_sizes = [256, 256]
        self.conditioning = cfg.get("conditioning", "film")
        self.t_keep_p = float(cfg.get("t_keep_p", 0.9))
        self.task_dim = t_dim
        self.solver_state_dim = solver_state_dim
        self.price_dim = flat_dim - t_dim - solver_state_dim
        if self.price_dim <= 0:
            raise ValueError(
                "obs must be laid out as [price_window_flat, t_embed, solver_state]"
            )

        trunk_layers = []
        in_dim = self.price_dim
        for hidden_size in hidden_sizes:
            trunk_layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU()])
            in_dim = hidden_size
        self.trunk = nn.Sequential(*trunk_layers)
        hidden = hidden_sizes[-1]
        proj = int(cfg.get("t_proj", 64))
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, proj),
            nn.ReLU(),
            nn.Linear(proj, 2 * hidden),
        )
        self.pi = nn.Linear(hidden, num_outputs)
        self.vf = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(p=1.0 - self.t_keep_p)

        self.state_encoder = None
        self.state_policy_residual = None
        self.state_value_residual = None
        if self.solver_state_dim:
            residual_init = str(
                cfg.get("solver_state_residual_init", "random_v1")
            ).strip().lower()
            if residual_init not in {"random_v1", "zero_v1"}:
                raise ValueError(
                    "solver_state_residual_init must be 'random_v1' or 'zero_v1'"
                )
            state_width = int(cfg.get("solver_state_width", 8))
            if state_width <= 0:
                raise ValueError("solver_state_width must be positive")
            self.state_encoder = nn.Sequential(
                nn.Linear(self.solver_state_dim, state_width, bias=False),
                nn.ReLU(),
            )
            self.state_policy_residual = nn.Linear(
                state_width, num_outputs, bias=False
            )
            self.state_value_residual = nn.Linear(state_width, 1, bias=False)
            if residual_init == "zero_v1":
                nn.init.zeros_(self.state_policy_residual.weight)
                nn.init.zeros_(self.state_value_residual.weight)

    def _split_obs(self, obs: torch.Tensor):
        price = obs[..., : self.price_dim]
        task_end = self.price_dim + self.task_dim
        task_embed = obs[..., self.price_dim : task_end]
        solver_state = obs[..., task_end:]
        return price, task_embed, solver_state

    def _condition_hidden(self, hidden: torch.Tensor, task_embed: torch.Tensor):
        if self.conditioning != "film":
            return hidden
        gamma_beta = self.t_mlp(task_embed)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = torch.tanh(gamma)
        return gamma * hidden + beta

    def value_from_embeddings(self, embeddings):
        """Apply the shared value head to trunk and optional state embeddings."""
        hidden, state_z = embeddings
        value = self.vf(hidden)
        if self.state_value_residual is not None and state_z is not None:
            value = value + self.state_value_residual(state_z)
        return value.squeeze(-1)

    def _solver_forward(self, obs: torch.Tensor, *, is_training: bool):
        if self.semantic_observation_contract is not None:
            expected_width = self.semantic_observation_contract["agent_obs_dim"]
            if obs.ndim != 2 or obs.shape[-1] != expected_width:
                raise ValueError(
                    f"semantic observation batch shape must be [batch, {expected_width}]"
                )
        x = torch.nan_to_num(obs.float())
        price, task_embed, solver_state = self._split_obs(x)
        if is_training and self.t_keep_p < 1.0:
            task_embed = self.dropout(task_embed)
        hidden = self._condition_hidden(self.trunk(price), task_embed)
        logits = self.pi(hidden)
        state_z = None
        if self.solver_state_dim:
            state_z = self.state_encoder(solver_state)
            logits = logits + self.state_policy_residual(state_z)
        embeddings = (hidden, state_z)
        return logits, self.value_from_embeddings(embeddings), embeddings


# Retain the old internal symbol for callers that imported the mixin directly.
SolverMLPFiLMMixin = SolverMLPFiLMTensorCore


__all__ = ["SolverMLPFiLMTensorCore", "SolverMLPFiLMMixin"]
