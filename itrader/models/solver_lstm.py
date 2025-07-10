"""
Minimal Solver-only LSTM (new RLModule API)
───────────────────────────────────────────
fc-128 ➜ LSTM-64 ➜ π / V heads
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from ray.rllib.core.rl_module.torch import TorchRLModule          # new stack ✔
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.utils.annotations import override


class SolverLSTM(TorchRLModule, nn.Module):
    """Tiny actor-critic that works with RLlib’s Learner API."""

    def __init__(self, config: RLModuleConfig):
        nn.Module.__init__(self)
        TorchRLModule.__init__(self, config)

        obs_space, action_space = config.observation_space, config.action_space
        self._T = config.model_config_dict.get("window_size", 30)

        in_dim = int(np.prod(obs_space.shape))          # flattened window
        self.fc   = nn.Linear(in_dim, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)

        self.pi_head = nn.Linear(64, action_space.n)   # logits
        self.v_head  = nn.Linear(64, 1)                # value

    # ── internal helper ─────────────────────────────────────────────── #
    def _extract_feat(self, obs: torch.Tensor) -> torch.Tensor:
        # obs shape: [B, in_dim]
        x   = torch.relu(self.fc(obs))
        x   = x.view(-1, self._T, 128)          # [B, T, 128]
        seq, _ = self.lstm(x)
        return seq[:, -1, :]                    # last timestep
    
    # ───────────────── public API (3 modes) ────────────── #
    def _forward_all(self, batch):
        feat = self._extract_feat(batch["obs"])
        return {
            "logits": self.pi_head(feat),
            "value":  self.v_head(feat).squeeze(-1),
        }, []                                   # no RNN state

    # ── public API (three modes share the same logic) ──────────────── #
    @override(TorchRLModule)
    def forward_exploration(self, batch, state=None, seq_lens=None):
        return self._forward_all(batch)

    @override(TorchRLModule)
    def forward_inference(self, batch, state=None, seq_lens=None):
        return self._forward_all(batch)

    @override(TorchRLModule)
    def forward_train(self, batch, state=None, seq_lens=None):
        return self._forward_all(batch)
