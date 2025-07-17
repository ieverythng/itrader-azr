# itrader/models/solver_lstm.py
from __future__ import annotations
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns 
from ray.rllib.utils.annotations import override


class SolverLSTM(TorchRLModule, nn.Module):
    """Tiny actor–critic that works with RLlib’s new Learner stack."""

    def __init__(self, *, observation_space, action_space,
                 model_config: dict | None = None, **kwargs):
        nn.Module.__init__(self)
        TorchRLModule.__init__(self,
                               observation_space=observation_space,
                               action_space=action_space)

        # ---- window & feature dims ------------------------------------
        self.win = (model_config or {}).get("window_size", 30)
        flat_dim = int(np.prod(observation_space.shape))
        self.n_feat = flat_dim // self.win

        # ---- network ---------------------------------------------------
        self.fc   = nn.Linear(self.n_feat, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.pi   = nn.Linear(64, action_space.n)
        self.vf   = nn.Linear(64, 1)

    # -------------------------------------------------------------------
    def _forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Expects `obs` shape (B, win * F).  
        Outputs exactly the columns the PPO Learner pipeline needs **per env-step**:
        ```
        action_dist_inputs , actions , action_logp , vf_preds
        ```
        """
        obs = batch[Columns.OBS]                         # (B, win·F)
        B   = obs.size(0)
        
        x   = obs.view(B, self.win, self.n_feat)         # (B, T, F)
        x   = F.relu(self.fc(x))                         # (B, T, 128)

        seq, _ = self.lstm(x)                            # (B, T, 64)
        h_last = seq[:, -1]                              # last hidden state

        logits = self.pi(h_last)                         # (B, A)
        value  = self.vf(h_last).squeeze(-1)             # (B,)

        dist    = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        logp    = dist.log_prob(actions)

        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.ACTIONS           : actions,
            Columns.ACTION_LOGP       : logp,
            Columns.VF_PREDS          : value,
            Columns.ADVANTAGES  : torch.zeros_like(value),
            Columns.VALUE_TARGETS: torch.zeros_like(value),
        }

    # same logic for all three forward modes#
    @override(TorchRLModule)
    def forward_exploration(self, batch, **kw):
        return self._forward(batch)

    forward_inference = forward_exploration
    forward_train     = forward_exploration

    def compute_values(self, batch, **kw):
        obs = batch[Columns.OBS]
        B   = obs.size(0)
        x   = obs.view(B, self.win, self.n_feat)
        x   = F.relu(self.fc(x))
        seq, _ = self.lstm(x)
        return self.vf(seq[:, -1]).squeeze(-1)