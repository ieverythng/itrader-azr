"""
Minimal Solver-only LSTM (new RLModule API)
"""

from __future__ import annotations

import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np
import torch, torch.nn as nn
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override


class SolverLSTM(TorchRLModule, nn.Module):
    """tiny actor-critic compatible with RLlib’s new Learner stack"""

    def __init__(
        self,
        *,
        observation_space,
        action_space,
        model_config: dict | None = None,
        **kwargs,
    ):
        nn.Module.__init__(self)
        TorchRLModule.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
        )

        # --- window length & per-timestep feature size ------------------
        win = (model_config or {}).get("window_size", 30)
        flat = int(np.prod(observation_space.shape))
        n_feat = flat // win                       # features per row

        self._win   = win                          # save for forward()
        self._nfeat = n_feat

        # network
        self.fc   = nn.Linear(n_feat, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.pi   = nn.Linear(64, action_space.n)   # policy head (logits)
        self.vf   = nn.Linear(64, 1)                # value head   (scalar)

    # ------------------------------------------------------------------ #
    def _forward_core(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = obs.size(0)
        x = obs.view(B, self._win, self._nfeat)             # (B,T,F)
        x = F.relu(self.fc(x))                      # (B,T,128)
        seq, _ = self.lstm(x)                       # (B,T,64)
        logits = self.pi(seq)                       # (B,T,A)
        values = self.vf(seq).squeeze(-1)           # (B,T)
        return logits, values
    
    def _forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        logits, values = self._forward_core(batch[Columns.OBS])
        ## Sample actions & log-probs *inside* the module so the learner
        ## gets all columns it needs.
        logits_last  = logits[:, -1, :]
        dist    = torch.distributions.Categorical(logits=logits_last)
        actions = dist.sample()
        logp    = dist.log_prob(actions)

        return {
            Columns.ACTION_DIST_INPUTS : logits_last,
            Columns.ACTIONS            : actions,
            Columns.ACTION_LOGP        : logp,
            Columns.VF_PREDS           : values,
        }                

    @override(TorchRLModule)  # three modes delegate to _forward
    def forward_exploration(self, batch, **kw): return self._forward(batch)

    forward_inference = forward_exploration
    forward_train     = forward_exploration
