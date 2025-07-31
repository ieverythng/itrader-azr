# itrader/models/solver_lstm.py
from __future__ import annotations
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.policy.sample_batch import SampleBatch 
from ray.rllib.models.torch.torch_distributions import TorchCategorical


class SolverLSTM(TorchRLModule, nn.Module):
    """Tiny actor–critic that works with RLlib’s new Learner stack."""

    def __init__(self, module_config, **kwargs):
        nn.Module.__init__(self)
        TorchRLModule.__init__(self, config=module_config)

        # pull out what we need from the config
        obs_space = module_config.observation_space
        act_space = module_config.action_space
        mc = getattr(module_config, "model_config_dict", {}) or {}

        self.win = mc.get("window_size", 30)
        flat_dim = int(np.prod(obs_space.shape))
        self.n_feat = flat_dim // self.win

        # ---- network ---------------------------------------------------
        self.fc   = nn.Linear(self.n_feat, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.pi   = nn.Linear(64, act_space.n)
        self.vf   = nn.Linear(64, 1)

    def _forward_exploration(
        self, batch: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, torch.Tensor]:
        obs = batch[SampleBatch.OBS]               # (B, win·F)
        B   = obs.size(0)
        x   = obs.view(B, self.win, self.n_feat)  # → (B, T, F)
        x   = F.relu(self.fc(x))                  # → (B, T, 128)
        seq, _ = self.lstm(x)                     # → (B, T, 64)
        h    = seq[:, -1]                         # (B, 64)

        logits = self.pi(h)                       # (B, A)
        values = self.vf(h).squeeze(-1)           # (B,)

        dist = TorchCategorical(logits=logits)
        actions = dist.sample()
        logp = dist.logp(actions)

        return {
            SampleBatch.ACTION_DIST_INPUTS: logits,            
            SampleBatch.ACTION_DIST       : dist,
            SampleBatch.ACTIONS           : actions,
            SampleBatch.ACTION_LOGP       : logp,
            SampleBatch.VF_PREDS          : values,
        }

    def forward_exploration(self, batch, **kwargs):
        return self._forward_exploration(batch, **kwargs)

    # for inference / evaluation, we use the same
    forward_inference = forward_exploration
    forward_train = forward_exploration

    # ------------------------------------------------------------- #
    # Value-only pass, used by the PPO learner when computing loss #
    # ------------------------------------------------------------- #
    def compute_values(
        self, batch: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        return self._forward_exploration(batch)[SampleBatch.VF_PREDS]
    
    # And RLModule needs to know your Distribution class:
    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    # same dist for train & inference
    get_train_action_dist_cls     = get_exploration_action_dist_cls
    get_inference_action_dist_cls = get_exploration_action_dist_cls
