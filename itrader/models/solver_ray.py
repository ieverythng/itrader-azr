# -----------------------------------------------------------
#  SolverRay – simple 2-layer MLP baseline (no recurrence)
# -----------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class SolverRay(TorchModelV2, nn.Module):
    """
    Minimal fully-connected network for ablations vs. LSTM.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        flat_obs = int(np.prod(obs_space.shape))

        self.network = nn.Sequential(
            nn.Linear(flat_obs, 256), nn.ReLU(),
            nn.Linear(256, 256),      nn.ReLU(),
        )

        self.policy_head = nn.Linear(256, action_space.n)
        self.value_head  = nn.Linear(256, 1)

        self.to(self.device)
        self._value_out = None

    # ------------------------------------------------------------------ #
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float().to(self.device)
        feats = self.network(x)
        logits = self.policy_head(feats)
        self._value_out = self.value_head(feats).squeeze(1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out
