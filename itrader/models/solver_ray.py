import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class SolverRay(TorchModelV2, nn.Module):
    """
    Vanilla 2-layer MLP that can be used instead of SolverLSTM:

    $ python -m itrader.rllib.train_solo --iters 200 --model solver_mlp
    """
    def __init__(self, obs_space, action_space, num_outputs,
                 model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs, model_config, name)
        nn.Module.__init__(self)

        in_dim = int(np.prod(obs_space.shape))
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.policy_head = nn.Linear(256, num_outputs)
        self.value_head  = nn.Linear(256, 1)
        self._value_out  = torch.tensor(0.0)

    def forward(self, input_dict, state, seq_lens):
        feat = self.backbone(input_dict["obs"].float().view(input_dict["obs"].size(0), -1))
        logits = self.policy_head(feat)
        self._value_out = self.value_head(feat).squeeze(1)
        return logits, state

    def value_function(self):
        return self._value_out
