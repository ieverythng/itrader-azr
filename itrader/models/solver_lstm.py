# itrader/models/solver_lstm_v2.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

class SolverLSTMModel(TorchModelV2, nn.Module):
    """
    LSTM SOLVER 
    """
        
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # pull window size out of your custom config
        mc = model_config.get("custom_model_config", {})
        self.win = mc.get("window_size", 30)
        flat_dim = int(np.prod(obs_space.shape))
        self.n_feat = flat_dim // self.win

        self.fc   = nn.Linear(self.n_feat, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.pi   = nn.Linear(64, num_outputs)
        self.vf   = nn.Linear(64, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        # input_dict["obs"] is (B, win·F)
        obs = input_dict["obs"].float()
        B = obs.size(0)
        x = obs.view(B, self.win, self.n_feat)    # → (B, T, F)
        x = F.relu(self.fc(x))                    # → (B, T, 128)
        seq, _ = self.lstm(x)                     # → (B, T, 64)
        h = seq[:, -1]                            # → (B, 64)

        logits = self.pi(h)                       # → (B, A)
        self._value_out = self.vf(h).squeeze(-1)  # → (B,)

        return logits, state

    def value_function(self):
        assert self._value_out is not None, "must call forward first"
        return self._value_out

# register it
ModelCatalog.register_custom_model("solver_lstm_v2", SolverLSTMModel)
