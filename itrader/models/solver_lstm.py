# -----------------------------------------------------------
#  SolverLSTM – recurrent policy/value head for Phase-A PPO
# -----------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class SolverLSTM(TorchModelV2, nn.Module):
    """
    obs  : flattened window  → FC → LSTM
    heads: policy logits & state-value
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -- hyper-params coming from the trainer config
        self.window_size = int(model_config.get("window_size", 32))

        # -- shared feature extractor
        flat_obs = int(np.prod(obs_space.shape))          # e.g. window_size * feat_dim
        self.fc1   = nn.Linear(flat_obs, 128)
        self.lstm  = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)

        # -- policy & critic heads
        self.policy_head = nn.Linear(64, action_space.n)
        self.value_head  = nn.Linear(64, 1)

        self.to(self.device)
        self._value_out = None

    # ------------------------------------------------------------------ #
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        input_dict["obs"]: (B, flat_obs)  where flat_obs = window_size * feat_dim
        We reshape to (B, T=window_size, 128) before feeding LSTM.
        """
        x = input_dict["obs"].float().to(self.device)          # (B, flat_obs)
        x = torch.relu(self.fc1(x))                            # (B, 128)
        B = x.shape[0]
        x = x.view(B, self.window_size, 128)                   # (B, T, 128)

        lstm_out, _ = self.lstm(x)                             # (B, T, 64)
        features = lstm_out[:, -1, :]                          # last step (B, 64)

        logits        = self.policy_head(features)             # (B, n_actions)
        self._value_out = self.value_head(features).squeeze(1) # (B,)

        return logits, state

    # ------------------------------------------------------------------ #
    @override(TorchModelV2)
    def value_function(self):
        assert self._value_out is not None, "Must call forward() first"
        return self._value_out
