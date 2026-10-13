"""Stateless-per-decision window LSTM solver with the R0 task FiLM readout."""
from __future__ import annotations

import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from itrader.models.observation_contract import validate_semantic_config
from itrader.models.task_film import apply_task_film


class SolverWindowLSTMFiLM(TorchModelV2, nn.Module):
    """Encode the ordered market window, then apply task FiLM and split heads.

    Input semantics are mandatory: ``[60x5 OHLCV | task embedding | optional
    two-value solver state]``. The LSTM receives only the market window and
    starts from zero hidden/cell state on every forward call. It returns no
    RLlib recurrent state; R1 is not episode-persistent recurrence.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", model_config)
        contract = cfg.get("semantic_observation_contract")
        if contract is None:
            raise ValueError("R1 requires an explicit semantic_observation_contract")
        self.contract = validate_semantic_config(
            cfg,
            contract,
            boundary="model",
            actual_obs_shape=obs_space.shape,
        )
        if self.contract["window_size"] != 60 or self.contract["solver_state_dim"] not in {0, 2}:
            raise ValueError("R1 requires a 60-bar window and zero or two state features")
        if num_outputs != 3 or getattr(action_space, "n", None) != 3:
            raise ValueError("R1 Family 00 requires exactly three discrete actions")

        self.window_size = 60
        self.feature_count = 5
        self.market_dim = self.window_size * self.feature_count
        self.task_dim = 32
        self.solver_state_dim = self.contract["solver_state_dim"]
        self.hidden_size = int(cfg.get("lstm_hidden_size", 0))
        self.projection_size = int(cfg.get("projection_size", 0))
        if (self.hidden_size, self.projection_size) != (156, 256):
            raise ValueError("R1 architecture requires LSTM hidden size 156 and projection size 256")
        if cfg.get("conditioning") != "film" or int(cfg.get("t_proj", 0)) != 64:
            raise ValueError("R1 requires the existing 64-wide task FiLM readout")
        if int(cfg.get("solver_state_width", 0)) != 8:
            raise ValueError("R1 requires the accepted width-8 solver-state residual")
        if "fcnet_hiddens" in cfg:
            raise ValueError("R1 must not inherit the R0 MLP hidden-layer configuration")

        self.t_keep_p = float(cfg.get("t_keep_p", 0.7))
        if not 0.0 <= self.t_keep_p <= 1.0:
            raise ValueError("t_keep_p must be between zero and one")
        self.dropout = nn.Dropout(p=1.0 - self.t_keep_p)
        self.window_lstm = nn.LSTM(
            input_size=self.feature_count,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.market_projection = nn.Linear(self.hidden_size, self.projection_size)
        self.t_mlp = nn.Sequential(
            nn.Linear(self.task_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.projection_size),
        )
        self.pi = nn.Linear(self.projection_size, num_outputs)
        self.vf = nn.Linear(self.projection_size, 1)
        self._value_out = None

        self.state_encoder = None
        self.state_policy_residual = None
        self.state_value_residual = None
        if self.solver_state_dim:
            if cfg.get("solver_state_residual_init") != "zero_v1":
                raise ValueError("R1 solver state requires zero_v1 residual initialization")
            self.state_encoder = nn.Sequential(
                nn.Linear(self.solver_state_dim, 8, bias=False),
                nn.ReLU(),
            )
            self.state_policy_residual = nn.Linear(8, num_outputs, bias=False)
            self.state_value_residual = nn.Linear(8, 1, bias=False)
            nn.init.zeros_(self.state_policy_residual.weight)
            nn.init.zeros_(self.state_value_residual.weight)

    def get_initial_state(self):
        """R1 uses a fresh in-window LSTM state per decision, never RLlib carry."""
        return []

    def forward(self, input_dict, state, seq_lens):
        if not isinstance(state, (list, tuple)) or len(state) != 0:
            raise ValueError("R1 is stateless across decisions and rejects carried state")
        observation = input_dict["obs"]
        if observation.ndim != 2 or observation.shape[-1] != self.contract["agent_obs_dim"]:
            raise ValueError(
                "R1 observation batch shape must be "
                f"[batch, {self.contract['agent_obs_dim']}]"
            )

        obs = torch.nan_to_num(observation.float())
        market = obs[:, :self.market_dim].reshape(-1, self.window_size, self.feature_count)
        task_end = self.market_dim + self.task_dim
        task_embed = obs[:, self.market_dim:task_end]
        solver_state = obs[:, task_end:]

        # Omitting hx/cx intentionally creates fresh zero state on each call.
        sequence, _ = self.window_lstm(market)
        hidden = torch.relu(self.market_projection(sequence[:, -1, :]))
        is_training = getattr(input_dict, "is_training", None)
        if is_training is None:
            is_training = input_dict.get("is_training", self.training)
        if isinstance(is_training, torch.Tensor):
            is_training = bool(is_training.reshape(-1)[0].item())
        if bool(is_training) and self.t_keep_p < 1.0:
            task_embed = self.dropout(task_embed)
        hidden = apply_task_film(hidden, self.t_mlp(task_embed))

        logits = self.pi(hidden)
        value = self.vf(hidden)
        if self.solver_state_dim:
            state_embedding = self.state_encoder(solver_state)
            logits = logits + self.state_policy_residual(state_embedding)
            value = value + self.state_value_residual(state_embedding)
        self._value_out = value.squeeze(-1)
        return logits, []

    def value_function(self):
        if self._value_out is None:
            raise RuntimeError("value_function requires a preceding forward call")
        return self._value_out
