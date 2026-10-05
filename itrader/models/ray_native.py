"""Ray 2.58 native-stack RLModule for the implemented R0/R4 MLP-FiLM solver."""
from __future__ import annotations

from typing import Any

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule

from itrader.models.solver_mlp_film_core import SolverMLPFiLMTensorCore


class SolverMLPFiLMTorchRLModule(
    SolverMLPFiLMTensorCore,
    TorchRLModule,
    ValueFunctionAPI,
):
    """R0/R4 module using the same named tensors and forward math as ModelV2."""

    def setup(self) -> None:
        config = self.model_config or {}
        self._setup_solver_mlp_film(
            self.observation_space,
            int(self.action_space.n),
            config,
        )

    def get_train_action_dist_cls(self):
        return self.action_dist_cls

    def get_exploration_action_dist_cls(self):
        return self.action_dist_cls

    def get_inference_action_dist_cls(self):
        return self.action_dist_cls

    def _forward_for_phase(self, batch: dict[str, Any], *, training: bool):
        logits, values, embeddings = self._solver_forward(
            batch[Columns.OBS], is_training=training
        )
        outputs = {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: values,
        }
        if training:
            outputs[Columns.EMBEDDINGS] = embeddings
        return outputs

    def _forward(self, batch: dict[str, Any], **kwargs):
        return self._forward_for_phase(batch, training=self.training)

    def _forward_train(self, batch: dict[str, Any], **kwargs):
        return self._forward_for_phase(batch, training=True)

    def _forward_exploration(self, batch: dict[str, Any], **kwargs):
        return self._forward_for_phase(batch, training=False)

    def _forward_inference(self, batch: dict[str, Any], **kwargs):
        return self._forward_for_phase(batch, training=False)

    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            _, values, _ = self._solver_forward(
                batch[Columns.OBS], is_training=False
            )
            return values
        return self.value_from_embeddings(embeddings)


__all__ = ["SolverMLPFiLMTorchRLModule"]
