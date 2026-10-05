"""RLModule-only deterministic inference helpers used by active evaluation."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ray.rllib.core.columns import Columns


def _native_module(algorithm: Any):
    config = getattr(algorithm, "config", None)
    if not (
        config is not None
        and getattr(config, "enable_rl_module_and_learner", False)
        and getattr(config, "enable_env_runner_and_connector_v2", False)
    ):
        raise TypeError("Active inference requires a native RLModule/Learner algorithm")
    return algorithm.get_module()


def deterministic_action_details(algorithm: Any, observation: np.ndarray) -> dict[str, Any]:
    """Return greedy action evidence through the native module inference API."""
    module = _native_module(algorithm)
    obs = torch.as_tensor(observation, dtype=torch.float32)
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    if obs.ndim != 2 or obs.shape[0] != 1:
        raise ValueError(
            "deterministic_action_details requires exactly one observation"
        )
    with torch.inference_mode():
        outputs = module.forward_inference({Columns.OBS: obs})
    logits_batch = outputs[Columns.ACTION_DIST_INPUTS]
    if logits_batch.ndim != 2 or logits_batch.shape[0] != 1:
        raise ValueError("Native RLModule returned unexpected action-logit shape")
    logits = logits_batch.squeeze(0)
    probabilities = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(
        probabilities * torch.log(torch.clamp(probabilities, min=1e-12))
    )
    values = outputs[Columns.VF_PREDS].reshape(-1)
    if values.numel() != 1:
        raise ValueError("Native RLModule returned unexpected value-prediction shape")
    return {
        "action": int(torch.argmax(logits, dim=-1).item()),
        "action_logits": logits.detach().cpu().numpy().astype(np.float64).tolist(),
        "action_probabilities": probabilities.detach().cpu().numpy().astype(np.float64).tolist(),
        "value_prediction": float(values[0].item()),
        "policy_entropy": float(entropy.item()),
    }


def deterministic_action(algorithm: Any, observation: np.ndarray) -> int:
    """Choose the greedy discrete action using the native RLModule."""
    return int(deterministic_action_details(algorithm, observation)["action"])


def algorithm_device(algorithm: Any) -> str:
    """Describe the device owned by the active RLModule."""
    module = _native_module(algorithm)
    return str(next(module.parameters()).device)


__all__ = ["algorithm_device", "deterministic_action", "deterministic_action_details"]
