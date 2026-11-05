"""Shared numerical operation for the repository's task FiLM readout."""
from __future__ import annotations

import torch


def apply_task_film(hidden: torch.Tensor, gamma_beta: torch.Tensor) -> torch.Tensor:
    """Apply the established tanh-scale plus bias FiLM formula."""
    gamma_raw, beta = torch.chunk(gamma_beta, 2, dim=-1)
    return torch.tanh(gamma_raw) * hidden + beta
