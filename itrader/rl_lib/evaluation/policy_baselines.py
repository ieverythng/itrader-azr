from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class ActionPolicy(Protocol):
    def action(self, observation: np.ndarray) -> int: ...


@dataclass
class FrozenCheckpointPolicy:
    """Use restored RLlib weights without re-enabling sampling on replay."""

    algorithm: Any

    def action(self, observation: np.ndarray) -> int:
        return int(self.algorithm.compute_single_action(observation, explore=False))


@dataclass
class ControlPolicy:
    """Deterministic fixed controls plus a reproducible random control."""

    name: str
    seed: int

    def __post_init__(self) -> None:
        if self.name not in {"flat", "long", "short", "random"}:
            raise ValueError(f"Unknown policy control: {self.name}")
        self._rng = np.random.default_rng(int(self.seed))

    def action(self, observation: np.ndarray) -> int:
        del observation
        fixed = {"short": 0, "flat": 1, "long": 2}
        if self.name in fixed:
            return fixed[self.name]
        return int(self._rng.integers(0, 3))


class UntrainedFiLMPolicy:
    """Seeded, deterministic argmax policy from the untrained Phase A model."""

    def __init__(self, observation_space, action_space, *, seed: int, model_config: dict):
        import torch

        from itrader.models.solver_mlp_film import SolverMLPFiLM

        torch.manual_seed(int(seed))
        self._torch = torch
        self._model = SolverMLPFiLM(
            observation_space,
            action_space,
            int(action_space.n),
            model_config,
            "untrained_solver_mlp_film",
        )
        self._model.eval()

    def action(self, observation: np.ndarray) -> int:
        tensor = self._torch.as_tensor(observation, dtype=self._torch.float32).unsqueeze(0)
        with self._torch.no_grad():
            logits, _ = self._model({"obs": tensor}, [], None)
        return int(self._torch.argmax(logits, dim=-1).item())


def evaluate_policy_control(env, policy: ActionPolicy, *, episode_count: int) -> list[dict[str, Any]]:
    """Evaluate a policy on the environment's frozen TaskSpec sequence."""
    rows: list[dict[str, Any]] = []
    for _ in range(int(episode_count)):
        observation, _ = env.reset()
        done = False
        reward_sum = 0.0
        action_counts = np.zeros(int(env.action_space.n), dtype=np.int64)
        terminal_info: dict[str, Any] = {}
        while not done:
            action = int(policy.action(observation))
            observation, reward, terminated, truncated, terminal_info = env.step(action)
            reward_sum += float(reward)
            action_counts[action] += 1
            done = bool(terminated or truncated)
        evidence = dict(terminal_info["episode_evidence"])
        steps = max(1, int(evidence["steps"]))
        evidence["optimization_reward"] = reward_sum
        evidence["action_fractions"] = (action_counts / steps).tolist()
        rows.append(evidence)
    return rows


def summarize_control_episodes(episodes: list[dict[str, Any]]) -> dict[str, float]:
    if not episodes:
        raise ValueError("At least one control episode is required")

    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in episodes]))

    report = {
        "training_iteration": 0,
        "pbt_progress_steps": 0,
        "episode_len_mean": mean("steps"),
        "evaluation/optimization_reward_mean": mean("optimization_reward"),
        "evaluation/gross_return_mean": mean("gross_return"),
        "evaluation/net_return_mean": mean("net_return"),
        "evaluation/net_sharpe_per_sqrt_bar_mean": mean("net_sharpe_per_sqrt_bar"),
        "evaluation/turnover_mean": mean("turnover"),
    }
    action_names = ("short", "flat", "long")
    for index, name in enumerate(action_names):
        value = float(np.mean([row["action_fractions"][index] for row in episodes]))
        report[f"custom_metrics/act_{index}_frac_mean"] = value
        report[f"custom_metrics/act_{name}_frac_mean"] = value
        report[f"evaluation/custom_metrics/act_{index}_frac_mean"] = value
        report[f"evaluation/custom_metrics/act_{name}_frac_mean"] = value
    return report
