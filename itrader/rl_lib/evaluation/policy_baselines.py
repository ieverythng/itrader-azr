from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from itrader.rl_lib.evaluation.ray_native_inference import (
    deterministic_action,
    deterministic_action_details,
)


class ActionPolicy(Protocol):
    def action(self, observation: np.ndarray) -> int: ...


@dataclass
class FrozenCheckpointPolicy:
    """Use restored RLlib weights without re-enabling sampling on replay."""

    algorithm: Any

    def action(self, observation: np.ndarray) -> int:
        return deterministic_action(self.algorithm, observation)

    def action_details(self, observation: np.ndarray) -> dict[str, Any]:
        """Return deterministic action plus model probabilities and value output."""
        return deterministic_action_details(self.algorithm, observation)


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
        from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from itrader.models.ray_native import SolverMLPFiLMTorchRLModule

        from itrader.models.registry import model_identity

        torch.manual_seed(int(seed))
        self._torch = torch
        identity = model_identity(model_config)
        self._model = RLModuleSpec(
            module_class=SolverMLPFiLMTorchRLModule,
            observation_space=observation_space,
            action_space=action_space,
            model_config=identity["custom_model_config"],
            catalog_class=PPOCatalog,
        ).build()
        self._model.eval()

    def action(self, observation: np.ndarray) -> int:
        tensor = self._torch.as_tensor(observation, dtype=self._torch.float32).unsqueeze(0)
        with self._torch.no_grad():
            from ray.rllib.core.columns import Columns

            outputs = self._model.forward_inference({Columns.OBS: tensor})
            logits = outputs[Columns.ACTION_DIST_INPUTS]
        return int(self._torch.argmax(logits, dim=-1).item())


def remaining_time_bin(remaining_fraction: float) -> str:
    """Map causal remaining-horizon state to fixed quarter bins."""
    remaining = float(np.clip(remaining_fraction, 0.0, 1.0))
    if remaining <= 0.25:
        return "q1_terminal"
    if remaining <= 0.5:
        return "q2"
    if remaining <= 0.75:
        return "q3"
    return "q4_early"


def counterfactual_state_details(
    policy: FrozenCheckpointPolicy,
    observation: np.ndarray,
    *,
    prior_position: float,
    remaining_fraction: float,
) -> dict[str, dict[str, Any]]:
    """Score fixed market/task inputs under named two-value state interventions."""
    base = np.asarray(observation).copy()
    if base.ndim != 1 or base.size < 2:
        raise ValueError("Counterfactual state diagnostics require a flat stateful observation")
    position = float(np.clip(prior_position, -1.0, 1.0))
    remaining = float(np.clip(remaining_fraction, 0.0, 1.0))
    states = {
        "observed": np.asarray(base[-2:], dtype=np.float64),
        "masked": np.asarray([0.0, 0.0]),
        "position_only": np.asarray([position, 0.0]),
        "time_only": np.asarray([0.0, remaining]),
        "both": np.asarray([position, remaining]),
        "position_flipped": np.asarray([-position, remaining]),
        "time_mirrored": np.asarray([position, 1.0 - remaining]),
    }
    details: dict[str, dict[str, Any]] = {}
    for name, state in states.items():
        intervened = base.copy()
        intervened[-2:] = state
        details[name] = policy.action_details(intervened)
    return details


def summarize_counterfactual_state_steps(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize action, logit, and value sensitivity to state interventions."""
    if not rows:
        raise ValueError("At least one counterfactual state step is required")
    observed = [row["counterfactuals"]["observed"] for row in rows]
    names = sorted(
        set.intersection(
            *(set(row["counterfactuals"]) for row in rows)
        )
        - {"observed"}
    )
    summary: dict[str, Any] = {"step_count": len(rows)}
    for name in names:
        logit_l2: list[float] = []
        logit_max_abs: list[float] = []
        value_abs: list[float] = []
        changed: list[bool] = []
        for actual, row in zip(observed, rows):
            counterfactual = row["counterfactuals"][name]
            logit_delta = np.asarray(
                counterfactual["action_logits"], dtype=np.float64
            ) - np.asarray(actual["action_logits"], dtype=np.float64)
            logit_l2.append(float(np.linalg.norm(logit_delta)))
            logit_max_abs.append(float(np.max(np.abs(logit_delta))))
            value_abs.append(
                abs(
                    float(counterfactual["value_prediction"])
                    - float(actual["value_prediction"])
                )
            )
            changed.append(int(counterfactual["action"]) != int(actual["action"]))
        summary[f"{name}_vs_observed"] = {
            "argmax_change_fraction": float(np.mean(changed)),
            "mean_logit_l2_delta": float(np.mean(logit_l2)),
            "mean_logit_max_abs_delta": float(np.mean(logit_max_abs)),
            "mean_abs_value_delta": float(np.mean(value_abs)),
        }
    return summary


def _summarize_step_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actions = np.asarray([int(row["action"]) for row in rows], dtype=np.int64)
    logits = np.asarray([row["action_logits"] for row in rows], dtype=np.float64)
    probabilities = np.asarray(
        [row["action_probabilities"] for row in rows], dtype=np.float64
    )
    values = np.asarray([row["value_prediction"] for row in rows], dtype=np.float64)
    ordered_logits = np.sort(logits, axis=1)
    logit_margins = ordered_logits[:, -1] - ordered_logits[:, -2]
    ordered_probabilities = np.sort(probabilities, axis=1)
    probability_margins = ordered_probabilities[:, -1] - ordered_probabilities[:, -2]
    return {
        "step_count": len(rows),
        "action_fractions": [float(np.mean(actions == action)) for action in range(3)],
        "mean_action_probabilities": probabilities.mean(axis=0).tolist(),
        "top_two_logit_margin_mean": float(np.mean(logit_margins)),
        "top_two_logit_margin_p01": float(np.quantile(logit_margins, 0.01)),
        "top_two_logit_margin_p05": float(np.quantile(logit_margins, 0.05)),
        "top_two_logit_margin_median": float(np.median(logit_margins)),
        "top_two_logit_margin_below_0_01_fraction": float(
            np.mean(logit_margins < 0.01)
        ),
        "top_two_probability_margin_mean": float(np.mean(probability_margins)),
        "top_two_probability_margin_p01": float(
            np.quantile(probability_margins, 0.01)
        ),
        "top_two_probability_margin_p05": float(
            np.quantile(probability_margins, 0.05)
        ),
        "top_two_probability_margin_median": float(np.median(probability_margins)),
        "switching_fraction": float(np.mean([bool(row["switched"]) for row in rows])),
        "policy_entropy_mean": float(
            np.mean([float(row["policy_entropy"]) for row in rows])
        ),
        "value_prediction_mean": float(np.mean(values)),
        "value_prediction_std": float(np.std(values)),
    }


def summarize_state_conditioned_steps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate policy behavior by actual prior position and causal time bin."""
    if not rows:
        raise ValueError("At least one state-conditioned step is required")

    position_names = {-1.0: "short", 0.0: "flat", 1.0: "long"}
    by_position = {
        name: _summarize_step_group(group)
        for position, name in position_names.items()
        if (group := [row for row in rows if float(row["prior_position"]) == position])
    }
    time_bin_order = ("q4_early", "q3", "q2", "q1_terminal")
    by_time = {
        time_bin: _summarize_step_group(group)
        for time_bin in time_bin_order
        if (group := [row for row in rows if row["remaining_time_bin"] == time_bin])
    }
    by_start_idx = {
        str(start_idx): _summarize_step_group(group)
        for start_idx in sorted({int(row["start_idx"]) for row in rows})
        if (group := [row for row in rows if int(row["start_idx"]) == start_idx])
    }
    return {
        "step_count": len(rows),
        "action_order": ["short", "flat", "long"],
        "overall": _summarize_step_group(rows),
        "by_prior_position": by_position,
        "by_remaining_time_bin": by_time,
        "by_start_idx": by_start_idx,
    }


def evaluate_checkpoint_state_diagnostics(
    env,
    policy: FrozenCheckpointPolicy,
    *,
    episode_count: int,
) -> list[dict[str, Any]]:
    """Replay a frozen policy while recording causal state-conditioned decisions."""
    rows: list[dict[str, Any]] = []
    action_positions = (-1.0, 0.0, 1.0)
    for episode_index in range(int(episode_count)):
        observation, reset_info = env.reset()
        done = False
        step_index = 0
        task = reset_info["task"]
        horizon = int(task["horizon"])
        while not done:
            prior_position = float(env.pos)
            remaining_fraction = max(
                0.0,
                (float(horizon) - float(step_index)) / float(horizon),
            )
            counterfactuals = counterfactual_state_details(
                policy,
                observation,
                prior_position=prior_position,
                remaining_fraction=remaining_fraction,
            )
            details = counterfactuals["observed"]
            action = int(details["action"])
            next_observation, _, terminated, truncated, _ = env.step(action)
            rows.append(
                {
                    "episode_index": episode_index,
                    "step_index": step_index,
                    "task_seed": int(task["seed"]),
                    "start_idx": int(task["start_idx"]),
                    "horizon": horizon,
                    "prior_position": prior_position,
                    "remaining_fraction": remaining_fraction,
                    "remaining_time_bin": remaining_time_bin(remaining_fraction),
                    "action": action,
                    "target_position": action_positions[action],
                    "switched": action_positions[action] != prior_position,
                    "turnover": abs(action_positions[action] - prior_position),
                    "terminal": bool(terminated or truncated),
                    "counterfactuals": counterfactuals,
                    **details,
                }
            )
            observation = next_observation
            step_index += 1
            done = bool(terminated or truncated)
    return rows


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
