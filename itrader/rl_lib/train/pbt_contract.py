from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any


_CHOICES = {
    "clip_param": (0.1, 0.2, 0.3),
    "minibatch_size": (128, 256, 512),
    "num_epochs": (10, 20, 30),
    "vf_clip_param": (5.0, 10.0, 20.0),
    "vf_loss_coeff": (0.5, 1.0, 2.0),
}


def parse_step_schedule(
    value: Any, *, name: str = "optimization_cost_schedule"
) -> tuple[tuple[int, float], ...]:
    """Parse ordered control points; the caller owns clock and interpolation."""
    if value is None or value == "":
        return ()
    raw_points = value if isinstance(value, (list, tuple)) else str(value).split(",")
    points: list[tuple[int, float]] = []
    try:
        for raw_point in raw_points:
            if isinstance(raw_point, (list, tuple)) and len(raw_point) == 2:
                raw_step, raw_value = raw_point
            else:
                raw_step, raw_value = str(raw_point).split(":", maxsplit=1)
            step = int(raw_step)
            scheduled = float(raw_value)
            if step < 0 or scheduled < 0.0 or not math.isfinite(scheduled):
                raise ValueError
            points.append((step, scheduled))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain nonnegative step:value pairs"
        ) from exc
    if not points or points[0][0] != 0:
        raise ValueError(f"{name} must start at step 0")
    if any(current[0] <= previous[0] for previous, current in zip(points, points[1:])):
        raise ValueError(f"{name} steps must be strictly increasing")
    return tuple(points)


def scheduled_value(schedule: Any, progress_steps: int) -> float:
    """Resolve a piecewise-constant value at monotonic sampled steps."""
    points = parse_step_schedule(schedule)
    if not points:
        raise ValueError("optimization_cost_schedule must not be empty")
    progress = max(0, int(progress_steps))
    resolved = points[0][1]
    for step, value in points[1:]:
        if step > progress:
            break
        resolved = value
    return resolved


def economic_selection_score(
    *,
    net_return: float,
    net_sharpe: float,
    flat_action_fraction: float,
    min_net_sharpe: float = 0.0,
    max_flat_action_fraction: float = 0.98,
) -> float:
    """Rank economically eligible PBT donors by net return.

    Eligibility is lexicographic: a policy must have positive held-out net
    return and Sharpe and must not be effectively all-flat before its net
    return can outrank an ineligible policy. Directional action fractions are
    deliberately ignored.
    Ineligible members retain their net-return ordering inside a separate,
    finite score tier so PBT can still make progress before any member passes.
    """
    net = float(net_return)
    sharpe = float(net_sharpe)
    flat = float(flat_action_fraction)
    minimum_sharpe = float(min_net_sharpe)
    maximum_flat = float(max_flat_action_fraction)
    if not math.isfinite(minimum_sharpe):
        raise ValueError("min_net_sharpe must be finite")
    if not math.isfinite(maximum_flat) or not 0.0 <= maximum_flat <= 1.0:
        raise ValueError("max_flat_action_fraction must be between 0 and 1")
    if (
        not all(math.isfinite(value) for value in (net, sharpe, flat))
        or not 0.0 <= flat <= 1.0
    ):
        return -1.1
    eligible = (
        net > 0.0
        and sharpe > minimum_sharpe
        and flat < maximum_flat
    )
    if eligible:
        return net
    return -1.0 + min(0.1, max(-0.1, net))


def economic_search_score(
    *,
    net_return: float,
    flat_action_fraction: float,
    max_flat_action_fraction: float = 0.98,
) -> float:
    """Rank non-flat search donors by held-out net return.

    Search eligibility is intentionally weaker than economic promotion. Any
    finite, non-flat policy may guide PBT while absolute net, Sharpe, control,
    replay, and uncertainty gates remain mandatory for promotion. The leading
    tier keeps every non-flat candidate above flat or invalid candidates, then
    preserves net-return ordering within that tier.
    """
    net = float(net_return)
    flat = float(flat_action_fraction)
    maximum_flat = float(max_flat_action_fraction)
    if not math.isfinite(maximum_flat) or not 0.0 <= maximum_flat <= 1.0:
        raise ValueError("max_flat_action_fraction must be between 0 and 1")
    if not math.isfinite(net) or not math.isfinite(flat) or not 0.0 <= flat <= 1.0:
        return -1.0
    if flat >= maximum_flat:
        return 0.0
    return 1.0 + min(0.5, max(-0.5, net))


@dataclass(frozen=True)
class CheckpointRetentionDecision:
    retain_stable: bool
    report_to_tune: bool


def checkpoint_retention_decision(
    *,
    cadence_due: bool,
    terminal_due: bool,
    selection_score: float,
    current_best_score: float,
    synchronous_pbt: bool = False,
) -> CheckpointRetentionDecision:
    """Keep exact synchronous snapshots without changing Tune's cadence."""
    report_to_tune = bool(cadence_due or terminal_due)
    score = float(selection_score)
    best = float(current_best_score)
    eligible_improvement = (
        math.isfinite(score)
        and score > 0.0
        and (not math.isfinite(best) or score > best)
    )
    return CheckpointRetentionDecision(
        retain_stable=bool(synchronous_pbt) or report_to_tune or eligible_improvement,
        report_to_tune=report_to_tune,
    )


@dataclass
class PBTProgressClock:
    """Trial-owned exposure clock that is not replaced with donor state.

    RLlib's ``training_iteration`` is part of a restored checkpoint. A PBT
    target therefore needs a small sidecar clock in its stable trial directory
    so exploit/restore cannot move scheduling time backwards.
    """

    target_steps: int
    completed_steps: int = 0
    updates: int = 0
    schema_version: int = 1

    @classmethod
    def load(cls, path: Path, *, target_steps: int) -> "PBTProgressClock":
        target = max(1, int(target_steps))
        if not path.exists():
            return cls(target_steps=target)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid PBT progress state: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid PBT progress state: {path}")
        return cls(
            target_steps=target,
            completed_steps=max(0, int(payload.get("completed_steps", 0))),
            updates=max(0, int(payload.get("updates", 0))),
        )

    @property
    def remaining(self) -> bool:
        return self.completed_steps < self.target_steps

    def advance(self, result: dict[str, Any]) -> int:
        env_runners = result.get("env_runners")
        sampled = env_runners.get("num_env_steps_sampled") if isinstance(env_runners, dict) else None
        if isinstance(sampled, bool) or not isinstance(sampled, Integral) or sampled <= 0:
            raise ValueError(
                "Native PPO result requires positive integer "
                "env_runners.num_env_steps_sampled"
            )
        sampled = int(sampled)
        self.completed_steps += sampled
        self.updates += 1
        return sampled

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(asdict(self), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def crossed_interval(previous: int, current: int, interval: int) -> bool:
    """Return whether monotonic progress crossed a cadence boundary."""
    cadence = max(1, int(interval))
    return int(current) // cadence > int(previous) // cadence


def _nearest(value: Any, choices: tuple[Any, ...]) -> Any:
    numeric = float(value)
    return min(choices, key=lambda candidate: (abs(float(candidate) - numeric), float(candidate)))


def constrain_trial_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize one PBT mutation into the supported PPO search contract.

    Ray calls this after arithmetic perturbation. The function mutates and
    returns the supplied config, matching ``PopulationBasedTraining``'s seam.
    """
    config["gamma"] = min(0.999, max(0.95, float(config.get("gamma", 0.99))))
    config["lambda_"] = min(1.0, max(0.8, float(config.get("lambda_", 0.95))))
    learning_rate_min = float(config.get("pbt_lr_min", 1e-6))
    learning_rate_max = float(config.get("pbt_lr_max", 0.005))
    if (
        not math.isfinite(learning_rate_min)
        or not math.isfinite(learning_rate_max)
        or learning_rate_min <= 0.0
        or learning_rate_max <= learning_rate_min
    ):
        raise ValueError("pbt_lr_min and pbt_lr_max must be finite, positive, and ordered")
    config["lr"] = min(
        learning_rate_max,
        max(learning_rate_min, float(config.get("lr", 5e-5))),
    )
    for key, choices in _CHOICES.items():
        config[key] = _nearest(config.get(key, choices[0]), choices)
    # Keep rollout exposure fixed across population members. Only repair a
    # malformed imported config; batch size is deliberately not a mutation.
    train_batch_size = int(config.get("train_batch_size_per_learner", 4800))
    config["train_batch_size_per_learner"] = max(
        train_batch_size,
        2 * int(config["minibatch_size"]),
    )
    return config
