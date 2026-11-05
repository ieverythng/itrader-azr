from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_CHOICES = {
    "clip_param": (0.1, 0.2, 0.3),
    "sgd_minibatch_size": (128, 256, 512),
    "num_sgd_iter": (10, 20, 30),
    "vf_clip_param": (5.0, 10.0, 20.0),
    "vf_loss_coeff": (0.5, 1.0, 2.0),
}


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


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

    def advance(self, result: dict[str, Any], *, fallback_steps: int) -> int:
        candidates = (
            result.get("num_env_steps_sampled_this_iter"),
            result.get("num_agent_steps_sampled_this_iter"),
            result.get("sampler_results", {}).get("num_env_steps_sampled_this_iter"),
        )
        sampled = next(
            (parsed for value in candidates if (parsed := _positive_int(value)) is not None),
            max(1, int(fallback_steps)),
        )
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
    config["lambda"] = min(1.0, max(0.8, float(config.get("lambda", 0.95))))
    config["lr"] = min(0.005, max(1e-6, float(config.get("lr", 5e-5))))
    for key, choices in _CHOICES.items():
        config[key] = _nearest(config.get(key, choices[0]), choices)
    # Keep rollout exposure fixed across population members. Only repair a
    # malformed imported config; batch size is deliberately not a mutation.
    train_batch_size = int(config.get("train_batch_size", 4800))
    config["train_batch_size"] = max(
        train_batch_size,
        2 * int(config["sgd_minibatch_size"]),
    )
    return config
