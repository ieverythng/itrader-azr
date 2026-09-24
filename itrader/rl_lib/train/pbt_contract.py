from __future__ import annotations

from typing import Any


_CHOICES = {
    "clip_param": (0.1, 0.2, 0.3),
    "sgd_minibatch_size": (128, 256, 512),
    "num_sgd_iter": (10, 20, 30),
    "vf_clip_param": (5.0, 10.0, 20.0),
    "vf_loss_coeff": (0.5, 1.0, 2.0),
}


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
