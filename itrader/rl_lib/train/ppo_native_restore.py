"""Reconcile target PPO settings after a native Ray 2.58 learner restore."""
from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from numbers import Integral

import ray
import torch
from ray.rllib.utils.schedules.scheduler import Scheduler


def _config_fields(target_config: dict) -> dict:
    fields = {
        "gamma": float(target_config["gamma"]),
        "lambda_": float(target_config["lambda_"]),
        "clip_param": float(target_config["clip_param"]),
        "vf_loss_coeff": float(target_config["vf_loss_coeff"]),
        "vf_clip_param": float(target_config["vf_clip_param"]),
        "lr": deepcopy(target_config["lr"]),
        "entropy_coeff": deepcopy(target_config["entropy_coeff"]),
        "num_epochs": int(target_config["num_epochs"]),
        "minibatch_size": int(target_config["minibatch_size"]),
    }
    return fields


def _updated_config(config, fields):
    updated = config.copy(copy_frozen=False)
    updated.update_from_dict(deepcopy(fields))
    updated.freeze()
    return updated


def _state_fingerprint(learner) -> str:
    digest = hashlib.sha256()
    for optimizer_name in sorted(learner._named_optimizers):
        optimizer = learner._named_optimizers[optimizer_name]
        digest.update(optimizer_name.encode("utf-8"))
        for group_index, group in enumerate(optimizer.param_groups):
            digest.update(f"group:{group_index}".encode("ascii"))
            for param_index, parameter in enumerate(group["params"]):
                digest.update(f"param:{param_index}".encode("ascii"))
                state = optimizer.state.get(parameter, {})
                for key in sorted(state, key=str):
                    digest.update(str(key).encode("utf-8"))
                    value = state[key]
                    if torch.is_tensor(value):
                        tensor = value.detach().cpu().contiguous()
                        digest.update(str(tensor.dtype).encode("ascii"))
                        digest.update(str(tuple(tensor.shape)).encode("ascii"))
                        digest.update(tensor.numpy().tobytes())
                    else:
                        digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _call_result_value(value):
    return value.get() if hasattr(value, "get") else value


def _reconcile_learner(learner, fields, progress_steps):
    state_before = _state_fingerprint(learner)
    learner.config = _updated_config(learner.config, fields)

    optimizer_lrs = []
    for optimizer, _params in learner._optimizer_parameters.items():
        scheduler = Scheduler(
            fixed_value_or_schedule=fields["lr"],
            framework=learner.framework,
            device=learner._device,
        )
        value = float(scheduler.update(timestep=progress_steps))
        if not math.isfinite(value) or value < 0:
            raise ValueError("Target native PPO learning rate must be finite and nonnegative")
        learner._optimizer_lr_schedules[optimizer] = scheduler
        learner._set_optimizer_lr(optimizer, value)
        optimizer_lrs.extend(float(group["lr"]) for group in optimizer.param_groups)

    entropy_coefficients = {}
    for module_id in learner.module.keys():
        scheduler = Scheduler(
            fixed_value_or_schedule=fields["entropy_coeff"],
            framework=learner.framework,
            device=learner._device,
        )
        value = float(scheduler.update(timestep=progress_steps))
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "Target native PPO entropy coefficient must be finite and nonnegative"
            )
        learner.entropy_coeff_schedulers_per_module[module_id] = scheduler
        entropy_coefficients[module_id] = value

    state_after = _state_fingerprint(learner)
    if state_before != state_after:
        raise RuntimeError("Native PPO restore reconciliation changed optimizer moments")
    if not optimizer_lrs or any(value != optimizer_lrs[0] for value in optimizer_lrs):
        raise RuntimeError("Native PPO optimizer learning rates failed reconciliation")
    return {
        "learner_index": int(getattr(learner, "_learner_index", 0)),
        "learning_config": deepcopy(fields),
        "optimizer_learning_rates": optimizer_lrs,
        "entropy_coefficients": entropy_coefficients,
        "optimizer_state_sha256": state_after,
        "optimizer_state_preserved": True,
    }


def reconcile_native_ppo_after_restore(algorithm, target_config, progress_steps):
    """Restore target settings and schedule values without replacing Adam state."""
    if ray.__version__ != "2.58.0":
        raise RuntimeError("Native PPO restore reconciliation requires Ray 2.58.0")
    from itrader.rl_lib.train.ray_native_config import NATIVE_API_STACK_ID

    if target_config.get("api_stack_id") != NATIVE_API_STACK_ID:
        raise ValueError("Native PPO restore requires a qualified native target identity")
    if (
        not algorithm.config.enable_rl_module_and_learner
        or not algorithm.config.enable_env_runner_and_connector_v2
        or algorithm.config.framework_str != "torch"
    ):
        raise ValueError("Native PPO restore requires the Torch RLModule/Learner stack")
    if (
        isinstance(progress_steps, bool)
        or not isinstance(progress_steps, Integral)
        or progress_steps < 0
    ):
        raise ValueError("progress_steps must be a nonnegative integer")
    progress_steps = int(progress_steps)
    fields = _config_fields(target_config)
    training_records = [
        _call_result_value(result)
        for result in algorithm.learner_group.foreach_learner(
            lambda learner: _reconcile_learner(learner, fields, progress_steps)
        )
    ]
    if not training_records:
        raise RuntimeError("Native PPO restore found no learners")

    algorithm.config = _updated_config(algorithm.config, fields)
    if getattr(algorithm, "evaluation_config", None) is not None:
        algorithm.evaluation_config = _updated_config(
            algorithm.evaluation_config,
            fields,
        )

    evaluation_group = getattr(algorithm, "eval_env_runner_group", None)
    if evaluation_group is not None:
        evaluation_group.sync_weights(
            from_worker_or_learner_group=algorithm.learner_group,
            inference_only=True,
        )

    return {
        "version": "ppo_native_restore_reconciliation_v1",
        "ray_version": ray.__version__,
        "progress_steps": progress_steps,
        "target_learning_config": fields,
        "training_learners": training_records,
        "optimizer_state_preserved": all(
            record["optimizer_state_preserved"] for record in training_records
        ),
    }


__all__ = ["reconcile_native_ppo_after_restore"]
