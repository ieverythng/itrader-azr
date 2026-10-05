"""Reconcile target PPO settings after a Ray 2.58 legacy donor restore.

Ray restores donor policy configuration and optimizer parameter groups while
retaining the target's schedule objects. Reconcile these owners before sampling.
Weights, Adam moments, adaptive KL, and RLlib's native counters remain untouched.
The explicit progress selects schedule values for the next update; this helper
does not change subsequent RLlib clock semantics or move schedule boundaries.
"""

import math
from copy import deepcopy
from numbers import Integral

import ray
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.torch_mixins import EntropyCoeffSchedule, LearningRateSchedule

LEARNING_FIELDS = (
    "gamma",
    "lambda",
    "clip_param",
    "vf_loss_coeff",
    "vf_clip_param",
    "lr",
    "lr_schedule",
    "entropy_coeff",
    "entropy_coeff_schedule",
    "num_epochs",
    "minibatch_size",
)


def _updated_config(config, fields, explore=None):
    updated = config.copy(copy_frozen=False)
    values = deepcopy(fields)
    if explore is not None:
        values["explore"] = explore
    updated.update_from_dict(values)
    updated.freeze()
    return updated


def _reconcile_policy(policy, policy_id, fields, progress_steps, explore, worker_index):
    if not isinstance(policy, PPOTorchPolicy):
        raise TypeError("PPO restore reconciliation requires legacy PPOTorchPolicy")
    policy.config = {**policy.config, **deepcopy(fields), "explore": explore}
    LearningRateSchedule.__init__(policy, fields["lr"], fields["lr_schedule"])
    EntropyCoeffSchedule.__init__(
        policy, fields["entropy_coeff"], fields["entropy_coeff_schedule"]
    )
    if policy._lr_schedule is not None:
        policy.cur_lr = policy._lr_schedule.value(progress_steps)
    if policy._entropy_coeff_schedule is not None:
        policy.entropy_coeff = policy._entropy_coeff_schedule.value(progress_steps)
    if not math.isfinite(policy.cur_lr) or policy.cur_lr < 0:
        raise ValueError("Restored target learning rate must be finite and nonnegative")
    if not math.isfinite(policy.entropy_coeff) or policy.entropy_coeff < 0:
        raise ValueError(
            "Restored target entropy coefficient must be finite and nonnegative"
        )
    for optimizer in policy._optimizers:
        for group in optimizer.param_groups:
            group["lr"] = policy.cur_lr

    actual_fields = {name: deepcopy(policy.config[name]) for name in LEARNING_FIELDS}
    actual_lrs = [
        float(group["lr"])
        for optimizer in policy._optimizers
        for group in optimizer.param_groups
    ]
    if (
        actual_fields != fields
        or not actual_lrs
        or any(value != policy.cur_lr for value in actual_lrs)
    ):
        raise RuntimeError("PPO restored policy settings failed verification")
    return {
        "worker_index": worker_index,
        "policy_id": policy_id,
        "learning_config": actual_fields,
        "cur_lr": float(policy.cur_lr),
        "entropy_coeff": float(policy.entropy_coeff),
        "optimizer_lrs": actual_lrs,
        "adaptive_kl_coeff": float(policy.kl_coeff),
        "explore": policy.config["explore"],
        "policy_global_timestep": int(policy.global_timestep),
    }


def _reconcile_group(group, fields, progress_steps, explore):
    if group is None:
        return []

    def reconcile_worker(worker):
        worker.config = _updated_config(worker.config, fields, explore)
        return worker.foreach_policy(
            lambda policy, policy_id: _reconcile_policy(
                policy, policy_id, fields, progress_steps, explore, worker.worker_index
            )
        )

    expected_workers = group.num_remote_workers() + int(
        group.local_env_runner is not None
    )
    worker_results = group.foreach_env_runner(reconcile_worker, healthy_only=False)
    if len(worker_results) != expected_workers:
        raise RuntimeError(
            "PPO restore reconciliation did not reach every environment runner"
        )
    return [record for records in worker_results for record in records]


def reconcile_ppo_after_restore(algo, target_config, progress_steps):
    """Apply target learning settings and return measured pre-update evidence.

    Call after donor/target checkpoint restore and recovery of target-owned
    sampled steps, before any further sampling. Representation compatibility
    must already have been checked by the model registry.
    """
    if ray.__version__ != "2.58.0":
        raise RuntimeError("PPO restore reconciliation requires Ray 2.58.0")
    if (
        algo.config.enable_rl_module_and_learner
        or algo.config.enable_env_runner_and_connector_v2
        or algo.config.framework_str != "torch"
    ):
        raise ValueError("PPO restore reconciliation requires the legacy Torch stack")
    if (
        isinstance(progress_steps, bool)
        or not isinstance(progress_steps, Integral)
        or progress_steps < 0
    ):
        raise ValueError("progress_steps must be a nonnegative integer")
    progress_steps = int(progress_steps)
    fields = {
        name: deepcopy(target_config.get(name, algo.config[name]))
        for name in LEARNING_FIELDS
    }
    explore = target_config.get("explore", algo.config.explore)
    if type(explore) is not bool:
        raise ValueError("Target exploration setting must be boolean")
    training = _reconcile_group(algo.env_runner_group, fields, progress_steps, explore)
    if not training:
        raise RuntimeError("PPO restore reconciliation found no training policies")
    evaluation = _reconcile_group(
        algo.eval_env_runner_group, fields, progress_steps, False
    )
    algo.config = _updated_config(algo.config, fields)
    if algo.evaluation_config is not None:
        algo.evaluation_config = _updated_config(algo.evaluation_config, fields, False)
    return {
        "version": "ppo_restore_reconciliation_v1",
        "ray_version": ray.__version__,
        "progress_steps": progress_steps,
        "target_learning_config": fields,
        "algorithm_learning_config": {
            name: deepcopy(algo.config[name]) for name in LEARNING_FIELDS
        },
        "training_policies": training,
        "evaluation_policies": evaluation,
    }
