"""Construct the repository's sole active Ray RLlib configuration."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import ray
from gymnasium.spaces import Box, Discrete
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from itrader.models.ray_native import SolverMLPFiLMTorchRLModule
from itrader.rl_lib.api_stack import NATIVE_API_STACK_ID


_RETIRED_LEGACY_KEYS = {
    "rllib_api_stack",
    "num_gpus",
    "num_workers",
    "evaluation_num_workers",
    "num_gpus_per_worker",
    "train_batch_size",
    "lambda",
    "entropy_coeff_schedule",
    "sgd_minibatch_size",
    "num_sgd_iter",
}


def build_native_ppo_config(application_config: dict):
    """Build a native RLModule/Learner/EnvRunner PPO config.

    The application config is already expressed in Ray's current API vocabulary.
    Old stack selectors and field aliases are rejected, never translated into a
    second trainer implementation.
    """
    if ray.__version__ != "2.58.0":
        raise RuntimeError(
            "The active RLModule/Learner trainer requires the pinned Ray 2.58.0 build"
        )
    retired = sorted(_RETIRED_LEGACY_KEYS.intersection(application_config))
    if retired:
        raise ValueError(
            "Legacy API-stack selection is retired; migrate this run config to "
            f"Ray 2.58 native fields: {', '.join(retired)}"
        )
    if not hasattr(PPOConfig, "api_stack"):
        raise RuntimeError("The active trainer requires Ray's native PPO API stack")
    if application_config.get("api_stack_id") != NATIVE_API_STACK_ID:
        raise ValueError("Run config is missing the qualified native api_stack_id")
    if application_config.get("observation_filter", "NoFilter") != "NoFilter":
        raise ValueError(
            "Ray 2.58 native stack requires observation_filter=NoFilter; "
            "MeanStdFilter is not installed as a native connector"
        )
    if application_config.get("solver_model", "solver_mlp_film") != "solver_mlp_film":
        raise NotImplementedError(
            "The native Ray trainer currently implements solver_mlp_film only"
        )

    values = deepcopy(application_config)
    values.pop("api_stack_id")
    # This application-level declaration is validated above; it is not a
    # native AlgorithmConfig field and must not rely on an OldAPIStack path.
    values.pop("observation_filter", None)
    model_config = values.pop("model")["custom_model_config"]
    env_config = values["env_config"]
    observation_width = int(env_config["agent_obs_dim"])
    action_width = int(env_config["task_sampler"]["action_dim"])
    values.pop("metrics_sinks", None)
    values.pop("prometheus_export_port", None)
    module_spec = RLModuleSpec(
        module_class=SolverMLPFiLMTorchRLModule,
        observation_space=Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_width,),
            dtype=np.float32,
        ),
        action_space=Discrete(action_width),
        model_config=deepcopy(model_config),
        catalog_class=PPOCatalog,
    )
    values["enable_rl_module_and_learner"] = True
    values["enable_env_runner_and_connector_v2"] = True
    values["rl_module_spec"] = module_spec

    from itrader.rl_lib.train.ray_native_ppo import DiagnosticPPOTorchLearner

    config = PPOConfig().api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    config.update_from_dict(values)
    return config.learners(learner_class=DiagnosticPPOTorchLearner)


__all__ = ["NATIVE_API_STACK_ID", "build_native_ppo_config"]
