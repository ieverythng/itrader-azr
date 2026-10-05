"""Ray-free solver identity and native restore validation."""
from __future__ import annotations

from copy import deepcopy

SOLVER_SELECTORS = ("solver_mlp_film", "solver_window_lstm_film")
REPRESENTATION_PROFILES = ("legacy", "r0_r1_v1")


def validate_solver_selector(selector: str) -> str:
    """Validate the active solver selector without importing any model adapter."""
    if selector == "solver_window_lstm_film":
        raise NotImplementedError(
            "solver_window_lstm_film is reserved for a qualified native R1 implementation"
        )
    if selector != "solver_mlp_film":
        raise ValueError(f"Unknown or compatibility-only solver_model: {selector!r}")
    return selector


def model_identity(config: dict) -> dict:
    """Return framework-neutral solver identity from an application config."""
    model = config.get("model", config)
    nested = model.get("custom_model", "solver_mlp_film")
    selector = config.get("solver_model", nested)
    if selector != nested:
        raise ValueError("solver_model conflicts with model.custom_model")
    validate_solver_selector(selector)
    cfg = deepcopy(model.get("custom_model_config", {}))
    return {
        "identity_version": "solver_identity_v1",
        "solver_model": selector,
        "custom_model_config": cfg,
    }


def validate_representation_config(config: dict) -> dict:
    identity = model_identity(config)
    profile = config.get("representation_profile", "legacy")
    if profile not in REPRESENTATION_PROFILES:
        raise ValueError(f"Unknown representation_profile: {profile!r}")
    if profile == "legacy":
        return identity
    from .observation_contract import validate_semantic_config

    model_cfg = config["model"]["custom_model_config"]
    contract = model_cfg.get("semantic_observation_contract")
    validate_semantic_config(model_cfg, contract, boundary="model")
    if config.get("observation_filter") != "NoFilter":
        raise ValueError("Fixed-recipe observation_filter must be NoFilter")
    for env in (config["env_config"], config["evaluation_config"]["env_config"]):
        if env.get("semantic_observation_contract") != contract:
            raise ValueError("Model and environment semantic observation contracts differ")
        validate_semantic_config(env, contract, boundary="env")
    if model_cfg.get("solver_state_residual_init") != "zero_v1":
        raise ValueError("Fixed-recipe solver state initialization must be zero_v1")
    expected_architecture = {
        "fcnet_hiddens": [256, 256],
        "conditioning": "film",
        "t_proj": 64,
        "solver_state_width": 8,
    }
    if any(model_cfg.get(key) != value for key, value in expected_architecture.items()):
        raise ValueError("Fixed-recipe architecture differs from the declared R0 contract")
    return identity


def validate_native_restored_model(algorithm, expected_config: dict) -> None:
    """Check that a checkpoint restored into the declared native RLModule."""
    from itrader.rl_lib.train.ray_native_config import NATIVE_API_STACK_ID

    if expected_config.get("api_stack_id") != NATIVE_API_STACK_ID:
        raise ValueError("Native restore requires a qualified native api_stack_id")
    expected = model_identity(expected_config)
    config = algorithm.config
    if not (
        getattr(config, "enable_rl_module_and_learner", False)
        and getattr(config, "enable_env_runner_and_connector_v2", False)
    ):
        raise ValueError("Restored algorithm is not using the declared native API stack")
    if expected["solver_model"] != "solver_mlp_film":
        raise ValueError("The native RLModule does not implement the declared solver")
    from .ray_native import SolverMLPFiLMTorchRLModule

    module = algorithm.get_module()
    if type(module) is not SolverMLPFiLMTorchRLModule:
        raise ValueError("Restored RLModule class does not match declared solver")
    module_spec = getattr(config, "rl_module_spec", None)
    actual_model_config = getattr(module_spec, "model_config", None)
    if actual_model_config != expected["custom_model_config"]:
        raise ValueError("Restored RLModule settings do not match checkpoint manifest")


__all__ = [
    "REPRESENTATION_PROFILES",
    "SOLVER_SELECTORS",
    "model_identity",
    "validate_native_restored_model",
    "validate_representation_config",
    "validate_solver_selector",
]
