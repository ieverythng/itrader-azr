"""Structural solver selection shared by training, controls and restore checks."""
from __future__ import annotations

from copy import deepcopy

SOLVER_SELECTORS = ("solver_mlp_film", "solver_window_lstm_film")
REPRESENTATION_PROFILES = ("legacy", "r0_r1_v1")
_LEGACY_SELECTORS = ("solver_lstm_v2", "solver_mlp")


def resolve_solver_model(selector: str):
    if selector == "solver_window_lstm_film":
        from .solver_window_lstm_film import SolverWindowLSTMFiLM

        return SolverWindowLSTMFiLM
    if selector != "solver_mlp_film":
        raise ValueError(f"Unknown or restore-only solver_model: {selector!r}")
    from .solver_mlp_film import SolverMLPFiLM

    return SolverMLPFiLM


def model_identity(config: dict, *, historical_restore: bool = False) -> dict:
    model = config.get("model", config)
    nested = model.get("custom_model", "solver_mlp_film")
    selector = config.get("solver_model", nested)
    if selector != nested:
        raise ValueError("solver_model conflicts with model.custom_model")
    if selector not in SOLVER_SELECTORS and not (historical_restore and selector in _LEGACY_SELECTORS):
        raise ValueError(f"Unknown solver_model: {selector!r}")
    cfg = deepcopy(model.get("custom_model_config", {}))
    return {"identity_version": "solver_identity_v1", "solver_model": selector,
            "custom_model_config": cfg}


def register_solver_models() -> None:
    from ray.rllib.models import ModelCatalog
    from .solver_lstm import SolverLSTMModel
    from .solver_MLP import SolverMLPModel

    ModelCatalog.register_custom_model("solver_mlp_film", resolve_solver_model("solver_mlp_film"))
    ModelCatalog.register_custom_model(
        "solver_window_lstm_film", resolve_solver_model("solver_window_lstm_film")
    )
    ModelCatalog.register_custom_model("solver_lstm_v2", SolverLSTMModel)
    ModelCatalog.register_custom_model("solver_mlp", SolverMLPModel)


def validate_representation_config(config: dict) -> dict:
    identity = model_identity(config)
    resolve_solver_model(identity["solver_model"])
    profile = config.get("representation_profile", "legacy")
    if profile not in REPRESENTATION_PROFILES:
        raise ValueError(f"Unknown representation_profile: {profile!r}")
    if profile == "legacy":
        if identity["solver_model"] == "solver_window_lstm_film":
            raise ValueError("R1 requires the explicit r0_r1_v1 representation profile")
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
    if identity["solver_model"] == "solver_window_lstm_film":
        expected_architecture = {
            "lstm_hidden_size": 156,
            "projection_size": 256,
            "conditioning": "film",
            "t_proj": 64,
            "t_keep_p": 0.7,
            "solver_state_width": 8,
        }
    else:
        expected_architecture = {
            "fcnet_hiddens": [256, 256],
            "conditioning": "film",
            "t_proj": 64,
            "solver_state_width": 8,
        }
    if any(model_cfg.get(key) != value for key, value in expected_architecture.items()):
        raise ValueError("Fixed-recipe architecture differs from the declared R0 contract")
    return identity


def validate_restored_model(algorithm, expected_config: dict) -> None:
    """Check restored implementation and effective semantics, not just tensor shapes."""
    expected = model_identity(expected_config, historical_restore=True)
    actual_config = algorithm.config.to_dict() if hasattr(algorithm.config, "to_dict") else algorithm.config
    actual = model_identity(actual_config, historical_restore=True)
    if expected["solver_model"] in SOLVER_SELECTORS:
        validate_representation_config(expected_config)
    if actual["solver_model"] in SOLVER_SELECTORS:
        validate_representation_config(actual_config)
    # RLlib fills model defaults; identity uses only custom configuration.
    if actual != expected:
        raise ValueError("Restored solver identity does not match checkpoint manifest")
    selector = expected["solver_model"]
    if selector in SOLVER_SELECTORS:
        policy = algorithm.get_policy()
        policy_config = getattr(policy, "config", None)
        if not isinstance(policy_config, dict):
            raise ValueError("Restored policy configuration is unavailable")
        policy_identity = model_identity(policy_config, historical_restore=True)
        if policy_identity != expected:
            raise ValueError("Restored policy identity does not match checkpoint manifest")
        validate_representation_config(policy_config)

        restored_model = policy.model
        if type(restored_model) is not resolve_solver_model(selector):
            raise ValueError("Restored model class does not match declared solver")
        resolved_model_config = getattr(restored_model, "model_config", None)
        if not isinstance(resolved_model_config, dict):
            raise ValueError("Restored model configuration is unavailable")
        resolved_custom_config = resolved_model_config.get("custom_model_config", {})
        if resolved_custom_config != expected["custom_model_config"]:
            raise ValueError("Restored model configuration does not match checkpoint manifest")
