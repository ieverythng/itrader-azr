"""Versioned, opt-in semantic boundary shared by environments and solvers.

Historical configurations without this contract keep their original defaults.
This version pins the fixed-recipe observation, not a particular market encoder.
"""

from collections.abc import Mapping, Sequence
from typing import Any


SEMANTIC_OBSERVATION_CONTRACT_VERSION = "ohlcv_window_v1"
_STATE_MODES = {"off", "masked", "position", "time", "both"}


def semantic_observation_contract(solver_state_mode: str = "off") -> dict[str, Any]:
    if solver_state_mode not in _STATE_MODES:
        raise ValueError(f"Unknown solver_state_mode: {solver_state_mode!r}")
    state_dim = 0 if solver_state_mode == "off" else 2
    return {
        "version": SEMANTIC_OBSERVATION_CONTRACT_VERSION,
        "feature_cols": ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"],
        "window_size": 60,
        "window_order": "oldest_to_newest",
        "window_end": "idx-1",
        "t_embed_dim": 32,
        "task_encoder_version": "semantic_v2",
        "t_embed_mode": "concat",
        "observation_transform": "market_relative_v1",
        "observation_filter": "NoFilter",
        "observation_schema_id": "r0_base" if state_dim == 0 else "r4_v1_min",
        "base_obs_dim": 332,
        "solver_state_dim": state_dim,
        "agent_obs_dim": 332 + state_dim,
        "solver_state_mode": solver_state_mode,
        "solver_state_features": (
            ["position", "remaining_horizon_fraction"] if state_dim else []
        ),
        "state_lifecycle": "stateless_per_decision",
    }


def validate_semantic_observation_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("semantic_observation_contract must be a mapping")
    mode = value.get("solver_state_mode")
    if not isinstance(mode, str) or mode not in _STATE_MODES:
        raise ValueError("semantic_observation_contract has invalid solver_state_mode")
    expected = semantic_observation_contract(mode)
    for key in expected.keys() | value.keys():
        if key not in expected or key not in value or value[key] != expected[key]:
            raise ValueError(f"semantic_observation_contract incompatible {key}")
        if isinstance(expected[key], int) and type(value[key]) is not int:
            raise ValueError(f"semantic_observation_contract requires integer {key}")
    return expected


def validate_semantic_config(
    cfg: Mapping[str, Any],
    contract: Any,
    *,
    boundary: str,
    actual_feature_columns: Sequence[str] | None = None,
    actual_obs_shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate declarations against effective config and available runtime facts."""
    expected = validate_semantic_observation_contract(contract)
    if boundary not in {"env", "model"}:
        raise ValueError(f"Unknown observation boundary: {boundary!r}")
    defaults = {
        "window_size": 30, "t_embed_dim": 32,
        "task_encoder_version": "legacy_v1", "t_embed_mode": "concat",
        "observation_transform": "raw", "observation_filter": "MeanStdFilter",
        "solver_state_mode": "off",
    }
    for key, default in defaults.items():
        if cfg.get(key, default) != expected[key]:
            raise ValueError(f"semantic observation contract conflicts with {key}")
    # Dimensions and temporal declarations may be derived, but never contradicted.
    for key in expected:
        if key in cfg and cfg[key] != expected[key]:
            raise ValueError(f"semantic observation contract conflicts with {key}")
    if actual_feature_columns is not None:
        if list(actual_feature_columns) != expected["feature_cols"]:
            raise ValueError("semantic observation contract conflicts with actual feature_cols")
    elif boundary == "model" and cfg.get("feature_cols") != expected["feature_cols"]:
        raise ValueError("semantic model requires explicit feature_cols")
    if actual_obs_shape is not None:
        if tuple(actual_obs_shape) != (expected["agent_obs_dim"],):
            raise ValueError(
                "semantic observation shape must be "
                f"({expected['agent_obs_dim']},), got {tuple(actual_obs_shape)}"
            )
    if boundary == "model" and cfg.get("solver_state_dim", 0) != expected["solver_state_dim"]:
        raise ValueError("semantic observation contract conflicts with solver_state_dim")
    return expected
