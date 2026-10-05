"""Stable identities for deliberately separate RLlib API-stack profiles."""

NATIVE_API_STACK_ID = "ray258_native_rlmodule_learner_envrunner_v1"
LEGACY_API_STACK_ID = "rllib_modelv2_torchpolicy_v1"

__all__ = ["LEGACY_API_STACK_ID", "NATIVE_API_STACK_ID"]
