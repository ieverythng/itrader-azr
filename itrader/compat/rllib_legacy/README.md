# Historical RLlib legacy API adapters

This is a non-default compatibility capsule for Ray's `ModelV2` / `TorchPolicy`
API. New training, evaluation, inference, checkpoint validation, and reporting
must use the native Ray 2.58 `RLModule` / `Learner` / `EnvRunner` path in the
active `itrader.models` and `itrader.rl_lib` packages.

The adapters here are retained for checkpoint/API parity investigations only.
They are not a second supported trainer, and active modules must not import
them. The former stack selector and mixed `ray_api_stack` bridge are retired.
Historical experiment recipes that selected this stack live beside this file
and are rejected by the active ablation runner.

Checkpoint digests are stack-qualified and never inferred from directory
contents: native runs use `ray258_native_rlmodule_learner_envrunner_v1`; legacy
parity inspection must explicitly use `rllib_modelv2_torchpolicy_v1`. The
legacy identity validates Policy state only and cannot authorize an active
native restore.

`compat/ray258` remains the isolated Ray 2.58 environment fixture. It is not
the name for legacy source. `compat/ray29` remains the distinct Ray 2.9 runtime
reproduction profile. For exact historical training/replay, use the source
revision and dependency profile recorded by that run's manifest; do not try to
replay an old run by switching the active trainer back to `ModelV2`.

The compatibility tests are intentionally outside the normal
`itrader/tests` acceptance suite. Run them explicitly when investigating the
old API: `pytest itrader/compat/rllib_legacy/tests`.
