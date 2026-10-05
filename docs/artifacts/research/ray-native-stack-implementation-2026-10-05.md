# Ray 2.58 Native RLlib Stack Migration

Date: 2026-10-05

Branch: `feat(R)/Ray_upgrade`

Source: isolated feature worktree populated from the dirty `agent/Watson-dev`
working tree at commit `d3eb4c517075b1b85506f5500a9a5552d6186e3f`. The imported
`itrader/**/*.py` snapshot was verified by SHA-256
(`b2739469a5a601aceb95b69e19d5f08ce44e8b8763b94300ebb7ea4f79234e95`) before
native-stack edits; the live Watson checkout was not modified.

Status: **active training is native-only on this branch; historical legacy
adapters/recipes are separated for explicit parity. Four-cell scientific
qualification and economic promotion remain pending.**

## Scope and implementation

The authoritative package migration remains Ray/RLlib 2.58.0, Gymnasium 1.2.2,
and PyArrow 17.0.0. This work does not introduce an additional Ray version or
unnecessary pandas, NumPy, or PyTorch upgrades. `uv.lock` is imported from the
dirty Watson source snapshot.

The active trainer always selects the Ray 2.58 RLModule/Learner/EnvRunner stack;
there is no `rllib_api_stack` CLI/config selector and no ModelV2/TorchPolicy
fallback. The shared solver math is a PyTorch-only core, hosted by the native
RLModule. Historical ModelV2/TorchPolicy adapters and Ray 2.58 legacy recipes
are isolated in `itrader/compat/rllib_legacy/`; the Ray 2.9 runtime profile stays
separate in `itrader/compat/ray29/`. The native adapter supports
`solver_mlp_film` only and fails closed on other solver selectors or Ray
versions. `TaskSpec`,
`ITraderEnv`, `RewardEngine`, and online risk/execution seams are unchanged.

The native bridge includes:

- a TorchRLModule with exploration, training, and inference forwards sharing
  the existing MLP-FiLM parameter layout, plus PPO value-function support;
  the PyTorch-only solver core owns the shared trunk/state value readout used by
  the native `ValueFunctionAPI`;
- conversion of stable repository experiment inputs to Ray 2.58 Learner field
  names before native configuration reaches RLlib;
- a `SingleAgentEpisode`/`MetricsLogger` callback adapter preserving project
  episode-evidence and action/reward metric names;
- one validated `sum`/`count` mean reducer shared by callbacks, report
  projection, and fresh-process replay;
- Learner-side clipping diagnostics from the same forward pass as PPO loss;
- native deterministic checkpoint inference, model-identity validation, and
  restore reconciliation that reapplies target settings/schedules while
  verifying optimizer moments remain unchanged;
- one shared checkpoint replay-input contract for both replay entrypoints,
  verifying actual checkpoint and dataset bytes plus the frozen source-panel
  identity before Ray starts;
- LearnerGroup-to-evaluation-EnvRunner synchronization for checkpoint-consistent
  evaluation.

## Bounded validation

The synthetic parquet integration test exercises one real PPO update, native
learner diagnostics and metrics, checkpoint save/restore, deterministic action
replay, changed-target donor restore reconciliation, and the native
checkpoint-consistent evaluation sync path. It runs R0 and R4-v1 state modes;
module-level tests check PyTorch-core/native logits and values plus native
optimizer configuration mapping. A separate real-Ray subprocess test invokes
the official trainer CLI/Tune path for one update and checks completed run
status, native stack identity, persisted reporting, and RLModule checkpoint
structure. Both use synthetic data and are included in the repository test
suite; the full command and results are recorded in the companion iteration
record.

The resulting checks are mechanical compatibility evidence only. They do not
establish economic improvement, full-data readiness, PBT search effectiveness,
or promotion. In particular, R1 is still unimplemented and this branch has not
completed the R0, R0 + R4-v1, R1, R1 + R4-v1 four-cell comparison. Keep the
native trainer as the sole active stack. Do not claim four-cell qualification
until that separate gate is completed and reviewed.

The pre-cutover baseline validation on the feature worktree completed with 416
tests passing. The required-dependency healthcheck passed on Python 3.11.13, Ray
2.58.0, Gymnasium 1.2.2, PyArrow 17.0.0, and CPU Torch 2.3.1; optional
`llama_cpp` and `transformers` are not installed in this environment.

## Runtime notes

Ray 2.58 emits its own RLModule deprecation warning while restoring a checkpoint
through Ray internals; direct construction of the application RLModule uses the
modern constructor contract. Ray's [Learner API](https://docs.ray.io/en/latest/rllib/rllib-learner.html)
is documented as alpha, so keep the exact runtime pin and bounded
restore/evaluation probes in the qualification matrix. Historical reproduction
uses the explicit compatibility capsule or the source revision recorded by the
run manifest; it is not a selectable active trainer mode.
