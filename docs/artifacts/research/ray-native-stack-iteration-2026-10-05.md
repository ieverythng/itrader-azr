# Ray 2.58 native migration iteration record

Date: 2026-10-05<br>
Branch: `feat(R)/Ray_upgrade`<br>
Objective: make the active training stack native-only while retaining a small,
explicitly non-default compatibility boundary for historical parity. Keep the
Ray 2.9 dependency profile distinct. No economic or full-data experiment is part
of this migration loop.

## Baseline

- Before the native-only cutover edits, the feature worktree had a reported
  **416-passing-test** full suite and a successful required-dependency
  healthcheck on Python 3.11.13, Ray 2.58.0, Gymnasium 1.2.2, PyArrow 17.0.0,
  and CPU Torch 2.3.1. This is pre-mutation baseline evidence, not current final
  validation.
- The then-current native bridge was still opt-in beside a legacy default. That
  failed the user's full-migration contract even though bounded native PPO,
  restore, and inference probes existed.
- The shared `SolverMLPFiLM` math had already been split into a Ray-free PyTorch
  core, and the native RLModule consumed that core directly.

## Mutation and focused checks

The active `train_solo_V2` path now builds only the pinned Ray 2.58 native
configuration, uses native `lambda_`, `entropy_coeff`, `minibatch_size`,
`num_epochs`, `train_batch_size_per_learner`, and `num_env_runners` fields, and
rejects the retired stack selector. Legacy ModelV2/PPO helpers and corrected
Ray 2.58 recipes were moved into `itrader/compat/rllib_legacy/`; the mixed
`ray_api_stack.py` bridge was removed. The Ray 2.9 runtime profile was not
changed.

Reported focused validation during this mutation:

- `pytest itrader/tests/test_train_config.py`: **108 passed** after converting
  stale config, evaluation-sync, cost-schedule, and PBT assertions to native
  contracts.
- `pytest itrader/tests/test_policy_baselines.py itrader/tests/test_run_ablation.py`:
  **46 passed** after removing ModelV2-style inference fakes and rejecting
  historical selectors in active loadouts.
- Core/semantic/inference/native-diagnostics selection, with the real
  Ray-initializing native PPO integration test excluded:
  **62 passed, 2 deselected**.
- A read-only Ray 2.58 source/API audit found that scheduled reward overrides
  assumed `.envs` on the outer EnvRunner wrapper and incorrectly required an
  evaluation group. The implementation now calls through Ray's Gymnasium
  vector-wrapper to per-env `call`, and a synthetic `DictInfoToList` /
  `SyncVectorEnv` regression covers multiple training envs with no eval group.
- Native metric projection tests now distinguish an empty wrapper from a
  genuinely missing metric namespace; absent economics remain unavailable, not
  fabricated zeroes.

## Latest API-boundary checks

The active Ray 2.58 trainer now consumes only the native result tree and Tune
function-trainable contract: `learners` / `env_runners`, `Trial.local_path`,
`storage_path`, `tune.get_context()`, `tune.get_checkpoint()`, and `tune.report()`.
Legacy `sampler_results`, `info.learner`, `episode_reward_mean`, `local_dir`,
`Trial.logdir`, and Ray AIR session fallbacks were removed from the active path.
Ray 2.58's documented Function API uses the Tune entrypoints; the pinned runtime
also rejects `ray.train.get_checkpoint()` outside a Ray Train worker. See the
[Ray Tune Function API](https://docs.ray.io/en/latest/tune/api/trainable.html).

A synthetic `training_fn` integration probe now exercises the active orchestration
without starting Ray or training a policy: it supplies the native configuration,
projects native metrics, writes a structurally valid checkpoint, verifies the
checkpoint index is durable and hash-matched before Tune is reported, and checks
the terminal trial status. Separate real-Ray tests cover an in-process Algorithm
train/save/restore/inference round trip and an official CLI/Tune subprocess that
writes native run, metric, report, and checkpoint evidence.

Earlier focused validation before the independent-review follow-up:

- `pytest itrader/tests/test_train_config.py itrader/tests/test_pbt_scheduler.py
  itrader/tests/test_ps_metrics_native.py
  itrader/tests/test_solver_mlp_film_core.py
  itrader/tests/test_semantic_observation_contract.py
  itrader/tests/test_no_nan_forward.py`: **164 passed**.
- The real Ray-initializing PPO/checkpoint integration was not run during DEV's
  supervised bridge; no new Ray worker, replay, or market-data experiment was
  launched from this worktree.

## Independent-review follow-up

An independent source review found native-boundary gaps that the earlier
synthetic orchestration test did not cover: native custom metrics did not reach
the report projector, episode reductions were last-episode/mean-of-means,
progress used the retired sampled-step counter, GPU trials reserved but did not
assign a GPU to the learner, native checkpoints were checked against the legacy
Policy file layout, and `MeanStdFilter` was declared but not installed as a
native connector. A small inference helper also flattened batched logits into a
single action record.

The follow-up closes these boundaries with weighted `__sum`/`__count` metrics,
the native per-update sampled-step delta, `num_gpus_per_learner`, mandatory
`api_stack_id`-keyed checkpoint authority and matching replay provenance,
explicit native `NoFilter`, and a single-observation inference guard. Historical
TorchPolicy checkpoint checks remain available under their explicit legacy
stack identity. The solver's structural identity remains separate from full
run/checkpoint identity, which now includes the stack and observation schema.

Latest validation on the pinned Ray 2.58.0 Python 3.11 environment:

- `pytest -q itrader/tests/test_ray_native_stack.py -k
  test_native_ppo_checkpoint_restore_replays_deterministically -x`: **2 passed,
  5 deselected**. This starts real Ray, trains PPO, saves/restores, and verifies
  deterministic native inference and donor optimizer reconciliation.
- `pytest -q
  itrader/tests/test_ray_native_stack.py::test_native_training_cli_completes_and_persists_a_restorable_checkpoint
  -x`: **1 passed**. The official CLI subprocess completed one native Tune
  update and persisted run identity, report/metrics, and an RLModule checkpoint.
- Full test suite excluding those three Ray-starting cases: **414 passed,
  3 deselected**. This also confirms the duplicate-variant regression checks
  after fixing their undefined test-local fixture.
- Required-dependency healthcheck passed on Python 3.11.13, Ray 2.58.0,
  Gymnasium 1.2.2, PyArrow 17.0.0, and CPU Torch 2.3.1. Optional
  `llama_cpp`/`transformers` are absent; CUDA is unavailable.
- Working-tree `git diff --check` passed. Final full suite and pre-commit remain
  scheduled outside the supervised terminal-replay window; knowledge freshness
  and independent review are also still required.

## Holdout and acceptance decision

No market-data holdout or economic study was run. The holdout for this code-only
iteration is the installed Ray 2.58/Gymnasium 1.2 vector-wrapper contract plus
the separate synthetic Algorithm and CLI integration tests; no scientific model
claim is inferred. The separately supervised compatibility control/replay stays
outside this worktree's execution scope. Ray-starting checks must not overlap
that supervised run.

**Status: integration gates pass; delivery gates remain.** Still required on the
final diff: full repository suite and pre-commit with the pinned environment,
`git diff --check`, knowledge freshness, and the independent reviews required by
`REVIEW.md`. Then commit and push this feature branch for main DEV review.

## Independent-review closure pass

The adversarial review found and the follow-up closed three split-ownership
gaps before delivery:

- Both native checkpoint replay commands now use
  `validate_checkpoint_replay_inputs`; that shared contract verifies checkpoint
  bytes, dataset bytes, frozen evaluation-panel identity, resolved configuration,
  and native stack identity before Ray starts. A new regression first accepts a
  valid fixture, changes only the dataset bytes, then asserts rejection before
  restore.
- Callback, report projection, and fresh-process replay now share one validated
  `mean_from_totals` contract. Boolean/non-numeric and invalid denominator data
  cannot produce an accepted mean in one path while failing in another.
- Value-readout math lives in the PyTorch-only solver core and is called by the
  native RLModule's `ValueFunctionAPI`; the adapter no longer reproduces the
  residual calculation.

The review also caught two documentation lineage issues: the September 12 audit
now retains its original `bec5c775d3adaafe758202c7175791f6e8b323a8` source and
labels later evidence as a continuation, and the September 20 legacy-stack
guidance in current-state is explicitly historical/superseded. The compatibility
control record and current-state handoff carry the latest supervised closeout:
all eight training cells completed; after the first cell-8 replay was safely
stopped and preserved, a separate user-authorized retry completed all eight
terminal replays (256 fresh episodes total). Those are compatibility replay
outcomes, not native migration or economic-promotion evidence.

Follow-up focused validation with the pinned environment: the replay-integrity,
checkpoint, native-metric, and solver-core selection passed **16 tests**; the
separate native RLModule/core parity selection passed **2 tests**. Final
`pre-commit run --all-files` passed, including the healthcheck and full
repository test hook. All three independent source reviews returned
`VERDICT: APPROVE`. No new training or economic evaluation was launched from
this branch.
