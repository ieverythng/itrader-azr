# Current State

Last updated: 2026-09-14
Branch: `feat(R)/Ray_upgrade`
Base HEAD: `42d1b3a`

## Working-tree boundary

This dirty branch now includes the Phase A measurement/PBT implementation round.
The previously staged Sharpe challenger is preserved as an experimental reward;
`pnl_linear` remains the sole default. The supplied AZR and recurrence audits are
source-grounded evidence, not executable instructions. Recurrent/LSTM solver work
remains out of scope for this implementation.

The same working tree now includes the Ray 2.58 dependency migration and its
acceptance evidence. The package/build migration is complete. R0 deliberately
keeps the legacy ModelV2/RolloutWorker stack selected until the native
RLModule/EnvRunner/Learner/ConnectorV2 port clears a separate parity gate.

## Current decision boundary

The user's Phase B scope includes both offline/curriculum work and online
replay/shadow work; Phase C is not the first replay boundary. Families `00`–`04`
remain a roadmap. `00`–`02` may share `Discrete(3)` width without settling input
schemas, task/reward semantics, or state ownership. Shared recurrent trunk plus
family-specific policy/value heads, recurrence placement, and restored R-ID
reconciliation remain proposed until the ongoing grill-me review confirms them.
`RewardEngine` remains the environment-owned reward seam, not a mandatory learned
reward head.

## Implemented baseline

- `train_solo_V2.py` selects `SolverMLPFiLM` directly.
- `ITraderEnv` emits chronological flattened OHLCV window plus task embedding.
- `TaskSpec` is canonical; `RewardEngine` owns reward calculation.
- `pnl_linear` is the only proven/default reward; Sharpe variants remain
  experimental.
- `train_AZR.py` generates/validates a task pool and hands it to solver training.
  This is bootstrap, not a co-adaptive AZR loop.
- Only source family `00` is enabled. Families `01`/`02` are bootstrap contracts;
  `03` is trade management; `04` is abstention/WAIT semantics.

## Phase A performance evidence

The current R0 implementation is a compatibility baseline, not a validated
performance winner. The 2026-09-12 audit restored both historical iteration-200
PBT checkpoints and measured each on 288 symbol-safe deterministic diagnostic
episodes. Checkpoint `00002` averaged +0.0163% gross and -0.0728% net at 1 bp per
turnover unit; checkpoint `00004` averaged -0.0112% gross and -0.0950% net. The
policies were 93.6% and 96.3% flat, respectively.

This does not disprove that `pnl_linear` is learnable as a PPO optimization
signal. It invalidates the stronger claim that the current training metric proves
economic performance: `custom_metrics/episode_return_mean` is accumulated shaped
reward, evaluation was in-sample, and the peak target has invalid `gamma=1.188`.

The initial official random/generated probe reproduced identical population
members, missed exploit boundaries, and zero-byte Ray-owned files. The repaired
probe now uses distinct training seeds/configs, one matched validation seed,
fixed rollout exposure, held-out net return at 1 bp as the PBT objective, bounded
mutations, aligned checkpoint/perturbation cadence, and restore-aware lineage.
Both PBT members completed exactly 12 unique iterations and 13,520 timesteps;
real exploit/restore events and mutated configs were recorded. Repository-owned
manifests, status, config history, checkpoint indices, metrics, plots, and
capability reports are non-empty outside Tune-owned paths.

The smoke probe does not validate performance. Four validation episodes per
iteration and two training seeds are deliberately insufficient; isolated
positive observations reversed by the final iteration. The overnight baseline-
controlled, multi-seed study and untouched final test remain the promotion gate.

On 2026-09-14, a parity-shaped preflight completed with four PBT members, nine
iterations, 16 held-out episodes per iteration, and approximately 44.3k–44.5k
environment steps per member. Three members recorded checkpoint-restore events
after the non-terminal iteration-6 exploit boundary. Aggregate net-return and
per-square-root-bar Sharpe plots completed headlessly. The best final burst point
was still negative (-0.115% net; Sharpe -0.0081), so it is seam evidence only.

The 200-iteration parity run
`pbtA26_parity200_20260914_000736_pbt200` completed all four members at roughly
0.99M environment steps each. Final held-out net returns ranged from -1.185% to
-0.598%; final per-square-root-bar Sharpes ranged from -0.1983 to -0.1415. The
run is valid evidence of negative economic outcomes under its reported policy
state, but not of solver uplift or reward learnability.

A checkpoint-boundary audit froze the 16-task evaluation panel and traced the
abrupt turnover jumps to Ray 2.9 restore semantics. Policy hashes were identical,
but `Algorithm.restore()` changed the evaluation policy's `explore` flag from
`false` to `true`; PBT therefore ranked stochastic post-restore actions. The
trainer now forces deterministic evaluation, explicitly synchronizes weights,
accepts Ray 2.9's direct evaluation payload, and limits smoothing to the 16-panel
episodes. PBT additionally uses checkpoint-free `market_relative_v1` observations
with `NoFilter` and copies checkpoints outside Tune ownership.

The final two-member, six-iteration official probe exercised a non-terminal
restore. Source turnover 56.625 became 56.25 after the clone's first update, not
roughly 127, and a fresh process reproduced the stable checkpoint's net return,
Sharpe, and turnover exactly. Checkpoint/evaluation parity is now green. The
four-member, 200-iteration run
`pbtA26_parity200_checkpoint_valid_20260914_041907_pbt200_checkpoint_valid`
completed on 2026-09-14 using `stream_worker`, random/generated tasks, the frozen
16-task held-out panel, deterministic evaluation, and 10-iteration
checkpoint/perturbation cadence. All members reached iteration 200 and roughly
990k retained environment steps. Terminal net returns were -0.0556%, -0.1069%,
-0.1866%, and -0.3906%; terminal Sharpes were -0.00060, -0.03145, -0.04089,
and -0.05376. The best member was 91.3% flat and its durable terminal checkpoint
reproduced net return, Sharpe, and turnover exactly in a fresh process. This is
valid candidate-generation evidence, not performance promotion.

The aggregate reporter initially aligned PBT histories by row position, which
misplaced terminal points after lineage rewinds. A red/green regression now
joins on the recorded `training_iteration`; regenerated CSVs and plots contain
all four iteration-200 observations. The next P0 is the deterministic
flat/long/short/random/untrained baseline panel on identical starts, followed by
multi-seed uncertainty and cost scenarios. Phase B must not consume R0 as a
performance winner yet.

Ray 2.58.0 is now the authoritative build across the root uv lock, Conda files,
constraints, and cluster setup; Gymnasium 1.2.2 and PyArrow 17.0.0 are aligned
with it. The isolated compatibility environment first proved the dependency
set, then the root environment passed control training, a real six-iteration PBT
exploit/restore, and zero-delta fresh-process replay. Unsloth remains relevant
only to a future LLM proposer fine-tune, not the numerical PPO/PBT solver.

## Solver truth

R0 is the only implemented/default solver. `SolverMLPFiLM` uses
`tanh(gamma(task))*hidden + beta(task)` and task dropout; non-`film` conditioning
currently ignores task features rather than concatenating them.

The checked-in `SolverLSTMModel` infers feature width from the full observation,
reshapes incorrectly when task embedding is appended, discards LSTM state, and
returns incoming state unchanged. It is not a stateful recurrent solver.

R-series design identities are preserved from commit `4732a98`:

- R1 reset-window LSTM + FiLM — unimplemented;
- R2a concat — unimplemented;
- R2b init-only FiLM — deferred because current initial-state hook has no task;
- R2c per-step FiLM — active candidate, unimplemented;
- R2d init+per-step value-head-only — deferred;
- R3 GRU per-step — later;
- R4-v1 matched explicit solver-state feature study — planned and versioned.

The old `bec5c77` per-step label R2b maps to original R2c; the old init R4 maps
to original R2b. `R2c-CARRY`/`R2c-RESET` are named same-input controls, not new
IDs.

## Required evidence before recurrence claims

- explicit `[price_window_flat | task_embedding]` split and newest-bar order;
- cold-start versus causal initial-window warmup documented;
- separate input-window, unroll/fragment, and episode/reward-stat windows;
- native pinned Ray/RLlib sequence/state contract with valid-length packing,
  padding-loss masks, and last-valid h **and c** preservation;
- real termination versus truncation bootstrap and fragment detach semantics;
- train/inference state parity, vector-worker ownership, checkpoint/replay state;
- same-input/same-conditioning train-from-scratch carry/reset control;
- R4-v1 fields added identically to FF and recurrent controls;
- chronological held-out validation/final test and cost-aware metrics.

A zero-state probe is an OOD diagnostic, not a causal memory proof. Any
post-cell/readout FiLM claim is readout-only only when no other task path (such as
concat) reaches the cell; with concat, the transition is already task-conditioned.
The placement and memory-conditioning order remain open. OHLCV-only recurrence
cannot reconstruct sampled action/fill/position.

## Task encoding contract

Historical checkpoints retain `legacy_v1`. New training uses `semantic_v2`, which
represents solver-relevant task semantics and excludes episode seed/start as a
location shortcut. Reset seeding, declared reward ranges, and normalized
overrides are covered by regression tests.

## Dependency and validation status

`pyproject.toml`/`uv.lock` are authoritative: Python 3.11, Ray/RLlib 2.58.0,
Gymnasium 1.2.2, PyArrow 17.0.0, and setuptools 69.5.1. The old 2.9/2.47 lines are
historical reproduction references.

The working Windows `.venv` runs the current suite and Ray 2.58 legacy stack. No recurrent
training result is reported here. The bounded Phase A probe validates effective
PBT search and evidence wiring, not reward learnability or trading performance.
The accepted migration smoke recorded a real PBT policy log and restore, while
stable repository-owned provenance remains authoritative. Ray child processes
also emitted Windows teardown access-violation messages despite a successful
driver; long runs still require Linux/WSL2 validation.

## Next priority

Pause all performance claims that depend on either historical PBT scores or the
completed negative parity run. Run the corrected PBT only as candidate
generation, then add deterministic flat/long/short/random/untrained baselines,
block-aware uncertainty, cost scenarios, and the untouched final-test evaluation.
Preserve R0 while this happens; recurrence work may continue in its separate
thread but must use the repaired evidence contract.
