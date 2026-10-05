# iTrader Architecture Entry Point

Last reconciled: 2026-10-05
Branch snapshot: `feat(R)/Ray_upgrade`, sourced from the dirty `agent/Watson-dev` tree
Status: **active working architecture; R0 + R4-v1 is architecture-qualified but economically unpromoted, and R1-R3 remain unimplemented**

This is the compact compatibility entrypoint. Detailed architecture is in
`docs/architecture/itrader_system_architecture.md` (+ `.html`); phase ordering is
in `docs/plans/itrader_masterplan.md` (+ `.html`); research identities and
evidence navigation are in
`docs/research/itrader_research_dashboard.md` (+ `.html`).

## Implemented baseline

Ray 2.58.0 and its RLModule/Learner/EnvRunner API are the sole active training
stack. The active solver is a PyTorch-only core in
`models/solver_mlp_film_core.py`, hosted by the native adapter in
`models/ray_native.py`; the active trainer has no legacy-stack selector or
ModelV2/Policy fallback. Historical ModelV2/TorchPolicy adapters and corrected
Ray 2.58 legacy recipes are isolated in `compat/rllib_legacy/` for explicit
parity work. `compat/ray29/` remains the separate Ray 2.9 runtime reproduction
profile, not the home for Ray 2.58 legacy source. This is a mechanical platform
migration, not four-cell parity qualification or economic promotion. R1 remains
a separate factor and its selector fails closed until implemented.
`models/registry.py` owns framework-neutral solver identity and native restore
validation; `models/observation_contract.py` owns semantic observation checks.
`r0_r1_v1` pins OHLCV order, the 60-bar chronological window, task encoding,
NoFilter/market_relative_v1 and zero_v1 state initialization. The representation
profile named `legacy` is historical feature metadata, not an RLlib API choice.
PPO diagnostics and restore reconciliation belong to the native training
adapter; replay/config/data/source identity belongs to evaluation provenance,
not the env.

```text
itrader/rl_lib/train/train_solo_V2.py
-> ITraderEnv
-> TaskSpec / TaskSampler00
-> RewardEngine
-> SolverMLPFiLM
-> PPO/GAE metrics and manifests
```

`SolverMLPFiLMTensorCore` is the active/default solver math. Its native
`SolverMLPFiLMTorchRLModule` host splits
`[price_window_flat | task_embed]`, applies an MLP to the price slice, then uses
its implemented FiLM path:

```text
gamma = tanh(gamma_projection(task))
conditioned = gamma * hidden + beta_projection(task)
```

Training-time task dropout is part of that implementation. The current model
returns the unconditioned hidden representation when `conditioning != "film"`;
that setting is not a task-concat control. Do not document identity-centred
`1 + scale` semantics as if they were current behavior.

`RewardEngine` remains the environment-owned reward seam; it is not a mandatory
learned reward head or a reason to change policy/value-head architecture.

`ITraderEnv` now emits terminal `EpisodeEvidence` schema v1. Optimization reward
and named reward components remain separate from arithmetic gross return, cost,
net return, turnover including liquidation, drawdown, exposure, and Sharpe. A
symbol-safe `DataPartition` owns per-symbol chronological train/validation/test
intervals and purge. PBT selects only held-out validation net return; the final
test remains outside tuning.

Synchronous PBT schedules on trial-owned cumulative sampled environment
steps. Every evaluation report has an immutable stable sidecar checkpoint, while
Tune receives only cadence/terminal checkpoints. A donor is valid only when its
scored `pbt_progress_steps` exactly matches the selected sidecar; absence is a
fail-closed contract error. RLlib iteration and actor-local `timesteps_total`
remain lineage-local diagnostics, not population exposure clocks.

Promotion loadouts may bind an explicit content-hashed evaluation `TaskSpec`
panel. Every task must declare `start_idx`, the panel count must equal the
evaluation episode count, and reset uses those task-owned starts. The current P0
crosses each anchor with every horizon to prevent horizon from identifying a
different start. Restore provenance is generation-bound: scheduler selection,
one target receipt with the same full-tree checkpoint digest, a target report,
and a retained target checkpoint must agree before an exploit counts.
The native result callback records per-episode metric sums and counts; report
means are computed from their totals so unequal EnvRunner episode counts remain
correctly weighted. Native PBT progress uses Ray's sampled-step delta. Checkpoint
authority is selected by explicit `api_stack_id`: native RLModule state and
legacy TorchPolicy state have separate required-file schemas, and the ID is
copied through run manifests, checkpoint indexes, restore receipts, and replay
selection. The active native path uses explicit `NoFilter` observations and
rejects the legacy inert `MeanStdFilter` field.

Ray 2.58 is the accepted forward R-series runtime; Ray 2.9 remains a frozen
historical reproduction profile. The original migration gate passed 254 tests
on both runtimes and matched initial policy weights and deterministic R0 controls.
The later compatibility/evidence and fixed-entropy round passes 406 tests on Ray 2.58;
this is not a current dual-runtime suite claim. The
[historical optimizer audit](artifacts/research/ray258-historical-optimizer-audit-2026-10-05.md)
qualifies earlier PBT effectiveness: checkpoint transfers did not guarantee
that requested policy/optimizer settings reached the learner. Corrected-stack
R0/R4 fixed-recipe controls are a separate prospective experiment. RLlib stochastic
rollout streams differ under the same seed, so results remain runtime-labelled
and R1 must include a matched Ray-2.58 R0 control. Ray 2.58 native RLModule/
EnvRunner is the active stack on this migration branch; Ray 2.9 is a historical
reproduction profile. Keep runtime identity explicit and do not mix a runtime
change into a solver-representation attribution cell.
See the [native-stack implementation note](artifacts/research/ray-native-stack-implementation-2026-10-05.md)
for scope and bounded evidence; it does not claim economic promotion. Migration
coverage includes separate real-Ray tests for Algorithm-level train/save/restore
and inference, plus the official training CLI/Tune subprocess through a persisted
native run and checkpoint. These synthetic integration tests validate wiring,
not trading performance.

## Market-data authority

```text
provider adapter -> retained raw pages -> immutable Parquet snapshot
-> DuckDB query/catalog -> filtered Ray environment load
```

The implemented dataset-v2 acquisition adapter is read-only and separate from
future Alpaca brokerage/execution support. The immutable Parquet snapshot, not a
mutable DuckDB table, is the data authority. Physical partitions are
`session/symbol/year`; `ITraderEnv` requests only configured sessions, symbols,
and columns. `DataPartition` additionally prevents episodes from crossing
session or one-minute-continuity boundaries. Phase A uses RTH only. Pre-market
and after-hours are retained for named Phase B factors. Split-adjusted prices are
accepted; dividend cashflows remain unimplemented and must not be inferred from
silently dividend-adjusted prices.

Snapshot manifests retain retrieval time, usage classification, missing-bar
policy, and hashes for raw provider pages. Raw and manifest writes are atomic;
licensed data and credentials remain ignored while aggregate evidence is tracked.

## R-series implementation/status matrix

| ID | Historical identity | Current status | Attribution rule |
|---|---|---|---|
| R0 | `SolverMLPFiLM`, no cross-step state | **Implemented; immutable parity control** | Historical control for all solver studies |
| R1 | Reset-every-step window LSTM + FiLM, explicit market/task split | **Unimplemented** | Within-window temporal encoder only |
| R2a | Stateful LSTM + task concatenation | **Unimplemented / planned** | Original recurrence baseline |
| R2b | Stateful LSTM + FiLM initial state only | **Deferred** | RLlib `get_initial_state()` does not receive the task under the current contract |
| R2c | Stateful LSTM + per-step FiLM | **Active candidate / unimplemented** | Original per-step identity; placement remains open pending grill-me review |
| R2d | Init + per-step FiLM on value head only | **Deferred** | Requires matched policy/value placement controls |
| R3 | Stateful GRU + per-step FiLM | **Planned after LSTM controls** | Cell-type/compute comparison |
| R4-v1 | Matched explicit solver state: position + remaining horizon | **Architecture-qualified; 40-update replication complete** | `R0 + R4-v1` is the forward FF training baseline; paired net uplift passed, absolute economic promotion failed |

After the core R-series representation/lifecycle study, causal C0/C1/shuffled-
C1 remains a separate train-fitted market-representation factor. The
System-One/CLM/Jev S-series remains research-only until the R-series solver
interface, platform, and evidence contract are frozen.

Provenance is fixed by `git show 4732a98:docs/research/recurrent-solver-ablation.md`:
that document named init-only FiLM **R2b**, per-step FiLM **R2c**, and value-head-only
**R2d**. The simplified `bec5c77` naming of per-step FiLM as R2b maps to original
R2c; its old R4 init maps to original R2b. Do not silently repurpose those IDs.
Names are design identities, not executable flags.

## Causal controls and limits

- R1 versus a newest-bar stateful R2 cell changes input information, cold-start
  history, conditioning, and state lifetime. It is a useful engineering path,
  not an isolated carry-state proof.
- Add a train-from-scratch same-input/same-conditioning pair named
  `R2c-CARRY` and `R2c-RESET`: both use the same newest-bar input and explicit
  per-step FiLM readout; only the carry/reset lifecycle differs. These suffixes
  are controls, not new R-series identities.
- A zero-state evaluation probe is diagnostic only. It is an out-of-distribution
  intervention on a policy trained with carried state, not proof of causal memory
  use. Report it separately from matched training comparisons.
- Same tensor width does not mean the same information was consumed. Record
  window length, feature order, task fields, warmup policy, and state ownership.

## Recurrence contract

Separate the input window, BPTT/unroll (`max_seq_len`), episode horizon, and
reward-statistics window. A causal warmup may feed the prior `T-1` bars one at a
time before the first scored action; that is different from a cold start. Do not
count overlapping bars from successive observations as new chronological timesteps.
At real episode termination reset state; a truncated bootstrap difference is not
itself a reset, and rollout-chunk detach is not an episode reset. Valid sequence
lengths and padding must protect the last valid hidden **and cell** state and mask
padded losses. Training and inference must use the same state ordering, feature
timing, and reset rules. Vector workers own separate states; checkpoint and replay
formats must identify state ownership.

Post-cell/readout FiLM is readout-only only when no other task path (such as
concat) reaches the cell. If concat reaches the cell, the transition is already
task-conditioned and FiLM is an additional factor. Pre-cell/input FiLM remains a
separate placement factor. Readout-first is a proposed comparison order, not a
settled memory-conditioning decision.

## Phase B architecture decision status

Phase B covers both offline/curriculum work and online replay/shadow work. A
shared recurrent trunk with family-specific policy/value heads is a proposed
option under review, not a settled architecture. Families `00`–`04` remain the
roadmap; `00`–`02` may share `Discrete(3)` output width without sharing input
schemas, task semantics, reward assignments, or state ownership. Do not implement
these choices until the ongoing grill-me review confirms them.
## Task observability and online state

position, previous action, realized reward, unrealized PnL, or episode progress.
No market-only RNN can reconstruct sampled actions/fills/position exactly. The
accepted `R4-v1-min` FF seam is the fixed
`[position, remaining_horizon_fraction]` block. Its blocking three-seed,
12-update masked/position/time/both gate is replay-valid: time-only showed a
  positive paired net effect with an anchor interval excluding zero, but no arm
  had positive three-seed net and Sharpe or beat the controls. A later decision
  architecture-qualifies the complete `R0 + R4-v1` pair as the forward
  feed-forward training baseline because it removes the known endogenous
  position/termination alias; this does not reverse the failed economic verdict.
  R0 remains the parity control. R1 must keep state-free and explicitly named
  `R1 + R4-v1` cells so recurrence and observability remain attributable.

  The completed 4x3x40 replication used the named `zero_v1` residual
  initialization profile and preserved update-zero policy/value parity. Combined
  state improved net over masked in all seeds and passed the pooled relative net
  gate, but failed control superiority, Sharpe consistency, and the near-flat
  gate. Episode PnL, accounting, holding,
  drawdown, turnover-budget, and session-clock features are registered future
  factors rather than current inputs. Nonlinear market/task × state late fusion
  is reserved for separately named R4-v2.

`TaskEncoder` is versioned. `legacy_v1` preserves historical checkpoint behavior.
New runs use `semantic_v2`, which represents solver-relevant normalized task
semantics and excludes episode seed/start as a location shortcut. Do not restore
an old checkpoint under the new encoding without an explicit migration.

## AZR boundary and online shell

`train_AZR.py` currently creates a task pool and hands it to PPO. It is a
bootstrap handoff, not a co-adaptive proposer/solver loop. iTrader intentionally
uses a language-model proposer and numerical PPO solver with historical OHLCV;
call this **AZR-inspired adaptive curriculum learning**, not zero-data AZR.

The future online path remains deterministic at the shell:

```text
market window -> MarketFramer -> CandidateTask set + WAIT
-> solver scoring -> DecisionGate -> RiskEngine -> Executor -> DecisionRecord
```

A shared recurrent state must not be mutated repeatedly while scoring multiple
task candidates. Use task-independent market state or candidate-specific
state/replay ownership before online deployment.

## Promotion gates

R1 must be correct and measurable; it need not win before true recurrence is
tested. No recurrent cell becomes the default until lifecycle tests, matched
train-from-scratch seeds, held-out chronological evaluation, throughput/memory
measurements, and checkpoint/replay state checks pass. No claim of calibrated
confidence, expected net utility, causal memory, or profitability follows from
PPO logits/value outputs alone.
