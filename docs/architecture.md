# iTrader Architecture Entry Point

Last updated: 2026-09-24
Branch snapshot: `agent/Watson-dev` with the uncommitted dataset-v2 implementation round
Status: **active working architecture; recurrence and trunk/head design pending grill-me review**

This is the compact compatibility entrypoint. Detailed architecture is in
`docs/architecture/itrader_system_architecture.md` (+ `.html`); phase ordering is
in `docs/plans/itrader_masterplan.md` (+ `.html`).

## Implemented baseline

```text
itrader/rl_lib/train/train_solo_V2.py
-> ITraderEnv
-> TaskSpec / TaskSampler00
-> RewardEngine
-> SolverMLPFiLM
-> PPO/GAE metrics and manifests
```

`SolverMLPFiLM` is the active/default solver. The model splits
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

Synchronous Ray 2.9 PBT schedules on trial-owned cumulative sampled environment
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
| R0 | `SolverMLPFiLM`, no cross-step state | **Implemented/default** | Control for all solver studies |
| R1 | Reset-every-step window LSTM + FiLM, explicit market/task split | **Unimplemented** | Within-window temporal encoder only |
| R2a | Stateful LSTM + task concatenation | **Unimplemented / planned** | Original recurrence baseline |
| R2b | Stateful LSTM + FiLM initial state only | **Deferred** | RLlib `get_initial_state()` does not receive the task under the current contract |
| R2c | Stateful LSTM + per-step FiLM | **Active candidate / unimplemented** | Original per-step identity; placement remains open pending grill-me review |
| R2d | Init + per-step FiLM on value head only | **Deferred** | Requires matched policy/value placement controls |
| R3 | Stateful GRU + per-step FiLM | **Planned after LSTM controls** | Cell-type/compute comparison |
| R4-v1 | Matched explicit solver-state features on feed-forward and recurrent controls | **Planned, versioned** | Not the historical task-conditioned-init R4 |

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
No market-only RNN can reconstruct sampled actions/fills/position exactly. When
studying solver-state observability, add typed, normalized fields to both the
feed-forward and recurrent controls. Keep `R4-v1` separate from R-series
conditioning claims.

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
