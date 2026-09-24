# iTrader Masterplan

Last updated: 2026-09-14
Status: **Phase A checkpoint/evaluation parity repaired and corrected 200-iteration PBT completed negative; deterministic baseline-controlled revalidation is the next P0**
Branch snapshot: `agent/Watson-dev` at `bec5c77`

## North star

iTrader is a governed proposer/curriculum plus numerical solver system. Keep
learning, framing, deterministic gating, and execution separate.

```text
offline: historical OHLCV -> proposer/curriculum -> validated TaskSpec pool
-> ITraderEnv -> numerical PPO solver -> held-out capability evidence
online: market window -> MarketFramer -> CandidateTask + WAIT -> solver score
-> DecisionGate -> RiskEngine -> Executor -> DecisionRecord
```

This is AZR-inspired adaptive curriculum learning, not literal zero-data AZR:
historical OHLCV is external data, the proposer and numerical PPO solver are
intentionally separate, and `train_AZR.py` is currently a bootstrap handoff.

## Phase A — implemented solver baseline under performance revalidation

**Status: measurement and PBT seams repaired; optimizer health is provisional and trading performance is not validated.**

- `train_solo_V2.py` selects `SolverMLPFiLM`.
- `ITraderEnv` emits chronological `price_window_flat + task_embedding`.
- `TaskSpec` is canonical; `RewardEngine` owns finite reward computation.
- `pnl_linear` remains the default optimization reward. It is learnable in the
  limited sense that PPO produced finite training signals and restorable policies;
  it is not yet proven to rank economic performance. Sharpe variants remain
  challengers and are not a recurrence prerequisite.
- Family `00` is the only enabled learned family. `01`/`02` are bootstrap
  contracts; `03` is trade management; `04` is abstention/WAIT semantics.

The 2026-09-12 performance-validity audit restored both historical iteration-200
PBT checkpoints and replayed each on 288 symbol-safe diagnostic episodes. The
final-score target averaged +0.0163% gross but -0.0728% net at 1 bp per turnover
unit; the peak-score target averaged -0.0112% gross and -0.0950% net. Both were
overwhelmingly flat. These are in-sample diagnostics because the historical run
used the full Parquet file, but they disprove the interpretation that the shaped
PBT score was itself profitable return.

The different historical gammas confirm that the long PBT run did mutate
configurations after checkpoints became available. Ray's default numeric
perturbation multiplied categorical gamma values by 0.8/1.2 without domain
validation: one target ended at `gamma=0.6272`, while the other ended at the
invalid `gamma=1.188`.

### Mandatory Phase A revalidation gate

The engineering portion of this gate is now implemented and smoke-validated on
the official random/generated ablation path. `EpisodeEvidence`, symbol-safe
purged partitions, `semantic_v2` task encoding, held-out net-return selection,
bounded/diverse PBT, aligned checkpoints, equal exposure, restore-aware lineage,
atomic stable provenance, all-member summaries, and the flat-schema capability
report are present. The bounded two-member run ended at exactly 12 unique
iterations and 13,520 timesteps per member and recorded real exploit/restore
events. This establishes seam correctness, not trading performance.

Before Phase B may rely on R0 behavior as a performance baseline:

1. **Implemented:** emit a public versioned `EpisodeEvidence` record with independent optimization
   reward, raw gross return, explicit costs/net return, turnover including terminal
   liquidation, drawdown, action counts, task identity, symbol/split, and provenance.
2. **Implemented:** make reset/evaluation symbol-safe and use per-symbol chronological
   train/validation/final-test ranges with lookback/outcome purge. The final test
   remains unavailable to PBT and proposer updates.
3. **Implemented:** constrain PBT domains, diversify member seeds and initial configurations,
   align checkpoint cadence with perturbation cadence, and persist lineage.
4. **Implemented:** replace zero-byte Tune manifests/trial records with atomic non-empty evidence
   outside Tune-owned paths; summarize every population member.
5. **Partly implemented:** the capability-report schema now consumes live flat
   trainer records; matched flat/long/short/random/untrained baselines under
   multiple cost assumptions and training seeds remain required.
6. **Completed but not promoted:** the four-member, 200-iteration Phase A parity
   PBT run `pbtA26_parity200_20260914_000736_pbt200` selected on held-out net
   return, never shaped reward. Every final net-return and Sharpe observation was
   negative. A follow-up isolated a Ray 2.9 restore defect: `Algorithm.restore()`
   silently re-enabled exploration on the evaluation policy. PBT now forces
   deterministic post-restore evaluation, uses the deterministic
   `market_relative_v1` transform with `NoFilter`, limits smoothing to the frozen
   panel, and writes a durable checkpoint copy outside Tune ownership. A final
   six-iteration probe and fresh-process replay reproduced the selection metrics
   exactly. The corrected candidate-generation run
   `pbtA26_parity200_checkpoint_valid_20260914_041907_pbt200_checkpoint_valid`
   completed all four members at iteration 200 and approximately 990k retained
   environment steps each. Every terminal held-out net return and Sharpe was
   negative; the best was -0.0556% net with Sharpe -0.00060 and 91.3% flat
   actions. Its durable checkpoint reproduced net return, Sharpe, and turnover
   exactly in a fresh process. Promotion remains blocked on frozen baselines,
   uncertainty, costs, and final test.
7. **Dependency migration completed; native stack still gated:** Ray/RLlib
   2.58.0, Gymnasium 1.2.2, and PyArrow 17.0.0 are aligned across the root lock,
   Conda files, constraints, and cluster setup. A control plus six-iteration PBT
   run exercised a real exploit/restore, and a fresh process reproduced the
   post-exploit checkpoint's held-out metrics exactly. R0 remains explicitly on
   the legacy ModelV2/RolloutWorker stack until the
   `RLModule`/EnvRunner/Learner/ConnectorV2 port passes tensor/update, metrics,
   evaluation, checkpoint, and PBT parity.

The detailed findings and TDD slices are in
`docs/artifacts/research/phase-a-performance-validity-audit-2026-09-12.md`.

## Phase B — offline + replay/shadow work

**Status: active scope; implementation and ordering remain under review.**

Phase B includes both offline/curriculum work and online replay/shadow validation.
It is not limited to proposer integration, and replay/shadow is not deferred until
Phase C. Live broker execution remains outside this Phase B scope.

The solver/trunk/head structure is still an open design question. A shared
recurrent trunk with family-specific policy/value heads is a proposal for the
ongoing grill-me review, not an accepted architecture. Families `00`–`04` remain
the roadmap; `00`–`02` may share `Discrete(3)` output width without settling
input schemas, task semantics, reward assignments, or state ownership.


### Solver standard (proposed, not settled)

Keep R0 as the control. Do not describe model registration as model selection.
R1–R3 are unimplemented, and R4 is versioned as `R4-v1`. The shared-trunk and
family-head arrangement remains open pending user review.

Historical provenance is authoritative:

| Cell | Identity | Status |
|---|---|---|
| R0 | MLP + implemented FiLM (`tanh(gamma)*hidden + beta`) | Implemented/default |
| R1 | Reset-every-step window LSTM + FiLM; explicit market/task split | Unimplemented; must be correct/measurable |
| R2a | Stateful LSTM + task concat | Unimplemented recurrence baseline |
| R2b | Stateful LSTM + init-only FiLM | Deferred: task unavailable to current initial-state hook |
| R2c | Stateful LSTM + per-step FiLM | Active candidate; readout placement first |
| R2d | Init + per-step FiLM on value head only | Deferred; matched placements required |
| R3 | Stateful GRU + per-step FiLM | Later cell-type comparison |
| R4-v1 | Matched explicit solver-state features on FF + recurrent controls | Separate observability study |

The `bec5c77` “R2b per-step” label maps to original R2c. The older
init-oriented R4 maps conceptually to R2b; `bec5c77` instead used R4 for
matched observation features. IDs are not executable flags, and this
proposed reconciliation has not been silently promoted to a new canonical schema.

### Conditioning placement (open proposal)

Use explicit placement names: R2a input concat, `R2c-precell` input/gate FiLM,
R2c post-cell/readout FiLM, R2b initial-state FiLM, and R2d value-only placement.
A post-cell/readout FiLM claim is readout-only only when no other task path (such
as concat) reaches the cell; with concat, the transition is already task-
conditioned and FiLM is an additional factor. Readout-first is a proposed order,
not a settled memory-conditioning decision.

### Attribution controls

R1 versus newest-bar R2 changes input history, cold-start/warmup, conditioning,
and state lifetime; it is an engineering progression, not an isolated memory
proof. Add train-from-scratch same-input/same-conditioning `R2c-CARRY` versus
`R2c-RESET` before attributing gains to carry state. A zero-state evaluation probe
is an OOD diagnostic, not a causal estimate.

### Lifecycle contract

Keep input-window, BPTT/unroll (`max_seq_len`), episode horizon, and
reward-statistics window distinct. Define cold start versus causal initial-window
warmup. Do not count overlapping bars as new steps. Reset state when the
environment actually resets at real termination; a truncated bootstrap difference
is not itself a reset, and rollout-chunk detach is not an episode reset. Specify
vector-worker ownership, training/inference parity, checkpoint/replay state, and
reset semantics before training.

### Task and feature controls

The default encoder omits reward parameters, bar resolution, regime tags,
difficulty, metadata, and start index while exposing a bounded seed feature.
Freeze omitted fields initially or create a separately versioned semantic encoder;
report observability. R4-v1 adds explicit position/progress/PnL fields identically
to feed-forward and recurrent controls. OHLCV-only recurrence cannot know sampled
actions, fills, or position exactly.

### Phase B implementation/evidence order

1. Complete the mandatory Phase A revalidation gate and preserve R0 as a
   compatibility control, not a validated performance winner.
2. Produce truthful task/family/regime metadata through the shared episode-evidence
   contract.
3. Complete alignment/reward controls on R0; keep reward changes independent of
   recurrence. Treat the staged reward file as existing/unreviewed here.
4. Implement and test R1 shape splitting, chronological newest-bar convention,
   FiLM readout, finite forward/backward, and smoke configuration.
5. Implement R2a native recurrent LSTM with state lifecycle tests.
6. Implement R2c readout FiLM and the named `R2c-CARRY`/`R2c-RESET` pair.
7. Only then consider the named pre-cell factor, R2b init-only, R2d value-only,
   and R3 GRU according to evidence.
8. Run `R4-v1` with identical explicit state features on R0 and the winning
   recurrent control.
9. Promote only after held-out chronological evaluation, multi-seed results,
   state norms, sequence-mask checks, compute/memory, and checkpoint/replay tests.

R1 need not win before R2 is tested. It must be correct and measurable.

## Phase B — AZR-inspired curriculum boundary

**Status: bootstrap handoff implemented; co-adaptive loop planned.**

The local AZR v2 source's Eq. 4 proposer reward is zero at `p=0` or `p=1`,
otherwise `1-p`; it is not a symmetric 50% peak and not measured learning
progress. iTrader should retain deterministic validation, reference/task buffers,
and role-specific evidence without claiming reproduction.

Proposer study is separate from R-series solver study:

1. fixed rule/random validated curriculum;
2. frozen LLM proposer;
3. frozen LLM with solver-feedback/reference-conditioned selection;
4. adaptive proposer only after repeatable utility/uplift evidence.

Use frozen-versus-adaptive reference-buffer updates as a named factor. Adaptation
uses validation; reserve an untouched final chronological test. Keep a cost-aware
verifier and report generation calls, rejected tasks, tokens/latency, accepted
solver timesteps, cost, turnover, and drawdown.

## Phase C — operational coupled shell

**Status: future operationalization; not the first replay/shadow phase.**

Phase B already includes offline/curriculum work and online replay/shadow
validation. Phase C means hardening the coupled stack for broader operational use:

- `CurriculumProposer` and `MarketFramer` are operational;
- online `DecisionGate`/`RiskEngine`/`Executor` shell is hardened;
- `WatsonOW` can consume iTRADER as a skill;
- proposer fine-tuning can use an accumulated, audited corpus later.

Phase C does NOT require immediate replacement of the solver with an LLM.

## Phase D — optional Market-JEPA

**Status: future/advisory.**

Keep representation learning behind the stable market-window and candidate-task
contracts. Do not expand to JEPA/MoE as a substitute for causal recurrence
controls or truthful metrics.

## Promotion rules

No reward, solver, proposer, or online claim is promoted from training curves
alone. Use matched data/task identities, chronological validation/test separation,
explicit costs, drawdown, turnover, state lifecycle evidence, and a written
attribution statement for every comparison.

`custom_metrics/episode_return_mean` is currently accumulated shaped optimization
reward and must not be interpreted as economic return. Historical checkpoints
`00002` and `00004` are quarantined as compatibility/negative-test fixtures until
the Phase A gate passes. Phase B architecture work may continue independently,
but no Phase B capability or uplift claim may use their old PBT scores as a
validated performance baseline.
