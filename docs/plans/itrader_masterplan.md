# iTrader Masterplan

Last reconciled: 2026-10-03
Status: **dataset-v2 R0 is the immutable negative/parity control; R0 + R4-v1 is architecture-qualified but economically unpromoted after G0 artifact-pair closure; fixed-recipe R1 attribution is next**
Branch snapshot: `agent/Watson-dev` plus the uncommitted R4-v1 implementation/evidence round

## North star

The [R1 development-readiness contract](../project-state/r1-development-readiness.md)
defines the next diagnostics and replay-identity work before implementation.
The representation screen includes R0, R0 + R4-v1, R1, and R1 + R4-v1 under
each explicit curriculum factor, completing the encoder-by-state comparison.
G0 artifacts establish paired byte parity; separate producer processes cannot
be independently verified from the historical receipts.

iTrader is a governed proposer/curriculum plus numerical solver system. Keep
learning, framing, deterministic gating, and execution separate.

The [iTrader Research Dashboard](../research/itrader_research_dashboard.md)
indexes wing identities, controls, dependencies, and evidence. It is a companion
registry, not a competing plan: this masterplan remains authoritative for phase
order and promotion.

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
- `pnl_linear` remains the historical/default optimization reward. It is learnable in the
  limited sense that PPO produced finite training signals and restorable policies;
  it is not yet proven to rank economic performance. Sharpe variants remain
  challengers and are not a recurrence prerequisite.
- `net_return` is now an explicit challenger objective: gross arithmetic return
  minus configured turnover and terminal-liquidation costs, multiplied by one
  constant PPO scale. At the default 1 bp its episode sum divided by the scale
  exactly matches `EpisodeEvidence.net_return`; it does not alter broker or live
  execution semantics. PPO discounting with `gamma < 1` remains an explicit
  difference from the undiscounted episode sum.
- Family `00` is the only enabled learned family. `01`/`02` are bootstrap
  contracts; `03` is trade management; `04` is abstention/WAIT semantics.

### Dataset-v2 baseline reset

Dataset v1 is now characterized rather than assumed. In an authenticated Alpaca
SIP overlap, over 99.7% of audited closes matched after timestamp correction,
but every v1 OHLC row was flat, volumes commonly appeared as exact multiples,
and naive local timestamps shifted summer data by one hour. Old close-to-close
PnL was therefore based on a genuine close path, not synthetic prices, while the
input representation, volume, provenance, and earlier evidence seams were not
sufficient for an economic promotion claim.

The immutable v2 snapshot contains 10,370,064 unique one-minute Alpaca SIP bars
for AAPL, MSFT, AMZN, NVDA, AMD, AVGO, GOOGL, META, SPY, QQQ, IWM, and TLT from
2022-01-03 through 2026-09-22. It is split-adjusted, not dividend-adjusted, and
partitioned by session/symbol/year. Phase A selects 5,515,061 RTH rows; extended
hours are retained for Phase B. The snapshot hash is
`fac41f6d42f8c5b1eac739755667bb38407461f82c168d7955de26bb2e5fb79c`.

The original fixed 32-episode v2 panel scores long-only at +0.05970% net and +0.00757
Sharpe/sqrt(bar), flat at zero, and short-only at -0.09970% net. A fresh 12-update
no-cost R0 bridge remained active but ended at -0.80750% net. Its matched 1 bp
optimization arm cut terminal turnover from 80.69 to 18.06, but did so by moving
to 95.23% short and still lost -0.29511% net. This validates reward
responsiveness, not economic edge. See the dated dataset-v2 R0 audit.

That original panel is retained for bridge comparability, not current selection:
each horizon used different starts while `semantic_v2` exposes horizon. The
canonical P0 panel instead crosses eight explicit RTH starts with all four
horizons. Its 1 bp controls are flat 0.00000% net/0 Sharpe, long-only -0.09785%/
+0.00827, short-only +0.05785%/-0.01558, random -1.31143%/-0.25857, and a
seeded untrained FiLM policy exactly equal to short-only. This removes the
horizon/start identity shortcut but remains a selection-informed development
panel, so P1 still requires new chronological blocks.

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
   align checkpoint cadence with perturbation cadence, and persist lineage. The
   PBT time attribute is now trial-owned cumulative sampled environment steps,
   so restoring a donor checkpoint cannot rewind the target's burn-in, cadence,
   or total exposure as restored `training_iteration` did near iteration 130.
4. **Implemented:** replace zero-byte Tune manifests/trial records with atomic non-empty evidence
   outside Tune-owned paths; summarize every population member.
5. **Implemented and run:** the capability-report schema consumes live flat
   trainer records. The official frozen-panel harness evaluates matched
   flat/long/short/random/untrained controls. On the 32 starts used by the latest
   bridge, long-only scored +0.03034% net with Sharpe +0.01077, flat scored zero,
   and the seeded untrained MLP-FiLM policy scored -0.31137% net. Multiple costs,
   independent training seeds, and block-aware uncertainty remain required.
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
7. **Implemented for the next probe:** fixed validation tasks are balanced over
   configured horizons; Ray 2.9 KL, value explained variance, total loss,
   gradient norm, learning rate, and KL coefficient are first-class report
   columns. Clip fraction remains explicitly unavailable on this legacy learner.
   Cadence checkpoints write a best-held-out pointer for mandatory fresh-process
   replay, and the ablation harness supports explicit baseline-relative terminal
   promotion gates.
8. **Platform seam accepted:** Ray 2.58.0 is the forward R-series authority and
   Ray 2.9 is a frozen historical reproduction profile. Both runtime suites,
   deterministic initialization/control, synchronous exploit/restore lineage,
   and exact fresh replay pass. A three-seed smoke bridge confirms that both
   short runs remain economically negative and that same-seed PPO endpoints are
   not a valid cross-version parity requirement. R1 must use a Ray-2.58 R0
   control; a later `RLModule`/EnvRunner/Learner/ConnectorV2 modernization remains
   a separate factor.
9. **Completed but not promoted:** the matched four-member-per-arm 120-update
   bridge compared exact 1 bp `net_return` with historical `pnl_linear` under
   identical seeds, starts, data, exposure, and MLP-FiLM architecture. Mean
   terminal net return was -0.3309% for `net_return` and -0.8850% for
   `pnl_linear`. The former reduced damage primarily by moving toward inactivity;
   one member became exactly flat. The latter preserved action diversity but paid
   away gross edge through 66.6--94.6 turnover. Its best cadence checkpoint
   reached +0.04948% net but had negative Sharpe and 42.94 turnover, so it did
   not pass the matched long-only gate. PPO is active and reward-sensitive;
   economic reward learnability remains unproven.
10. **Ray 2.9 promotion-mode correction:** asynchronous PBT remains the legacy
    reproduction default, but promotion-oriented loadouts should set
    `pbt_synch: true`. Synchronous scheduling orders donor checkpoint creation
    before exploitation. Checkpoint indices are persisted before
    `session.report`, because a synchronous pause may not return through the
    reporting call. The final 12-update integration probe produced three durable
    checkpoint/index records per member, monotonic progress, real mutations, and
    no missing-donor warning.
11. **Completed but not promoted — synchronous bridge:** the 2026-09-17
    four-member-per-arm 120-update comparison used the same initial seeds,
    frozen 32-task validation identities, `stream_worker`, dataset, and 1 bp
    reporting cost. The first 1 bp arm completed; a user-requested pause left
    one PnL-linear member without a first model checkpoint, so only that arm
    was restarted from zero under a separate experiment name. The final
    all-member report includes the completed restart, not the interrupted
    prefix. Each member reached 576k–580k trial-owned steps and 12 durable
    checkpoints. Actual configuration changes numbered 10 and 8 across the
    respective arms; gamma stayed in 0.98–0.995. Terminal mean net returns
    were -0.14219% and -1.00048%, with all terminal Sharpes negative. Fresh
    replay exactly reproduced each arm's best checkpoint. PnL-linear member
    00001 briefly scored +0.06718% net at 1 bp, but Sharpe was -0.02301,
    turnover 26.1875, and its paired 14-block exploratory interval against
    long-only crossed zero. It turned negative at 2 bp. This is real episodic
    evidence, not a stable economic winner. Keep the final test sealed.
12. **Completed candidate discovery; independent promotion failed:** the
    2026-09-19 matched P0 compared 0 bp discovery, staged cost, and exact 1 bp
    optimization while evaluating every policy at 1 bp. Discovery members
    `00001` and `00003` ended at +0.33846% and +0.35476% net with positive
    Sharpe, positive results at every horizon, exact fresh replay, positive
    2 bp repricing, and regime-varying actions. They share ancestry. A broad
    independent population and a four-member independent winner-recipe
    population produced no qualifying checkpoint; the closest independent
    point barely beat short-only net but not Sharpe and did not persist. A
    subsequent eligibility-aware 40-update probe proved the opt-in PBT donor
    gate works but also failed. Keep both candidates as research checkpoints,
    not promoted winners, and keep the final test sealed.
13. **Completed 24k raw-net diagnostic and Ray-2.9 tail hardening:** a
    memory-safe four-member run produced two genuine exploits before the prior
    eight-actor topology exhausted the 16 GB Windows host. Sequential immutable-
    checkpoint recovery identified member `00002` at 372,216 steps as a strong
    transient challenger: +0.39258% 1 bp net, +0.02959 Sharpe, exact replay,
    +0.30008% at 2 bp, positive bullish/bearish regime returns, and two
    consecutive reports above short-only. Its paired block interval against
    short-only crossed zero and it later regressed, so it is not promoted.
    Ray now respects one rollout worker, accepts a declared local CPU cap,
    retries an empty scheduled donor save from the durable checkpoint index,
    reattaches paused upper donors to their own checkpoint, and suppresses
    mutation at terminal exposure. A final four-member live proof terminated
    every member with one legitimate boundary exploit, zero terminal exploits,
    and zero errored trials.

P0 candidate discovery and the repaired PBT mechanics are now sufficiently
measured to define P1, but not to claim promotion. P1 retains 1 bp evaluation
truth and tests independent lineages on genuinely new, non-overlapping
chronological validation blocks, with selection-aware block uncertainty,
regime-conditioned action attribution, and fresh per-step 0.5/1/2 bp replay.
Direction remains diagnostic rather than punitive; only effectively all-flat
behavior is an automatic action gate. Require sustained independent net and
Sharpe before freezing a candidate and opening the sealed final test. For any
externally paused overnight run, still use coordinated all-member checkpoints
or budget a fresh-arm restart; a saved Ray experiment state alone is not exact
recovery. The matched NVMe/current-storage A/B remains deferred and must not
relocate an active run.

The detailed findings are in
`docs/artifacts/research/phase-a-mlp-film-autoresearch-2026-09-19.md`, with the
earlier repair chronology in
`docs/artifacts/research/phase-a-performance-validity-audit-2026-09-12.md`.

Dataset-v2 supersedes dataset-v1 for new performance selection. The matched RTH
no-cost/staged/exact-cost P0 is complete and not promoted. The Ray-2.58 R0
platform gate is accepted; after review, run R1 beside a matched Ray-2.58 R0
control. P1 uses independent, non-overlapping chronological
blocks, paired block-aware uncertainty, regime-conditioned actions, and 0.5/1/2
bp replay. Frozen proposer/family uplift cannot be claimed until the R-series
solver/evidence contract is frozen. A frozen Qwen3.8 proposer then compares
against random/generated tasks at equal checkpoint and solver compute; adaptive
proposer reward remains later Phase B work.

The first dataset-v2 P0 launch stopped safely during the no-cost arm after four
genuine exploits: Ray 2.9 scheduled a save on an exploited actor before its
first post-restore result. A verification gate showed that the initial guard
covered exploit-target restoration but not stable self-reattachment. A second
fresh gate proved both restore paths through ~97k steps for all members, then
exposed a final Ray 2.9 ordering: a redundant actor save was scheduled while an
exact same-step durable donor fallback was available. Synchronous upper donors
now use that immutable sidecar checkpoint directly and fail closed when it is
missing. Gate C then showed that Ray can deliver one in-flight update after a
synchronous pause, leaving the selection row one update beyond a cadence-only
snapshot. Synchronous PBT therefore retains an immutable snapshot at every
report while exposing only cadence/terminal checkpoints to Tune. Gate D then
completed all four members at 170,152--170,416 trial-owned steps with 140
reports/checkpoints, five genuine exploits, five exact same-step donor receipts,
and zero trial errors, skips, or receipt lag. It crossed the former 102,080 and
120k failure regions and closes the first Ray 2.9 checkpoint/search seam. A later
forensic pass matched every retained Gate D restore copy to the selected donor
tree hash. New arms make the proof generation-bound: scheduler selection,
exactly one target restore receipt, a target report, and a target retained
checkpoint must all agree. Its strongest
point (`00000`, 29,232 steps) reached +0.08116% 1 bp net and +0.01576 Sharpe for
three consecutive positive-net/positive-Sharpe reports, then regressed; every
terminal row was negative. Gate D is not a matched economic result.

The crossed-panel staged-cost and exact-1-bp follow-ups are now complete through
their predeclared approximately-96k-step review gates. Several early probes were
invalidated when a synchronous target was pre-staged before the population
barrier and later restored target-local state despite a donor exploit. The
scheduler now vetoes early boundary resume, removes orphan target staging after
exploit, and fails closed if a paused exploit target already owns an actor. The
final exact-cost arm passed strict inspection with four members, three exploits,
three verified generation-bound restore lineages, and no stale or pending
evidence.

Economically, staged cost converged all four members to the matched short-only
control (+0.05785% net, -0.01558 Sharpe, turnover 2). Exact 1 bp ended with a
population mean -0.05317% net, -0.03815 Sharpe/sqrt(bar), 67.33% flat, 32.67%
short, 0% long, and turnover 6.609. Across 84 exact-cost reports only three had
positive net return, three had positive Sharpe, and none had both. PBT mutations
and weight transfers were valid; the result is a mechanically credible R0 basin,
not absent search. Further exact-cost R0 selection against the same panel is
stopped. R0 remains the compatibility/economic control for a matched R-series
challenger. The accepted Ray-2.58 platform gate now permits a Ray-2.58 R0/R1
comparison without confounding framework and representation.

The dataset reset and first v2 R0 evidence are in
`docs/artifacts/research/dataset-v2-r0-audit-2026-09-24.md`.

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

Keep R0 as the immutable parity control. Do not describe model registration as
model selection. R1–R3 remain unimplemented. `R4-v1-min` has a
backwards-compatible position/remaining-horizon seam, a completed blocking 4x3
gate, and a completed fresh 4x3x40 replication. On the fresh block, combined
state improved terminal net over masked in all seeds and passed the pooled
relative net gate, but every arm failed the absolute economic gate because of
control inferiority, Sharpe inconsistency, and near-flat collapse. The complete
connector is nevertheless architecture-qualified: `R0 + R4-v1` is the forward
feed-forward training baseline because it removes the known endogenous
control-state alias. Economic promotion remains blocked. The shared-trunk and
family-head arrangement remains open pending user review.

Historical provenance is authoritative:

| Cell | Identity | Status |
|---|---|---|
| R0 | MLP + implemented FiLM (`tanh(gamma)*hidden + beta`) | Implemented; immutable parity control |
| R1 | Reset-every-step window LSTM + FiLM; explicit market/task split | Unimplemented; must be correct/measurable |
| R2a | Stateful LSTM + task concat | Unimplemented recurrence baseline |
| R2b | Stateful LSTM + init-only FiLM | Deferred: task unavailable to current initial-state hook |
| R2c | Stateful LSTM + per-step FiLM | Active candidate; readout placement first |
| R2d | Init + per-step FiLM on value head only | Deferred; matched placements required |
| R3 | Stateful GRU + per-step FiLM | Later cell-type comparison |
| R4-v1 | Position + remaining-horizon state on matched controls | Architecture-qualified with R0; fresh 4x3x40 replication passed paired net attribution but failed absolute promotion |

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

The historical `legacy_v1` encoder omits reward parameters, bar resolution,
regime tags, difficulty, and metadata while exposing a seed feature. New runs
use versioned `semantic_v2`, which represents normalized solver-relevant task
fields while excluding episode seed/start location shortcuts; checkpoint
compatibility remains explicit. R4-v1-min adds position and normalized remaining
horizon through a fixed two-value block on the FF control. New replication uses
the named `zero_v1` residual initialization so the update-zero policy/value
matches R0; historical checkpoints retain `random_v1`. R1 must preserve a clean
state-free cell and an explicitly named `R1 + R4-v1` cross. Episode PnL, cost
basis, unrealized/realized PnL, holding duration, drawdown, accumulated turnover,
and market-session time are registered future factors with explicit ownership
and reset semantics, not current inputs. OHLCV-only recurrence cannot know
sampled actions, fills, or position exactly.

### Phase B implementation/evidence order

1. Preserve the completed dataset-v2 R0 cost study as a compatibility/economic
   control, not a validated performance winner. Do not continue selecting on
   the same crossed 32-episode panel.
2. Preserve the accepted Ray-2.58 gate. The local-worker two-member run closed a
   genuine exploit, exact donor digest, generation-bound target receipt,
   monotonic target clock, post-restore report, and exact fresh replay. The
   three-seed smoke bridge ended negative on both versions. Keep Ray 2.9 only for
   historical reproduction, run R1 beside a Ray-2.58 R0 control, and never change
   solver representation plus native RLlib stack in the same attribution cell.
3. Produce truthful task/family/regime metadata through the shared episode-evidence
   contract.
4. Keep reward, dataset, tasks, seeds, PBT cadence, and economic gates fixed while
   changing solver representation; do not attribute a cost-objective change to
   recurrence.
5. Implement and test R1 shape splitting, chronological newest-bar convention,
   FiLM readout, finite forward/backward, and smoke configuration.
6. Implement R2a native recurrent LSTM with state lifecycle tests.
7. Implement R2c readout FiLM and the named `R2c-CARRY`/`R2c-RESET` pair.
8. Only then consider the named pre-cell factor, R2b init-only, R2d value-only,
   and R3 GRU according to evidence.
9. Preserve the completed R4-v1 economic failures while using `R0 + R4-v1` as the
   architecture-qualified forward FF baseline. The fresh December-2025 4x3x40
   replication is complete with zero-init parity, exact replay, block-aware
   uncertainty, and counterfactual diagnostics. Combined state improved net over
   masked in all seeds and passed the pooled relative net gate, but failed
   long-only superiority, Sharpe consistency, and the near-flat ceiling. The
   subsequent January-2026 four-member PBT pair is also complete: causal cost
   staging converged exactly to short-only, while exact 1 bp from update zero
   converged exactly to flat; neither produced a replay-valid
   directional-control-beating candidate. The checkpoint-level follow-up found
   that all 81 jointly positive causal reports were at least 98.014% short and
   that the unexploited 77,832-step checkpoint combined high stochastic entropy
   with narrow greedy action margins and a repeatable one-action replay
   mismatch. Do not spend another full PBT population on the same January
   panel. The fresh-panel G0 diagnostic is complete: five unexploited member
   `00002` checkpoints have paired byte-identical action and reward artifacts,
   but every November-panel net return and Sharpe was negative. Historical
   receipts lack producer identity, so separate-process provenance is unknown.
   The least-negative 102,040-step checkpoint repeated essentially the same
   75/0/25 short/flat/long mix at every anchor, while later checkpoints were pure
   short. Freeze the lower-LR recipe across matched R0, R0 + R4-v1, clean R1,
   and R1 + R4-v1.
   Treat causal cost staging as an explicit matched factor and run PBT only
   after representation attribution. Cross fixed 1 bp and causal staging as
   explicit curriculum cells; neither is part of an R identity. Keep state-free R1 and the named
   `R1 + R4-v1` cross separate. R4-v2 nonlinear late fusion remains a later
   independent factor.
10. Run causal conditioning only as a separate C-series factor: C0 current
   features, C1 trailing train-fitted regime summaries, and shuffled-C1. Never
   train on the current future-derived `regime_tags`.
11. Keep the System-One/CLM/Jev S-series research-only until the R-series solver
   interface, platform, and evidence contract are frozen. Then run it as a named
   independent factor rather than mixing it into an R or C comparison.
12. Promote only after held-out chronological evaluation, multi-seed results,
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
