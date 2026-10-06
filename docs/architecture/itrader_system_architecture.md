# iTrader System Architecture

Last reconciled: 2026-10-06
Branch snapshot: `agent/Watson-dev` with the uncommitted R4-v1 implementation/evidence round
Status: **active working architecture; R0 + R4-v1 is architecture-qualified but economically unpromoted, and R1-R3 remain unimplemented**

Research identities and evidence status are indexed in the
[iTrader Research Dashboard](../research/itrader_research_dashboard.md); the
[masterplan](../plans/itrader_masterplan.md) remains the phase-order and
promotion authority.

## 1. Source-grounded runtime

Current platform: Ray 2.58 with explicit ModelV2/TorchPolicy compatibility APIs.
The compatibility repair round adds shared structural model selection, opt-in
semantic OHLCV contracts, same-loss clip diagnostics and measured post-restore
learning-setting reconciliation. Scoped Sol/Astra evidence review approved these
fixes; no performance promotion or R1 implementation follows automatically.
The native RLModule/Learner/EnvRunner migration is deferred until R1 is compared
on this corrected stack; then repeat R0, R0 + R4-v1, R1 and R1 + R4-v1.
See [R1 readiness](../project-state/r1-development-readiness.md) for ownership,
the accepted Q15/Q18 model contract, Q19 selection and Q20 budgets. R1 remains
unimplemented; Q21 two-tier evaluation is accepted, with fresh-panel construction
and execution gates still pending. Native API development may proceed in an
isolated worktree; integration remains after the corrected-stack R1 comparison.

Q22 freezes the same corrected optimizer recipe across all 16 screen cells;
PBT stays off and entropy retains its original sampled-step schedule. Q23 panel
sampling is accepted; eligibility/freeze gates remain pending. The completed
R0/R4 one-update smoke and fresh replay
qualify only that bounded compatibility path, not R1 or economic performance.

The implemented offline path is:

```text
historical OHLCV parquet
-> ITraderEnv
-> chronological flattened market window + task embedding
-> SolverMLPFiLM
-> PPO/GAE update
-> EpisodeEvidence -> held-out metrics, stable provenance, capability report
```

Source anchors:

- `itrader/rl_lib/train/train_solo_V2.py` uses the shared model registry;
  `solver_mlp_film` remains the default. The R1 selector is reserved and fails
  closed until its implementation lands; registration alone is not availability.
- `itrader/envs/itrader_env.py` owns rolling-window assembly, task-conditioned
  reset, episode state, and reward context.
- `itrader/rl_lib/tasks/spec.py` owns `TaskSpec`.
- `itrader/models/solver_mlp_film.py` owns the active FiLM baseline.
- `itrader/rl_lib/train/train_AZR.py` generates a task pool then launches the
  solver: this is bootstrap handoff, not co-adaptive AZR training.

## 2. Observation and conditioning contract

The current observation is:

```text
obs_t = [price_window_flat(T * F) | task_embedding(D) | optional solver_state(S)]
price_window = reshape(price_window_flat, T, F)
newest_bar = price_window[-1]
```

The environment fills the window chronologically. R0 uses `S=0`; R4-v1 uses the
fixed `S=2` block. A candidate model must split these dimensions explicitly;
inferring `F` from the full observation width is invalid once conditioning or
solver state is appended. The legacy `SolverLSTMModel` does exactly that and is
both shape-incompatible for the usual `T=30,F=5,D=32` layout and non-stateful.

The active `SolverMLPFiLM` implementation is source-defined as:

```text
hidden = MLP(price_window_flat)
(gamma_raw, beta) = task_projection(task_embedding)
gamma = tanh(gamma_raw)
readout = gamma * hidden + beta
```

It also applies task dropout during training when configured. When its
`conditioning` is not `film`, `_condition_hidden` returns `hidden`; this is not
concat. An identity-centred `(1 + scale) * hidden + shift` parameterization would
be a new factor and must retain R0 as a control.

### Task embedding observability

`TaskEncoder` exposes an explicit compatibility boundary. `legacy_v1` retains
the historical family/price/reward/horizon/dimension/seed representation for
checkpoint replay. New training uses `semantic_v2`, which hashes canonical
solver-relevant fields including normalized reward parameters, bar resolution,
regime, difficulty, and metadata while excluding episode seed/start. Reward
overrides normalize before encoding. An encoding version is part of the resolved
configuration and must not change silently across checkpoint restore.

### Phase A evidence and partition contract

`ITraderEnv` owns the public terminal `EpisodeEvidence` record because it already
owns action-to-position transitions and episode accounting. `RewardEngine` emits
optimization reward; it does not define economic return. The evidence record
independently contains gross/cost/net arithmetic returns, turnover including
terminal liquidation, drawdown, exposure/activity, Sharpe per square-root bar,
task identity, symbol, split, and row bounds.

`DataPartition` precomputes valid contiguous starts per symbol for `train`,
`validation`, and `test`, accounting for lookback, task horizon, and purge rows.
Training samples only `train`; PBT evaluates with `explore=False` on a matched
validation seed. The final `test` partition is reserved for post-selection.
Annualized Sharpe is optional and requires an explicit periods-per-year contract;
the invariant metric is the unannualized per-square-root-bar value.

An evaluation run may load one explicit, content-hashed `TaskSpec` panel. Its
task count must equal `evaluation_num_episodes`, each task must own `start_idx`,
and reset uses those starts instead of sampler-generated identities. The current
P0 crosses eight starts with all four horizons so task horizon cannot identify a
different market path. This is a development-panel correction, not a substitute
for independent chronological validation.

Promotion-oriented PBT uses synchronous scheduling and a target-owned
cumulative sampled-environment-step clock. Every report writes an immutable
stable sidecar checkpoint before `session.report`; Tune receives only
cadence/terminal checkpoints. Donor selection must resolve an exact
`pbt_progress_steps` match or fail closed. RLlib iteration and actor-local
`timesteps_total` can change with lineage restoration and are not authoritative
population-exposure clocks.

An exploit is accepted only through a generation-bound chain: the scheduler's
selected donor path/digest, exactly one target restore receipt with that digest,
a later target report carrying the same restore generation/exploit ID, and a
same-generation retained target checkpoint. Historical event schemas require an
explicit legacy inspection mode and cannot satisfy a new-arm gate.

### Dataset-v2 authority and session contract

The historical-data path is a deep module with one public outcome—an immutable,
content-addressed snapshot—and provider-specific pagination/retry details hidden
behind an adapter:

```text
AlpacaHistoricalDataClient
  -> retained raw gzip response pages + build_request.json
  -> split-adjusted Parquet partitions (session/symbol/year) + manifest.json
  -> in-memory DuckDB catalog/query
  -> projected, predicate-filtered pandas frame
  -> ITraderEnv / DataPartition
```

Parquet is authoritative. DuckDB is a catalog/query engine, not a mutable source
of truth. The retained v2 snapshot records provider, feed, adjustment, time zone,
symbols, bounds, session counts, schema version and a content hash. Its
2026-09-22 manifest predates the builder's raw-file hash ledger. New manifests
also retain retrieval/usage/missing-bar provenance and raw-page hashes; old raw
calendar/pages require separate reviewed binding. Raw pages and manifests are
written atomically. Credentials
are read from ignored environment variables and must never enter manifests or
experiment artifacts.

`ITraderEnv` owns only the requested market slice and observation assembly. It
does not fetch network data. `DataPartition` respects configured boundary
columns and exact one-minute continuity so an episode cannot bridge an overnight
gap, session transition, missing interval, or early close. Phase A selects
`regular`; pre-market and after-hours are retained for a predeclared Phase B
ablation. Dividend accounting remains a future explicit cashflow seam in
environment evidence/reward, not an implicit price rewrite.

### Platform migration gate

Ray 2.58 is the accepted forward R-series runtime; Ray 2.9 is a frozen historical
reproduction profile. The original migration gate passed both then-current full
suites, deterministic initialization/control, synchronous exploit/restore lineage
and fresh replay. The subsequent
[optimizer audit](../artifacts/research/ray258-historical-optimizer-audit-2026-10-05.md)
found effective-setting drift despite successful checkpoint transfer. Current
compatibility repairs have scoped independent approval; the suite after the
fixed-run entropy-schedule addition passes 406 tests on
Ray 2.58; new PBT-off R0/R4 controls do not establish PBT search effectiveness.
The first
stochastic rollout differs under the same seed, so results stay runtime-labelled
instead of requiring bit-identical PPO paths. R1 must run beside a Ray-2.58 R0
control; the native RLModule/EnvRunner modernization remains a later independent
factor.

## Phase B architecture decision status

Phase B covers both offline/curriculum work and online replay/shadow work. A
shared recurrent trunk with family-specific policy/value heads is a proposed
option under review, not a settled architecture. Families `00`–`04` remain the
roadmap; `00`–`02` may share `Discrete(3)` output width without sharing input
schemas, task semantics, reward assignments, or state ownership. Do not implement
these choices until the ongoing grill-me review confirms them.

`RewardEngine` remains the environment-owned solver reward seam. It is not a
mandatory learned reward head, and reward computation must not be silently
conflated with policy/value-head design.

Ordering is explicit: platform parity first; R1/R2 representation and lifecycle
controls second; causal C0/C1/shuffled-C1 only after a stable R-series contract;
and System-One/CLM/Jev S-series work only after the solver interface, platform,
and evidence contract are frozen. C and S are named independent factors, not
ways to rescue an R-series result.


The historical identities below preserve the original file at commit `4732a98`.
They are design names, not current CLI flags, and do not settle the architecture.

| ID | Head and conditioning | State | Status and purpose |
|---|---|---|---|
| R0 | `SolverMLPFiLM` | None | **Implemented; immutable parity control** |
| R1 | Corrected reset-every-step window LSTM + FiLM | None | Unimplemented; test temporal inductive bias inside the supplied window |
| R2a | Stateful LSTM + task concat at recurrent input | `(h,c)` across steps | Unimplemented; original recurrence baseline |
| R2b | Stateful LSTM + FiLM initial state only | `(h,c)` across steps | Deferred; `get_initial_state()` lacks task input under current RLlib contract |
| R2c | Stateful LSTM + per-step FiLM | `(h,c)` across steps | **Active candidate; unimplemented**; placement remains open pending grill-me review |
| R2d | Init + per-step FiLM on value head only | `(h,c)` across steps | Deferred; needs matched policy/value placement controls |
| R3 | Stateful GRU + per-step FiLM | `h` across steps | Planned after LSTM lifecycle controls |
| R4-v1 | Explicit position + remaining-horizon state on matched cells | Cell-dependent | Architecture-qualified with R0 as the forward FF baseline; 4x3x40 replication complete, economic promotion failed |

Migration note: the simplified `bec5c77` docs called per-step FiLM R2b; that
label maps to original R2c. The former init-oriented R4 maps to original R2b.
The current R4 label is retained only as `R4-v1` for the matched state-feature
study. Historical IDs are not silently repurposed.

## 4. Causal attribution design

### 4.1 What R1 versus newest-bar R2 does and does not prove

R1 consumes a full window at every decision and resets its window encoder. A
newest-bar recurrent cell consumes one bar and carries state. The comparison
therefore changes accessible input history, cold-start warmup, conditioning path,
and state lifetime. It is a useful engineering progression, not an isolated
proof that carry-state caused an uplift.

Use two separately named same-input/same-conditioning controls, trained from
scratch:

- `R2c-CARRY`: newest-bar input, the chosen explicit per-step FiLM placement,
  carried `(h,c)` within real episodes.
- `R2c-RESET`: the same architecture, task embedding, input timing, optimizer,
  data, and seeds, but reset `(h,c)` at every scored call.

These suffixes do not create new R-series identities. They isolate the lifecycle
factor after the architecture is held fixed. Compare R1 to R2 as an engineering
path and `R2c-CARRY` to `R2c-RESET` as the carry/reset control.

A zero-state evaluation probe is a separate diagnostic. It intervenes on a
trained carry model and is out of distribution relative to training; a return
drop is not a causal estimate by itself. Report state permutation, carry/reset,
and zero-state results with that distinction.

### 4.2 FiLM placement vocabulary

| Placement | Task reaches | Returned recurrent state | Interpretation |
|---|---|---|---|
| Input concat (R2a) | Cell input | Raw `(h,c)` | Task is an input to the transition |
| Pre-cell/input FiLM (`R2c-precell`, planned factor) | New-bar projection or cell/gate input | Raw `(h,c)` | Candidate task-dependent transition; hold concat fixed for attribution |
| Post-cell/readout FiLM (R2c candidate) | Policy/value readout after cell | Raw `(h,c)` | Readout-only only if no other task path reaches the cell; placement remains open |
| Initial-state FiLM (R2b) | Initial `(h,c)` | Carried state | Deferred because task is unavailable to `get_initial_state()` in the current contract |
| Value-head-only init + per-step FiLM (R2d) | Value readout only by design | Matched raw state | Deferred until policy/value placement and capacity are matched |

For any placement, state the complete task path before attributing causality. A
post-cell/readout FiLM path is only a readout-only claim when no other task path
(such as concat) reaches the cell. If task concatenation reaches the cell, the
transition is already task-conditioned and the FiLM placement is an additional
factor. Readout-first is a proposed comparison order, not a settled user
decision; a memory-conditioning path remains open.
## 5. Time, state, and sequence lifecycle

Keep these windows distinct:

1. **Input window**: the market bars visible in one observation.
2. **BPTT/unroll (`max_seq_len`)**: the training sequence/chunk span; chunk
   boundaries may detach gradients without resetting environment state.
3. **Episode horizon**: the task/environment episode length and termination scope.
4. **Reward-statistics window**: history used by a reward statistic such as Sharpe;
   it is distinct from both the episode horizon and BPTT span.

A newest-bar cell may begin with a **cold start** (zero initial state and first
bar) or with a **causal initial-window warmup** (feed prior bars sequentially
before scoring the first action). The warmup is input processing, not extra
future information. Repeating overlapping observation bars as separate
chronological steps is invalid.

The current forward stack is Ray/RLlib 2.58.0 with Gymnasium 1.2.2 from
`pyproject.toml`/`uv.lock`; `itrader/compat/ray29/` preserves the historical
reproduction profile. The old `phase_b_plan.md` Ray 2.47.1 line is historical
ordering, not an active lock. Do not invent `recurrent_policy=True` or other
flags. Use the native recurrent model/state contract of the accepted platform.

Required lifecycle semantics:

- define state ordering and shapes for `(h,c)`;
- provide native initial-state and recurrent-forward hooks;
- use valid sequence lengths and padding masks;
- preserve the last valid **h and c**, not only h, when packing/chopping;
- mask padded policy/value losses;
- reset state when the environment actually resets at real episode termination;
- do not treat a truncated bootstrap difference as an episode reset;
- distinguish rollout-chunk detach/truncated BPTT from a real environment reset;
- keep vector-worker state ownership separate;
- use identical feature timing/state ordering in training and inference;
- test checkpoint restore and replay state ownership; weights alone are not a
  proof that hidden rollout state was restored.

`batch_mode="complete_episodes"` describes collection boundaries. It does not
imply recurrent state handling or full-episode backpropagation.

## 6. Solver-state observability

State-free R0 does not include position, prior actions, episode accounting, or
progress. OHLCV-only recurrence cannot reconstruct sampled actions, fills, or
position exactly. `R4-v1-min` fixes the auxiliary order as
`[position, remaining_horizon_fraction]`, including reset and terminal semantics,
and appends it through a separate bias-free residual branch. The blocking
three-seed 4x3 FF gate is replay-valid. Time-only passed the relative net-return
factor gate, but no arm passed the absolute economic gate.

The complete connector is nevertheless architecture-qualified with R0 as the
forward `R0 + R4-v1` feed-forward baseline because it removes the known
endogenous action-cost and finite-horizon alias. R0 remains the immutable parity
control, and this does not constitute economic promotion or full observability of
the market. New replication uses zero-initialized final residual heads to preserve
the update-zero R0 policy/value. The completed 4x3x40 replication found a
seed-consistent positive terminal net delta for combined state and a positive
pooled seed-then-anchor interval, but not control superiority or seed-consistent
Sharpe; one seed collapsed near-flat. R1 must retain both state-free and
explicitly named `R1 + R4-v1` cells.

Episode PnL, cost basis, unrealized/realized PnL, position age, accumulated
turnover, drawdown, and market-session time are registered future causal factors.
They enter only with explicit accounting, reset, reward/transition dependency,
and inference-availability contracts. Market/task × state nonlinear late fusion
is reserved for separately named R4-v2.

## 7. AZR-inspired proposer/solver boundary

The local AZR v2 source supports a useful but bounded analogy: proposal and
solving, deterministic validation, task/reference buffers, and role-specific
feedback. Eq. 4 is `0` when observed solve rate `p` is `0` or `1`, otherwise
`1-p`; it is asymmetric and is not a symmetric 50% peak or measured pre/post
learning uplift. The source's Table 2 averages are AZR coding/math results, not
iTrader returns.

iTrader intentionally differs: historical OHLCV is external data; a language
model may propose tasks while numerical PPO solves them; `train_AZR.py` is a
bootstrap handoff rather than a jointly updated self-play loop. Use the label
**AZR-inspired adaptive curriculum learning**.

The proposer study is separate from R-series solver attribution:

1. fixed rule/random validated curriculum control;
2. frozen LLM proposer under matched validation and solver budget;
3. frozen LLM plus solver-feedback/reference-conditioned selection;
4. learned proposer only after repeatable utility/uplift evidence.

Use a reference buffer study with frozen versus adaptive updates as a named
factor. Adaptation feedback is validation, not an untouched final chronological
test. Keep a final time-separated test unavailable to proposer updates, tuning,
and head selection. Record accepted task counts, rejected tasks, generation
calls, tokens/latency, compute, costs, turnover, and drawdown. A cost-aware
verifier must not allow proposer-controlled reward parameters to redefine the
metric.

## 8. Online state ownership

For multiple candidate tasks on one market window, do not mutate one shared
recurrent state once per candidate. Either maintain task-independent market state
and candidate-specific readouts, or own separate candidate state/replay streams.
The online shell remains:

```text
market window -> MarketFramer -> CandidateTask(s) + WAIT
-> solver scoring -> deterministic DecisionGate -> RiskEngine -> Executor
-> DecisionRecord
```

PPO logits/value outputs are not automatically calibrated confidence or expected
net utility. Capability profiles require held-out evaluation slices and explicit
calibration evidence.

## 9. Promotion gates

- R0 remains the default.
- R1 needs correct shape semantics, finite forward/backward behavior, and
  measurable matched evaluation; it need not win before R2 is tested.
- R2c only advances when `R2c-CARRY` beats `R2c-RESET` and matched controls on
  held-out chronological slices, with stable state norms, valid sequence masks,
  and recorded compute/memory cost.
- R3 waits for a meaningful LSTM result.
- R4-v1-min is implemented and replay-valid on FF. Its fresh 4x3x40 replication
  found a relative combined-state net effect but failed absolute promotion. The
  subsequent actor-robustness PBT, checkpoint forensics, and two-process
  fresh-panel G0 diagnostic are complete and economically rejected. The G0
  checkpoints have byte-identical paired artifacts but historical receipts lack
  producer identities. All had negative fresh-panel net return and Sharpe.
  Next work compares matched R0, R0 + R4-v1, clean R1, and separately named
  `R1 + R4-v1`, with causal staging retained as an explicit matched curriculum
  factor. R4-v2 remains a distinct later joint-fusion factor.

Implementation prerequisites and tests are specified in
[R1 development readiness](../project-state/r1-development-readiness.md).
- No recurrent cell becomes operational guidance from an unimplemented design,
  a zero-state probe, PPO logits/value scores, or a training-only curve.
