# R0 + R4-v1 architecture qualification and replication audit — 2026-09-25

## Status

**The fixed 4x3 replication, exact replay, and counterfactual diagnostics are
complete. R4-v1 is architecture-qualified; economic promotion failed.**

The completed 12-update R4-v1 gate is not reinterpreted as an economic win. It
did establish that the fixed-cost, finite-horizon Family 00 control problem was
missing agent-visible position and remaining-horizon state. The forward Phase A
feed-forward baseline is therefore named **R0 + R4-v1**, while state-free R0 is
retained unchanged as the historical and parity control.

The authorized fixed-recipe 40-update replication is complete. It does not
authorize R4-v2, a sealed-test read, or economic promotion from relative uplift
alone. Its result does authorize the complete `R0 + R4-v1` pair as the forward
feed-forward training baseline while R0 remains the immutable parity control.

## Consolidated decisions

| Identity | Meaning | Decision |
|---|---|---|
| R0 | State-free `SolverMLPFiLM` | Immutable historical/parity control |
| R4-v1 | Two-value `[position, remaining_horizon_fraction]` connector with additive actor/value residuals | Architecture-qualified solver-state factor |
| R0 + R4-v1 | R0 market/task path plus the complete R4-v1 state block | Forward Phase A feed-forward training baseline; not economically promoted |
| R4-v1.1 evidence profile | R4-v1 with zero-initialized final residual heads | Required for update-zero policy/value parity in the new replication |
| R4-v2 | Future nonlinear market/task × state late-fusion factor | Research only; separately named and gated |

R0 is not renamed or overwritten. Future R1 work must preserve a clean
state-free R1 cell and name the state-complete cross explicitly as
`R1 + R4-v1`. This keeps recurrence attribution separate from observability.

## What has and has not become Markov

For the present exact-cost, finite-horizon `net_return` task, current position
determines the cost of a new target action and remaining horizon determines the
termination/liquidation boundary. Exposing both removes the known endogenous
control-state alias. The exogenous market process remains partially observed;
therefore the correct claim is **control-state completion for this task
contract**, not that trading as a whole is a fully observed MDP.

`ITraderEnv` owns both values and observation assembly. `TaskSpec` remains the
problem/curriculum contract, `RewardEngine` remains the reward seam, and neither
the proposer nor deterministic risk/execution shells acquire mutable portfolio
state.

## State-feature semantics registry

Only position and remaining horizon enter the current baseline. Later
feature-engineering rounds must use named semantics rather than an expanding
anonymous input vector.

| Candidate state | Domain meaning | Activation condition | Planned factor |
|---|---|---|---|
| Current position | Inventory immediately before the action | Active now; action cost depends on it | R4-v1 |
| Remaining episode horizon | Fraction of the declared task horizon remaining | Active now; terminal liquidation and finite return-to-go depend on it | R4-v1 |
| Episode gross/net PnL | Path-dependent episode accounting to the current decision | Add only when a declared reward, risk budget, or stopping rule makes future utility depend on accumulated PnL | R4-F1 candidate |
| Cost basis and unrealized return | Inventory entry state and mark-to-market displacement | Add only with explicit lot/accounting semantics; never infer it from reward history | R4-F1 candidate |
| Realized PnL | Closed-position episode accounting | Add only with an explicit realized/unrealized decomposition | R4-F1 candidate |
| Position age / time since change | Duration of the currently held target position | Add only for a holding-duration reward, constraint, or causal ablation | R4-F2 candidate |
| Accumulated turnover / remaining turnover budget | Path-dependent consumption of a declared activity budget | Add only when such a budget affects termination, reward, or admissible actions | R4-F2 candidate |
| Running peak and drawdown | Path-dependent risk state | Add only when drawdown changes reward, gating, or termination | R4-F2 candidate |
| Market-session clock | Causal position inside RTH/pre/after-hours, distinct from task horizon | Add after Phase A RTH closure with explicit cyclic and remaining-session encodings | Causal session factor |
| Continuous portfolio lifecycle | State that survives artificial episode boundaries | Requires a separately specified continuing environment and accounting lifecycle | Future portfolio factor |

Every addition must name its owner, normalization, reset rule, inference
availability, and corresponding reward or transition dependency.
Episode-relative time must not be confused with market-session time, and
cumulative PnL must not become a reward shortcut.

## R4-v1.1 initialization correction

The completed R4-v1 run used ordinary random initialization for its residual
heads. Although masked `[0, 0]` observations were inert, nonzero state arms did
not begin from the same action preferences as R0. The new
`solver_state_residual_init=zero_v1` profile zero-initializes only the final
actor and value residual heads. The base market trunk, task FiLM path, state
encoder, parameter count, and observation contract remain unchanged.

The update-zero gate requires matching base weights, action argmax parity on
nonzero state inputs, logits and values equal within `1e-7`, zero residual-head
weights, and arm-matched untrained replays. Historical checkpoints continue to
restore with the compatibility default `random_v1`.

The official control pass now closes this gate. For each of seeds 20260924,
20260925, and 20260926, the complete 32-episode JSONL receipt is byte-identical
across `off`, `masked`, `position`, `time`, and `both`. The per-seed SHA-256
digests are `9ad3af...12e8`, `af316b...3a2`, and `da53da...ed49` respectively.
This proves action/economic update-zero parity on the actual panel, not only a
tensor-level unit test.

## Tooling gates closed before launch

1. The absolute analyzer now encodes an all-seed turnover ceiling of `20.0` and
   an exclusive near-flat ceiling of `0.98`, in addition to positive net,
   positive Sharpe, control superiority, and exact replay.
2. The policy-baseline runner accepts the solver-state arm and residual
   initialization profile, namespaces its artifacts, and can generate matched
   update-zero controls.
3. Checkpoint diagnostics score identical market/task inputs under observed,
   masked, position-only, time-only, both, position-flipped, and time-mirrored
   state interventions. They report argmax changes plus logit and value deltas.
4. The analyzer accepts an explicit update count and records initialization and
   counterfactual evidence.

Focused affected tests passed (`165`), the full repository suite passed (`288`),
the Ray-2.58 healthcheck passed, and `git diff --check` reported no errors.

## Fresh chronological development block

The new panel is
`dataset_v2_r4_dec2025_crossed_8x4_20260925.json`. Before inspecting returns,
eight RTH anchors were fixed at 15:30 UTC across AAPL, GOOGL, META, MSFT, NVDA,
QQQ, SPY, and TLT on distinct December-2025 sessions. Each is crossed with
horizons 96/128/160/200.

All tasks lie inside the purged validation partition; keep the 60-bar lookback
and 200-bar maximum outcome in one RTH session; are exposure-disjoint from the
prior panel; remain development rather than sealed-test evidence; and were
selected by symbol/date/time schedule rather than return or action outcome.

## Fixed 40-update replication

The loadout is
`phase_a_r0_plus_r4_v1_1_dec2025_4x3_40_20260925.json`. It freezes four arms,
three seeds, Ray 2.58 `workers=0`, dataset-v2 RTH, Family 00, `semantic_v2`,
exact fixed 1 bp `net_return`, scale 1,000, 40 PPO updates, batch 4,800, and
evaluation on the new 32-task panel every update. PBT is disabled.

Primary analysis remains terminal and seed-paired against `masked`; updates
38–40 form the durability slice. Exact terminal replay, seed-then-anchor block
uncertainty, arm-matched update-zero controls, counterfactual diagnostics, and
complete provenance are mandatory.

## Completed replication result

All 12 cells reached exactly 40 PPO updates on one frozen source, dataset, and
panel identity. All 12 terminal checkpoints restored in fresh processes and
reproduced the recorded 32-episode scores exactly. The update-zero controls were
also byte-identical across `off`, `masked`, `position`, `time`, and `both` for
each seed, so the comparison starts from one policy rather than four accidental
initial policies.

| Seed | Masked net / Sharpe | Position net / Sharpe | Time net / Sharpe | Both net / Sharpe |
|---|---:|---:|---:|---:|
| 20260924 | -0.3701% / -0.1020 | -0.1731% / -0.0398 | -0.1050% / -0.0460 | -0.1749% / -0.0377 |
| 20260925 | -0.0030% / -0.0090 | +0.0123% / -0.0345 | -0.0280% / -0.0173 | +0.0659% / -0.0024 |
| 20260926 | 0.0000% / 0.0000 | +0.0011% / -0.0044 | +0.0032% / +0.0029 | +0.0014% / -0.0092 |
| Mean | -0.1243% / -0.0370 | -0.0532% / -0.0262 | -0.0433% / -0.0201 | -0.0359% / -0.0164 |

The strongest deterministic control was long-only at +0.1752% net and +0.0409
Sharpe per square-root bar. Flat returned zero. No trained arm beat long-only,
and no arm had positive net and positive Sharpe in all three seeds.

### What improved

The combined state arm improved terminal net return over masked in every paired
seed: +0.1952, +0.0689, and +0.0014 percentage points. Its pooled
seed-then-anchor mean effect was +0.0885 percentage points with a 95% interval
of [+0.0037, +0.1889], so the net-return factor gate passed. Its mean turnover
fell from 11.40 to 4.71. Position-only also improved net in every seed, while
time-only produced the strongest mean gross return and critic fit.

This is not merely cost avoidance. Mean gross return moved from -0.0104% for
masked to +0.0064% for position, +0.0196% for time, and +0.0112% for both.
However, the paired Sharpe effect was not seed-consistent and its pooled interval
crossed zero.

### What failed

Seed 20260926 converged to a near-flat basin: terminal flat fractions were
100.00%, 99.77%, 99.97%, and 99.39% for masked, position, time, and both. The
explicit `<0.98` ceiling correctly rejects all four arms. Seed 20260924 remained
economically negative, and masked also exceeded the turnover ceiling at 29.25.
Updates 38–40 do not rescue the verdict; the combined arm's final-three mean is
negative for seeds 20260924 and 20260926.

The useful transient is seed 20260925. Combined state retained +0.0672% mean net
over updates 38–40, but terminal Sharpe remained slightly negative. This is a
real learned checkpoint, not provenance contamination, yet it is insufficient
for promotion.

### Actor and critic diagnostics

Masked-state counterfactuals are exactly inert. In active position-only arms,
flipping position changed the greedy action on 18.21%, 6.06%, and 0.11% of
steps across the three seeds. In time-only arms, mirroring remaining horizon
changed far fewer greedy actions, but changed the value estimate by a mean
absolute 0.959, 0.578, and 0.358. Combined arms used both coordinates in the
non-collapsed seeds. These results show that the state branch is learned and
that episode phase materially improves the critic, while actor use remains
seed-sensitive and disappears in the near-flat basin.

Interventions on coordinates disabled for an arm are out of distribution and
are excluded from interpretation. PPO logits are not Q-values; the diagnostics
demonstrate state sensitivity, not a decomposed cost-aware action-value model.

## Final gate decision

The combined position-plus-time connector passes architecture qualification and
the relative paired net-return factor gate. It fails the absolute economic gate.
Accordingly:

1. preserve state-free R0 as the immutable parity control;
2. use `R0 + R4-v1` as the forward Phase A feed-forward training baseline;
3. do not claim economic promotion, a profitable solver, or a fully observed
   market MDP;
4. retain `position + remaining_horizon_fraction` together even though their
   effects differ, because both are required by the exact-cost finite-horizon
   transition/reward contract;
5. keep episode PnL, accounting, holding-age, turnover-budget, drawdown, and
   market-session features in the semantics registry until their corresponding
   transition, reward, or risk semantics exist;
6. make the next bounded solver step an actor-robustness search over the already
   authorized baseline—learning rate, entropy pressure, and causal cost staging,
   with retained-best replay and the same flat/turnover guards—before R4-v2;
7. preserve clean state-free `R1` and named `R1 + R4-v1` cells when recurrence
   begins, so recurrence and observability remain attributable.

Authoritative evidence is in [`analysis.json`](solver-state-r4-v1-replication-plan-2026-09-25-evidence/analysis.json).
The comparison plots include [net return](solver-state-r4-v1-replication-plan-2026-09-25-evidence/training_comparison/ablation_compare.png),
[Sharpe](solver-state-r4-v1-replication-plan-2026-09-25-evidence/training_comparison/ablation_compare_validation_sharpe.png),
[action fractions](solver-state-r4-v1-replication-plan-2026-09-25-evidence/training_comparison/ablation_compare_action_fractions.png),
and [PPO diagnostics](solver-state-r4-v1-replication-plan-2026-09-25-evidence/training_comparison/ablation_compare_diagnostics.png).

## Decision boundary

Architecture qualification is accepted: future feed-forward Phase A work uses
`R0 + R4-v1`, with R0 retained in every parity-sensitive comparison. Economic
promotion is rejected and cannot be inferred from critic fit or relative
uplift. Run the bounded actor-robustness search before spending on R4-v2. Carry
both `R1` and `R1 + R4-v1` long enough to attribute recurrence separately.

## Deferred R4-v2 design question

The leading research shape is centered nonlinear late fusion after task FiLM:

```text
h_market = MLP(market_window)
h_task = FiLM(h_market, task)
z_state = state_encoder(agent_state)
delta = joint_connector(h_task, z_state) - joint_connector(h_task, zero_state)
logits = policy_head(h_task) + policy_residual(delta)
value = value_head(h_task) + value_residual(delta)
```

Centering preserves masked-state parity while allowing market/task × state
interactions. It remains separate because it changes the function class and the
causal interpretation of the residual.
