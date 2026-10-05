# Decisions

## 2026-10-05 — Full native API cutover authorized for active training

- Later human direction for `feat(R)/Ray_upgrade`: this is a full migration of
  the active trainer to Ray 2.58 RLModule/Learner/EnvRunner, not a dual-stack
  rollout. Keep shared solver math in the PyTorch-only core; place historical
  Ray 2.58 ModelV2/TorchPolicy adapters and recipes in
  `itrader/compat/rllib_legacy/`, and Ray 2.9 runs in the separate
  `itrader/compat/ray29/` profile.
- Legacy APIs remain available only as explicit, non-default parity tools for
  old runs. Do not route new runs through them or blend old/new contracts in the
  active trainer. This engineering delivery does not make an economic claim or
  require the scientific four-cell study to be completed before main DEV review.
- The implementation gate includes separate real-Ray Algorithm
  train/save/restore/inference and official training CLI/Tune subprocess tests.
  Preserve the unrelated supervised compatibility-control sequence and its
  evidence; do not rewrite historical runs.

## 2026-10-05 — Q21 two-tier evaluation accepted; native development isolated

- Human approval verified in GRILL-IMPLEMENT: inspected 32-task monitoring at
  initialization/every update; fresh validation-only 32 symbol/session anchors
  crossed with horizons 96/128/160/200 for 128 terminal tasks. Predeclare/hash
  without outcome inspection and share identical tasks across all cells/controls.
- Use predetermined terminal checkpoints, not selected transients. Open the
  fresh panel only after all 16 target-screen runs finish. Final test stays sealed.
  Protocol acceptance does not mean the panel or enforcement already exists.
- Reporting separates raw episode/run economics, directional and other matched
  controls, encoder/state/interaction effects, PPO/critic diagnostics, actual
  causal-cost exposure, resource use and provenance. Retain raw traces; crossed
  horizons are not independent and two seeds do not establish significance.
  Evaluation is outside the 921,600 nominal training-step budget.
- The existing eight-cell corrected-compatibility preparation remains its own
  development-panel bridge; do not silently count it as the complete target screen.
- Native Ray 2.58 API development is authorized in the separate migration
  worktree on `feat(R)/Ray_upgrade`, with commits/pushes confined there. The
  scientific order remains corrected compatibility R0/R4 controls followed by
  R1 comparison; the later full-cutover direction above separates API migration
  delivery from any future four-cell parity experiment. No main-tree source
  overwrite is authorized by this separate work.
- See [R1 readiness](r1-development-readiness.md) for the implementation contract.

## 2026-10-05 — Earlier sequence: compatibility first; native APIs after R1

This records the earlier scientific ordering decision. Its statement that the
active API migration should wait for R1 is superseded by the later full-cutover
direction above; the corrected-control/R1 attribution sequence remains separate.

- Human decision: finish the Ray 2.58 compatibility repairs, rerun matched R0
  and R0 + R4-v1, then implement/compare R1 and R1 + R4-v1 on the same stack.
  Native RLModule/Learner/EnvRunner migration follows, with a complete four-cell
  platform bridge. It is not a prerequisite for R1 implementation.
- Preserve historical runs, checkpoints and raw outcomes. Do not infer effective
  learning settings from historical PBT mutation logs: integration reproduced
  obsolete minibatch/epoch aliases and donor-restored optimizer/policy settings.
- Q19 structural selectors and Q20 screen budget are accepted, superseding the
  pending wording in the earlier entry: four cells x two seeds x two curricula,
  57,600 nominal steps each; cost boundaries 14,400 / 28,800 / 43,200.
  Q21 was pending at this decision and is superseded by the accepted entry
  above; final test remains sealed.
- Correctness approval, representation attribution and economic promotion are
  separate decisions. No new performance claim is made by synthetic probes.


## 2026-10-05 — GRILL Q18 accepts shared R1 encoder and FiLM

- Human decision relayed by GRILL: R1 actor and critic share one market window
  LSTM, projection and task-FiLM path, with separate pi/vf output heads. Both
  losses update the shared representation, matching R0's sharing arrangement.
- Q15 remains fixed: unidirectional LSTM(5,156), zero h/c every decision, final
  chronological output, Linear(156,256,bias=True), ReLU, existing task FiLM and
  dropout semantics. R4-v1 zero_v1 additive residuals remain unchanged.
- Capacity under this accepted design is 178,324 parameters for R1 and 178,372
  with R4-v1, versus 179,268 and 179,316 for their R0 counterparts. These are
  measured component counts, not an implemented or trained R1 result.
- Q19 structural selector design and the run budget remain pending. Proposed
  `solver_model` values are `solver_mlp_film` and `solver_window_lstm_film`;
  historical registrations retain their original restore meanings.
- The four-cell matrix and compressed cost curriculum remain accepted. No
  performance run can bypass the open evidence/diagnostics gates.
- Implementation acceptance details: [R1 readiness](r1-development-readiness.md).

## 2026-10-03 — Close G0 without promotion and enter fixed-recipe R1 attribution

- Decision: close the member-`00002` checkpoint-only diagnostic and proceed to
  matched Ray-2.58 R0 / clean R1 / `R1 + R4-v1` representation attribution.
- Evidence: five retained unexploited checkpoints replayed twice on one
  predeclared non-overlapping November-2025 8x4 RTH panel. Episode/reward and
  per-step action/logit outputs were byte-identical across clean processes.
- Economic result: every fresh-panel mean net return and Sharpe was negative.
  The least-negative 102,040-step checkpoint reached -0.22822% net and -0.00620
  Sharpe with an approximately 75/0/25 short/flat/long mix repeated across all
  anchors. The 106,968-step and terminal policies were pure short.
- Interpretation: causal staging produced real, deterministic transients on the
  January panel, but not portable conditional edge. Preserve R0 + R4-v1 as the
  architecture-qualified feed-forward baseline without economic promotion.
- Next boundary: freeze the lower-LR recipe across R0, clean R1, and R1 + R4-v1.
  Cross fixed 1 bp and causal staging explicitly; do not hide curriculum inside
  an R identity. Keep PBT off until fixed-recipe representation attribution is
  complete.
- Evidence navigation: use the generated experiment browser for concrete
  run/member/checkpoint search and comparison-factor checks. Its registry owns
  provenance metadata and pointers only; source summaries own metric values,
  and dated dashboards stay immutable.

## 2026-10-01 — Canonical research registry without replacing the masterplan

- Decision: use `docs/research/itrader_research_dashboard.md` (+ `.html`) as the
  canonical registry of research identities, controls, dependencies, evidence
  status, and open decisions.
- Authority boundary: `docs/plans/itrader_masterplan.md` remains authoritative
  for cross-phase order and promotion; active architecture owns invariants;
  source/tests/config own implementation truth; dated artifacts remain evidence.
- Taxonomy: R, C, and S are stable research wings. ATX/PF/WM/TTS and the new AZR
  labels are provisional experiments until separately accepted. Reward, cost,
  task families, data, platform, and online contracts remain orthogonal axes or
  seams rather than new solver wings.
- Organization: preserve dated reports/evidence in place and organize them
  through the dashboard and categorized artifact index. This avoids breaking
  provenance links and historical snapshot citations.
- Drift recorded, not hidden: actor robustness is complete/rejected; Ray 2.58 is
  forward authority; R4-v2 keeps its fixed joint-fusion identity while the
  reported parallel staging remains unverified on this checkout.

## 2026-09-29 — Stop same-panel R0 + R4-v1 PBT and freeze R1 attribution recipe

- Decision: do not launch another full R0 + R4-v1 PBT population against the
  January development panel. Run only the bounded fresh-panel transient/logit
  diagnostic, then proceed to fixed-recipe R1 attribution.
- Why: causal staging materially preserved activity and increased positive
  reports, but all 81 jointly positive reports were at least 98.014% short and
  the terminal result exactly tied short-only. The unexploited 77,832-step
  checkpoint combined 99.36% greedy short actions with high entropy and narrow
  top-two margins, then repeated the one-action replay mismatch.
- Consequence: freeze member `00002`'s lower-LR unexploited recipe across matched
  Ray-2.58 R0, clean R1, and R1 + R4-v1. Keep causal cost staging as an explicit
  matched factor; do not silently embed it in R1 and do not re-open PBT until the
  representation comparison is complete.
- Instrumentation boundary: add all-equal/all-ineligible PBT exhaustion,
  candidate logit margins, training-block causal attribution, and valid PPO
  clip-fraction capture. The static optimization-cost reporting omission is
  fixed for future evidence and does not change the completed exact-cost result.

## 2026-09-29 — Close R4-v1 actor robustness without economic promotion

- Decision: retain `R0 + R4-v1` as the architecture-qualified feed-forward
  baseline, retain R0 unchanged for parity, and close the bounded actor search
  without an economic winner.
- Why: both four-member/40-update arms passed strict PBT lineage mechanics. The
  causal-cost population became exactly short-only and tied the strongest
  directional control; the exact-1-bp population became exactly flat. The one
  marginal causal report above short-only failed exact checkpoint replay.
- Keep: lower-learning-rate and entropy search ranges, causal cost staging as a
  named curriculum factor, exact provenance, near-flat/turnover guards, and the
  December fixed-recipe state-arm attribution reference.
- Reject/defer: exact 1 bp from update zero as the default curriculum, economic
  promotion, a new independent population, 0.5/1/2-bp stress on a non-candidate,
  independently searched mask arms, and R4-v2.
- Next boundary: after code review, compare clean R1 with matched Ray-2.58 R0 and
  keep `R1 + R4-v1` separately named so recurrence and observability remain
  attributable.

## 2026-09-25 — Close the R4-v1 replication without economic promotion

- Decision: retain `R0 + R4-v1` as the architecture-qualified Phase A
  feed-forward training baseline and R0 as the immutable state-free parity
  control. Do not economically promote any R4-v1 arm.
- Why: on the fresh December-2025 4x3x40 replication, combined state improved
  terminal net return over masked in all three seeds and its pooled
  seed-then-anchor interval excluded zero. All terminal replays were exact and
  the state counterfactuals were active. Nevertheless, no arm beat long-only or
  achieved positive net and Sharpe across seeds, and seed 20260926 breached the
  near-flat ceiling.
- Consequences: keep position and remaining horizon together for the exact-cost,
  finite-horizon task; run a bounded actor-robustness search with retained-best
  replay before R4-v2; preserve separate `R1` and `R1 + R4-v1` cells.
- Non-goals: this does not claim a profitable solver, a fully observed market
  MDP, R4-v2 approval, or final-test eligibility.
- Revisit when: the actor-robustness search produces a replay-valid candidate
  that passes the flat/turnover guards and deterministic-control gate.

## 2026-09-25 — Architecture-qualify R0 + R4-v1 without economic promotion

- Decision: preserve R0 unchanged as the historical/parity control and use the
  complete position-plus-remaining-horizon connector as the forward feed-forward
  Phase A baseline named `R0 + R4-v1`.
- Why: those variables remove the known endogenous action-cost and finite-horizon
  alias. Improved critic fit and relative economics support state use, while the
  completed 12-update gate still fails absolute economic promotion.
- Consequences: the new replication uses the named `zero_v1` residual
  initialization, matched update-zero controls, counterfactual diagnostics,
  explicit turnover/near-flat gates, and a fresh December-2025 validation block.
  R1 remains state-free versus an explicitly named `R1 + R4-v1` cross.
- Non-goals: no PBT, R4-v2, sealed-test read, recurrence implementation, or
  economic promotion is authorized by this architecture decision.
- Revisit outcome: the fixed 40-update 4x3 replication and exact replay are
  complete; the architecture decision stands and economic promotion remains
  rejected by the newer decision above.

## 2026-09-25 — Do not economically promote R4-v1 after the blocking 4x3 gate

- Decision: retain the implemented R4-v1 connector as an available named
  observability factor, but do not economically promote it from this
  development-panel result. The later architecture qualification above
  supersedes the earlier default-input restriction without changing this
  economic verdict.
- Why: time-only produced directionally consistent net uplift and a positive
  seed-then-anchor interval, but its absolute three-seed net and Sharpe were
  negative. The complete position-plus-time arm improved every seed relative to
  masked but its pooled intervals crossed zero, and it also failed flat,
  directional, and untrained controls.
- Consequences: the special R4-before-R1 execution block is closed, but R1 still
  requires user review and its lifecycle/placement gates. R4 must not be silently
  bundled with recurrence. No PBT, sealed-test, or 2-bp promotion follows.
- Revisit when: a new chronological development block is predeclared and the
  remaining-time factor can be tested without reselecting on the same anchors.

## 2026-09-25 — Accept Ray 2.58 for the R-series; freeze Ray 2.9 for reproduction

- Decision: coordinated root dependencies target Ray/RLlib 2.58.0, Gymnasium
  1.2.2, and PyArrow 17.0.0. Ray 2.9 remains a version-pinned historical
  reproduction profile under `itrader/compat/ray29/`.
- Why: current dataset-v2 source passes both 254-test suites, deterministic
  initialization/control, a local-worker synchronous exploit/restore gate,
  strict generation-bound lineage, and exact fresh replay. Cross-version PPO
  trajectories are not seed-identical, so numerical endpoint identity is not an
  acceptance condition.
- Consequences: future R0/R1 comparisons use Ray 2.58 and carry runtime labels;
  the validated topology on this host is `workers=0`. Native
  RLModule/EnvRunner modernization remains a separate factor.
- Revisit when: a measured remote-worker topology has safe memory headroom or
  the native RLlib-stack migration has its own R0 parity gate.

## 2026-09-24 — Cross starts and horizons in the P0 development panel

- Decision: evaluate the next matched P0 on eight explicit RTH starts crossed
  with all four horizons, with every task carrying `start_idx` and the panel
  bound by content hash.
- Why: the original 32-task panel used different starts for each horizon while
  `semantic_v2` exposes horizon, so a policy could route on horizon as a proxy
  for a different market path.
- Consequences: the old panel remains historical bridge evidence. The crossed
  panel is selection-informed development evidence, not an independent or final
  test; P1 still requires predeclared new chronological blocks.
- Revisit when: a P0 candidate is frozen and the independent P1 panel is declared.

## 2026-09-24 — Require generation-bound PBT restore provenance

- Decision: a new-arm exploit counts only when scheduler selection, exactly one
  target restore receipt, a later same-generation target report, and a retained
  same-generation target checkpoint agree on exploit identity and full-tree
  checkpoint digest.
- Why: step/path-only inspection could falsely accept a lagged target after a
  restore even when the donor selection itself was exact.
- Consequences: strict inspection is the default. Older Gate D evidence remains
  forensically valid through direct retained-copy hashing, but requires an
  explicit legacy-inspection flag because it predates restore generations.
- Revisit when: the Ray migration provides native immutable lineage receipts
  with an equally strong or stronger contract.

## 2026-09-24 — Use exact per-report sidecars for synchronous Ray 2.9 PBT

- Decision: synchronous PBT writes an immutable stable checkpoint for every
  evaluation report, while Tune receives only cadence/terminal checkpoints.
  Donor selection must resolve the checkpoint at exactly the scored
  `pbt_progress_steps`; missing exact evidence fails closed.
- Why: Ray 2.9 can admit an in-flight update after a synchronous pause and can
  request actor saves before a recreated actor has a local result. Cadence-only
  snapshots therefore allowed score/model lag or actor-save assertions.
- Evidence: Gate D terminated four members at 170,152--170,416 steps with 140
  report/checkpoint pairs and five exploit receipts, all with zero donor lag,
  including the former 102,080-step failure.
- Consequences: `pbt_progress_steps`, not RLlib `training_iteration` or local
  `timesteps_total`, is the authoritative exposure/scheduling clock. Stable
  sidecars are donor/model evidence; Tune-owned zero-byte moved artifacts remain
  non-authoritative on Windows.
- Revisit when: the Ray migration reproduces exact selection/model identity on
  the new API stack and can replace this compatibility adapter.

## 2026-09-24 — Make immutable Parquet the dataset-v2 authority

- Decision: acquire provider data through a read-only adapter, retain raw pages,
  write content-addressed Parquet partitions by session/symbol/year, and use
  DuckDB only for catalog/query. Ray loads selected partitions and projected
  columns; it never owns or mutates the canonical dataset.
- Why: dataset v1 collapsed OHLC, duplicated/summed volume, and lost DST-safe
  provenance. A mutable database authority would make exact experiment replay
  and provider-regime attribution harder.
- Consequences: manifests record feed, adjustment, timezone, dates, symbols,
  counts, schema, and content hash. Raw/licensed bars and credentials remain
  ignored; tracked artifacts contain only reproducibility metadata and aggregate
  evidence.
- Revisit when: a second provider is certified or a forward collector requires
  snapshot compaction. New snapshots must remain immutable.

## 2026-09-24 — Keep Phase A RTH-only and dividends explicit

- Decision: retain pre-market/RTH/after-hours in dataset v2, but train and
  evaluate Phase A only on RTH. Use split-adjusted, non-dividend-adjusted prices.
  Model dividends later as explicit signed cashflows in environment evidence and
  reward accounting.
- Why: extended hours change liquidity, spread, volatility, and annualization;
  silently adjusted prices obscure ownership and short-side dividend effects.
- Consequences: `DataPartition` prevents cross-session episodes. Extended hours
  and dividend cashflows are named Phase B factors, not hidden changes to R0.
- Revisit when: R0 passes the RTH data/evidence gate and ex-date lifecycle tests
  are specified.

## 2026-09-12 — Separate optimization reward from promotion evidence

- Decision: `ITraderEnv` emits terminal `EpisodeEvidence` schema v1 with
  independent gross/cost/net arithmetic return, liquidation-inclusive turnover,
  drawdown, activity/exposure, and Sharpe. `RewardEngine` continues to own only
  optimization reward and finite components.
- Why: historical `episode_return` fields were shaped/normalized reward and could
  not support economic performance claims.
- Consequences: PBT selects `evaluation/net_return_mean` on a purged validation
  partition; annualized Sharpe requires an explicit convention; the final test is
  post-selection only.
- Revisit when: an evaluation-run aggregator adds baseline deltas, block-aware
  uncertainty, and checkpoint/dataset fingerprints to the promotion record.

## 2026-09-12 — Use fair, bounded Ray 2.9 PBT with independent provenance

- Decision: independently sample training seeds/hyperparameters, share the
  validation seed, keep train batch fixed under an iteration clock, align
  checkpoint and perturbation cadence, constrain every mutation, and rewind
  active metrics on checkpoint restore while retaining append-only config history.
- Why: the historical population admitted invalid gamma, and the first smoke
  population performed no search. Function-trainable restores also repeated a
  full local loop and retained superseded metric rows.
- Consequences: the final smoke members each contain exactly iterations 1–12 and
  13,520 timesteps. Tune-owned Windows logs are non-authoritative; atomic stable
  run/trial artifacts outside Tune are required.
- Revisit when: Ray is upgraded and a timestep-based PBT clock or PB2 search is
  evaluated through a matched study.

## 2026-09-12 — Version task semantics without breaking checkpoint replay

- Decision: historical restores use `legacy_v1`; new runs use `semantic_v2`,
  which represents solver-relevant canonical task semantics without episode
  seed/start shortcuts.
- Why: hidden reward/regime fields aliased tasks while the seed exposed a market-
  location proxy.
- Consequences: encoding version is a resolved-config/checkpoint compatibility
  field. Overrides normalize before encoding and reset seeds control task/start
  streams reproducibly.
- Revisit when: the TaskSpec schema changes or learned task representations are
  proposed.

## 2026-09-12 — User clarification and open review boundaries

**Confirmed by the user in this review:** Windows is the working project; Phase B covers both offline/curriculum and online/replay work. Families `00`–`04` are the intended decomposition, with `00`–`02` sharing `Discrete(3)` while inputs and task/reward semantics may differ. Source currently enables only `00`; roadmap scope is not implementation evidence.

**Still open in the ongoing `grill-me` workflow:** whether to use a shared recurrent trunk selected through future ablations, family-specific policy/value heads, the input adapter/schema, recurrent state ownership, FiLM placement, and reward-specific components. This documentation pass does not accept these choices, authorize solver implementation, or stage/commit the working tree. `RewardEngine` remains distinct from learned policy/value heads.

**Proposed correction to earlier R-series naming:** commit `4732a98` used R2a concat, R2b init-only, R2c per-step FiLM, and R2d value-focused conditioning. `bec5c77` relabeled per-step FiLM R2b and used R4 for matched observation features. Earlier init-oriented R4 belongs to the older design, not `bec5c77`. The reviewed draft restores original labels for discussion and proposes `R4-v1` to disambiguate the observation study; those draft labels are not executable flags or approved experiment registrations.

The older claim that R1→newest-bar R2 isolates recurrence is superseded: it changes input history and conditioning as well as state lifetime. The draft adds same-input/same-conditioning carry/reset controls. It also separates readout FiLM from task conditioning of the recurrent transition. Full details and limitations are in `../research/recurrent-solver-ablation.md`.

Historical entries below remain provenance, not fresh test results or authority over this clarification.

## 2026-09-03 - Isolate Reward RNG From Task Identity

- Decision: task samplers and the rule-based proposer use a dedicated RNG for
  reward parameters, separate from price, horizon, and task-seed sampling.
- Why: reward functions consume different numbers of random draws. A shared RNG
  changes later task seeds and market windows, invalidating a reward ablation.
- Consequences: matched reward variants preserve task identity while still
  sampling reward-specific parameters deterministically.
- Revisit when: task generation adopts immutable pre-generated evaluation pools.

## 2026-09-03 - Keep PnL Default And Test Two Sharpe Challengers

- Decision: `pnl_linear` remains the sole proven/default reward. `sharpe_only`
  tests bounded incremental strategy Sharpe alone; `pnl_sharpe` adds the same
  component to normalized `pnl_linear`. Both challengers are unannualized during
  optimization, have a warmup floor, and are not family defaults.
- Why: a full rolling Sharpe level paid every step double-counts prior path
  quality and minute-scale annualization saturates a bounded signal. The
  incremental form is finite, inspectable, and approximately telescoping.
- Consequences: family `00` can run a three-way matched comparison without
  changing its default or semantics. Family-specific reward assignments remain
  deferred until a challenger wins held-out evaluation.
- Revisit when: held-out multi-seed reward comparisons are available.

## 2026-09-03 - Treat Recurrent FiLM As A Factorized Hypothesis

- Decision: compare R0 MLP-FiLM, R1 reset-every-step window-LSTM-FiLM, R2a
  stateful LSTM plus task concatenation, and R2b stateful LSTM plus per-step FiLM
  in that order.
- Why: this separates within-window temporal bias, persistent state, and task
  modulation. The current observation also lacks explicit trade state, so plan
  tracking cannot be attributed to recurrence without a later matched feature
  ablation.
- Consequences: the legacy `SolverLSTMModel` is not a valid recurrent baseline.
  RLlib 2.9 state lifecycle and truncated BPTT need TDD coverage before training
  claims. The newest bar is the last row of the chronological market window.
- Revisit when: the alignment and reward baselines are measured.

## 2026-09-03 - Preserve Canonical Family Semantics

- Decision: documentation and experiments use the source registry: `00`
  directional, `01` breakout, `02` mean reversion, `03` trade management, and
  `04` abstention.
- Why: prior research prose swapped `01`/`02` and treated `03` as trend
  following, which would make reward and head routing ambiguous.
- Consequences: no reward is assigned by an obsolete family label.
- Revisit when: a deliberate, migrated family-registry change is proposed.

## 2026-09-03 - Keep All Ablation Variants On FiLM; Defer LSTM Head Fix

- Decision: The episode/regime alignment ablation runs entirely on `SolverMLPFiLM`. It now contains mixed-horizon controls plus a matched random-128 versus stream-128 pair. The `SolverLSTMModel` obs-shape fix (`flat_dim // win` breaking with task embeddings) is deferred to the solver-head ablation.
- Why: Mixing an uncorrected LSTM head into the alignment ablation would confound two variables at once (alignment + head architecture). The FiLM baseline handles arbitrary flat obs via `flat_dim - t_dim`, so all 4 variants get a clean comparison.
- Consequences: fixed-horizon alignment is measured by comparing v3 and v4,
  rather than comparing a mixed-horizon control with a single-horizon cell. If
  alignment helps, solver-head experiments use the winning mode.
- Revisit when: Solver-head ablation (item 4) is scoped, or if alignment results show horizon-aligned tasks benefit disproportionately from temporal inductive bias.

## 2026-09-03 - Stash@{0} proposer_llm.py Is Superseded; Stash Can Be Dropped After Verification

- Decision: The 199-line Transformers+bitsandbytes `proposer_llm.py` that exists only in `stash@{0}` is considered superseded by the lighter ollama/openai_compat proposers integrated in commits `4921708` and `bd6453e`.
- Why: The old proposer required a local 4-bit quantized Qwen2.5-7B via HuggingFace Transformers, which is heavier and less flexible than the OpenAI-compatible endpoint path already wired into the proposer factory.
- Consequences: No branch needs the stashed file. After Juan verifies nothing else in stash@{0} is needed, all 3 stashes can be dropped to clean up the reflog.
- Revisit when: If a future task requires offline Transformers inference without a server (e.g., edge deployment), the file can be recovered via `git show "stash@{0}:itrader/llm/proposer_llm.py"`.

## 2026-05-02 - Treat Episode-Aligned PPO Optimization As An Experiment, Not A Default

- Decision: Keep the current PPO `batch_mode = "complete_episodes"` baseline, and evaluate stricter episode/regime alignment as an ablation instead of making it the default training path.
- Why: RLlib already collects complete episodes, so rewards/returns/GAE are not being computed across episode boundaries. The remaining mixing happens when complete episodes are aggregated into `train_batch_size` and shuffled into SGD minibatches, which is normal PPO behavior and often improves gradient variance.
- Consequences: Solver quality work should focus first on controlled evaluation by market regime, horizon, reward family, and start-index mode rather than forcing every gradient update to represent only one episode.
- Revisit when: A/B runs show the baseline mixed-batch solver underperforms a stream/regime-aligned variant on held-out rolling windows, or a true recurrent solver is introduced that needs stricter sequence semantics.

## 2026-05-02 - Capability Profiles Should Be Updated From Evaluation Slices

- Decision: `CapabilityProfile` should represent solver capability on held-out or rolling evaluation slices by family/regime/horizon, not raw training-batch returns alone.
- Why: Training batches are influenced by curriculum sampling and PPO updates; online gating needs calibrated estimates of what the solver can handle in comparable market windows.
- Consequences: Add a capability updater/report before making online `DecisionGate` thresholds depend on solver scores.
- Revisit when: Shadow-mode online replay produces enough `DecisionRecord` outcomes to calibrate live candidate selection.

## 2026-05-02 - Proper Recurrent Solver Work Needs A Separate Check

- Decision: Do not assume the current `SolverLSTMModel` behaves like a stateful RLlib recurrent policy across environment steps.
- Why: The current default config uses `SolverMLPFiLM`; the existing LSTM model consumes the flattened rolling window and returns no recurrent state. Episode-bound minibatching is therefore less relevant than it would be for a true RLlib recurrent policy.
- Consequences: If LSTM becomes the main solver head again, validate sequence handling, `max_seq_len`, hidden-state reset behavior, and minibatch sequencing before attributing performance changes to episode alignment.
- Revisit when: Solver head selection changes for task families `00`, `01`, or `02`.

## 2026-05-02 - Test A Window Encoder Before A Fully Stateful Recurrent Solver

- Decision: Add a corrected time-series window encoder head before investing in a fully stateful recurrent PPO policy.
- Why: The current environment already provides a rolling OHLCV window plus task embedding. A native window encoder can test whether temporal inductive bias improves solver quality without adding recurrent rollout-state complexity immediately.
- Consequences: Keep `SolverMLPFiLM` as the baseline, then compare it against a window LSTM/GRU/attention head that explicitly splits price features from task embedding. Implement true recurrent state only after that comparison shows benefit or exposes a memory limitation.
- Revisit when: Window-encoder results are available on held-out rolling windows and capability slices.
