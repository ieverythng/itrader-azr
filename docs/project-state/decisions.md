# Decisions

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
