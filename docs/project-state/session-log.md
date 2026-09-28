# Session Log

## 2026-09-25 — R4-v1 fresh 4x3x40 replication closed

- Completed all 12 fixed cells on the predeclared December-2025 32-task panel at
  exactly 40 PPO updates with PBT off and one frozen source/data/panel identity.
- Replayed all 12 terminal checkpoints exactly in fresh processes and completed
  arm-aware state counterfactual diagnostics. Update-zero controls remained
  byte-identical across all state masks for each seed.
- Combined state improved terminal net over masked in all seeds; the pooled
  seed-then-anchor net interval was positive. Mean turnover also fell from 11.40
  to 4.71, and critic/state sensitivity was genuine.
- Withheld economic promotion: no arm beat long-only, Sharpe uplift was not
  seed-consistent, and seed 20260926 breached the 98% near-flat ceiling.
- Froze `R0 + R4-v1` as the architecture-qualified forward FF baseline while R0
  remains the parity control. Next P0 targets actor robustness, not R4-v2.
- Evidence and final audit:
  `docs/artifacts/research/solver-state-r4-v1-replication-plan-2026-09-25.md`.

## 2026-09-25 — R4-v1 architecture qualification and 40-update prelaunch

- Split architecture qualification from economic promotion: R0 remains the
  immutable parity control; `R0 + R4-v1` becomes the forward FF baseline, while
  the completed 12-update economic verdict remains a failure.
- Added the named `zero_v1` residual initialization and an update-zero regression
  proving nonzero state inputs preserve R0 actions/logits/values within
  floating-point tolerance.
- Added arm-aware untrained controls, executable turnover/near-flat gates, and
  identical-input counterfactual diagnostics with logit/value deltas.
- Predeclared eight return-uninspected December-2025 RTH anchors crossed with all
  four horizons and a fixed 4x3 40-update, non-PBT replication loadout.
- Registered episode accounting, holding, risk-budget, and session-clock state
  semantics for later named factors; none enter the current observation.

## 2026-09-25 — R4-v1 blocking 4x3 economic decision

- Completed 12 sequential Ray-2.58 cells across masked/position/time/both and
  seeds 20260924–20260926 at 12 PPO updates each on the frozen crossed panel.
- Replayed every update-12 checkpoint exactly and produced terminal
  state-conditioned diagnostics. All schema, dataset, panel, and source
  fingerprints matched; no PBT, final-test access, 2-bp work, or R1 launch
  occurred.
- Time-only passed the relative net gate: all seed deltas were positive and the
  pooled seed-then-anchor 95% interval was [+0.0265%, +0.1752%]. Its absolute
  mean net and Sharpe remained negative. Both and position also failed, so no
  R4-v1 arm is promoted.
- Added tested hard-RAM gating, explicit terminal replay/diagnostic selection,
  namespaced control evidence, and an eight-anchor crossed-horizon bootstrap.
- Audit and evidence:
  `docs/artifacts/research/solver-state-r4-v1-blocking-ablation-2026-09-25.md`.

## 2026-09-25 — R4-v1-min Markov-state seam and bounded smoke

- Implemented the backwards-compatible 334-wide R4-v1-min observation contract:
  the historical 332-wide market/task base plus a fixed position/remaining-
  horizon block, with a bias-free actor/value residual and exact masked parity.
- Completed sequential masked, position, time, and combined three-update cells
  on the crossed dataset-v2 RTH panel at fixed 1 bp. All four checkpoints replay
  exactly, but every terminal result is negative and economically ineligible.
- Added matched controls and state-conditioned checkpoint diagnostics. Position
  input changes action probabilities but caused severe switching in its cell;
  time-only was least negative. Early positive values did not beat the seeded
  untrained/directional explanation.
- Recorded the full audit and evidence under
  `docs/artifacts/research/solver-state-r4-v1-min-implementation-note-2026-09-25.md`.
  The next gate is a longer masked-versus-combined probe with best-update
  retention, not PBT or automatic R1 promotion.
- Validation before launch: 268 tests, healthcheck, and scoped pre-commit passed;
  focused diagnostic tests pass after the evidence seam. No commit, staging,
  deletion, final-test access, PBT, or R1 launch occurred.

## 2026-09-25 — Ray 2.58 R0 platform acceptance

- Re-ran the current dataset-v2 R0 source on Ray 2.58 with a memory-safe local
  rollout-worker topology. The 2x12 synchronous run reached about 58.6k sampled
  steps/member and closed one exact exploit/restore lineage plus fresh replay.
- Ran the matched Ray-2.9 mechanical gate and three fixed-hyperparameter seeds per
  runtime. All six terminal rows and all best-checkpoint Sharpes were negative;
  the bridge is compatibility evidence, not economic uplift.
- Diagnosed replay drift to task-embedding dropout keyed to PyTorch module mode.
  Updated `SolverMLPFiLM` to honor RLlib `is_training`; corrected replays have
  zero delta.
- Promoted coordinated Ray 2.58/Gymnasium 1.2.2/PyArrow 17 dependency pins and
  retained a Ray-2.9 reproduction profile under `itrader/compat/ray29/`.
- Validation: 126 focused and 254 full tests pass on both runtimes; both
  healthchecks pass. No commit, stage, deletion, R1 launch, or raw-artifact
  cleanup was performed.

## 2026-09-24 — Ray 2.58 current-R0 compatibility gate

- Reviewed and bounded the separate Ray-2.58 migration; ported only API adapters
  into current dataset-v2 source rather than cherry-picking the stale experiment
  boundary.
- Passed 248 tests on both Ray 2.9 and Ray 2.58 plus both healthchecks.
- Proved exact cross-runtime initial policy weights and fixed-panel evaluation;
  localized same-seed divergence to the first stochastic rollout.
- Completed repeatable one-update controls on both runtimes.
- Started the two-member 12-update synchronous Ray-2.58 probe and stopped it at
  one update/member when free memory reached 1.04 GiB during concurrent game use.
  No exploit occurred before the 24k cadence.
- Corrected its status to `incomplete` and added a tested target-step completion
  guard so Tune `TERMINATED` trials cannot contaminate ablation evidence.
- Audited `D:/iTRADER-offload`: retain it; it is live authoritative storage, not
  a disposable cache.

## 2026-09-24 — dataset-v2 R0 goal closeout

- Closed R0 as a mechanically credible but economically negative dataset-v2
  baseline. Measurement, controls, PPO activity, PBT mutation/transfer, replay,
  and generation-bound lineage pass; no v2 R0 arm passes net-plus-Sharpe.
- Kept the earlier replay-exact `00001`/`00003` checkpoints explicitly scoped to
  dataset-v1 mechanism evidence. Their shared ancestry and failed independent
  replication block promotion, and dataset-v2 supersedes them for selection.
- Fixed three future-run provenance defects test-first: atomic snapshot hashes
  exclude `.tmp` files, manifests serialize resolved PBT learning-rate domains,
  and best-checkpoint economic eligibility comes from the explicit promotion
  gate rather than the ranking metric. Healthcheck, 236 tests, diff check, HTML
  parsing, and file-scoped pre-commit pass.
- Set the next-goal order: review Ray 2.58 in isolation and prove an R0-only
  platform bridge; then R1/R2; then causal C0/C1/shuffled-C1; keep the
  System-One/CLM/Jev S-series research-only until the R-series solver/platform/
  evidence contract is frozen.
- Preserved the user's no-commit boundary: the branch remains at `0894116`, the
  index is empty, and no Jev/CLM research artifact was removed or rewritten.

## 2026-09-24 — final dataset-v2 R0 exact-cost gate

- Closed the pre-staged-target Ray-2.9 contamination path test-first. The final
  exact-1-bp arm passed strict synchronous-PBT lineage inspection with four
  members at the 96k gate and three verified exploits.
- Stopped deliberately after the continuation gate: population mean net return
  -0.05317%, Sharpe/sqrt(bar) -0.03815, 67.33% flat, and no jointly positive
  net/Sharpe report across 84 evaluations.
- Compared the valid staged arm: all members converged to short-only at
  +0.05785% net but -0.01558 Sharpe. PPO/PBT are responsive; economic promotion
  remains closed.
- Preserved compact per-member checkpoints/config/report/receipt evidence,
  comparison CSVs, the strict gate decision, and net-return/Sharpe/action plots
  under the dated dataset-v2 audit evidence directory.
- Updated the roadmap: freeze R0, review Ray 2.58 through an R0-only parity
  bridge, then begin bounded R1 as a matched representation challenger; retain
  P1 independent chronological blocks and defer causal regime conditioning to
  an explicit C-series study.

# 2026-09-24 — Dataset-v2 acquisition and R0 bridge

- Authenticated old Alpaca SIP access and built an immutable, split-adjusted,
  12-symbol one-minute snapshot for 2022-01-03 through 2026-09-22.
- Retained pre-market/RTH/after-hours raw and Parquet partitions; made Phase A
  loads RTH-only and session/interval-safe.
- Verified 10,370,064 unique bars, zero duplicate keys/OHLC failures/negative
  volumes, and a 14.97% naturally flat OHLC rate versus 100% in audited v1.
- Added filtered Parquet/DuckDB loading, data selectors, directory hashing,
  acquisition/build contracts, and focused TDD coverage.
- Completed 32-start deterministic controls plus matched fresh 12-update no-cost
  and exact-1-bp R0 arms. PPO remained active; neither arm cleared flat/long-only.
- Published the Markdown/HTML audit, evidence plots, exact snapshot manifest,
  and updated Phase A/Phase B roadmap.
- Started the matched v2 synchronous P0 and observed four genuine no-cost-arm
  exploits, two durable-checkpoint fallbacks, and no exploit skips before a
  Ray 2.9 save-after-restore race errored member `00001`. Enforced the declared
  stop gate before either cost arm began and preserved 107 report rows plus 22
  stable checkpoint-index records.
- Reproduced the exact actor-save ordering in a focused test and repaired the
  adapter: an exploited actor awaiting its first post-restore report reuses the
  attached donor checkpoint, then returns to ordinary checkpointing. The
  scheduler file passes 10 tests; the combined PBT/training/ablation slice
  passes 80 tests; healthcheck is green.
- The first verification gate reached ~97k steps per member with three genuine
  exploits before the same Ray assertion exposed a second restore path: stable
  self-reattachment did not mark the fresh actor as awaiting its first result.
  The stop gate fired, preserving all evidence. A second red/green regression
  moved the guard to checkpoint attachment itself; the 10 scheduler tests and
  80-test focused slice remain green. Another fresh gate is required.
- The next fresh gate reached ~97k for all four members with three genuine
  exploits and proved both restore paths could emit new results. It then stopped
  on the declared trial-failure gate: an upper donor's scheduled Ray save
  resolved empty, the exact same-step durable checkpoint was recovered, and Ray
  nevertheless scheduled a redundant save on the recreated actor before its
  first result. The scheduler now treats the exact immutable sidecar checkpoint
  as authoritative for synchronous upper donors and fails closed if it is
  absent. Two new red/green regressions bring the scheduler file to 12 tests;
  the focused 82-test slice, full 173-test suite, and healthcheck are green.
- Gate C proved the exact-sidecar selection path twice and completed two clean
  exploits before all four members reached ~97k. The fail-closed guard then
  caught a one-update in-flight overshoot: member `00002` had a 102,080-step
  selection row but only the preceding 97,248-step cadence snapshot. The
  training contract now retains an immutable snapshot for every synchronous-PBT
  report while reporting only cadence/terminal checkpoints to Tune. This keeps
  donor score and model state exact without changing PBT cadence or using actor
  saves. The 83-test focused slice, full 174-test suite, healthcheck, and diff
  check pass; Gate D is required before the matched three-arm search.
- Gate D completed the official 35-update four-member no-cost ablation with all
  trials `TERMINATED` at 170,152--170,416 trial-owned steps. It produced 140
  reports, 140 non-empty stable checkpoints, five genuine exploits, five exact
  `stable_checkpoint_selected` receipts, and zero donor lag, exploit skips,
  trial errors, required-metric nonfinites, or storage stops. The run crossed the
  former 97k, 102,080, and 120k failure regions and both exploited targets emitted
  fresh post-restore results. Member `00000` briefly reached +0.08116% 1 bp net
  and +0.01576 Sharpe for a three-report positive streak, but every terminal row
  was negative. Gate D passes checkpoint/search integrity and fails economic
  promotion; the matched three-arm P0 was not launched automatically.
- Audited Gate D restore provenance directly: all five retained restore copies
  match their selected donor checkpoint tree hashes. Hardened future inspection
  to require a generation-bound scheduler selection, exactly one target restore
  receipt, a same-generation target report, and a same-generation target
  checkpoint. Historical Gate D now requires an explicit legacy-inspection flag.
- Diagnosed a development-panel confound: horizon groups used different starts
  while `semantic_v2` exposes horizon. Added an explicit 8-start x 4-horizon
  crossed panel and made both controls and training consume it by content hash.
- Ran the final crossed 1 bp controls: flat 0.00000% net/0 Sharpe; long-only
  -0.09785%/+0.00827; short-only +0.05785%/-0.01558; random
  -1.31143%/-0.25857; seeded untrained FiLM exactly matched short-only. The panel
  removes the horizon/start shortcut but remains selection-informed.
- Prepared the matched 60-update synchronous P0 with `stream_worker`, family 00,
  random/generated tasks, four members, a narrower learning-rate range, and
  bounded entropy schedules. Focused tests and schema/CLI validation pass. No PBT
  arm was launched; the goal stops paused immediately before `--variant no_cost`.

# 2026-09-20 — Phase A raw-24k recovery and Ray-2.9 tail closure

- Recovered all four raw-net 24k lineages from immutable checkpoints after the
  original eight-actor topology exhausted 16 GB Windows RAM. The watchdog found
  no Ubuntu/WSL crash and no storage threshold breach.
- Retained member `00002` at 372,216 steps as a replay-exact transient challenger:
  +0.39258% 1 bp net, +0.02959 Sharpe, positive 2 bp repricing, and positive
  bullish/bearish regime returns. It failed the five-report, paired-block, and
  independent-lineage promotion gates.
- Added a declared Ray CPU cap, respected one rollout worker, durable
  fallback/exploit/skip receipts, retry from stable checkpoints when a scheduled
  save resolves empty, paused-upper-donor checkpoint reattachment, and a hard
  no-mutation guard at terminal exposure through red/green tests.
- Completed the final four-member synchronized live proof with all trials
  `TERMINATED`, one legitimate exploit at the 24k boundary, no terminal exploit,
  no errored trial, and exit code 0. Defined P1 as independent new-block
  replication plus selection-aware block/cost/regime evidence.
- Closed validation with 158 repository tests passing, a green healthcheck, and
  a clean second all-files pre-commit pass. The first hook pass only added
  missing final newlines to three historical generated summary JSON files.
- Aborted the required GitNexus refresh after the watchdog caught its WSL Node
  process creating an 82.46 GB crash dump that briefly filled C. The writer
  eventually released it and the verified move to E restored C to 76.84 GB.
  Ubuntu and all completed PBT evidence remained intact; GitNexus stays stale
  until the coordinated WSL restart activates dump routing to E.
- Preserved this work as an uncommitted review surface; the earlier requested
  commits remain `f045fde6` and `88361f05`.

# 2026-09-19 — Storage incident and NVMe offload

- Diagnosed the C-drive block with a deterministic 5 GB free-space check. The
  immediate cause was a 77.94 GB WSL crash dump for `/usr/bin/node`; Phase
  experiment outputs and `ray_results` contributed about 5.82 GB.
- Preserved the three dumps on E, moved `ray_results` and 37 experiment run
  directories to the D NVMe, and restored every active path through verified
  NTFS junctions. C/D/E ended with 78.25/163.85/259.79 GB free.
- Added a user-level `.wslconfig` that sends future dumps to E and retains one,
  effective on the next normal WSL restart. The live 76.95 GB Ubuntu VHDX was
  not moved because Hermes, OpenClaw, ZeroTier, and the browser harness are
  active inside that distro.
- Recorded byte counts, verification receipts, prevention settings, and the
  coordinated WSL move procedure in
  `docs/artifacts/research/phase-a-storage-incident-audit-2026-09-19.md`.

# 2026-09-19 — Phase A MLP-FiLM P0 autoresearch

- Added environment-step cost scheduling, independent PBT population seeds,
  training-only reward overrides, promotion streak/Sharpe/flat diagnostics,
  checkpoint cost/block/regime analysis, and an opt-in eligibility-aware donor
  metric through TDD.
- Completed the matched 0 bp/staged/exact-1-bp PBT study, exact checkpoint
  replay, matched deterministic controls, block/cost/regime diagnostics, two
  independent 70-update populations, and one eligibility-aware 40-update probe.
- Retained two shared-lineage discovery candidates with positive 1--2 bp
  economics, but independent replication failed. The sealed final test remains
  untouched; Phase A promotion is still blocked.
- Preserved the requested commit boundary: prior work was pushed as `f045fde6`
  and `88361f05`; this dated autoresearch remains uncommitted for separate
  review.

## 2026-09-12 — Phase A measurement and PBT implementation loop

- Used the iTrader architecture guardrails, TDD, deep-module design vocabulary,
  design-report structure, and a repo-specific adaptation of the NAO stack loop.
- Implemented `EpisodeEvidence`, `DataPartition`, truthful callback metrics,
  reset/task bounds, `semantic_v2`, held-out evaluation, Ray 2.9-compatible PBT,
  stable provenance, restore-aware lineage, all-member ablation reporting, and
  flat-schema capability reporting. Recurrent/LSTM solvers were not changed.
- Repeated the official JSON ablation path until the final run
  `phase_a_performance_validity_probe_20260912_031358` satisfied the contract:
  distinct starts, matched validation seed, fixed batch, real mutations/restores,
  12 unique iterations and 13,520 timesteps per PBT member, non-empty stable
  evidence, and generated capability reports.
- The bounded probe is not a performance claim. Historical winners remain
  quarantined; deterministic baselines, multiple seeds/costs, block uncertainty,
  overnight training, and untouched final-test evaluation remain next.
- Final validation: 87 tests passed with one Ray deprecation warning; healthcheck
  and the Phase A loadout schema passed on the Windows-native environment.

## 2026-09-12 — Windows recurrence/FiLM/AZR audit and documentation draft

- Source of truth: Windows clone `C:\Users\Admin\PROJECTS\iTRADER\itrader-azr`, `agent/Watson-dev`, starting HEAD `bec5c775d3adaafe758202c7175791f6e8b323a8`. The WSL clone was identified as stale and not used for current conclusions.
- Read local sibling `AZR-paradigm.pdf` (v2), method, reward equation, ablations and Appendix D; source findings and paper adaptation are attached as dated Markdown reports under `docs/artifacts/research/`.
- Revised active architecture, masterplan, implementation spec, recurrence/Sharpe design and continuity drafts. Phase B includes offline and online/replay; the shared recurrent trunk, family heads and FiLM placement remain open in the user's `grill-me` workflow.
- Preserved the existing staged `itrader/rl_lib/tasks/rewards.py` change; no solver code, defaults, environment configuration, branch, staging or commits changed by this pass.
- Failures: two documentation delegates each timed out at 600 seconds after partial writes. The finalization was split into bounded HTML synchronization and independent consistency review, rather than treating the partial writes as complete.
- Runtime validation blocked before pytest/healthcheck: Windows Python, PowerShell and CMD each failed with `Exec format error` in this WSL session. A separate static AST check passed for rewards, healthcheck and the three focused test files; it is not a runtime test pass. GitNexus remains stale.
- Exact coverage, final verification and remaining review status: `../artifacts/research/recurrence-docs-revision-audit-2026-09-12.md`. Earlier test counts below are historical, not repeated here.

## 2026-09-03 - Sharpe And Recurrent Proposal Reconciliation

- Changed: Kept `pnl_linear` as the sole proven/default reward, added bounded
  incremental `sharpe_only` and blended `pnl_sharpe` challengers, and added
  environment strategy-return history. Also added task-horizon/reward CLI
  overrides, repaired the Lazarus launcher, and created matched experiments.
- Added: reward selection CLI/config and independent reward RNG streams so a
  reward comparison does not silently compare different task/window sequences.
- Research: Preserved the recurrent-FiLM and risk-adjusted reward hypotheses,
  while correcting family IDs, newest-bar indexing, RLlib state-reset/BPTT
  claims, and the missing trade-state observation fields.
- Docs: Synchronized research Markdown/HTML, architecture, masterplan, decisions,
  current state, next actions, and handoff.
- Validation: The recreated lock-synchronized environment passes healthcheck and
  all 67 tests. Pre-commit also passes after routing its Python hooks through
  the repo launcher and teaching the launcher to prefer `.venv`. The ambient
  Python still has a NumPy versus pandas/pyarrow ABI mismatch.
- Follow-up: Execute alignment and reward ablations before implementing R1/R2
  recurrence.

## 2026-09-03 - Episode/Regime Alignment Ablation Scaffolded + Stash Audit

- Changed: Created ablation config (4 variants v0-v3), regime tagger, capability reporter, and Lazarus launcher script. Recovered `p_solve` hook in ps_metrics.py from stash@{1}. Updated next-actions.md, current-state.md, decisions.md.
- Validation: All Python files pass syntax check. Bash script passes `bash -n`. JSON config matches run_ablation.py schema.
- Notes: Training not yet run (OOM constraint with llama.cpp). Branch is clean fast-forward over Phase-B-Training. Stash@{0} proposer_llm.py confirmed superseded; stash can be dropped after user verification.
- Follow-up: Run ablation via Lazarus script, generate capability reports, compare variants on held-out windows.

## 2026-05-02 - Episode Alignment Feasibility Check

- Changed: Added `docs/project-state/` as a durable handoff layer and recorded the PPO episode-alignment assessment.
- Validation: Inspected recent commits, training config, environment reset/episode logic, reward computation, task contracts, current README/docs, and relevant tests.
- Notes: Current PPO uses `batch_mode = "complete_episodes"`, so the immediate risk is not reward computation crossing episode boundaries. The open question is whether optimization should mix episodes from different regimes/horizons inside each PPO update.
- Follow-up: Run a controlled ablation before changing defaults.

## 2026-05-02 - Recurrent Solver Frontier Clarification

- Changed: Added `solver-recurrence-assessment.md` and updated project-state docs to distinguish the FiLM MLP baseline, the existing window-encoder LSTM, and a future true recurrent RLlib policy.
- Validation: Inspected `SolverLSTMModel`, `SolverMLPFiLM`, active training model config, local Ray version, and RLlib recurrent helper source.
- Notes: The existing LSTM processes each rolling observation window but does not carry hidden state across environment steps.
- Follow-up: Prototype a corrected window encoder head before implementing fully stateful recurrence.
