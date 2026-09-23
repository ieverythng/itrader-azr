# Session Log

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
