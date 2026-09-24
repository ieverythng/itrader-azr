# Session Log

## 2026-09-14 — Ray 2.58 dependency migration and PBT acceptance

- Used the iTrader architecture guardrails and kept `TaskSpec`, `ITraderEnv`,
  `RewardEngine`, proposer validation, R0, and the future deterministic online
  shell boundaries unchanged.
- Resolved Ray/RLlib 2.58.0 first in an isolated uv project. PyArrow 13 failed
  resolution; PyArrow 17.0.0 plus Gymnasium 1.2.2 resolved and passed.
- Promoted the tested versions into the root uv lock, Conda declarations,
  constraints, and cluster setup. Added Ray 2.58 bridges for Tune context,
  storage, reporting/checkpoints, trial paths, worker/EnvRunner properties,
  evaluation metric nesting, and removed UnifiedLogger.
- Ran a one-iteration control and a two-member, six-iteration PBT acceptance.
  The PBT run recorded a real exploit/restore at iteration 5 from an iteration-4
  donor. A fresh process replayed the post-exploit iteration-6 checkpoint with
  zero delta for optimization reward, gross/net return, and net Sharpe.
- Native Windows emitted Ray child-process access-violation teardown messages
  while all drivers and acceptance checks still exited successfully. WSL2/Linux
  long-run validation remains required.
- No RLModule, recurrent/LSTM, proposer, risk, or execution architecture was
  introduced. The native RLlib stack remains separately parity-gated.

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
