# Session Log

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
