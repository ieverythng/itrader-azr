# Session Log

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
