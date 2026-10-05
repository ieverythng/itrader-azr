# Solver Recurrence Assessment (historical evidence, non-executable)

Last reconciled: 2026-10-01
Status: **superseded as implementation recipe; retained as provenance and constraints**

This file records the earlier assessment. The active R-series provenance and
experiment proposal is `docs/research/recurrent-solver-ablation.md`, while the
current status registry is `docs/research/itrader_research_dashboard.md`; do not
copy an executable recipe or treat this historical proposal as settled
architecture.

## Branch reality

The active solver is `SolverMLPFiLM`. `SolverLSTMModel` is a reset-every-forward
window encoder, not a stateful RLlib policy. It reshapes the full observation and
infers feature width from `flat_dim // window`; with appended task embedding this
is incompatible with the current `[T*F | task_embedding]` contract. It discards
LSTM state and returns incoming state unchanged.

At this assessment's snapshot, R1, R2a, R2b, R2c, R2d, R3, and R4-v1 were design
cells only. R4-v1 has since been implemented in the working tree and qualified
with R0, but failed economic promotion; R1-R3 remain unimplemented. Model
registration is not model selection; `train_solo_V2.py` still selects R0.

## Durable conclusions

- Split market and task dimensions explicitly. The newest chronological bar is
  the final row of the reshaped window.
- Preserve the original R-series identities from commit `4732a98`: R2b is
  init-only FiLM, R2c is per-step FiLM, and R2d is value-head-only init+per-step.
  The simplified `bec5c77` per-step R2b label maps to original R2c; old R4 init
  maps to original R2b. Current solver-state observability is `R4-v1`.
- R1 versus newest-bar R2 changes input information, cold-start history,
  conditioning, and state lifetime. It is not an isolated memory proof.
- Add same-input/same-conditioning train-from-scratch `R2c-CARRY` and
  `R2c-RESET`; treat zero-state evaluation as OOD diagnostic evidence only.
- Distinguish input concat, pre-cell FiLM, post-cell/readout FiLM, and initial-
  state FiLM. A post-cell/readout claim is readout-only only when no other task
  path (such as concat) reaches the cell; with concat, the transition is already
  task-dependent and FiLM is an additional factor. Placement/order remains open.
- Keep input window, BPTT/unroll (`max_seq_len`), episode horizon, and
  reward-statistics window separate. Overlapping observations do not create new
  chronological timesteps. Fragment detach is not reset; reset only when the
  environment actually resets at real episode termination, while truncated
  bootstrap is handled separately.
- Valid sequence lengths, padding masks, and last-valid h/c preservation are
  mandatory. `complete_episodes` is not full-episode BPTT. Do not invent
  recurrent flags.
- R4-v1 must add explicit position/progress/PnL features to matched feed-forward
  and recurrent controls. OHLCV-only recurrence cannot reconstruct actions,
  fills, or position exactly.
- The default task encoder omits reward parameters, bar resolution, regime tags,
  difficulty, metadata, and start index while exposing a bounded seed feature.
  Freeze omitted fields initially or version a semantic encoder.
- Current lock authority is Ray/RLlib 2.58.0 and Gymnasium 1.2.2 from
  `pyproject.toml`/`uv.lock`; the old Ray 2.9/2.47 plan lines are historical.
  R0 still selects the legacy API stack until the native new-stack parity gate.
- Candidate-specific state/replay is required for online multi-task scoring; do
  not mutate one shared recurrent state repeatedly.

## Current user-review boundary

Phase B includes offline/curriculum work and online replay/shadow work; Phase C is
later operationalization. Families `00`–`04` are a roadmap, and `00`–`02` sharing
`Discrete(3)` output width does not settle input schemas, task/reward semantics, or
state ownership. Shared recurrent trunk plus family-specific policy/value heads
remains proposed, not accepted. `RewardEngine` remains the environment-owned
reward seam, not a mandatory learned reward head.

## Evidence boundary

No recurrent training, PPO smoke, full-suite pass, calibrated capability claim,
profitable strategy, or reproduced AZR result is reported. The independent dated
code audit is the evidence source for exact source findings and the blocked WSL
runtime attempt. This document is intentionally not an implementation guide.
