# Next Actions

Last updated: 2026-05-02

## Now

- Define an `episode/regime alignment` ablation around the existing PPO baseline.
  - Scope: `itrader/rl_lib/train/train_solo_V2.py`, experiment configs, docs.
  - Compare: current `random_episode` baseline, `stream_worker` contiguous windows, `ret_norm_reset=stream`, and horizon-aligned task sets.
  - Validation: same seed set, same dataset slice, same train budget, held-out rolling-window evaluation.

- Add capability reporting by task family, regime tags, and horizon.
  - Scope: `itrader/rl_lib/tasks/capability_profile.py`, metrics/report scripts, `docs/iTRADER_Codex_Implementation_Spec.md`.
  - Expected output: a small JSON/CSV report with `p_solve`, expected utility, calibration error, sample count, and horizon/regime keys.
  - Validation: unit tests for validation/coercion and a smoke report from existing metrics.

- Record the solver-head reality before deeper training claims.
  - Scope: `README.md`, `docs/project-state/current-state.md`, maybe a training config comment.
  - Fact to preserve: current default solver is `SolverMLPFiLM`; LSTM exists but is not the default training head.

- Design a solver-head ablation for recurrence.
  - Scope: `itrader/models/`, `itrader/rl_lib/train/train_solo_V2.py`, tests, and experiment docs.
  - Compare: `SolverMLPFiLM`, corrected window LSTM/GRU/attention encoder, and later a true stateful recurrent policy.
  - Validation: identical task pool, identical dataset slices, held-out rolling-window evaluation, and capability metrics by family/regime/horizon.

## Next

- Add a proposer-on versus proposer-off comparison run template.
  - Scope: experiment config docs and/or a lightweight script.
  - Validation: reports task-pool acceptance rate, invalid task rate, solver return, drawdown, turnover, and capability uplift.

- Implement a shadow-mode online replay skeleton.
  - Scope: future `itrader/online/` package and existing task contracts.
  - Validation: replayed market window -> candidate tasks -> validation -> solver score placeholder -> `DecisionRecord`.

## Later

- Evaluate a fully stateful recurrent solver head after a corrected window encoder has been measured.
- Explore LoRA/QLoRA primarily on the proposer after enough validated candidate/outcome data exists.
- Add stronger regime labeling for market windows so capability estimates are not only horizon-based.

## Blocked

- True solver-quality conclusions require actual training/evaluation runs on the target workstation.
- Online decision thresholds require calibrated capability reports, not just training reward curves.
