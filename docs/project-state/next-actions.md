# Next Actions

Last updated: 2026-09-03

## Now

- Define an `episode/regime alignment` ablation around the existing PPO baseline. **[SCAFFOLDED 2026-09-03]**
  - Scope: `itrader/rl_lib/train/train_solo_V2.py`, experiment configs, docs.
  - Compare: current `random_episode` baseline, `stream_worker` contiguous windows, `ret_norm_reset=stream`, and horizon-aligned task sets.
  - Validation: same seed set, same dataset slice, same train budget, held-out rolling-window evaluation.
  - Config: `itrader/scripts/experiments/configs/episode_regime_alignment_ablation.json` (4 variants v0-v3).
  - Launcher: `bash itrader/scripts/experiments/run_ablation_lazarus.sh` (checks llama.cpp is down first).
  - Regime tagging: `itrader/scripts/tag_pool_regimes.py`. Capability report: `itrader/scripts/generate_capability_report.py`.
  - Known limitation: v3 uses FiLM head (not LSTM) — SolverLSTMModel has an obs-shape assumption (`flat_dim // win`) that breaks with task embeddings. Defer LSTM to solver-head ablation (item 4).

- Add capability reporting by task family, regime tags, and horizon.
  - Scope: `itrader/rl_lib/tasks/capability_profile.py`, metrics/report scripts, `docs/iTRADER_Codex_Implementation_Spec.md`.
  - Expected output: a small JSON/CSV report with `p_solve`, expected utility, calibration error, sample count, and horizon/regime keys.
  - Validation: unit tests for validation/coercion and a smoke report from existing metrics.

- Record the solver-head reality before deeper training claims.
  - Scope: `README.md`, `docs/project-state/current-state.md`, maybe a training config comment.
  - Fact to preserve: current default solver is `SolverMLPFiLM`; LSTM exists but is not the default training head.

- Design a solver-head ablation for recurrence. **[PRIORITY ELEVATED 2026-09-03]**
  - Scope: `itrader/models/`, `itrader/rl_lib/train/train_solo_V2.py`, tests, and experiment docs.
  - Compare: `SolverMLPFiLM` (R0), corrected window LSTM encoder (R1), true recurrent LSTM with persistent hidden state (R2), recurrent GRU (R3), recurrent + task-FiLM on initial state (R4).
  - **Key requirement**: R2-R4 must use RLlib's native recurrent policy interface — hidden state persists across environment steps within an episode, resets at episode boundaries. This is the "compounding effect" where reward bias, position memory, and regime awareness carry forward through windows.
  - Prerequisite: Fix obs-shape split (explicit `price = obs[:, :win*n_feat]`, `task = obs[:, win*n_feat:]`) — no more `flat_dim // win` inference.
  - PPO config for recurrent variants: `shuffle_sequences=False`, `recurrent_policy=True`, sequence-aware minibatching.
  - Validation: identical task pool, identical dataset slices, held-out rolling-window evaluation, capability metrics by family/regime/horizon. New metric: **memory utilization** (does long-horizon performance improve more than short-horizon?).
  - Full design: `docs/project-state/solver-recurrence-assessment.md` (§ "2026-09-03 Update").
  - Ablation doc: `docs/research/recurrent-solver-ablation.md` + `.html`.

## Next

- Wire `pnl_sharpe` to Family 01 and implement Sharpe-aware rewards for Families 02-03. **[DESIGN COMPLETE 2026-09-03]**
  - Phase 1 (~2h): Update `families.py` — `secondary_01` default_reward_id → `pnl_sharpe`. Tune params: `w_sharpe=0.5`, `sharpe_window=32`, `sharpe_target=1.0`. Unit test + smoke training.
  - Phase 2 (~1d): Implement `reward_sharpe_dominant` + `_compute_vol_expansion_bonus` for Family 02. Register, unit test, smoke train.
  - Phase 3 (~1-2d): Implement `reward_sharpe_trend` + `_compute_trend_persistence_bonus` for Family 03. Requires `ctx["bars_in_position"]` and `ctx["entry_price"]` from env.
  - Full design: `docs/research/sharpe-reward-design.md`.
  - Validation: capability reports by family show Sharpe-aware solvers outperform PnL-only on families 01-03.

- Add a proposer-on versus proposer-off comparison run template.
  - Scope: experiment config docs and/or a lightweight script.
  - Validation: reports task-pool acceptance rate, invalid task rate, solver return, drawdown, turnover, and capability uplift.

- Implement a shadow-mode online replay skeleton.
  - Scope: future `itrader/online/` package and existing task contracts.
  - Validation: replayed market window -> candidate tasks -> validation -> solver score placeholder -> `DecisionRecord`.

## Later

- **Proposer model selection + loop closure (Phase B closure criterion).** Per Notion log (23/08/26): lock the proposer model, hook it to the task generation loop, and validate end-to-end.
  - Candidate models: Qwen3.8-27B 4-bit (~14GB VRAM — tight on 16GB card), Gemma-class with hardware LoRA (~6GB), or CLIProxyAPI GPT-5 (no local VRAM, API cost).
  - OOM constraint: proposer must run alongside OR replace the llama.cpp server. Separate process + API communication avoids VRAM contention.
  - Phase B closure = proposer hooked to loop + solver head 00 fully developed (FiLM R0 for family 00, recurrent R2c for families 01-03) + all four families have working reward functions.
  - Additional ablations needed: proposer-on vs proposer-off, proposer model comparison, task pool acceptance rate by model.

- Evaluate a fully stateful recurrent solver head after a corrected window encoder has been measured.
- Explore LoRA/QLoRA primarily on the proposer after enough validated candidate/outcome data exists.
- Add stronger regime labeling for market windows so capability estimates are not only horizon-based.

## Blocked

- True solver-quality conclusions require actual training/evaluation runs on the target workstation.
- Online decision thresholds require calibrated capability reports, not just training reward curves.
