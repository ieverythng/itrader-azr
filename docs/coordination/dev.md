# DEV Board

- Owner chat: `01a092eb-9aa9-7571-b723-d91a84e78ea3`.
- Seeded by GRILL: 2026-10-09 08:42 UTC; DEV becomes sole writer at dispatch.
- Status: `active`; focused synthetic-test lease `LEASE-DEV-R1-FOCUSED-01`.
- Branch / base: `agent/Watson-dev` / `3a3583d`; dirty/untracked snapshot unsealed.

## Current Scope: R1-IMPLEMENT-01

Implement the already accepted R1 window encoder and explicit model-selection /
restore identity using TDD. Read [readiness](../project-state/r1-development-readiness.md),
[R-series research](../research/recurrent-solver-ablation.md), nearby source/tests,
and [coordination](README.md). Keep corrected Ray-2.58 compatibility, R0 defaults
and state-dict identity; no recurrent state across decisions, no PPO recipe change.

Write scope: R1 model/registry, required selection/loadout/training/restore
integration, focused tests, and active R1 availability/next-action docs. Do not
write other parent boards, `AGENTS.md`, coordination index/grill/review docs,
or the pinned skill port. Do not integrate native Ray migration or new families.

## Pre-edit Snapshot: R1-IMPLEMENT-01

- Started: 2026-10-09 08:50 UTC. Implementation stop: 09:25 UTC; GRILL focused
  test lease expires 09:30 UTC. Maximum implementation window: 35 minutes;
  focused test runtime is cumulatively capped at 10 minutes by the lease.
- Base and HEAD: `3a3583d109af7a4c094124e73ad84008f371d8e3`; branch
  `agent/Watson-dev`; staged diff empty. Shared worktree is dirty: 35 tracked
  paths modified and 54 untracked path entries at dispatch; all pre-existing
  changes are preserved and remain outside this scope unless explicitly named.
- Scope intent: implement only the accepted stateless-per-decision R1 window
  LSTM/FiLM model, structural selection, train/loadout construction, and current
  R1 implementation-status docs. No native Ray migration, PPO recipe change,
  Q23/Q26 work, real data, training, replay, staging, commit, or push.
- Write set: `itrader/models/solver_window_lstm_film.py` (new),
  `itrader/models/task_film.py` (new), `itrader/models/solver_mlp_film.py`,
  `itrader/models/registry.py`, `itrader/rl_lib/train/train_solo_V2.py`,
  `itrader/tests/test_solver_selection.py`,
  `itrader/tests/test_semantic_observation_contract.py`,
  `docs/architecture.md`, `docs/architecture/itrader_system_architecture.md`,
  `docs/project-state/r1-development-readiness.md`,
  `docs/project-state/current-state.md`, `docs/project-state/next-actions.md`,
  and `docs/project-state/handoff.md`. Generated HTML companions may be updated
  only if the existing documented renderer is deterministic and scoped.
- Read dependencies: `AGENTS.md`, `REVIEW.md`,
  `docs/coordination/review-workflow.md`, `docs/coordination/grill.md`,
  `docs/coordination/README.md`, `docs/project-state/decisions.md`,
  `docs/plans/itrader_masterplan.md`, `itrader/models/observation_contract.py`,
  `itrader/rl_lib/evaluation/policy_baselines.py`,
  `itrader/rl_lib/evaluation/run_policy_baselines.py`,
  `itrader/rl_lib/evaluation/screen_completion.py`, and related tests.
- Excluded: every other pre-existing dirty/untracked path, parent coordination
  boards except this DEV board, `AGENTS.md`, review rules, Q23/Q26 artifacts,
  evidence/data/run outputs, and all historical checkpoints.
- Baseline SHA256 (current working bytes before this scope):
  `solver_mlp_film.py` `96bf09a5e51c5661e89d417bde6a60ec910f4c3aa940278352703ab3f260bd38`;
  `registry.py` `64a16d191cd9d6f6de9597729bba6a21b3c3baca91bab0563ec11eabd4ca24af`;
  `observation_contract.py` `e3deb74a2b1c82513383797e871d10784a61b398b5608af0649963d193a4d741`;
  `train_solo_V2.py` `8428d4b8737cae2fe1839c6776be0a5697e010ab5a8ec11705330fa88afda060`;
  `test_solver_selection.py` `a51bd3850f6c517dd03c0c33e1684c7f9f3e06ab17ea326b2f1994df525775a2`;
  `test_semantic_observation_contract.py` `b0b7225b24df4934a34a2616ecbb442a5c6a2261d8139ec7815c06d779d740d1`;
  `REVIEW.md` `b92c8e19dbf9f5ca5dbc5ec632afe4be4084d9d1acfe313e1021f5d1321ad167`.
- Test strategy: compact public-contract coverage, not a growing matrix of
  implementation-shaped tests. First preserve the existing fail-closed selector
  and R0 tests. Add one tagged synthetic-Parquet path from semantic observation
  through selected R1 model to logits/value, batch-order and prior-call
  invariance, and zero-state R1/R1+R4 initial parity; add one bounded adversarial
  config/restore-identity check for unknown/conflicting selectors and malformed
  schema. Keep focused architecture math/gradient assertions only where that
  public integration cannot establish the accepted Q15/Q18 contract. No Ray
  algorithm/checkpoint job in this lease; a synthetic lifecycle smoke requires
  a later review and separate lease.
- Allowed checks under `LEASE-DEV-R1-FOCUSED-01`: named synthetic R1/R0 contract
  and public-interface tests only, with each source hash captured before the
  command and no concurrent writes; cumulative test runtime <=10 minutes; stop
  on expiry, source drift, contention, or resource breach. No full suite,
  pre-commit, Ray jobs, server, real data/outcomes, or performance claims.
- Commands to run after edits are frozen: `./itrader/tools/run_repo_python.sh
  -m pytest -q itrader/tests/test_solver_selection.py
  itrader/tests/test_semantic_observation_contract.py`; then `git diff --check`.
  Record exact source hashes, command, exit code, wall time, and test count below.

## Queue

1. `terminal_screen_only` Q23 role adapter and fail-closed consumer tests, after
   the current R1 scope and its prerequisite decision review.
2. Independent R1 gate reviews and exact snapshot receipts.

## Pending Evidence / Handoff

Q27 repair evidence is in
[the mechanical repair receipt](../artifacts/research/evaluation-mechanical-repairs-2026-10-07.md).
Prior green tests do not qualify new R1 bytes. Record new commands/results,
requested/observable reviewer identity, changed paths/hashes, and remaining gates.
Implementation may proceed; real screen execution is still closed on Q23/Q26.
