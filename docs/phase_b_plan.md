# Phase B Plan (Concrete Execution)

Date: February 24, 2026
Status: **historical ordering; Phase B includes offline and replay/shadow tracks; solver choices remain under review**

This document records the original concrete Phase B milestone order. The current
scope is broader than the historical slice: Phase B includes offline/curriculum
work and online replay/shadow validation. Live broker execution remains out of
scope. The ordering below is a proposal pending the user's grill-me review; the
source lock and current active docs win on source-defined facts.

## Goals

1. Preserve Phase A PPO stability and reproducibility.
2. Establish typed task/proposer validation and an AZR-inspired bootstrap loop.
3. Measure solver quality with matched controls before expanding architecture.
4. Include online replay/shadow validation in Phase B while keeping live execution
   out of scope.

## Scope

In scope: proposer integration, task validation, task-pool generation, solver
metrics/evaluation, dependency stabilization, and the documented R-series design.
Out of scope: multi-node production, live trading, and unmeasured model-family
expansion.

## Milestones

### M0 — stabilization

Remove secrets, repair entrypoints, align dependency manifests, remove tracked
cache artifacts, and keep architecture/runbook docs current.

### M1 — proposer contract

Define/retain `BaseProposer`, shared validation/clamping, deterministic fallback,
and invalid-output telemetry.

### M2 — bootstrap AZR handoff

Keep `train_AZR.py` as a bounded task-pool generation plus solver handoff until a
separate proposer-evidence study supports a co-adaptive loop. Do not call the
bootstrap handoff jointly optimized AZR.

### M3 — evaluation guardrails

**Phase A foundation implemented:** chronological purged validation/final-test
separation, cost-aware episode evidence, bounded PBT, and stable manifest/config
lineage. Phase B still needs proposer-on/off controls, accepted/rejected task
accounting, baseline deltas, and promotion-scale uncertainty.

### M4 — solver recurrence (planned, not historical implementation)

Follow the proposed R0 -> R1 -> R2a -> R2c controls only after grill-me review,
preserving original R2b/R2c/R2d meanings and adding `R2c-CARRY`/`R2c-RESET`.
Require native state lifecycle tests, sequence-length/padding masks for both h and
c, reset/truncate semantics, and matched feature controls before any result is
promoted.

### M5 — replay/shadow validation (Phase B)

Exercise `MarketFramer` on replayed windows through solver scoring and deterministic
`DecisionGate`/`RiskEngine`/`Executor` seams. This is Phase B evidence work, not a
Phase C-only activity, and does not authorize live execution.

## Dependency authority

The old target line below is retained as history only:

```text
historical proposal: Ray/RLlib 2.47.1, Gymnasium >=0.29
```

The pre-migration lock captured by this plan was:

```text
Python >=3.11,<3.12
ray[rllib] == 2.9.0
gymnasium == 0.28.1
setuptools == 69.5.1
```

That block is the pre-migration reproduction baseline. The active source lock is
now Ray/RLlib 2.58.0, Gymnasium 1.2.2, and PyArrow 17.0.0. See
`docs/environment_strategy.md` and the dated Ray migration audit. R0 remains on
the explicit legacy API stack until the native RLModule/EnvRunner port passes its
separate parity gate.

## Execution order

1. Keep family `00`/R0 runnable and metrics truthful.
2. Run alignment/reward controls with fixed task identity; no recurrence/reward
   interaction claim before independent evidence.
3. Implement R1 through TDD and measure it; R1 need not win, but it must be
   correct and measurable.
4. Implement native R2a, then R2c and the carry/reset pair.
5. Add R4-v1 explicit solver-state fields identically to FF/recurrent controls.
6. Run proposer study separately from solver R-series cells.
7. Enter replay/shadow online work only after capability evidence is held out and
   calibrated.

## Definition of done for the historical Phase B slice

- task generation and validation are selectable;
- bootstrap proposer -> validated pool -> solver handoff is reproducible;
- health/test/pre-commit checks pass in a compatible environment;
- proposer claims report accepted tasks, rejection, compute, and final-test use;
- solver and online promotion gates remain evidence-gated.

The WSL documentation audit could not execute tests because this session's runner
resolved the Windows `.venv/Scripts/python.exe` and returned `Exec format error`;
this is a session/environment observation, not a universal WSL limitation. See the
dated revision audit.
