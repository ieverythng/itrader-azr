---
name: itrader-architecture-check
description: Apply iTrader-specific architecture guardrails for solver, proposer, task contracts, capability profiles, MarketFramer, DecisionGate, RiskEngine, Executor, and PPO/RLVR training seams.
---

# iTrader Architecture Check

## Overview

Use this skill before or after meaningful iTrader architecture, training, solver,
task-contract, proposer, or online/shadow-mode changes. The goal is to preserve
working PPO/RLVR paths while making future proposer/framer/risk seams explicit.

## Quick Start

1. Read `AGENTS.md`.
2. Read `docs/architecture.md` and `docs/architecture/itrader_system_architecture.md`.
3. Read `docs/plans/itrader_masterplan.md` for phase context.
4. Run `tools/knowledge/status.sh` and treat stale graph output as advisory only.
5. Inspect nearest source and tests before proposing changes.
6. Apply the narrowest useful fix and validate with focused checks.

## Guardrails

- Keep `TaskSpec` as the canonical task/problem contract.
- Keep online execution semantics in `ExecutionIntent`, `CandidateTask`,
  `DecisionRecord`, and deterministic gate/risk/executor layers.
- Do not make solver code responsible for broker semantics, live risk limits, or
  execution policy.
- Preserve `SolverMLPFiLM` as the implemented baseline unless a task explicitly
  asks to change it.
- Treat `SolverWindowLSTMFiLM` as in development until implemented and tested on
  the current branch.
- Keep `WAIT` / `NO_ACTION` as a first-class online outcome.
- Keep LLM/proposer output behind validation; invalid candidates must not crash
  env reset or training.
- Use capability profiles from held-out or rolling evaluations before trusting
  online gating.

## Validation

Choose the narrowest relevant checks:

- `git diff --check`
- `./itrader/tools/run_repo_python.sh itrader/tools/healthcheck.py`
- `./itrader/tools/run_repo_python.sh -m pytest -q itrader/tests/test_task_contracts.py`
- `./itrader/tools/run_repo_python.sh -m pytest -q itrader/tests/test_train_config.py`
- `./itrader/tools/run_repo_python.sh -m pytest -q itrader/tests`
- `tools/knowledge/index_repo.sh` after major docs/code changes when GitNexus is stale

## Stop Conditions

- Do not broaden a docs pass into model/training code without explicit approval.
- Do not port experimental solver code across branches as a side effect.
- Do not normalize generated GitNexus wiki pages or imported research artifacts
  unless the task explicitly targets them.
