---
name: itrader-architecture-check
description: Trace existing iTrader entrypoints, owners and tests before meaningful implementation, research, review, environment or run/debug work; preserve solver, proposer, evaluation, runtime and online boundaries.
---

# iTrader Architecture Check

## Overview

Use this skill at the start of every meaningful implementation, research,
review, environment or run/debug scope. Reuse existing owners before adding
helpers; skill use grants no execution, qualification or Git authority.

## Quick Start

1. Read `AGENTS.md`.
2. Read `docs/architecture.md` and `docs/architecture/itrader_system_architecture.md`.
3. Read `docs/plans/itrader_masterplan.md` for phase context.
4. Run `tools/knowledge/status.sh` and treat stale graph output as advisory only.
   On native Windows use the installed Git Bash explicitly; missing `python3`
   or a stale graph is not permission to activate WSL or reindex during a run.
5. Trace public entrypoint -> owner -> existing helper -> nearest contract tests
   using [the source seam map](references/seam-map.md); record the route on the
   owned board before proposing another wrapper. Confirm against current source.
6. Apply the narrowest useful fix and validate with focused checks.

Run a bounded SkillOpt-style iteration (baseline -> mutate -> holdout gate -> accept/reject log) before finalizing major wording changes.
Use the seam map's bounded wording check; this is not a runtime benchmark.

## Guardrails

- Keep `TaskSpec` as the canonical task/problem contract.
- Keep online execution semantics in `ExecutionIntent`, `CandidateTask`,
  `DecisionRecord`, and deterministic gate/risk/executor layers.
- Do not make solver code responsible for broker semantics, live risk limits, or
  execution policy.
- Preserve `SolverMLPFiLM` as the implemented baseline unless a task explicitly
  asks to change it.
- R1 is implemented as an opt-in, stateless-per-decision window LSTM. Its CPU
  synthetic lifecycle receipt does not qualify GPU execution or economic edge;
  verify source and current receipts before reporting availability.
- Keep `WAIT` / `NO_ACTION` as a first-class online outcome.
- Keep LLM/proposer output behind validation; invalid candidates must not crash
  env reset or training.
- Use capability profiles from held-out or rolling evaluations before trusting
  online gating.

## Validation

Choose the narrowest relevant checks:

For the CPU/default control on native Windows with WSL off, use
`.venv\Scripts\python.exe itrader/tools/run_repo_python.py` instead of the shell
runner below. For CUDA follow the explicit reviewed profile invocation in
`docs/environment_strategy.md`; do not assume the default `.venv` has CUDA.
Verify the profile and current resource lease before execution; a CPU hook pass,
tensor smoke or synthetic lifecycle is not a market-performance certificate.

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
