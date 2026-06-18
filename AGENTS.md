# Agent Guide

This repository is an autonomous trading research platform for an AZR-inspired
proposer-solver loop. Its main design constraint is separation of task
generation, task-conditioned policy learning, deterministic online gating, and
future execution/risk shells.

## Start Here

Before a meaningful change:

1. Inspect the working tree and branch.
2. Read `docs/README.md` for the current documentation map.
3. Read `docs/architecture.md` and `docs/architecture/itrader_system_architecture.md`
   for active architecture.
4. Read `docs/plans/itrader_masterplan.md` for phase ordering and current seams.
5. Read nearby tests before touching training, environment, model, reward, or
   task-contract code.
6. Run `tools/knowledge/status.sh` before architecture-sensitive work.

Use `docs/project-state/` for handoff continuity. Treat `docs/knowledge/wiki/`,
dated handoffs, and research artifacts as evidence and provenance, not as the
current contract unless an active doc points to them.

When documentation disagrees, use this precedence:

1. current source, tests, configs, and this file
2. active architecture and masterplan docs
3. project-state handoff/current-state docs
4. generated GitNexus wiki pages
5. historical handoffs and imported research artifacts

Call out meaningful drift instead of silently choosing a convenient version.

## Architectural Invariants

- `TaskSpec` remains the canonical task/problem contract.
- `ITraderEnv` owns market-window observation assembly, task-conditioned reset,
  and reward-context plumbing.
- `RewardEngine` owns solver reward calculation and finite reward components.
- The solver learns policy/value behavior; it does not define broker semantics,
  risk limits, or live execution rules.
- `SolverMLPFiLM` is the current implemented baseline solver head.
- `SolverWindowLSTMFiLM` is a Phase B frontier under investigation and must not
  be documented as available on this branch until code and tests land here.
- `BaseProposer` and proposer backends generate candidate tasks; invalid output
  must be contained by validation rather than crashing training.
- `CandidateTask`, `ExecutionIntent`, `CapabilityProfile`, and `DecisionRecord`
  are explicit future online/shadow-mode seams. Do not fold broker/risk semantics
  back into `TaskSpec`.
- `MarketFramer`, `DecisionGate`, `RiskEngine`, and `Executor` are Phase C online
  roles. Risk and execution must stay deterministic at the shell.
- `WAIT` / `NO_ACTION` is a first-class online outcome; no future online path
  should force a trade when confidence or edge is absent.

Preserve the intended control split:

```text
offline: market data -> proposer/curriculum -> TaskSpec -> sampler/env -> solver -> metrics/capability
online: market window -> MarketFramer -> CandidateTask -> solver scoring -> DecisionGate -> RiskEngine -> Executor
```

## Deslop And Scope Guardrails

- Prefer behavior-preserving cleanup unless behavior is clearly unsafe or
  self-contradictory.
- Keep orchestration code focused on routing and sequencing; push parsing,
  validation, and backend-specific details into explicit helpers or adapters.
- Replace repeated magic branching with small registries only when supported
  cases need to be obvious.
- Do not deslop generated artifacts, data files, caches, or imported research
  material as collateral work.
- Do not port experimental solver code between branches unless the task
  explicitly includes code implementation and validation.

## GitNexus Knowledge Layer

This repo has GitNexus tooling under `tools/knowledge/`.

- Check freshness with `tools/knowledge/status.sh`.
- Explore with `query -> context -> source`.
- Debug with `query symptom -> context suspect -> source`.
- Refactor with `context + impact` before edits when the graph is fresh.
- Confirm graph findings with direct source reads; source wins on disagreement.
- Refresh with `tools/knowledge/index_repo.sh` after major docs/code changes or
  when the graph is stale.
- If a shared HTTP backend is running, prefer
  `GITNEXUS_USE_HTTP=1 tools/knowledge/codex_with_gitnexus.sh` and avoid local
  graph commands that contend for its database.

See `docs/knowledge/WORKFLOWS.md`.

## Validation

Run the narrowest relevant checks first, then widen based on affected seams:

1. `git diff --check`
2. focused tests for touched code
3. `./itrader/tools/run_repo_python.sh itrader/tools/healthcheck.py`
4. `./itrader/tools/run_repo_python.sh -m pytest -q itrader/tests`
5. `./itrader/tools/run_repo_python.sh -m pre_commit run --all-files`
6. `tools/knowledge/status.sh` and, when needed, `tools/knowledge/index_repo.sh`

For docs-only passes, still run the repo healthcheck and full test suite when
practical because pre-commit includes both. If local environment resolution
fails, report the exact interpreter issue and the closest successful check.

## Documentation Hygiene

- Keep top-level docs small and navigable.
- Keep durable operational truth in active docs listed by `docs/README.md`.
- Keep Markdown as source of truth for active review docs and maintain an HTML
  companion when the docs contract calls for it.
- Put dated investigations, imported research references, and generated reports
  under `docs/artifacts/`.
- Keep `docs/plans/itrader_masterplan.md` as the canonical cross-phase plan.
- Update architecture docs when behavior or intended seams change.
- Stage AGENTS, docs, skills, and moved/removed docs together when asked to
  deliver a complete docs/architecture pass.
