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

## Chat Coordination

Use `docs/coordination/README.md` for the board-based workflow ported from UAH.
GRILL-IMPLEMENT owns human decisions, cross-chat dispatch, sequencing, and run
authorization. DEV owns implementation, RESEARCH owns investigations, and
TRAINING/TESTING owns authorized validation and experiment execution.

- Each parent writes only its own board. GRILL checks boards at turn starts and
  dependency gates; boards do not wake chats or provide continuous monitoring.
- GRILL may send finite scoped dispatches. DEV, RESEARCH, and TRAINING/TESTING
  must not message or delegate to one another. Routine findings, blockers, and
  final handoffs belong on their owned boards, not unsolicited messages to GRILL.
- Urgent safety/resource incidents may be escalated directly to GRILL. A routine
  completion or stale board is not an urgent incident. Direct human instructions
  remain authoritative; record a resulting scope change on the owned board.
- Encourage bounded subagents with disjoint write sets. Children report to their
  own parent; that parent integrates the result and remains accountable. This is
  an organizational tree, not a new runtime hierarchy or a fixed agent count.
- Declare branch, HEAD, dirty/untracked snapshot, write scope, evidence, budget,
  stop rules, and next decision. A changed relevant snapshot invalidates its
  review/run authorization; idle or elapsed time alone does not prove staleness.
- Only one heavy local job may hold the explicit GRILL resource lease. Do not
  overlap full tests, training, PBT, model serving, or collection by assumption.
  Run leases are scoped human/GRILL records, not an implemented atomic lock.
- Boards do not grant Git staging, commits, pushes, data-access rights, statistical
  qualification, or release authority. Preserve existing dirty work and WSL-off
  constraints. UAH runtime/provider integration remains deferred.

Keep one current scope, a short queue, and compact final receipts per board.
Durable decisions and qualification evidence stay tracked; repetitive local
logs may go in ignored `docs/coordination/_local/`. Do not hide existing evidence
or duplicate whole reports across boards, plans, and handoffs.

## Review Tool Boundaries

`REVIEW.md` remains the independent review authority; the `code-review` skill
adds Standards/Spec checks. High-risk changes require its two-reviewer gate.
No-Mistakes is a pinned, experimental complementary workflow, not a qualified
replacement. Importing or invoking a skill grants no permission to commit,
rebase, reset, push, create a PR, initialize a daemon, change hooks, auto-fix
unrelated files, or bypass a failed review. Local instructions and human scope
override broader vendor defaults. See `docs/coordination/review-workflow.md` for
the bounded pilot contract; no actual tool pilot is authorized by this port.

For new behavior, prefer a small named set of public-contract integration and
end-to-end scenarios with independent oracles: episode/reset semantics, selector
and restore identity, runner evidence and fail-closed data gates. Keep focused
deterministic tests for distinct invariants; do not grow private-method tests or
delete existing tests to meet a numerical target. Synthetic E2E can certify a
lifecycle without proving trading edge. Live models/data are not ordinary CI
dependencies. Track the boundary caught, not just a larger green test count.

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

On native Windows with WSL off, use `.venv\Scripts\python.exe
itrader/tools/run_repo_python.py` in place of the shell launcher below. Keep
the same check scopes; do not start WSL to resolve an interpreter.

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
