# Coordination

This is the iTrader adaptation of UAH's board-based development workflow.
It coordinates people and chats; it does not couple the UAH runtime to iTrader.
Architecture and experiment authority remain in [AGENTS](../../AGENTS.md),
[the masterplan](../plans/itrader_masterplan.md), and [REVIEW](../../REVIEW.md).

## Boards And Ownership

| Parent | Owned board | Responsibility |
|---|---|---|
| GRILL-IMPLEMENT | [grill.md](grill.md) | Human decisions, dependencies, dispatch, integration, and run leases |
| DEV | [dev.md](dev.md) | Test-first implementation and independent review integration |
| RESEARCH | [research.md](research.md) | Primary-source investigations and decision-ready proposals |
| TRAINING/TESTING | [training-testing.md](training-testing.md) | Frozen-source validation, experiment execution, and run evidence |

GRILL initially seeds boards; after dispatch, each parent is their sole writer.
The human can change any scope directly. The receiving parent records that
change and stops conflicting work; a board never overrides human instructions.

## Dispatch And Handoff

GRILL sends finite scopes to parents. Outside the bounded repair exception below,
parents do not message/delegate to peers or send routine unsolicited completion messages to GRILL.
They update boards at scope start, actionable blocker, and final handoff.
Direct escalation to GRILL is reserved for urgent safety/resource incidents
(for example unintended outcome access or an unauthorized running job).
Routine missing decisions stay on the board.

### Bounded DEV--TRAINING Repair Loops

Human-approved exception (2026-10-13): GRILL may name a finite paired scope under
`itrader-stack-iteration-loop`. DEV owns production implementation and test-first
repairs; TRAINING owns reproduction and frozen-source execution. Concise direct
messages may carry the failing command/evidence or a hash-bound repair handoff,
but each parent records scope changes, evidence and closure on its own board.
No peer delegation to RESEARCH or unsolicited routine messages to GRILL follow.
The pair cannot authorize another run, change a lease, choose unresolved research
terms, or ship Git changes. A source change invalidates the affected run approval.
GRILL's heartbeat checks actionable board changes, not every message; it is not
continuous monitoring. Stop on unresolved decisions or failed prerequisites;
record routine blockers on the board. Escalate directly only urgent safety/resource incidents.

Every meaningful scope must start with `itrader-architecture-check`: trace the
existing public entrypoint and owner before adding scripts. A launcher failure
belongs in DEV, with TRAINING providing a minimal reproduction, not an expanding
set of experiment-local converters. Native argument lists and existing repository
launchers are preferred. Distinguish a failed launch from a CUDA/kernel failure.

Each scope records:

- Scope ID, parent chat ID, status, update timestamp, and exact next action.
- Branch and HEAD plus the relevant dirty/untracked file hashes or manifest.
- Allowed write set, read dependencies, and excluded behavior.
- Time/resource budget, stop rules, and any GRILL run-lease ID.
- Evidence links, commands and exit codes, independent review status, and the
  exact pending human decision. Unknown identity or qualification stays unknown.

Keep one active scope, at most three queued items, and at most three compact
closed summaries. Link detailed evidence instead of duplicating it. Supersede
decisions explicitly; do not silently delete unresolved question IDs.
GRILL's decision journal links durable accepted decisions to the project-state
log when implementation/promotion is integrated.

Parents may use bounded subagents with disjoint write sets and resource budgets.
Children report upward; only the parent writes its board and integrates patches.
No fixed binary branching, new runtime agents, or authority escalation is implied.

## Freshness And Resource Gates

GRILL checks boards at the start of a turn, after a handoff, and before advancing
a dependency. No board read schedules a watcher or wakes an idle chat. Check
scope ID, human decision, branch/HEAD, dependency hashes, and artifact availability;
age alone does not establish correctness or staleness.

Source changes in the reviewed/run scope invalidate its receipt until compared
and requalified. Distinguish implementation complete, independently reviewed,
test verified, run authorized, and statistically qualified. A final chat message
does not replace those receipts.

GRILL records one heavy-job lease with owner, exact job, source/config hashes,
budget, start/expiry or completion rule, and stop conditions. Full tests, training,
PBT, inference serving, and collection must not overlap without an explicit
resource exception. This is coordination, not an atomic filesystem lock.
Light read-only planning can continue in parallel. Stop an owned job on lease
expiry, source drift, unauthorized data access, or a failed prerequisite; report
the reason rather than silently retrying a new configuration.

## Evidence Without Churn

Boards, accepted decisions, source/config hashes, qualification certificates,
and compact review/validation receipts are durable evidence. Raw repeated local
logs can use ignored `_local/`; ignored content is not sufficient for a durable
qualification claim. Promote the minimal receipt needed to reproduce a verdict.
Never ignore existing research/review directories wholesale or delete their
history. Boards do not need HTML copies or per-probe dated report files.

See [review-workflow.md](review-workflow.md) for independent review and the
experimental No-Mistakes boundary. No board grants staging, commit, push, model
API spending, paid feed access, acquisition, or statistical qualification.

Use the existing masterplan, next-actions, readiness and research evidence as
scope inputs; boards record ownership and deltas, not parallel roadmaps. New
test work should favor named public-contract integration/E2E scenarios with
independent oracles. Retain small deterministic invariant tests and existing
coverage, not a numerical target of fewer tests or a happy-path-only replacement.
Do not infer measured token savings from fewer tests or fewer report files.

## Current Dependency Order

1. R1 source and synthetic contract tests may proceed under the accepted
   capacity/readout/selector contract. Real performance runs remain closed.
2. Q23: implement and review `terminal_screen_only`, bind the approved 2020-2021
   candidate envelope to exposure clearance and metadata evidence. Window/method
   approval is not an assertion that the data are unused.
3. Q26: declare the confirmatory endpoint, calendar dependence/blocks, intervals,
   support, calibration failures, training-seed treatment, and multiplicity.
   Calibrate on layout and synthetic outcomes before inspecting market outcomes.
4. Seal source/config/data/review receipts; authorize the matched terminal screen
   only after qualification. Volatility normalization remains optional and off.
