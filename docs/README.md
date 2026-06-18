# Documentation Index

This docs surface is intentionally split between active contracts, active plans,
generated knowledge, and research artifacts.

## Active Architecture

- `architecture.md` - compact compatibility entrypoint for the current system.
- `architecture/itrader_system_architecture.md` (+ `.html`) - richer Phase B/C
  architecture view for solver, proposer, task contracts, and online seams.
- `environment_strategy.md` - environment and dependency strategy.
- `ops_runbook.md` - operational runbook for training, metrics, and recovery.

## Active Plans

- `plans/itrader_masterplan.md` (+ `.html`) - canonical cross-phase plan.
- `phase_b_plan.md` - original concrete Phase B execution plan.
- `iTRADER_Codex_Implementation_Spec.md` - detailed Codex implementation spec
  for coupled offline/online architecture.

## Project State

- `project-state/current-state.md` - current state and architecture notes.
- `project-state/handoff.md` - continuation handoff for future agents.
- `project-state/decisions.md` - accepted decisions and rationale.
- `project-state/session-log.md` - session-level implementation log.

## Knowledge Layer

- `knowledge/README.md` - GitNexus tracked docs and workflow entrypoint.
- `knowledge/WORKFLOWS.md` - graph-guided exploration, debugging, and refresh.
- `knowledge/wiki/` - generated GitNexus wiki snapshots.

Generated knowledge is advisory. If it disagrees with source, tests, or active
architecture docs, trust the code and call out the mismatch.

## Artifacts And Research

- `artifacts/research/` - imported or exploratory HTML references, including
  Neural Workbench, AB-object theory, LoRA/PPO/GAE/RLVR notes, and Market-JEPA.
- `daily_logs/` - ongoing engineering logs.
- dated handoffs and historical plans remain in place until folded into the
  active masterplan or archived more aggressively.

## Notion Status

No local `notion` or `notion-cli` executable was available on PATH during the
2026-06-17 docs cleanup. Treat Notion publishing as a future connector/plugin
workflow rather than a shell-CLI step for this repo.
