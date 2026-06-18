---
name: itrader-experiment-handoff
description: Maintain iTrader project-state and experiment handoff docs across Codex, Cursor, and workstation sessions, including solver/proposer status, validation commands, run artifacts, and next actions.
---

# iTrader Experiment Handoff

## Overview

Use this skill when preparing a continuation note, updating project memory, or
handing iTrader work between Mac, main PC, Codex, Cursor, or Watson-adjacent
agents. Keep the handoff short, factual, and tied to runnable commands.

## Quick Start

1. Inspect `git status --short --branch`.
2. Read `docs/project-state/current-state.md`, `docs/project-state/handoff.md`,
   and `docs/plans/itrader_masterplan.md`.
3. Summarize implemented state separately from planned/frontier work.
4. Record validation commands and whether they passed, failed, or were skipped.
5. List exact next actions with repo-relative paths and expected outputs.

## Required Handoff Fields

- Branch and commit context.
- Current solver/proposer/task-contract status.
- Latest validation commands and results.
- Any stale or refreshed GitNexus state.
- Active blockers and environment assumptions.
- Next commands for Mac-safe checks and main-PC training runs.

## iTrader-Specific Notes

- Never describe `SolverWindowLSTMFiLM` as implemented unless the file, training
  config, tests, and docs are present on the current branch.
- Distinguish smoke validation from meaningful solver-quality training.
- Record task-pool paths, experiment names, seeds, and Ray output locations when
  discussing training results.
- Treat `CapabilityProfile` as an empirical evaluation target, not a promise
  derived from training reward alone.

## Validation

For a docs-only handoff update:

- `git diff --check`
- `tools/knowledge/status.sh`

For code or training seam changes:

- focused pytest for touched seam
- `./itrader/tools/run_repo_python.sh itrader/tools/healthcheck.py`
- `./itrader/tools/run_repo_python.sh -m pytest -q itrader/tests`
