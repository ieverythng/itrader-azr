# Documentation Index

Markdown is the source of truth for active documents. HTML companions accompany review surfaces that require them; their synchronization and verification status is recorded in the dated audit. Current architecture claims are source-grounded against branch `agent/Watson-dev`; dated reports preserve evidence and line references from their audit snapshots. Solver/trunk/head choices remain subject to explicit implementation and evidence gates.

## Active architecture

- `architecture.md` — compact current baseline, R-series status, and control rules.
- `architecture/itrader_system_architecture.md` (+ `.html`) — detailed solver, recurrence, task, proposer, and online-shell architecture.
- `environment_strategy.md` — dependency authority and Windows/WSL execution boundary.
- `ops_runbook.md` — operational runbook for compatible environments.

## Active plans

- `plans/itrader_masterplan.md` (+ `.html`) — active cross-phase order and promotion gates; Phase B/C boundaries are being reconciled.
- `phase_b_plan.md` — historical concrete Phase B ordering, reconciled with the locked 2.9 stack and the current offline+replay/shadow scope.
- `iTRADER_Codex_Implementation_Spec.md` — working interface/workstream draft; solver/trunk/head order is proposed, not settled by this index.

## Project state

- `project-state/current-state.md` — branch and implementation snapshot.
- `project-state/handoff.md` — continuation handoff.
- `project-state/next-actions.md` — ordered implementation and evidence queue.
- `project-state/decisions.md` — accepted decisions and supersessions.
- `project-state/session-log.md` — dated engineering log.
- `project-state/solver-recurrence-assessment.md` — historical assessment rewritten as non-executable evidence and constraints.

## Research

- `research/recurrent-solver-ablation.md` (+ `.html`) — R0–R4 provenance, R2a–R2d placement matrix, same-input carry/reset controls, and RLlib lifecycle questions; not an accepted architecture contract.
- `research/sharpe-reward-design.md` (+ `.html`) — reward evidence boundary, telescoping limitation, and proposer/solver separation.

## Dated artifacts and audit

- `artifacts/research/phase-a-performance-validity-audit-2026-09-12.md` (+ `.html`) — Phase A audit, corrected parity runs, frozen controls, synchronous reward bridge, raw-24k closure, replay/block/cost evidence, and P1 gates.
- `artifacts/research/phase-a-mlp-film-autoresearch-2026-09-19.md` (+ `.html`) — matched P0 cost/cadence study, shared-lineage and transient challengers, exact replay, cost/block/regime evidence, failed independent replication, and live-validated Ray-2.9 PBT tails.
- `artifacts/research/phase-a-storage-incident-audit-2026-09-19.md` (+ `.html`) — WSL crash-dump root cause, verified C/D/E offload layout, live watchdog/RAM attribution, junction receipts, and deferred coordinated WSL-distro move.
- `artifacts/research/market-data-provider-evaluation-2026-09-23.md` — primary-source comparison of Alpaca, Databento, Cboe, Tradier, and Polygon/Massive for dataset v2.
- `artifacts/research/dataset-v2-r0-audit-2026-09-24.md` (+ `.html`) — authenticated Alpaca SIP snapshot, dataset-v1 diagnosis, immutable Parquet/DuckDB/Ray contract, original and crossed RTH controls, matched R0 bridges, strict synchronous-PBT provenance, launch-ready matched P0, plots, and the Phase A-to-Phase B gate.
- `artifacts/research/azr-stack-review-2026-09-12.md` — local AZR v2 synthesis and source-grounded adaptation recommendations.
- `artifacts/research/recurrence-code-audit-2026-09-12.md` — independent read-only source audit; do not treat its attempted WSL test as a pass.
- `artifacts/research/recurrence-docs-revision-audit-2026-09-12.md` — this docs coverage inventory, edit rationale, and validation results.
- `daily_logs/2026-09-12.md` — dated docs-only work log.

## Historical/advisory surfaces intentionally excluded from rewrite

- `knowledge/wiki/` is generated GitNexus material and remains advisory.
- Imported HTML under `artifacts/research/`, including `Neural_Workbench_Formal_Masterplan_Extended.html`, remains historical/reference material.
- Dated handoffs and older research snapshots remain provenance unless an active document explicitly promotes their conclusion.

When documents disagree, use this precedence: current source/tests/config and `AGENTS.md`; active architecture/masterplan; project-state continuity docs; independent/dated research artifacts; generated/imported material.
