# Documentation Index

Markdown is the source of truth for active documents. HTML companions are
review surfaces. Dated reports preserve evidence and provenance; they do not
override current source, architecture, or plan decisions.

## Authority order

When documents disagree, use this order:

1. current source, tests, configs, and `AGENTS.md`;
2. active architecture and the canonical masterplan;
3. project-state decisions, current state, handoff, and next actions;
4. active research registry and focused design notes;
5. dated research artifacts;
6. generated GitNexus pages and imported references.

Call out meaningful drift rather than silently selecting a convenient version.

## Start here

| Need | Document |
|---|---|
| Current system boundary | [Compact architecture](architecture.md) |
| Detailed solver/task/online architecture | [iTrader System Architecture](architecture/itrader_system_architecture.md) ([HTML](architecture/itrader_system_architecture.html)) |
| Cross-phase order and promotion gates | [Canonical Masterplan](plans/itrader_masterplan.md) ([HTML](plans/itrader_masterplan.html)) |
| All research wings, dependencies, statuses, and decisions | [iTrader Research Dashboard](research/itrader_research_dashboard.md) ([HTML](research/itrader_research_dashboard.html)) |
| Search runs, members, checkpoints, and matched factors | [Experiment Browser](research/itrader_experiment_browser.html) ([registry](research/experiment_registry.json)) |
| Independent change-review contract | [Review Contract](../REVIEW.md) |
| GRILL dispatch, parent boards, resource leases, and review workflow | [Coordination](coordination/README.md) |
| Current branch and experiment state | [Current State](project-state/current-state.md) |
| Ordered continuation work | [Next Actions](project-state/next-actions.md) |
| R1 diagnostics, replay and implementation acceptance | [R1 development readiness](project-state/r1-development-readiness.md) |
| Accepted/rejected decisions | [Decision Log](project-state/decisions.md) |
| Complete dated research catalog | [Research Artifacts](artifacts/research/README.md) |

## Active architecture and operations

- [architecture.md](architecture.md) — compact baseline, R-series status, and
  control rules.
- [architecture/itrader_system_architecture.md](architecture/itrader_system_architecture.md)
  ([HTML](architecture/itrader_system_architecture.html)) — detailed solver,
  recurrence, task, proposer, capability, and online-shell architecture.
- [environment_strategy.md](environment_strategy.md) — Ray-2.58 forward
  authority, Ray-2.9 historical reproduction, and Windows/WSL boundaries.
- [ops_runbook.md](ops_runbook.md) — compatible-environment operations.

## Plans

- [plans/itrader_masterplan.md](plans/itrader_masterplan.md)
  ([HTML](plans/itrader_masterplan.html)) — canonical cross-phase plan and
  promotion authority.
- [phase_b_plan.md](phase_b_plan.md) — historical Phase-B milestone proposal;
  use the masterplan for current order and runtime authority.
- [iTRADER_Codex_Implementation_Spec.md](iTRADER_Codex_Implementation_Spec.md) —
  working interface/workstream draft; proposed wording is not implementation
  status.

See [plans/README.md](plans/README.md) for plan ownership and archival rules.

## Project state

- [project-state/current-state.md](project-state/current-state.md) — branch and
  implementation snapshot.
- [project-state/handoff.md](project-state/handoff.md) — continuation handoff.
- [project-state/next-actions.md](project-state/next-actions.md) — ordered
  implementation and evidence queue.
- [project-state/decisions.md](project-state/decisions.md) — accepted decisions
  and supersessions.
- [project-state/session-log.md](project-state/session-log.md) — dated
  engineering log.

## Coordination

- [coordination/README.md](coordination/README.md) defines board ownership,
  bounded dispatch, resource leases, and freshness checks.
- [coordination/grill.md](coordination/grill.md) records current orchestration
  and decision dependencies; DEV, RESEARCH, and TRAINING/TESTING have separate
  owned boards linked from the index.
- [coordination/review-workflow.md](coordination/review-workflow.md) bounds the
  experimental No-Mistakes port without replacing `REVIEW.md`.

Boards are operational Markdown, not a second architecture or experiment
ledger. They do not require generated HTML companions.

## Research

- [research/itrader_research_dashboard.md](research/itrader_research_dashboard.md)
  ([HTML](research/itrader_research_dashboard.html)) — canonical research
  registry. It distinguishes R/C/S identities, provisional ATX/PF/WM/TTS/AZR
  experiments, orthogonal reward/task/platform axes, masterplan placement, and
  the evidence/decision queue.
- [research/itrader_experiment_browser.html](research/itrader_experiment_browser.html)
  ([registry](research/experiment_registry.json)) — generated run/checkpoint
  browser over authoritative result JSON. It is an evidence navigator, not a
  second result ledger or roadmap.
- [research/recurrent-solver-ablation.md](research/recurrent-solver-ablation.md)
  ([HTML](research/recurrent-solver-ablation.html)) — focused R-series identity,
  placement, and lifecycle design provenance.
- [research/sharpe-reward-design.md](research/sharpe-reward-design.md)
  ([HTML](research/sharpe-reward-design.html)) — focused reward-axis design note.

See [research/README.md](research/README.md) for the directory boundary and
[artifacts/research/README.md](artifacts/research/README.md) for all dated
investigations, experiment receipts, evidence directories, and imported
references.

## Historical and advisory surfaces

- `docs/artifacts/` contains dated, imported, generated, or historical material.
- `docs/knowledge/wiki/` is generated GitNexus material and remains advisory.
- Imported HTML under `docs/artifacts/research/` is reference material unless an
  active document explicitly promotes a conclusion.
- Historical plans and recurrence assessments preserve provenance but do not
  supersede the active architecture, masterplan, or research dashboard.
