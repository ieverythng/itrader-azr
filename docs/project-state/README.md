# Project State

This folder is the repo-local handoff layer for `itrader-azr`.

Read order:

1. `handoff.md` for the fastest continuation point.
2. `current-state.md` for the durable project snapshot.
3. `next-actions.md` for the ordered work queue.
4. `decisions.md` for architecture and experiment decisions.
5. `session-log.md` for dated work notes.

For the active R1 boundary, read [development readiness](r1-development-readiness.md)
([HTML](r1-development-readiness.html)) before implementation. It specifies
diagnostic/replay repairs, model seams and the proposed matched experiment.

Update these files after meaningful architecture, training, task-contract, or operations changes. Keep long experiment outputs in their native run folders and summarize only the decisions, commands, and validation evidence here.

Phase B explicitly spans offline/curriculum work and online replay/shadow work.
Use the [research dashboard](../research/itrader_research_dashboard.md) for the
reconciled R/C/S registry and provisional axes; keep this directory focused on
checkout state, accepted decisions, handoff continuity, and the ordered queue.
