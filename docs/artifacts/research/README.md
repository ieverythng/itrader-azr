# Research Artifact Catalog

This directory contains dated investigations, experiment receipts, evidence
directories, and imported references. Use the active
[iTrader Research Dashboard](../../research/itrader_research_dashboard.md)
([HTML](../../research/itrader_research_dashboard.html)) to understand current
wing identities, dependencies, status, and decisions.
Use the generated [Experiment Browser](../../research/itrader_experiment_browser.html)
for indexed run/member/checkpoint navigation and explicit comparison factors;
the dated artifacts linked here remain the authoritative evidence.

Artifacts support claims; they do not override source, active architecture, the
canonical masterplan, or project-state decisions. Markdown is the source for
reports that have an HTML review companion.

## Foundation, measurement, and platform

- [Evaluation code audit](evaluation-code-audit-2026-10-06.md) ([HTML](evaluation-code-audit-2026-10-06.html)): completed bounded historical recovery and two CHANGES reviews; Q27/Q23/Q26 pending, no launch authority.
- [Evaluation quantitative methodology](evaluation-quant-methodology-2026-10-06.md) ([HTML](evaluation-quant-methodology-2026-10-06.html)): primary-source candidate statistical/attribution protocol and ordered Q23/Q26 decisions, not adopted defaults.
- [R1 panel qualification and terminal preflight](r1-panel-qualification-2026-10-06.md) ([HTML](r1-panel-qualification-2026-10-06.html)): outcome-independent calendar feasibility, historical-source proof limits, sixteen-cell draft and fail-closed preparation; no fresh panel or new run.
- [R1 prerequisite and browser closeout](r1-prerequisites-2026-10-05.md): Q21 completion components, Q23 historical-exposure limitation, Q24/Q25 reporting, and individual-run browser access.
- [Corrected-compatibility R0/R4 controls](corrected-compatibility-r0-r4-controls-2026-10-05.md)
  ([HTML](corrected-compatibility-r0-r4-controls-2026-10-05.html)) — supervised
  fixed-recipe control declaration and outcomes; distinct from historical PBT
  and the later native-API migration.

- [Historical Ray 2.58 optimizer/restore audit](ray258-historical-optimizer-audit-2026-10-05.md)
  ([HTML](ray258-historical-optimizer-audit-2026-10-05.html)) — ten terminal
  checkpoints, intended versus effective PPO settings, and member-00002
  self-resume entropy spikes; historical artifacts preserved.

- [R1 foundation and compatibility hardening](r1-foundation-hardening-2026-10-05.md)
  ([HTML](r1-foundation-hardening-2026-10-05.html)) — effective optimizer/restore
  settings, semantic observations, replay evidence, independent review and the
  compatibility-first / native-migration-later boundary.

- [R1 readiness independent review](r1-readiness-independent-review-2026-10-03.md)
  ([HTML](r1-readiness-independent-review-2026-10-03.html)) — fresh Sol/Astra
  review, reproduced evidence-gate blockers and development prerequisites.

- [Phase-A performance validity audit](phase-a-performance-validity-audit-2026-09-12.md)
  ([HTML](phase-a-performance-validity-audit-2026-09-12.html)) — corrected
  reward/evaluation semantics, checkpoint replay, controls, and promotion gate.
- [Phase-A MLP-FiLM autoresearch](phase-a-mlp-film-autoresearch-2026-09-19.md)
  ([HTML](phase-a-mlp-film-autoresearch-2026-09-19.html)) — matched cost/cadence
  study, exact replay, PBT behavior, and failed independent replication.
- [Phase-A storage incident audit](phase-a-storage-incident-audit-2026-09-19.md)
  ([HTML](phase-a-storage-incident-audit-2026-09-19.html)) — crash-dump root
  cause, offload layout, and operational receipts.
- [Market-data provider evaluation](market-data-provider-evaluation-2026-09-23.md)
  — primary-source provider comparison for dataset-v2.
- [Dataset-v2 R0 audit](dataset-v2-r0-audit-2026-09-24.md)
  ([HTML](dataset-v2-r0-audit-2026-09-24.html)) — authenticated immutable
  snapshot, v1 diagnosis, chronological controls, matched bridges, and PBT
  provenance.
- [Ray-2.58 R0 parity gate](ray-258-r0-parity-gate-2026-09-24.md)
  ([HTML](ray-258-r0-parity-gate-2026-09-24.html)) — accepted forward runtime
  boundary, dual-runtime evidence, exploit/restore/replay, and matched bridge.

The `.docx` performance-validity design report is a generated review artifact;
the dated Markdown audit above remains the textual source.

## R-series and R4-v1 chronology

Read these in order. They are a progressive evidence chain, not duplicate
reports.

1. [Solver-state Markov-completion rationale](solver-state-markov-completion-2026-09-25.md)
   — why endogenous position and horizon state are not recoverable from OHLCV.
2. [R4-v1 minimum implementation note](solver-state-r4-v1-min-implementation-note-2026-09-25.md)
   ([HTML](solver-state-r4-v1-min-implementation-note-2026-09-25.html)) — state
   contract, four-cell smoke, exact replay, and mechanistic controls.
3. [R4-v1 blocking ablation](solver-state-r4-v1-blocking-ablation-2026-09-25.md)
   ([HTML](solver-state-r4-v1-blocking-ablation-2026-09-25.html)) — masked,
   position, time, and combined arms across three seeds; failed absolute
   promotion.
4. [R4-v1 replication and architecture qualification](solver-state-r4-v1-replication-plan-2026-09-25.md)
   ([HTML](solver-state-r4-v1-replication-plan-2026-09-25.html)) — fresh 4×3×40
   replication, zero-init parity, controls, exact replay, and withheld economic
   promotion.
5. [R4-v1 actor-robustness PBT](solver-state-r4-v1-actor-robustness-2026-09-29.md)
   ([HTML](solver-state-r4-v1-actor-robustness-2026-09-29.html)) — causal-cost
   versus exact-1-bp populations, strict lineage, directional/hard-flat basins,
   and non-promotion.
6. [R4-v1 causal forensics](solver-state-r4-v1-causal-forensics-2026-09-29.md)
   ([HTML](solver-state-r4-v1-causal-forensics-2026-09-29.html)) — checkpoint
   anatomy, action/logit-margin collapse, control-normalized history, reporting
   repair, and the stop decision before R1.

Each paired `*-evidence/` directory contains machine-readable manifests,
summaries, replays, diagnostics, plots, or source snapshots associated with its
report. Preserve these paths because reports link to them directly.

## S-series: System One, CLM, Jev, and PPO

- [CLM and Jev for iTrader](system-one-clm-jev-trading-ablation-2026-09-24.md)
  ([HTML](system-one-clm-jev-trading-ablation-2026-09-24.html)) — primary S1
  proposal: exact CLM-8B probe, Discrete(3) state/action contract, projection
  heads, offline TaskSpec ranking, replay-only framing and capability matching,
  and bounded S1-P0…P5 ladder.
- [Open System-One models, PPO coupling, and AZR boundary](system-one-oss-ppo-azr-companion-2026-09-24.md)
  ([HTML](system-one-oss-ppo-azr-companion-2026-09-24.html)) — companion OSS
  landscape, exact CLM projection-head PPO, native numerical cosine head,
  semantic-transfer controls, and validated-task selector design.

Treat the first report as the primary S-series proposal and the second as its
model/PPO subexperiment companion. Neither report is R-series evidence or active
online architecture.

## Imperfect information, POMDP, self-play, and curriculum

- [Stratego, imperfect-information self-play, and iTrader](stratego-pomdp-self-play-itrader-2026-10-01.md)
  ([HTML](stratego-pomdp-self-play-itrader-2026-10-01.html)) — Ataraxos/Stratego
  and zero-data self-play review; all-wings placement matrix; provisional
  ATX/PF/WM/TTS/AZR experiments; bounded G0–G9 ladder; explicit protection of
  R4-v2's fixed identity.
- [AZR stack review](azr-stack-review-2026-09-12.md) — source-grounded AZR v2
  synthesis and bounded local adaptation notes.

The Oct-1 labels are proposed experiment names, not automatically accepted
masterplan branches. The paired evidence directory contains the machine-readable
experiment matrix and source snapshot.

## Source and documentation audits

- [Recurrence code audit](recurrence-code-audit-2026-09-12.md) — independent
  read-only source audit; its attempted WSL test is not a passing result.
- [Recurrence documentation revision audit](recurrence-docs-revision-audit-2026-09-12.md)
  — historical coverage inventory, rationale, and validation receipts.

## Imported reference-only material

These HTML files are historical/advisory inputs. They have no synchronized
Markdown authority in this repository and must not be treated as current runtime
contracts.

- `Neural_Workbench_Formal_Masterplan_Extended.html`
- `Neural_Workbench_AB_ML_Object_Theory.html`
- `LoRA, adapters, PPO, GAE and RLVR as operators over AB.html`
- `iTrader Market-JEPA.html`

In particular, the imported Market-JEPA page is a future hypothesis, not a
concrete implementation and not part of the S-series, PF/WM/TTS, or R4-v2.

## Catalog rules

- Keep dated reports and their evidence directories together.
- Do not edit historical numbers to match newer conclusions; add a newer report
  or an explicit supersession instead.
- Add each new dated report here and link it to the appropriate dashboard axis.
- Fold durable decisions into architecture, the masterplan, or project-state;
  leave raw investigation detail here.
