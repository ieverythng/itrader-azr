# iTrader Research Dashboard

Last reconciled: 2026-10-06<br>
Branch snapshot: `agent/Watson-dev` with the uncommitted R4-v1 research round<br>
Status: **active research registry; G0 closed and fixed-recipe R1 attribution next**<br>
Visual companion: [open the HTML dashboard](itrader_research_dashboard.html)<br>
Run evidence: [search the generated experiment browser](itrader_experiment_browser.html)
([registry](experiment_registry.json))

[History and operational log](#history-and-operational-log) preserves prior
notices without obscuring the research map. Eight corrected-control terminal
replays are accepted; no economic promotion or fresh-panel opening is implied.

The sixteen-cell scientific draft and read-only terminal preflight are now
implemented; real terminal execution and final config/source sealing are not.
Q23 has 2,052 provisional eligible symbol/session anchors; conservative possible
historical exposure leaves zero certifiably fresh candidates. Sixteen control
sidecars recover recorded starts, but 26 randomized runs remain source-unbound.
Q23's recovery/new-data choice and Q26's uncertainty decision remain gates.
No panel freeze/opening, R1 implementation or new market results. See the
[qualification audit](../artifacts/research/r1-panel-qualification-2026-10-06.md).

The [evaluation code audit](../artifacts/research/evaluation-code-audit-2026-10-06.md)
completes bounded Q23 recovery and current evaluator-gate review, not a reopening
of accepted controls: 48 recorded starts mapped conditionally, 26 randomized
configs still source-unbound, two independent CHANGES verdicts. The [quantitative-methodology synthesis](../artifacts/research/evaluation-quant-methodology-2026-10-06.md)
is candidate-only: synchronized calendar-block inference, explicit economic
attribution and future offline-to-online capability evidence remain separately
governed. Mechanical corrections require Q27 approval; Q23/Q26 choices still
block panel freeze. No new run is registered in the Experiment Browser.

## 1. Authority and scope

This document is the canonical index of iTrader research identities, controls,
dependencies, evidence, and open decisions. It is deliberately **not** a second
roadmap. The [masterplan](../plans/itrader_masterplan.md) remains authoritative
for cross-phase order and promotion gates. The architecture documents remain
authoritative for system boundaries, and source/tests/config remain authoritative
for what is implemented.

Only research branches, ablations, named controls, and their evidence belong in
this dashboard. Planned system construction already owned by the masterplan --
including closing the Phase-B proposer/solver loop, task-family rollout,
capability/replay contracts, the online shell, and data/platform foundations --
is referenced here only when it is a prerequisite or boundary for an
experiment.

| Question | Authority |
|---|---|
| What code and contracts exist on this checkout? | Source, tests, configs, and `AGENTS.md` |
| What must the system keep separate? | [Compact architecture](../architecture.md) and [detailed architecture](../architecture/itrader_system_architecture.md) |
| What runs first, and what is required for promotion? | [Canonical masterplan](../plans/itrader_masterplan.md) |
| What is the latest accepted, rejected, or pending outcome? | [Decisions](../project-state/decisions.md), then [current state](../project-state/current-state.md) and [next actions](../project-state/next-actions.md) |
| What does each research ID mean, depend on, and hold fixed? | **This dashboard** |
| What supports a claim? | Focused research notes and dated [research artifacts](../artifacts/research/README.md) |

The dashboard can register a hypothesis without placing it on the active
masterplan. A result becomes durable architecture or phase order only when the
corresponding authority document accepts it.

The generated experiment browser is the concrete evidence surface beneath this
research registry. It provides run tabs, checkpoint search, provenance links,
and an explicit eight-factor comparability gate. Its registry owns metadata and
source pointers only; linked JSON remains authoritative for metric values, and
dated one-off dashboards remain immutable provenance snapshots.
Matching factor labels are not sufficient: dataset, evaluation-panel, and
metric-contract identities must also be present and equal. Missing identities
produce a fail-closed `not directly comparable` result.
The schema is closed over the eight comparison factors, and every numeric
metric has an explicit display-unit contract. Unknown factors, unknown units,
duplicate JSON keys, non-finite values, malformed action fractions, and
incomplete adapter records fail the build instead of being guessed in the UI.

> **No deployment authority.** Nothing here authorizes live execution. Phase B
> online work is replay/shadow evidence. `DecisionGate`, `RiskEngine`, and
> `Executor` remain deterministic shell roles, and `WAIT` / `NO_ACTION` remains
> distinct from the solver's target-position action `flat`.

## 2. Decision cockpit

### Current boundary

- **Control:** R0 `SolverMLPFiLM` remains the immutable state-free parity and
  economic control; it is implemented but not a profitable-policy claim.
- **Qualified baseline:** `R0 + R4-v1` is the architecture-qualified
  feed-forward baseline. The state seam works and replays, but the completed
  replication and actor-robustness studies did **not** earn economic promotion.
- **Stop condition already reached:** do not run another same-panel
  `R0 + R4-v1` PBT population. The 1 bp run reached a hard-flat basin, while the
  staged-cost run converged to directional exposure rather than demonstrated
  conditional edge.
- **G0 evidence closure complete, with a provenance qualification:** five
  member-`00002` checkpoints produced byte-identical episode/reward and
  per-step action/logit artifact pairs on a predeclared non-overlapping panel.
  The historical receipts do not contain distinct producer/run identifiers, so
  the browser verifies artifact parity and records independent-process
  provenance as unknown rather than restating the source claim as independently
  proven. Every fresh-panel net return and Sharpe was negative; clip fraction
  remains explicitly unavailable in the legacy learner record and must be
  captured before performance interpretation. Future replay receipts must stamp
  producer identity at creation time.
- **Next solver attribution:** fixed-recipe Ray-2.58 comparison of matched R0,
  R0 + R4-v1, clean R1, and separately named `R1 + R4-v1`; then native R2 mechanics and the
  same-input `R2c-CARRY` versus `R2c-RESET` test.
- **PBT boundary:** keep PBT off during representation attribution. Reopen it
  only after a fixed architecture and trustworthy telemetry establish what is
  being optimized.

### Now, next, parallel, later

| Horizon | Work | Why it is placed here |
|---|---|---|
| **Now** | Close review/diagnostic gaps, implement R1, then run fixed-recipe `R0 / R0 + R4-v1 / R1 / R1 + R4-v1` screen with fixed 1 bp and causal staging | G0 is closed; complete encoder-by-state attribution before new search |
| **Next** | Implement native R2a lifecycle, then R2c placement and `CARRY / RESET` | This is the first valid carried-memory attribution |
| **Parallel research, not bundled** | S1-P0 frozen CLM diagnostic planning; `ATX-EMA-EVAL-v0`; tiny direct AZR learning-progress reference design | Each can be specified independently, but none may rescue or reinterpret an R result |
| **After stable solver interface** | C0/C1/shuffled-C1 and S projection/PPO cells | These need a frozen control and evidence contract |
| **Later** | PF predictive distribution, WM validation, and TTS diagnostics | Each depends on evidence missing today |
| **Phase D advisory** | Market-JEPA | Future MarketFramer/representation hypothesis; intentionally non-concrete |

## 3. Status language

Research records use three independent status fields. One word must never stand
in for all three.

| Field | Values | Meaning |
|---|---|---|
| **Identity class** | canonical research identity, named control/cross, provisional experiment, orthogonal factor | Whether the label is a stable research identity or a bounded ablation/supporting factor |
| **Implementation** | implemented, working-tree only, scaffold, unimplemented, deferred, external | What exists on this checkout; not whether it works economically |
| **Evidence decision** | baseline, architecture-qualified, screen pending, research-only, rejected, deferred, unresolved drift | What the evidence permits us to claim |

Promotion stages such as Phase-A `P0` and `P1` are evidence stages, not research
wings. They must not be confused with S-series cell names such as `S1-P0` or
companion loop names such as `P1`.

## 4. Masterplan boundary

| Owned by the canonical masterplan / architecture | Owned by this research dashboard |
|---|---|
| Phase ordering, core implementation, promotion gates, task-family rollout, data/platform/evidence foundations, replay/shadow contracts, and the deterministic online shell | Research identities, ablation cells, held-fixed factors, evidence gates, stop rules, and links to investigation artifacts |
| Closing the Phase-B proposer -> validated `TaskSpec`/buffer -> PPO solver -> feedback/`r_prop` -> proposer loop | Comparing alternative proposer-credit and curriculum mechanisms *inside* that loop: `AZR-LP-DIRECT-v0`, `AZR-LP-GRAD-v0`, and `AZR-QD-REPLAY-v0` |

The dashboard may track a research wing that the masterplan schedules, such as
R, without duplicating the plan itself. If a dashboard result is promoted, the
masterplan or architecture must record that decision explicitly.

The R4-v2 identity is fixed even though its branch status is not. This checkout
marks it deferred and absent. Parallel work has been reported as staged
elsewhere, but that claim has not been reconciled against this checkout. Do not
redesign, rename, or mark it implemented here until that branch is inspected.

## 5. Research dependency spine

```text
[masterplan-owned prerequisites]
dataset-v2 + Ray 2.58 + EpisodeEvidence/provenance + Phase-B training seams
  -> [research gate, complete] G0 fresh-panel / PPO telemetry closure
  -> fixed-recipe R0 vs R1 vs R1 + R4-v1
  -> native R2 lifecycle and replay
  -> R2c-CARRY vs identical R2c-RESET
  -> freeze solver interface and evidence contract
       |-- C-series causal summaries
       |-- S-series System-One heads
       `-- proposer-credit/curriculum ablations

accepted R/C evidence -> PF predictive-distribution ablation
accepted R/C evidence -> TTS hindsight ceiling
valid scenario model + calibrated blueprint -> TTS replay/shadow ablation

Market-JEPA remains a separate Phase-D line
```

Every attribution cell changes one factor. Keep runtime, task panel, reward,
transaction-cost schedule, solver representation, native RLlib migration, and
PBT behavior fixed unless that item is the named factor.

## 6. Research-axis registry

| Axis | Type | Question | Checkout status | Prerequisite / next gate | Must remain separate from |
|---|---|---|---|---|---|
| **R — solver representation** | canonical wing | Which numerical policy/value representation supplies useful temporal, task, and endogenous-state information? | R0 implemented; R4-v1 qualified but economically rejected; R1/R2 unimplemented | G0 closed; diagnostics/review repairs, matched R screen, then carry/reset | C, S, reward, optimizer, proposer, PF/WM/TTS |
| **C — causal market summaries** | canonical wing | Do trailing, train-fitted causal summaries add information beyond the current causal feature set? | C0 is the current causal-feature baseline; C1 and shuffled-C1 unimplemented | Stable R interface and exact causal fitting contract | Future-derived regime labels; R-memory claims |
| **S — System One / CLM / Jev style** | canonical research wing | Does explicit state-to-candidate geometry help the three-action solver, task selection, or later framing/matching? | Research-only; assets and designs exist, no iTrader S experiment completed | Frozen R interface; begin S1-P0 diagnostic | R promotion, Market-JEPA, PF/WM/TTS, execution |
| **ATX optimizer/evaluation** | provisional experiments | Can isolated EMA, schedule, filtering, or distributional-value factors improve a fixed solver recipe? | Proposal only | Trustworthy critic/advantage/clip/KL telemetry; R attribution first | Representation changes and PBT |
| **Reward** | orthogonal factor | Which solver reward ranks economic outcomes without hiding risk or cost? | `pnl_linear` default; `net_return` implemented challenger; Sharpe variants experimental | Matched held-out reward study | R/C/S architecture cells |
| **Cost curriculum** | orthogonal factor | Does causal cost staging help learning without changing the economic gate? | Named factor retained; exact 1 bp from update zero rejected as universal default | Use only as a matched ablation | R1 identity and promotion criteria |
| **AZR proposer-credit / curriculum** | Phase-B ablation axis | Which credit signal and task-memory policy best improve a fixed solver recipe from pinned starting checkpoints at equal accepted compute? | Core loop is masterplan-owned; current code is one-shot handoff; ablations unimplemented | Direct uplift reference before surrogate or adaptive QD | Core loop ownership, solver reward, online execution authority |
| **PF — predictive state** | provisional axis | Can a causal calibrated predictive distribution improve a frozen accepted solver? | Proposal only | R and C evidence, proper-score/calibration gate | Belief-state claims, R4-v2, WM/TTS |
| **WM — world/scenario model** | provisional axis | Can action-independent market scenarios support valid counterfactual portfolio rollouts? | Proposal only | Stable causal inputs plus multi-step scenario validity; PF is not a required component | Historical replay as counterfactual simulation |
| **TTS — test-time search** | provisional axis | Does branch-safe root-action search beat a frozen blueprint under equal compute? | Direct use rejected now; hindsight diagnostic then deferred | Valid WM and calibrated blueprint; replay/shadow only | Policy training and deterministic execution shell |
| **Market-JEPA** | future Phase-D line | Can a causal joint-embedding market representation improve future MarketFramer inputs? | Non-concrete advisory research | Stable causal contracts and a separately approved design | S-series, PF/WM/TTS, R4-v2 |

## 7. R-series: solver representation

| ID | Fixed identity | Implementation | Evidence / decision | Next gate |
|---|---|---|---|---|
| **R0** | State-free `SolverMLPFiLM`; current FiLM is `tanh(gamma) * hidden + beta` | Implemented and selected | Immutable parity/economic control; not economically promoted | Preserve beside every challenger |
| **R1** | Corrected reset-every-step LSTM over the supplied chronological window; no cross-step state | Unimplemented | Active next cell | Shape split, newest-bar convention, finite forward/backward, deterministic smoke, matched screen |
| **R2a** | Native stateful LSTM with task concatenated into recurrent input | Unimplemented | Planned after R1 | Native sequence lifecycle, masks, reset/truncation, state-aware replay |
| **R2b** | Stateful LSTM with task-conditioned initial `(h,c)` only | Deferred | Identity preserved | Explicit initial-state adapter/API |
| **R2c** | Stateful LSTM with per-step task FiLM; readout placement is the first candidate | Unimplemented, active design candidate | Memory claim not yet tested | `R2c-CARRY` versus otherwise identical `R2c-RESET` |
| **R2d** | Initial plus per-step FiLM restricted to value-head path | Deferred | No evidence | Matched policy/value placement and capacity controls |
| **R3** | Stateful GRU with per-step FiLM | Unimplemented, later | No evidence | Only after meaningful LSTM evidence |
| **R4-v1** | Exact endogenous state `[position, remaining_horizon_fraction]` through matched actor/value residuals | Working-tree implementation | Architecture-qualified with R0; economic promotion rejected | Preserve R0; cross separately as `R1 + R4-v1` |
| **R4-F1/F2** | Conditional future accounting/duration/risk features, only when semantics require them | Proposed semantics | No evidence | State-sufficiency audit tied to a real transition/reward/constraint |
| **R4-v2** | Centered nonlinear market/task × solver-state late fusion after task FiLM, with policy/value residuals | Deferred/absent locally; reportedly staged elsewhere | Unresolved branch-status drift; identity fixed | Inspect parallel branch; do not redesign |

`R2c-CARRY` and `R2c-RESET` are named controls, not new R identities. Likewise,
R4-v1 `masked`, `position`, `time`, and `both` are ablation arms. `R4-v1-min`
and `R4-v1.1` are implementation/evidence profiles, not replacements for R4-v1.

### R4-v1 evidence chain

1. [Markov-completion rationale](../artifacts/research/solver-state-markov-completion-2026-09-25.md)
2. [Minimal implementation note](../artifacts/research/solver-state-r4-v1-min-implementation-note-2026-09-25.md)
3. [Blocking four-arm ablation](../artifacts/research/solver-state-r4-v1-blocking-ablation-2026-09-25.md)
4. [Fresh replication and architecture qualification](../artifacts/research/solver-state-r4-v1-replication-plan-2026-09-25.md)
5. [Actor-robustness PBT](../artifacts/research/solver-state-r4-v1-actor-robustness-2026-09-29.md)
6. [Checkpoint-level causal forensics](../artifacts/research/solver-state-r4-v1-causal-forensics-2026-09-29.md)

This sequence is not redundant: it progresses from observability semantics to
implementation, attribution, replication, optimizer stress, and forensic stop
decision.

## 8. C-series: causal market summaries

| Cell | Meaning | Status | Required control |
|---|---|---|---|
| **C0** | Current causal feature set supplied to the solver | Implemented input baseline; no dedicated C comparison completed | Stable accepted solver recipe |
| **C1** | Trailing summaries computed only from information available at the decision point and fitted on training chronology only | Unimplemented | Same solver, task panel, chronology, reward, cost, and runtime as C0 |
| **shuffled-C1** | Same dimension and marginals as C1, but with useful temporal/state information destroyed | Unimplemented | Must accompany C1 in the same comparison |

Current future-derived `regime_tags` are ex-post diagnostics only. They cannot be
silently relabelled as C1 or supplied to a causal policy. C-series tests market
information: whether conditional context beyond raw causal features helps the
policy choose among the same actions. It does not test recurrent memory,
endogenous portfolio state, PF forecasts, WM scenarios, or R4-v2 fusion.

The first valid C experiment freezes the accepted R solver and all non-C factors,
then compares C0, C1, and shuffled-C1 on the same held-out chronology. Report
economic outcomes together with conditional-action dependence and calibration;
stop if C1 cannot beat both C0 and the information-destroyed control.

## 9. S-series: CLM/Jev-style System One

The primary question is whether explicit state/candidate scoring adds useful,
calibrated signal. S is a parallel ablation wing, not a replacement for the
R-series and not a shortcut around the economic gate.

| Cell | Fixed experiment | Implementation | Entry condition / verdict rule |
|---|---|---|---|
| **S1-P0** | Frozen `Contrastive-LM/CLM-v0.1-8B` ranks explicit short/flat/long target-position descriptions from deterministic state serialization | Not integrated; local asset receipt exists | Interface, ordering, latency, probability, and collapse diagnostic only |
| **S1-P1** | Train only released state/action projection heads; Qwen3-8B stays frozen | Design-only | Use per-question choice/cross-entropy, multi-positive contrastive loss, or PPO; forbid naive repeated-label pairwise InfoNCE |
| **S1-P2** | Cross explicit endogenous state into CLM and matched R control | Design-only; contract needs reconciliation | The broad `agent_state_v1` proposal is not automatically the narrow accepted R4-v1 contract |
| **S1-P3** | Offline ranker over already validated `TaskSpec` candidates | Design-only | Frozen solver, fixed candidate sets, equal accepted solver steps; measure downstream held-out uplift |
| **S1-P4a** | Replay/shadow MarketFramer ranks validated `CandidateTask + WAIT` | Future-only | Frozen candidates/solvers, deterministic eligibility mask, calibration and abstention; no execution authority |
| **S1-P4b** | Separate CapabilityMatcher ranks eligible versioned solver profiles or returns no match | Future-only | Held-out CapabilityProfile with uncertainty/provenance; never derive capability from PPO logits |
| **S1-P5** | Distil useful exact-CLM geometry into a numerical state projection plus three action prototypes | Conditional design | Only if exact CLM signal exists but resource/latency gates fail; not CLM evidence by itself |

### PPO coupling within S

The duplicate-label warning applies to generic single-positive batch InfoNCE:
many examples share the same three action labels, so same-action samples become
false negatives. It does **not** invalidate categorical PPO. PPO trains the
probability of the sampled action under the three-way policy.

The registered head-only sequence is:

| Cell | Trainable parameters | Purpose |
|---|---|---|
| `CLM-PPO-0` | value head only; released policy frozen | Interface/value-path diagnostic |
| `CLM-PPO-1` | state projection; action head and scale frozen | Cheapest market-state alignment test |
| `CLM-PPO-2` | state and action projections; scale frozen | Test whether action prototypes need domain movement |
| `CLM-PPO-3` | both projections plus bounded logit scale | Only after finite-logit, entropy, replay, and collapse gates |

The cheaper native numerical cosine-action head remains a distinct test. It may
reuse the existing iTrader trunk and PPO/GAE, but it is not evidence about the
released 8B CLM. The primary design report is [CLM and Jev for iTrader](../artifacts/research/system-one-clm-jev-trading-ablation-2026-09-24.md);
the [OSS/PPO companion](../artifacts/research/system-one-oss-ppo-azr-companion-2026-09-24.md)
owns the open-model comparison and exact head-coupling details.

## 10. Optimizer, reward, and cost factors

### ATX provisional experiments

| ID | Factor | Earliest admissible test |
|---|---|---|
| `ATX-EMA-EVAL-v0` | Raw versus exponential-moving-average weights at identical snapshots | After G0; evaluation-only, approximately zero extra solver steps |
| `ATX-ADV-FILTER-v0` | All samples versus top-absolute-advantage versus equal-count random subset | Only with a trustworthy critic and equal environment-query/retained-compute ledgers |
| `ATX-DAMP-SCHED-v0` | Isolated learning-rate and regularization schedules before any coupled schedule | After R attribution, with PBT disabled |
| `ATX-DIST-VALUE-v0` | Separately named distributional-return value-head factor | Deferred until scalar-critic diagnostics justify added complexity |

These ideas are inspired by the Stratego/Ataraxos review, but the names are
provisional experiment labels, not canonical roadmap additions.

### Reward and cost

- `pnl_linear` remains the baseline solver reward.
- `net_return` is an implemented challenger with explicit transaction cost and
  episode-evidence alignment; implementation is not promotion.
- `sharpe_only` and `pnl_sharpe` remain experimental challengers documented in
  [Sharpe-Aware Reward Design](sharpe-reward-design.md).
- Cost staging remains a named curriculum factor. It may be crossed with a
  frozen solver recipe, but it must not be hidden inside R1 or used to weaken the
  final exact-cost economic gate.

## 11. Phase-B proposer-credit and curriculum ablations

The **core proposer track is not optional research-dashboard scope**. The
[masterplan](../plans/itrader_masterplan.md) owns completion of the Phase-B
proposer -> validated `TaskSpec`/buffer -> PPO solver -> feedback/`r_prop` ->
proposer loop. Phase B commits to closing that loop, but does not yet commit to
one final `r_prop` formula.

On this checkout, `train_AZR.py` creates a validated, deduplicated task pool and
hands it to `train_solo_V2` once. It does not yet compute solver-feedback
`r_prop`, update the proposer, or operate a fresh/mutate/replay archive. That is
a bootstrap implementation state recorded by the masterplan, not a separate
research wing.

One terminology correction matters: [original AZR's local-v2 Eq. 4](https://arxiv.org/html/2505.03335v3#S3.SS1)
is not a symmetric raw-performance parabola. For mean solve rate `p`, it assigns
zero at `p = 0` and `p = 1`, and otherwise `1 - p`. It is therefore an asymmetric
learnability/difficulty proxy, not a direct measurement of pre/post solver
improvement. It remains the natural baseline control when the core loop lands;
the cells below test alternative credit and retention mechanisms.

| Cell | What changes | Reference, control, and gate |
|---|---|---|
| `AZR-LP-DIRECT-v0` | Measures task-induced uplift through repeated micro-updates of cloned solvers from pinned starting checkpoints | Expensive causal reference: freeze the solver recipe, optimizer start, task UID, evaluation panel, and accepted compute; require stable rankings and usable intervals |
| `AZR-LP-GRAD-v0` | Replaces direct micro-update scoring with a cheaper gradient-alignment surrogate | Admissible only if it predicts DIRECT across seeds and beats shuffled/random controls; it is not learning progress by declaration |
| `AZR-QD-REPLAY-v0` | Changes task retention/selection to typed fresh, mutate, and replay streams with quality-diversity coverage | Run fixed-QD first; enable adaptive scoring only after a credit signal is validated and compare at equal accepted solver compute |

These are alternatives for how validated tasks receive credit and are retained
inside the planned loop; they do not replace the loop. Proposer credit is also
separate from the solver's market reward. A System-One scorer may later rank
validated candidates (`S1-P3`), but it cannot generate unchecked tasks, modify
`TaskSpec`, update capability from logits, or acquire execution authority.

## 12. PF, WM, and TTS: provisional POMDP-facing axes

The Stratego review supports explicit uncertainty and search discipline, not a
direct game-to-market transplant. Trading remains partially observed, but there
is no supervised true hidden board. PF, WM, and TTS are distinct axes, not three
names for one stack.

| Axis | Input and output contract | First controls and promotion gate | Non-claim / stop boundary |
|---|---|---|---|
| **PF — `PF-PRED-DIST-v0`** | Causal market history -> calibrated distributions over predeclared future return, volatility, and/or liquidity targets and horizons. Future bars are training labels only. | Establish proper scores, calibration, coverage, and tails on new chronology; then freeze PF and compare `PF0`, `PF1`, zero-PF, and shuffled-PF with the solver fixed. | Not a hidden-board posterior, latent truth, action-conditioned PnL model, WM, or R4-v2. Stop before PPO spend if it cannot beat simple forecast baselines. |
| **WM — `WM-VALIDITY-v0`** | Causal market context -> coherent multi-step, action-independent exogenous market scenarios. Exact deterministic portfolio transitions apply each candidate action separately under the price-taker contract. | Beat bootstrap/empirical baselines on multi-horizon proper scores, coverage, tails, and regime slices; use common scenario seeds across all three actions. | Not a policy and not one replayed realized path. Stop if scenario validity or tail behavior fails; PF may inform a future design but is not a required WM input. |
| **TTS — `ATX-TTS-DIAG-v0`** | Frozen blueprint plus fully cloned branch state and, only after validation, WM scenarios -> an adjusted score for each of the three root target-position actions. It changes the current decision, not neural weights. | First measure a realized-path hindsight ceiling. Then, only with a valid WM and calibrated blueprint, compare frozen search-off/search-on at equal compute in replay/shadow. | Clone position, horizon, recurrent/PF state, reward context, equity, drawdown, and turnover. Stop on model-bias amplification or no equal-compute gain; never bypass the deterministic shell. |

The research-spend order is **PF -> hindsight ceiling -> WM -> TTS**, because
cheap forecast and ceiling tests can stop the expensive path early. This is not
a mandatory neural dependency chain: PF and WM answer different questions, and
WM need not consume PF output. TTS is the component that depends on a valid
scenario model and calibrated blueprint.

The core replay/shadow control flow, capability contracts, task families, and
deterministic `DecisionGate -> RiskEngine -> Executor` authority remain owned by
the [masterplan](../plans/itrader_masterplan.md) and
[architecture](../architecture/itrader_system_architecture.md). Research outputs
must respect those contracts; they are not registry axes here.

## 13. Cross-axis rules

### Allowed only after prerequisites

| Cross | Why it can be meaningful | Preconditions |
|---|---|---|
| `R1 + R4-v1` | Separates within-window sequence encoding from endogenous portfolio-state observability | Clean R1 and unchanged R4-v1 connector |
| `R2c-CARRY + R4-v1` vs `R2c-RESET + R4-v1` | Isolates carried memory with known endogenous state available in both arms | Native recurrent lifecycle and exact state replay |
| accepted solver × C1 | Tests new causal market summaries on a frozen representation | Causal train-only fitting and shuffled-C1 |
| accepted solver × PF1 | Tests calibrated predictive-state value | PF proper-score/calibration gate plus PF0/zero/shuffle controls |
| S1-P3 × fixed solver | Tests System-One task ranking | Stable capability gaps, fixed candidates, equal accepted compute |

### Forbidden conflations

- R4-v1 portfolio state is not recurrent memory, belief state, PF, or WM.
- R4-v2 joint fusion is not a container for PF, WM, TTS, or extra state fields.
- C1 causal summaries are not recurrence evidence.
- CLM/Jev action scoring is not Market-JEPA and is not an LLM-free result.
- The native cosine head or a distilled head is not released-CLM evidence.
- PPO action probabilities and value estimates are not CapabilityProfile entries.
- Solver `flat` is not online `WAIT`.
- Historical replay is not a counterfactual world model.
- PBT search activity is not economic promotion.
- Future-derived regime tags are not causal features.

## 14. Bounded gate ladder

This ledger estimates research loops; it does not authorize them. One screening
run unit (SRU) is one fixed 12-update, two-seed arm under the current 4,800
samples/update contract: 115,200 sampled environment steps. Recompute if the
accepted recipe changes.

| Gate | Question | Bounded budget | Advance condition |
|---|---|---|---|
| **G0** | Are formulas, configs, telemetry, lineage, panel, cost, and replay trustworthy? | No training | Source/checkpoint-backed receipt; critic, advantage, clip, KL, margin, replay all interpretable |
| **G1** | Are native recurrent selection, masks, reset, last-valid state, and replay correct? | Tests + deterministic smoke | Exact train/eval/checkpoint state parity |
| **G2a** | What are the encoder and solver-state effects across R0, R0 + R4-v1, R1 and R1 + R4-v1? | Accepted Q20: 8 SRU = 921,600 nominal training steps; Q21 monitoring and terminal evaluation budget separate | All 16 cells complete before fresh terminal evaluation; mechanics pass and a challenger merits extension; no promotion claim |
| **G2b** | Is there memory-specific signal? | 2 SRU = 230,400 steps | `CARRY` paired signal over identical `RESET`, with valid state norms/masks |
| **G2c** | Does a surviving R comparison clear economic gates? | 2 arms = 1,152,000; 3 arms = 1,728,000 steps | Held-out 1 bp, static-direction control, replay, block-aware uplift, then 2 bp stress |
| **G3** | Does EMA improve evaluation at identical snapshots? | Approximately zero extra solver steps | Lower variance or better paired economics without hidden collapse |
| **G4** | Does an isolated schedule factor help? | 3 SRU = 345,600 steps | One factor passes before any coupled schedule is predeclared |
| **G5** | Does advantage filtering beat equal-count random selection? | 3 SRU = 345,600 steps | Better at equal queries and efficient at equal retained compute |
| **G6a** | Is direct task-induced uplift repeatable enough to serve as curriculum reward? | 460,800 steps + evaluation | Repeated ranking is stable and intervals are usable |
| **G6b** | Does a gradient surrogate predict direct uplift? | Gradient/evaluation compute | Positive per-seed association and pooled interval excluding zero |
| **G7a/b** | Does fixed then adaptive QD improve coverage and economics at equal accepted compute? | 2 SRU per screen | Beats fixed/shuffled controls on fresh solver seeds |
| **G8** | Is PF calibrated, and does it add policy value? | Supervised budget separate; policy screen <= 691,200 steps | Proper scores/calibration first, then PF beats no/zero/shuffle controls |
| **G9** | Are WM and blueprint valid enough for search, and does search help? | No PPO spend until validity passes | Scenario model beats simple baselines; frozen search beats equal-compute controls |

Global economic promotion still requires held-out chronology, multiple seeds,
positive exact-1-bp net return and Sharpe, strongest static-direction control,
paired block-aware uplift, exact replay, 2 bp survival, and independent
replication. A policy that is nearly one-direction must beat that exact static
direction and show conditional state dependence.

## 15. Open decision queue

| Priority | Decision | Current default | Evidence that changes it |
|---|---|---|---|
| **D0** | Is G0 complete enough to begin R1 preparation? | G0 economic closure stands; training waits for diagnostics and review fixes | Artifact-pair parity verified; process independence unknown; capture clip fraction and harden replay/attribution gates |
| **D1** | Does R1 merit extended evaluation? | Screen, do not presume | G1 correctness + G2a paired measurable signal |
| **D2** | Does carried memory help? | Unknown | G2b `CARRY` over identical `RESET` |
| **D3** | Should PBT reopen? | No during attribution | Fixed representation plus trustworthy optimizer telemetry |
| **D4** | Should S1-P0 run in parallel? | Yes as a bounded diagnostic after interface freeze | Stable serialization/action order, resource receipt, and no impact on R runs |
| **D5** | Which S training path comes first? | Heads-only choice objective; CLM-PPO as a later matched branch | S1-P0 signal/latency/collapse results and label provenance |
| **D6** | Is direct AZR learning progress measurable? | Test tiny fixed reference before surrogate | Stable G6a ranks and usable intervals |
| **D7** | Should PF be promoted into the active plan? | Not yet | R/C closure plus G8 proper-score/calibration evidence |
| **D8** | Is deployable TTS worth testing? | No; hindsight diagnostic only | Valid WM and calibrated blueprint, then replay/shadow search-off/on evidence; PF remains an independent optional input ablation |
| **D9** | What is R4-v2's actual implementation state? | Deferred on this checkout | Direct inspection of the reported parallel branch |

## 16. Evidence ledger

### Foundations and Phase A

- [Phase-A performance validity audit](../artifacts/research/phase-a-performance-validity-audit-2026-09-12.md)
- [Phase-A MLP-FiLM autoresearch](../artifacts/research/phase-a-mlp-film-autoresearch-2026-09-19.md)
- [Dataset-v2 R0 audit](../artifacts/research/dataset-v2-r0-audit-2026-09-24.md)
- [Ray-2.58 R0 parity gate](../artifacts/research/ray-258-r0-parity-gate-2026-09-24.md)

### R-series

- Focused design ledger: [Recurrent Solver + FiLM Ablation](recurrent-solver-ablation.md)
- R4-v1 chronology: [Markov note](../artifacts/research/solver-state-markov-completion-2026-09-25.md) → [minimum implementation](../artifacts/research/solver-state-r4-v1-min-implementation-note-2026-09-25.md) → [blocking ablation](../artifacts/research/solver-state-r4-v1-blocking-ablation-2026-09-25.md) → [replication](../artifacts/research/solver-state-r4-v1-replication-plan-2026-09-25.md) → [actor robustness](../artifacts/research/solver-state-r4-v1-actor-robustness-2026-09-29.md) → [causal forensics](../artifacts/research/solver-state-r4-v1-causal-forensics-2026-09-29.md)

### S-series

- [Primary CLM/Jev integration report](../artifacts/research/system-one-clm-jev-trading-ablation-2026-09-24.md)
- [OSS System-One, PPO, and AZR companion](../artifacts/research/system-one-oss-ppo-azr-companion-2026-09-24.md)

### POMDP / self-play transfer

- [Stratego, imperfect-information self-play, and iTrader](../artifacts/research/stratego-pomdp-self-play-itrader-2026-10-01.md)

### Reward and proposer

- [Sharpe-Aware Reward Design](sharpe-reward-design.md)
- [AZR stack review](../artifacts/research/azr-stack-review-2026-09-12.md)

The complete categorized catalog, including imported reference-only HTML, lives
in [the artifact research index](../artifacts/research/README.md).

## 17. Documentation organization

```text
docs/
  README.md                         navigation and authority map
  architecture*.md                 active invariants and system boundaries
  plans/itrader_masterplan.md      canonical phase order and promotion gates
  project-state/                   current handoff, decisions, and next actions
  research/
    itrader_research_dashboard.*   canonical research registry (this pair)
    recurrent-solver-ablation.*    focused R identity/design ledger
    sharpe-reward-design.*         focused reward design ledger
  artifacts/research/
    README.md                       categorized evidence/provenance catalog
    dated reports + evidence dirs  immutable investigation history
    imported HTML                  advisory/reference material only
```

Existing reports remain in place because their relative evidence links and
snapshot citations are part of their provenance. Organization is achieved by
authority, registry, and catalog rather than risky mass moves.

## 18. Adding or changing a research identity

Before adding an ID:

1. state the single factor and falsifiable question;
2. classify it as canonical identity, control/cross, provisional experiment,
   system seam, or foundation;
3. identify the baseline, prerequisites, held-fixed factors, and stop rule;
4. estimate the bounded screen and promotion budget;
5. link implementation and evidence separately;
6. add it here without implying masterplan promotion; and
7. update architecture/masterplan only if their owning decision has changed.

Never recycle an existing R/C/S ID for a new meaning. If a historical label was
misapplied, preserve the provenance and write an explicit alias/migration note.

## 19. Reconciliation log

### 2026-10-01

- Created the canonical research registry and separated it from masterplan
  authority.
- Tightened scope to research branches and ablations only. Core Phase-B AZR
  loop construction, task families, capability/replay contracts, online shell,
  and platform foundations remain in the masterplan/architecture.
- Added the dedicated C0/C1/shuffled-C1 contract and expanded PF, WM, and TTS
  inputs, outputs, controls, gates, and non-claims without turning them into one
  mandatory model stack.
- Clarified that the AZR-LP/QD labels compare proposer credit and task retention
  inside the planned loop; they do not replace it, and original AZR's local-v2
  `r_prop` is an asymmetric learnability proxy rather than a parabola or direct
  pre/post learning-progress measure.
- Registered R, C, and S identities; separated provisional ATX/PF/WM/TTS/AZR
  experiment names from canonical wings.
- Recorded completed R4-v1 actor-robustness and causal-forensics stop decisions.
- Recorded unresolved local-versus-parallel R4-v2 implementation status without
  changing its fixed joint-fusion identity.
- Consolidated the evidence chain and added previously unindexed Markov and
  Stratego reports.
- Kept Market-JEPA as a separate, non-concrete Phase-D line.
- Made the correct CLM/PPO loss boundary explicit: repeated-label pairwise
  InfoNCE is invalid; per-question choice, multi-positive loss, or categorical
  PPO are valid alternatives.

## History and operational log

These dated notices preserve the operational sequence, not current run status.
The final closeout supersedes earlier holds; evidence is not removed.

**Final closeout, 16:38:** all eight corrected-compatibility terminal replays
are independently accepted (256 episodes); 416 tests, healthcheck and pre-commit
pass. No economic promotion. The positive staged R4 seed is reproducible, not
replicated across seeds. Next: accepted panel/completion/reporting prerequisites
for R1. All older pending-replay notices below are historical. WSL remains off.

**16:32 current:** all eight bridge terminals now have complete witnessed
replays (256 episodes) with indexed checkpoint/payload/score parity. Aggregate
review and repository validation are finishing. The positive staged R4 seed
reproduces +0.392761% net / +0.018892 Sharpe per sqrt-bar; the second seed is
flat, so there is no replicated economic promotion. Earlier holds below are
historical. WSL is off and the final closeout is Windows-only.

**16:10 replay renewal:** reviewed Windows job-object containment replaces the
failed external watchdog. Eight bounded terminal replays use the immutable
training evaluator snapshot and a separately stamped repaired launcher. The
human removed the global deadline; hard free-RAM reserve1GiB, proactive1.25GiB.
Full acceptance remains open, with no new training or economic promotion.
The historical safety hold below and failed15:37 attempt remain documented.

**13:48 safety hold:** all eight corrected-stack training cells completed;
three terminal replays reproduce their episode payloads. Replay 3 nevertheless
breached the 2 GiB RAM floor (1.47 GiB observed), then exited naturally.
Further replay is held and the heartbeat paused; full replay acceptance remains
open. No lost training, no OOM observed and no economic promotion. The first
two replay receipts also need witnessed repeats to close process provenance.

The [Experiment Browser's latest tab](itrader_experiment_browser.html#exp-0)
now indexes all eight bridge terminals separately from the smoke. It displays
official terminal net values; the report links Sharpe, action and PPO curves.
Staged R4 has positive net/Sharpe transients on both seeds, but terminal positive
performance only on the first. Replay hold and no-promotion labels remain explicit.

R1 preparation: [development contract](../project-state/r1-development-readiness.md).
The next screen includes R0 + R4-v1 as well as R0, R1 and R1 + R4-v1.
Replay/attribution review blockers and learner diagnostics precede training.

Current platform boundary: Ray 2.58 runtime with ModelV2/TorchPolicy compatibility
APIs. Finish corrected-stack R0/R4 controls and R1 attribution here first; defer
native RLModule/Learner/EnvRunner until a later four-cell platform bridge.
Historical PBT events are not proof of effective optimizer/policy mutations;
see the [compatibility/evidence audit](../artifacts/research/r1-foundation-hardening-2026-10-05.md).
Two corrected one-update smoke cells and fresh 32-task checkpoint replays now
pass. R0 net -0.059056%, Sharpe -0.044665 per sqrt-bar; combined-state R4 net
+0.441611%, Sharpe +0.024455, entirely short-only. These are mechanics checks,
not a completed attribution study or economic promotion.

The [retained-checkpoint audit](../artifacts/research/ray258-historical-optimizer-audit-2026-10-05.md)
confirms effective 30/128 optimizer settings, five exploited-target divergences,
and six member-00002 self-resume entropy spikes. Fresh corrected-stack training
has now been authorized under GPT-6 Luna/xhigh supervision; existing charts
remain historical observations. The [corrected-control audit](../artifacts/research/corrected-compatibility-r0-r4-controls-2026-10-05.md)
tracks preflight, frozen configuration, stops and eventual results. PBT is off;
this round cannot measure exploit/explore effectiveness.
Attempt04 now has all eight training cells complete (13:30:16, 467,584 sampled
steps); 96 reporting rows and 33 checkpoint hashes pass parent checks. Fresh
terminal replays and independent evidence closure remain pending. Exact-cost
terminals are all flat; staged R4's +0.392761% net / +0.018892 per-sqrt-bar Sharpe
on seed 20260924 does not replicate on seed 20260925 (flat). The linked audit
owns current replay status and official comparison plots. No economic promotion.
At 13:37, replay is not yet running: the eight terminal-input selections pass,
but RAM is approximately 4.12 GiB, below the declared 5 GiB startup gate.
By 13:42 headroom recovered and replay 1 passed exact episode/payload parity.
An uncaptured launcher/driver PID link is being resolved through the existing
process-witness helper and one bounded repeat; final replay acceptance is pending.

Q21 two-tier evaluation is now accepted for the later 16-cell representation
screen: the inspected 32-task panel monitors initialization/every update;
a separately predeclared fresh validation-only panel has 32 symbol/session
anchors x horizons 96/128/160/200 = 128 terminal tasks. Open it only after all
16 runs finish, using predetermined terminal checkpoints and identical tasks
for every cell/control. The final test remains sealed. Panel construction and
runner enforcement remain pending; the existing eight-cell bridge is distinct.
Q22's identical corrected optimizer recipe is accepted, with PBT off and the
original entropy clock. Q23 metadata-only fresh-panel sampling is approved;
eligibility audit and panel freeze remain pending. Q24 equal-symbol terminal-net
reporting is accepted but unimplemented; raw episode means remain secondary
and unchanged. Q25 secondary net-bps-per-actual-bar reporting is also accepted
but unimplemented: normalize each episode by its actual positive step count
before equal-symbol aggregation. It does not replace terminal net return.
See the [full reporting contract](../project-state/r1-development-readiness.md#accepted-q21-evaluation-and-reporting-contract).
The dashboard must separate economics, matched directional/flat/random/untrained
controls, encoder/state/interaction effects, PPO/critic diagnostics, actual
cost-stage exposure, and resource/provenance evidence. Preserve raw traces;
crossed horizons and two training seeds do not establish independent significance.

Native Ray 2.58 API development is staged separately on `feat(R)/Ray_upgrade`;
integration still follows the corrected-compatibility R1 comparison and requires
all four cells. No migration commit/push permission extends to this dirty tree.
