# CODEX IMPLEMENTATION SPEC
## Project: iTRADER Coupled Offline/Online Architecture
## Status: Working draft for Phase B and Phase C
## Decision state: User grill-me review ongoing; architecture and order below are proposals unless explicitly marked source-defined.
## Intent: High-level implementation plan only. Preserve existing working pieces wherever possible.

---

# 0. PRIMARY OBJECTIVE

Transform iTRADER from a Phase A / early Phase B research codebase into a coupled system with:

1. **Offline training mode**
   - `CurriculumProposer` generates Goldilocks-difficulty tasks.
   - `Solver` learns to solve trading-related task families.
   - `r_prop` is shaped by solver usefulness, not only output validity.
   - Multi-head solver architecture is introduced for task families beyond family `00`.

2. **Online trading mode**
   - Proposer is promoted into `MarketFramer` (alias allowed: `AdversarialProposer`).
   - `MarketFramer` transforms live market windows into a small set of candidate tasks, including `WAIT` / `NO_ACTION`.
   - Solver evaluates candidate tasks and only acts when it has sufficient edge and confidence.
   - A deterministic `RiskEngine` and `Executor` gate live actioning.
   - `WatsonOW` later orchestrates this stack as an external agent shell / skill consumer.

This spec is deliberately **architecture-first** and **interface-first**.
Codex should preserve current learnable, stable pieces and extend the system incrementally.

---

# 1. HARD CONSTRAINTS

## 1.1 Preserve working paths
Do not break:
- current Phase A training path,
- current `TaskSpec` usage,
- current smoke runs and healthchecks,
- current proposer backends that already produce validated JSON,
- current metrics/reporting outputs unless migrated compatibly.

## 1.2 Prefer extension over rewrite
If existing modules are close to the needed role, refactor or wrap them.
Avoid replacing known-good infrastructure unless clearly necessary.

## 1.3 Keep typed contracts central
All new moving parts must be introduced through explicit, typed contracts or schema-like validation layers.

## 1.4 Online execution must remain deterministic at the shell
LLM output may frame or suggest tasks.
LLM output must not directly bypass:
- solver evaluation,
- decision thresholds,
- risk limits,
- executor constraints.

## 1.5 WAIT / NO_ACTION is first-class
The online stack must support abstention natively.
No component should force trade activity when confidence/edge is absent.

---

# 2. TARGET ARCHITECTURE

## 2.1 Offline architecture

Historical market windows
-> Summarizer / feature assembly
-> CurriculumProposer
-> Candidate TaskSpecs
-> Validation / TaskSampler / TaskPool
-> Solver (shared trunk + family heads)
-> `r_solve`, metrics, capability updates
-> `r_prop`, golden task logging, dataset growth

## 2.2 Online architecture

Live rolling market window
-> Summarizer / market state encoder
-> MarketFramer (AdversarialProposer)
-> CandidateTasks (incl. WAIT / NO_ACTION)
-> Solver heads evaluate solvability / edge / confidence
-> DecisionGate
-> RiskEngine
-> Executor
-> Telemetry / memory / CapabilityProfile update
-> WatsonOW integration later

---

# 3. ROLE DEFINITIONS

## 3.1 Solver
Primary learned market-control component.

Responsibilities:
- consume observations + task conditioning,
- estimate action / value / confidence / solvability,
- support multiple task families through separate heads,
- remain trainable with PPO + GAE in current setup,
- provide outputs to offline training and online scoring.

Must NOT:
- directly place live trades,
- directly override risk limits,
- directly define broker semantics.

## 3.2 CurriculumProposer
Offline task generator.

Responsibilities:
- generate valid `TaskSpec` candidates,
- aim for Goldilocks difficulty,
- maximize solver improvement rather than raw task novelty,
- support later SFT/QLoRA training,
- log grouped candidate outputs for future ranking/GRPO.

## 3.3 MarketFramer
Online market-conditioned proposer.
Alias `AdversarialProposer` may remain as a naming option.

Responsibilities:
- take a live or replayed market window,
- decompose it into a small candidate set of tractable tasks,
- emit `WAIT` / `NO_ACTION` when market state is unsuitable,
- attach execution logic, TTL, confidence band, and rationale.

This role is NOT equivalent to the offline CurriculumProposer, though both may share a common backend/model family.

## 3.4 DecisionGate
Deterministic selection layer between solver outputs and risk approval.

Responsibilities:
- compare candidate tasks,
- apply minimum solver confidence thresholds,
- apply minimum expected utility thresholds,
- prefer `WAIT` if nothing clears thresholds,
- pass only approved intents to RiskEngine.

## 3.5 RiskEngine
Deterministic hard-guardrail layer.

Responsibilities:
- spread/slippage checks,
- max position sizing,
- exposure caps,
- instrument limits,
- daily loss / drawdown lockouts,
- cooldown rules,
- no-trade time windows,
- kill switch semantics.

Must remain rule-based.

## 3.6 Executor
Deterministic broker/order layer.

Responsibilities:
- map approved decisions to orders,
- handle sim / shadow / live modes,
- confirm order state,
- retry or fail safely,
- record order telemetry.

## 3.7 WatsonOW
External orchestrator / agent shell.

Responsibilities later:
- call `itrader-skill`,
- request offline analysis, shadow replay, or live evaluation,
- summarize outcomes,
- track memory/logs,
- never bypass RiskEngine for live execution.

---

# 4. CONTRACT MODEL

## 4.1 Rule
`TaskSpec` remains the canonical contract for “what problem is posed”.
Do not overload it with full broker semantics.
Introduce additional contracts for online execution and capability tracking.

## 4.2 New contracts to add

### A. `TaskSpec`
Purpose:
Defines the task/problem itself.

Required high-level fields:
- `task_family`
- `bar_resolution`
- `horizon_bars`
- `start_idx` or equivalent index reference
- `reward_config`
- `regime_tags`
- `difficulty_target`
- `metadata`

Illustrative structure:

```json
{
  "task_family": "00",
  "bar_resolution": "1m",
  "horizon_bars": 12,
  "start_idx": 12345,
  "reward_config": {
    "name": "composed_linear",
    "params": {}
  },
  "regime_tags": ["trend", "high_vol"],
  "difficulty_target": {
    "p_solve_min": 0.55,
    "p_solve_max": 0.80
  },
  "metadata": {
    "source": "curriculum_proposer",
    "dataset_fingerprint": "abc",
    "summary_id": "xyz"
  }
}
```

Notes:
- `bar_resolution` is now required to eliminate 1m vs 5m ambiguity.
- Difficulty target is declarative only; online gating should not rely on this field alone.

### B. `ExecutionIntent`
Purpose:
Defines how a candidate task would be traded if accepted in online mode.

Required fields:
- `entry_logic`
- `exit_logic`
- `ttl_bars`
- `confidence_band`
- `risk_envelope`
- `abstain_allowed`

Illustrative structure:

```json
{
  "entry_logic": {
    "type": "breakout_confirm",
    "params": {
      "confirm_bars": 2
    }
  },
  "exit_logic": {
    "type": "ttl_or_stop_take",
    "params": {
      "ttl_bars": 10,
      "stop_loss_bps": 18,
      "take_profit_bps": 35
    }
  },
  "ttl_bars": 10,
  "confidence_band": {
    "min": 0.62,
    "target": 0.74
  },
  "risk_envelope": {
    "max_position_pct": 0.05,
    "max_spread_bps": 8
  },
  "abstain_allowed": true
}
```

### C. `CandidateTask`
Purpose:
Online or offline candidate emitted by proposer/framer that combines a task with optional execution framing.

Required fields:
- `task_spec`
- `execution_intent`
- `expected_difficulty`
- `expected_utility`
- `rationale`
- `abstain_option`
- `created_at`
- `ttl_generated_at`

Illustrative structure:

```json
{
  "task_spec": {},
  "execution_intent": {},
  "expected_difficulty": 0.68,
  "expected_utility": 0.21,
  "rationale": "Compression followed by likely breakout regime.",
  "abstain_option": false,
  "created_at": "2026-03-22T00:00:00Z",
  "ttl_generated_at": "2026-03-22T00:10:00Z"
}
```

### D. `CapabilityProfile`
Purpose:
Track what the solver can actually solve under given conditions.

Required fields:
- `task_family`
- `bar_resolution`
- `regime_tags`
- `horizon_bars`
- `estimated_p_solve`
- `expected_net_utility`
- `calibration_error`
- `sample_count`
- `last_updated`

Illustrative structure:

```json
{
  "task_family": "00",
  "bar_resolution": "1m",
  "regime_tags": ["trend", "high_vol"],
  "horizon_bars": 12,
  "estimated_p_solve": 0.71,
  "expected_net_utility": 0.08,
  "calibration_error": 0.05,
  "sample_count": 423,
  "last_updated": "2026-03-22T00:00:00Z"
}
```

### E. `DecisionRecord`
Purpose:
Immutable record of one decision cycle.

Fields:
- market summary reference,
- candidate tasks,
- selected candidate or WAIT,
- solver scores,
- risk decisions,
- executor action,
- realized outcome.

This is the bridge between offline replay and live auditing.

---

# 5. TASK FAMILY STANDARD

## 5.1 General rule
Family `00` remains the current proven path.
Do not destabilize it.
Families `00`–`04` are the roadmap. Families `00`–`02` may currently use the same
`Discrete(3)` action width, but shared output width does not settle their input
schemas, task semantics, reward assignments, or state ownership. Introduce new
families progressively and record those differences explicitly.

## 5.2 Initial family set

### Family 00 — Directional predictor
Current proven family.
Problem shape:
- directional or short-horizon predictive task,
- current default action/reward path.

### Family 01 — Breakout / expansion
Problem shape:
- identify likely expansion or continuation after compression,
- entry often conditional,
- useful for framing trend-following opportunities.

### Family 02 — Mean reversion / snapback
Problem shape:
- identify likely reversion opportunities,
- generally different entry logic and sometimes different ideal action semantics.

### Family 03 — Trade management / hold-vs-exit
Problem shape:
- conditional on already open position,
- determines whether to hold, trim, or exit under evolving conditions.

### Family 04 — Abstain / unsuitable regime
Problem shape:
- detect when no-trade is preferable,
- useful both as explicit training target and online candidate type.

## 5.3 Family implementation rule
Do NOT build all five families at once. The `00`–`04` list is a roadmap, not an
accepted head or input architecture.

Implementation order proposal, pending grill-me review:
1. lock and clean `00`
2. add `01`
3. add `02`
4. optionally add `03`
5. add explicit `04` if not already implicit through WAIT

If action spaces differ significantly, evaluate whether to use shared or
family-specific policy/value heads; do not treat a shared backbone or head layout
as settled merely because `00`–`02` can emit `Discrete(3)`.
---

# 6. SOLVER ARCHITECTURE STANDARD

## 6.1 Open design question

The branch has one implemented/default solver, R0 `SolverMLPFiLM`. A shared
recurrent trunk with family-specific policy/value heads is a proposed option for
Phase B, not a settled architecture. Do not implement that choice until the
ongoing grill-me review confirms it. Shared `Discrete(3)` output width for
families `00`–`02` does not prove shared input schemas, task semantics, reward
assignments, or recurrent state ownership.

## 6.2 Candidate components to evaluate

Potential components include:
- shared observation encoder/trunk;
- task/family and optional regime conditioning paths;
- family-specific or shared policy/value heads;
- per-head metrics and calibration tracking.

The selected arrangement must be justified by matched controls rather than by
naming or registration. `SolverMLPFiLM` remains R0, and `RewardEngine` remains
the environment-owned reward seam; no learned reward head is required by this
spec.

## 6.3 Design rule while unresolved

Preserve R0 and keep candidate trunk/head layouts explicitly versioned. Treat
shared trunk + family heads, shared heads, and any memory-conditioning placement
as alternatives under review. Avoid MoE for now.

## 6.4 Why not MoE yet
MoE is deferred because:
- debugging complexity is high,
- routing errors can mask learning issues,
- current repo is still expanding task families,
- clear ablations are more important than capacity tricks right now.

## 6.5 Acceptance criteria
- family `00` still trains and evaluates successfully,
- adding family `01` does not break family `00`,
- the chosen routing/head arrangement is documented as an accepted decision before implementation,
- head-specific metrics are visible when heads exist,
- calibration and confidence claims are separately evidenced,
- existing training paths remain usable.

---

# 7. REWARD STANDARD

## 7.1 `r_solve`
Keep PPO + GAE for solver training.
Make `r_solve` explicitly decomposed and reportable.

Target conceptual form:

`r_solve =
 + w_pnl     * pnl_net
 + w_hit     * task_success
 + w_cal     * calibration_gain
 - w_dd      * drawdown_penalty
 - w_turn    * turnover_penalty
 - w_cost    * transaction_cost_penalty
 - w_invalid * invalid_action_penalty`

Notes:
- actual formulas may remain family-specific,
- maintain finite outputs and robust sanitization,
- report reward components separately,
- `RewardEngine` remains the environment-owned reward calculation seam; do not
  conflate it with a learned policy/value/reward head or make recurrence depend on
  a reward rewrite.

## 7.2 `r_prop`
Formal proposer reward standard:

`r_prop =
 + λ_band   * goldilocks_bonus
 + λ_uplift * solver_uplift_bonus
 + λ_util   * realized_task_utility_bonus
 + λ_div    * diversity_bonus
 - λ_easy   * too_easy_penalty
 - λ_hard   * too_hard_penalty
 - λ_inv    * invalid_task_penalty
 - λ_hack   * reward_hacking_penalty`

### Definitions

#### Goldilocks
A task is Goldilocks if it is:
- solvable enough to produce learning signal,
- not trivial,
- positive utility after costs/risk assumptions,
- within acceptable drawdown / stability profile.

Important:
Do not define Goldilocks purely as win rate.

#### Solver uplift
Measures whether the task or candidate group improved solver capability on relevant evaluation slices.

#### Diversity
Rewards useful task diversity across regimes and horizons, not random noise.

#### Invalid task penalty
Applied for:
- schema-invalid outputs,
- impossible execution semantics,
- mismatched family constraints,
- incoherent reward or horizon settings.

## 7.3 Online Goldilocks
For online selection, Goldilocks means:
- estimated solver capability is sufficient,
- expected utility after costs is positive,
- confidence is above threshold,
- risk is acceptable.

It is not just “50–90% win rate”.

---

# 8. PROPOSER TRAINING STANDARD

## 8.1 Stage 0 — rules and seeds
Maintain or improve:
- rule-based proposer,
- curated golden TaskSpecs,
- deterministic fallback generation.

This remains necessary for:
- smoke tests,
- failure isolation,
- bootstrap dataset creation.

## 8.2 Stage 1 — SFT / QLoRA proposer
Train proposer to emit:
- valid TaskSpecs,
- family-appropriate fields,
- reasonable difficulty targets,
- coherent rationales,
- later, valid CandidateTasks / ExecutionIntents.

Priority:
- correctness and structure first,
- sophistication second.

## 8.3 Stage 2 — grouped candidate optimization
After sufficient logged data exists:
- compare multiple proposer outputs per same market summary,
- score them by downstream usefulness,
- support ranking or later GRPO-like optimization.

## 8.4 Rule for now
Do NOT apply GRPO to the trading solver path.
If explored, GRPO belongs first on the proposer side once grouped candidate evaluation is robust.

---

# 9. OFFLINE TRAINING LOOP SPEC

## 9.1 Loop intent
Replace the current bootstrap-only proposer->solver handoff with a proper co-adaptive loop, but preserve bootstrap mode as a fallback or submode.

## 9.2 Offline loop
1. sample historical market window / summary
2. CurriculumProposer emits candidate tasks
3. validate and clamp outputs
4. construct or update task pool
5. train solver on accepted tasks
6. evaluate solver on held-out or rolling evaluation windows
7. update `CapabilityProfile`
8. compute `r_prop`
9. log candidate groups, accepted tasks, outcomes, metrics
10. optionally refresh proposer policy/training dataset

## 9.3 Dataset logging requirement
Every proposer cycle should write enough information for:
- SFT,
- ranking,
- capability analysis,
- regression debugging,
- future GRPO or alternative proposer training.

---

# 10. ONLINE LOOP SPEC

## 10.1 Core rule
Online mode must begin in replay/shadow mode before live execution.

## 10.2 Online loop
1. acquire rolling live/replay market window
2. build market summary
3. query MarketFramer
4. receive up to `K` candidate tasks plus WAIT
5. validate candidates
6. solver scores each candidate
7. DecisionGate filters by confidence/utility thresholds
8. RiskEngine approves or blocks
9. Executor places action or records no-action
10. write DecisionRecord and telemetry
11. update CapabilityProfile / online statistics

## 10.3 Candidate count
MarketFramer should emit a small set only.
Default recommendation:
- 1 to 3 trade candidates,
- 1 WAIT candidate.

## 10.4 WAIT semantics
WAIT must be selected automatically if:
- all candidates are invalid,
- all candidates are below confidence threshold,
- all candidates are below expected utility threshold,
- RiskEngine blocks every candidate.

## 10.5 Online outputs
Each decision cycle should log:
- market summary ID,
- candidate tasks,
- selected candidate or WAIT,
- solver scores,
- risk decisions,
- final action,
- realized outcome after TTL or exit.

---

# 11. DECISIONGATE STANDARD

## 11.1 Inputs
- candidate tasks,
- solver outputs,
- capability profile,
- current risk state,
- optional portfolio state.

## 11.2 Outputs
- selected candidate ID,
- WAIT,
- or “none approved”.

## 11.3 Decision logic
Use simple deterministic logic first:
- filter invalid candidates,
- filter below confidence threshold,
- filter below expected utility threshold,
- rank by utility-adjusted score,
- prefer WAIT if no strong candidate remains.

## 11.4 No hidden heuristics
Any ranking or threshold logic must be:
- inspectable,
- configurable,
- logged in telemetry.

---

# 12. RISKENGINE STANDARD

## 12.1 Required checks
- spread / slippage bounds,
- max position size,
- max per-asset exposure,
- max portfolio exposure,
- daily realized loss cutoff,
- drawdown lock,
- cooldown after stop-outs or loss streaks,
- session/no-trade windows if relevant.

## 12.2 Required modes
- replay mode,
- shadow mode,
- live mode.

Behavior should be as consistent as possible across modes.

## 12.3 Hard constraint
RiskEngine decisions must be final for live actioning.
Neither MarketFramer nor Solver nor WatsonOW can override them directly.

---

# 13. EXECUTOR STANDARD

## 13.1 Responsibilities
- map approved decisions to broker-compatible actions,
- support shadow/sim/live modes,
- track acknowledgements/fills/errors,
- fail safely,
- produce audit logs.

## 13.2 Non-goals
Executor should not:
- invent strategy,
- alter solver intent beyond deterministic normalization,
- change risk rules.

---

# 14. DATASET STANDARD

## 14.1 Golden dataset schema
Each row should include:
- market summary,
- raw window reference,
- task family,
- task spec,
- execution intent if applicable,
- proposer rationale,
- candidate group ID,
- validation results,
- solver pre metrics,
- solver post metrics,
- estimated p_solve,
- realized utility,
- acceptance label,
- golden label if curated.

Illustrative shape:

```json
{
  "market_summary": {},
  "raw_window_ref": "dataset/window/123",
  "task_family": "01",
  "task_spec": {},
  "execution_intent": {},
  "proposer_rationale": "Compression before expansion.",
  "candidate_group_id": "cg_001",
  "validation_errors": [],
  "solver_pre_metrics": {},
  "solver_post_metrics": {},
  "estimated_p_solve": 0.63,
  "realized_utility": 0.12,
  "accepted_for_training": true,
  "golden_label": "keep"
}
```

## 14.2 Capability dataset
Maintain derived tables or artifacts for:
- solve probability by family/regime/horizon,
- expected utility by family/regime/horizon,
- calibration quality,
- abstention success rate,
- candidate acceptance rate.

---

# 15. MODULE PLAN

## 15.1 Existing modules to preserve/extend
Codex should inspect the repo and reuse these where possible:
- envs
- task contracts/samplers/validation
- solver model path
- train scripts
- proposer abstraction layer
- plotting / metrics / manifests
- docs and ops patterns

## 15.2 Suggested new or extended modules

```text
itrader/
  rl_lib/
    tasks/
      spec.py                    # existing; extend carefully
      execution_intent.py        # new
      candidate_task.py          # new
      capability_profile.py      # new
      decision_record.py         # new
      families.py                # new or extend enums/registry
      validation.py              # extend existing
  llm/
    curriculum_proposer.py       # new wrapper/role layer
    market_framer.py             # new wrapper/role layer
    proposer_backends/           # reuse existing backend system
  models/
    solver_multitask.py          # new or refactor existing FiLM model
  online/
    decision_gate.py             # new
    risk_engine.py               # new
    executor.py                  # new
    broker_adapters/             # optional new
  rl_lib/train/
    train_phase_b.py             # new or refactor train_AZR toward this
    replay_online_shadow.py      # new
  tools/
    capability_report.py         # new
    replay_market_framer.py      # new
```

Use equivalent paths if repo conventions differ.

---

# 16. WORKSTREAMS

## Workstream 1 — Contract formalization
Deliverables:
- typed/validated `ExecutionIntent`
- typed/validated `CandidateTask`
- typed/validated `CapabilityProfile`
- typed/validated `DecisionRecord`
- `TaskSpec` extended with `bar_resolution` and cleaner metadata where needed

Acceptance criteria:
- unit tests for serialization/validation,
- invalid proposer outputs fail clearly,
- existing `TaskSpec` consumers still work or fail with clear migration messages.

## Workstream 2 — Family expansion
Deliverables:
- explicit family registry,
- family `00` locked,
- family `01` added,
- family `02` added,
- action-space and reward-path abstraction as needed.

Acceptance criteria:
- `00` regression-free,
- `01` and `02` can be sampled/validated,
- train/eval smoke path exists for each.

## Workstream 3 — Solver trunk/head alternatives (decision pending)
Deliverables:
- matched candidate layouts for shared trunk, family-specific heads, or other
  explicitly justified routing;
- task/family conditioning;
- per-head metrics where heads are used.

Acceptance criteria:
- the architecture is accepted through the user review before implementation;
- family-specific outputs are routed correctly if that option is selected;
- family `00` still converges;
- metrics can be split by family without assuming identical input semantics.

## Workstream 4 — Proposer split
Deliverables:
- `CurriculumProposer`
- `MarketFramer`
- shared backend compatibility
- grouped candidate logging

Acceptance criteria:
- both wrappers can call existing backends,
- both pass through validation,
- rule-based fallback exists for both.

## Workstream 5 — Capability profiling
Deliverables:
- capability estimator,
- persistence format,
- helper API for offline and online consumers.

Acceptance criteria:
- solver outcomes update profile,
- MarketFramer/DecisionGate can query it,
- profile supports family/regime/horizon granularity.

## Workstream 6 — Phase B loop evolution
Deliverables:
- `train_AZR.py` evolved or wrapped into fuller co-adaptation loop,
- bootstrap mode preserved,
- proposer reward logging,
- candidate-group logging.

Acceptance criteria:
- offline proposer->solver cycles can run,
- telemetry captures enough information for future proposer training,
- smoke tests remain CPU-safe where possible.

## Workstream 7 — Online shadow stack
Deliverables:
- `DecisionGate`
- `RiskEngine`
- `Executor`
- replay/shadow runner

Acceptance criteria:
- historical replay runs end to end,
- WAIT path is explicit,
- decisions are auditable.

## Workstream 8 — Training corpus and fine-tuning readiness
Deliverables:
- dataset writer,
- curation hooks,
- golden-task labeling support,
- grouped candidate scoring support.

Acceptance criteria:
- enough structure exists for later SFT/QLoRA,
- proposer data is reusable,
- failure cases are inspectable.

---

# 17. TESTING STANDARD

## 17.1 Unit tests
Must cover:
- contract validation,
- reward component finiteness,
- family routing,
- proposer output clamping,
- DecisionGate logic,
- RiskEngine blocking logic.

## 17.2 Integration tests
Must cover:
- offline proposer -> validation -> solver path,
- replay market window -> MarketFramer -> DecisionGate -> RiskEngine -> Executor path,
- manifest / metadata / telemetry output.

## 17.3 Regression tests
Must cover:
- NaN/Inf failures,
- bad proposer JSON,
- invalid family/action mismatch,
- checkpoint restore where relevant,
- backwards compatibility for family `00`.

## 17.4 Smoke profiles
Keep lightweight smoke configs that can run without full GPU training.
Do not tie correctness to expensive long runs.

---

# 18. METRICS STANDARD

## 18.1 Offline metrics
Track at minimum:
- reward total and components,
- episode return,
- family-specific success metrics,
- calibration/confidence,
- drawdown/turnover/cost stats,
- proposer acceptance rate,
- proposer invalid rate,
- solver uplift on evaluation windows.

## 18.2 Online metrics
Track at minimum:
- candidate count,
- WAIT rate,
- selected task family,
- solver confidence,
- expected utility,
- risk block rate,
- trade acceptance rate,
- realized PnL after TTL/exit,
- slippage/cost stats,
- calibration by family and regime.

## 18.3 Capability metrics
Track:
- `p_solve` estimates by family/regime/horizon,
- utility by family/regime/horizon,
- abstention quality,
- drift over time.

---

# 19. PHASE REMAP

## 19.0 Phase A evidence contract now implemented

New Phase A runs use terminal `EpisodeEvidence` schema v1, symbol-safe purged
train/validation/test partitions, `semantic_v2` task encoding, and PBT selection
on held-out net arithmetic return at an explicit cost. Population members start
distinctly but share a deterministic validation stream and fixed rollout batch;
mutations are bounded and checkpoint/perturbation cadence is aligned. Active
metrics rewind on exploit restore while append-only config history retains
lineage. Atomic manifests/status/config/checkpoint/metric artifacts outside Tune
are authoritative on Ray 2.9/Windows.

This closes the engineering seam, not the performance gate. Baseline-controlled
multi-seed overnight training, cost scenarios, block-aware uncertainty, and one
untouched final-test evaluation remain mandatory.

## 19.1 Updated Phase B — offline + online/replay/shadow

Phase B now includes both tracks:
- offline/curriculum work: proposer integration, family/task-contract work,
  solver evidence, multi-head/trunk alternatives, and candidate-group logging;
- online/replay/shadow work: `MarketFramer` shadow-mode evaluation on replayed
  windows, candidate scoring, WAIT/NO_ACTION, and deterministic shell evidence.

Replay/shadow is therefore not excluded from Phase B or deferred until Phase C.
Live broker execution remains outside the Phase B scope. The shared recurrent
trunk and family-specific policy/value heads remain proposed choices pending the
user's grill-me review.

## 19.2 Updated Phase C — operationalization

Phase C means hardening the coupled offline/online architecture after Phase B
replay/shadow evidence:
- `CurriculumProposer` and `MarketFramer` are operational;
- online `DecisionGate`/`RiskEngine`/`Executor` shell is hardened;
- WatsonOW can consume iTRADER as a skill;
- proposer fine-tuning can later use an accumulated, audited corpus.

Phase C does NOT require immediate replacement of the solver with an LLM.

---

# 20. IMPLEMENTATION ORDER (PROPOSAL UNDER REVIEW)

Codex should preserve the following dependency order unless the user's grill-me
review accepts a nearby variant. This is not an accepted architecture decision:

1. preserve R0/family `00` and make metadata truthful;
2. formalize contracts and validate family/task semantics;
3. run matched alignment/reward controls, keeping `RewardEngine` independent of
   policy/value-head and recurrence choices;
4. resolve the solver trunk/head and input-schema decision before implementation;
5. implement/test the selected R1/R2 recurrent controls and lifecycle contract;
6. add families `01` and `02` only with explicit input/task/reward semantics;
7. add any shared or family-specific heads only after that choice is accepted and
   matched controls are defined;
8. run the separate proposer/curriculum evidence loop;
9. run Phase B replay/shadow work through MarketFramer and deterministic gates;
10. harden Phase C operational shell only after replay/shadow evidence;
11. expand telemetry, checkpoint/replay coverage, and tests;
12. prepare proposer fine-tuning data only after the preceding evidence is credible.

Do not infer architecture from registration, shared output width, or a roadmap
family label. No design decision is implemented by this documentation update.

---

# 21. NON-GOALS FOR THIS IMPLEMENTATION PASS

Do not prioritize in this pass:
- full MoE solver,
- replacing PPO solver with GRPO,
- direct live broker trading without replay/shadow validation,
- unconstrained autonomous WatsonOW execution,
- premature optimization of all family types before family `00`/`01`/`02` are stable.

---

# 22. CODEX EXECUTION STYLE

Codex should:
- inspect the real repo structure first,
- preserve known-good flows,
- implement the smallest coherent vertical slices,
- keep diffs modular,
- add tests alongside interfaces,
- avoid speculative refactors unrelated to the new architecture,
- document migration points if existing callers break.

When tradeoffs appear, prefer:
1. preserving working Phase A
2. making contracts explicit
3. enabling replay/shadow validation
4. enabling future proposer training

---

# 23. DEFINITION OF DONE FOR THIS SPEC

This implementation standard is considered established when the repo can do all of the following:

1. still run current family `00` training reliably
2. define and validate new contracts (`ExecutionIntent`, `CandidateTask`, `CapabilityProfile`, `DecisionRecord`)
3. route at least families `00`, `01`, and `02`
4. run a multi-head solver path
5. run a true offline proposer->solver loop beyond simple bootstrap
6. replay a historical rolling window through:
   MarketFramer -> solver scoring -> DecisionGate -> RiskEngine -> Executor
7. emit WAIT / NO_ACTION explicitly
8. log enough structured data for future proposer SFT/QLoRA training

End of spec.
