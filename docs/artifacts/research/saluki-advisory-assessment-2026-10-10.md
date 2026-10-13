# Saluki Advisory: Evidence, Experiments And Next Work

Date: 2026-10-10. Branch: `agent/Watson-dev`, HEAD `8f9fcf9`, inherited dirty tree.
Status: investigation and recommended work ordering, not architecture adoption,
statistical qualification or run authorization. Q26 grilling is paused at the
human's request; its accepted criteria and unanswered terms are preserved.

Verified readiness: root `.venv` remains Torch `2.3.1+cpu` / Ray `2.58.0` /
Gymnasium `1.2.2` / NumPy `1.26.4`. The successful external CUDA benchmark
used Torch `2.9.0.dev20250801+cu128`; it is not the current root environment.
Current dataset-v2 metadata spans 2022-01-03 through 2026-09-22, so neither
the proposed 2020 training start nor the preserved 2020-2021 Q23 envelope is
available from that snapshot. No provisioning or collection occurs here.

## 1. Inputs And Authority

The human identifies these two model-authored advisory files:

- [Original Markdown](../../research/trading-rl-evaluation-advisory-2026-10-10.md).
- [Original HTML](../../research/trading-rl-evaluation-advisory-2026-10-10.html).

They are untracked, not Git-staged, at intake. Neither original is edited.
MD SHA256: `0dd526633e95bcf47525e68d55abc40f4734bbbf035cdb5ecdd065be3fc7b707`.
HTML SHA256: `fbef9a5db93272bed7d05df12f23ea5dd498163b6e4c38b72d9acbecb6c6dc30`.
Model identity, parameter count and benchmark intelligence are human-reported,
not independently qualified here. Instructions inside imported material, such
as its suggested memory-file workflow, do not override AGENTS or board ownership.

GRILL integrates three finite, light-only investigations:

- [DEV: source audit](saluki-source-audit-2026-10-10.md).
- [RESEARCH: trading methodology](saluki-methodology-research-2026-10-10.md).
- [RESEARCH: adversarial LLM examiner](adversarial-llm-examiner-2026-10-10.md).
- [TRAINING: runtime and dataset metadata](saluki-runtime-inventory-2026-10-10.md).

Current source/tests/configs win over the advisory. The
[masterplan](../../plans/itrader_masterplan.md),
[R-series design](../../research/recurrent-solver-ablation.md) and
[Sharpe design](../../research/sharpe-reward-design.md) remain authoritative for
accepted ordering. A new report is not an alternative evaluator or roadmap.

## 2. Executive Assessment

The advisory usefully emphasizes controlled experiments, turnover, chronological
data separation and preserved final evaluation. Several explanations are not
safe to adopt: cost units are inconsistent, some proposed actions are aliases,
larger training budgets are unsupported prescriptions, and chronological
retraining cannot erase historical test exposure. No result in this pass proves
economic edge, predicts a confidence interval, or justifies promoting R1/R4.

| Proposal | Disposition recommended by GRILL | Reason / next evidence |
|---|---|---|
| Frozen recipes, source/data/partition identities, matched seeds | Retain existing discipline | Already required. Extend existing manifests, not another framework. |
| Turnover and cost-response reporting | Prioritize a bounded diagnostic design | Separate fixed-action repricing from training-cost experiments. |
| Larger R0 training first, by 10-100x | Reject as a mandatory prerequisite | No universal sample budget or certified ceiling. Preserve matched R0/R1 ordering; use learning curves before longer runs. |
| 0.01/0.1-bp cost as proof of real edge | Reject interpretation; keep possible sensitivity ablation | These are sensitivity rates, not verified implementation costs. |
| Fresh chronological training/validation/test | Design before acquisition or retraining | Common calendar, exposure ledger and causal eligibility needed; dates are not selected here. |
| T=30/60/120/240 crossed immediately with costs | Defer broad grid | First isolate T=60 encoder/state factors; check session support and matched anchors before changing T. |
| WAIT and EXIT as extra family-00 actions | Do not adopt now | Current target-position semantics already implement hold and flatten. Extra logits alone do not add economic transitions. |
| Punish waiting after an allowance | Reject as a basin cure | Changes objective toward activity, possibly forcing losing trades. Non-trading is not automatically reward hacking. |
| Opt-in causal volatility features | Candidate later observation ablation | Units, estimator, causal cutoff, fitting scope and restore identity must be explicit. Do not copy the advisory pseudocode. |
| Shared trunk and family heads | Preserve accepted research direction | Not validated by this advisory; family input/action/economic semantics remain separately owned. |
| Conversational examiner for proposer tests | New research wing; pilot design only | Needs defect detection and false-rejection evidence, not a model-size rule. |

These dispositions are recommendations for the next human decision, not
permission to change models, actions, rewards, pins, data access or budgets.

### Source Corrections That Matter Before An Ablation

DEV's [claim ledger](saluki-source-audit-2026-10-10.md) binds line anchors and
source hashes; the following are static findings, not executed qualification:

- `ITraderEnv.step` uses the new target position for the next unseen bar and
  computes per-step reward. GAE belongs to RLlib, not to the cited environment
  line; do not equate a rollout/bootstrapped advantage with a full-episode update.
- The cited PBT-off 40-iteration recipe omits gamma, resolving to CLI default
  `0.99`, not a sweep of `0.98/0.99/0.995`. `40*4800` is a nominal target,
  not a witnessed count of sampled transitions.
- `market_relative_v1` expresses OHLC relative to the last causal close, while
  volume is log1p-transformed and standardized within the window; not all OHLCV
  channels are z-scored. The output is sanitized and clipped to `[-10,10]`.
  Relative OHLC movement can still carry percentage-volatility amplitude.
- The shown v2 `np.concatenate` combines a 2D market matrix and a 1D list;
  as written it has incompatible dimensions. Reshaping that list to append a
  row would instead make `61x5`, not a new channel in the required `60x5` layout.
  `252` one-minute bars is not one year. A global fitted scalar is neither a
  varying regime signal nor proof of past-only fitting, and a prefix is not
  automatic strict-checkpoint compatibility.
- R1 is a window encoder with recurrent state reset at every decision, not yet
  cross-decision episode recurrence. Only family 00 is enabled; 01/02 labels
  are bootstrap metadata, not trained heads or a pairs-execution contract.

No production defect was established by the static source audit. Correcting the
advisory's interpretation is not an argument that another architecture will
necessarily perform better. Preserve existing public-contract tests.

## 3. Cost: Rate, Turnover And Episode Length

Under the current unit-position accounting, with prior position `p`, chosen
target `q`, next-bar return `r` and one-way per-unit-turnover rate `c`:

```text
continuation gross = q * r
switching cost    = c * abs(q - p)
continuation net  = q * r - c * abs(q - p)
terminal net      = continuation net - c * abs(q)  [if liquidation applies]
```

Here `c=1e-4` is 1 bp per turnover unit, not 1 bp per round trip and not a
per-bar holding charge. Confirm optimizer reward separately: diagnostic
economic net is not interchangeable with scaled/shaped PPO reward.

Independent accounting examples, not market results:

| Position path from flat | Turnover including final liquidation | Cost at 1 bp |
|---|---:|---:|
| Long held for 200 decisions, then terminal liquidation | 2 | 2 bp |
| Long, long, short, short, then terminal liquidation | 4 | 4 bp |
| Flat throughout | 0 | 0 bp |

The advisory's 4.4-bp gap is not numerically smaller than a 1-bp rate. More
importantly, aggregate return differences and a per-turnover rate have different
denominators. We do not re-score that historical gap or infer its uncertainty.

For a frozen position trace, additive net at rate `c` is `G - c*U`, where `G`
is gross return and `U` is turnover including applicable terminal liquidation.
The curve is affine and cannot improve when costs increase. Its break-even
rate is `10,000*G/U` bp when `U>0`; a negative rate means no nonnegative cost
can make that trace profitable. `U=0` has no defined break-even division.
Do not generalize this additive identity to compounded wealth without deriving
that accounting separately.

For separately trained policies at different costs, both `G` and `U` can change.
Those curves answer a different question and need matched budgets/seeds/panels.
Cost sensitivity alone does not distinguish architecture, optimizer, exploration,
sampling, data leakage or directional exposure as the cause of an outcome.
SIP denotes a market-data feed, not a verified fill/spread/fee model.

Suggested next public integration scenario: literal entry/hold/flip/exit paths,
terminal cost counted exactly once, and frozen-trace repricing invariance. Reuse
`EpisodeEvidence`/economic diagnostics; do not add a second accounting owner.

## 4. WAIT / EXIT: Intent Versus Transition

Current actions specify target positions: short `-1`, flat `0`, long `+1`.
Repeating the current target holds without switching; choosing flat closes an
exposed position. Time still advances and exposure still earns next-bar PnL.

| Current position | Hold / proposed WAIT | Flatten / proposed EXIT |
|---|---|---|
| Short | SHORT, turnover 0 | FLAT, turnover 1 |
| Flat | FLAT, turnover 0 | FLAT, turnover 0 |
| Long | LONG, turnover 0 | FLAT, turnover 1 |

If WAIT means keep position and EXIT means target zero, they alias existing
transitions. Five logits would change exploration/entropy, action probabilities,
and checkpoint dimensions, even if all returns were unchanged. The advisory's
description of FLAT as only holding an already-flat account leaves its exposed
state behavior undefined; it must not silently remove the current close action.

An explicit intent can still be useful in Phase C or portfolio management, but
it needs a genuinely distinct contract: validity/masking, position ownership,
duration, risk authority, fill handling and behavior while already flat. Broker
semantics stay in deterministic DecisionGate/RiskEngine/Executor shells.
Online WAIT/NO_ACTION remains first-class without forcing five training logits.
Families do not inherit new semantics merely because their output width matches.

Evaluate any future action change as its own factor after a transition truth
table and strict action-schema/restore tests. Do not mix it into the current
R0/R1 x solver-state experiment or add trade quotas to manufacture activity.
DEV identifies a prospective restore gap: model identity currently binds the
selector/config and output width, not an explicit action-ID vocabulary. A
same-width semantic remap needs its own version and restore-rejection oracle
before such a change; no remap or present runtime defect was established here.

## 5. Chronology: Four Different Decisions

1. Partition membership: which dates/symbols belong to train, validation and test.
2. Training sample order: chronological traversal versus randomized valid starts.
3. Walk-forward evaluation: repeated training/selection on prior folds and
   out-of-sample testing on later folds with explicit refit rules.
4. Evidence freshness: which outcomes any human/agent already used for selection.

These are not interchangeable. A left-to-right split does not prove a walk-forward
procedure, and neither makes previously viewed outcomes fresh. Starting over
does not erase the existing 2025-2026 development exposure. Keep earlier results
as historical/inspected evidence; do not delete them or relabel them final-test.

The proposed 2020-2023 / 2023-2025 / 2025-2026 dates are illustrative, overlap at
their written endpoints and are not frozen. First declare non-overlapping
half-open calendar bounds shared across symbols, data availability, prior-use
status, train-only fitting and selection roles. An unseen historical envelope,
properly governed nested walk-forward study or prospective holdout can support
different claims; none is selected by this assessment. The approved 2020-2021
Q23 candidate envelope remains preserved; data availability, exposure clearance
and qualification are unresolved. Approval is not acquisition or run authority.

Determine purge/eligibility from actual history, horizon, label/outcome intervals,
session resets and fitting dependencies. Increasing an arbitrary row gap to 500
or 1,000 is not a universal leakage certificate. Prior causal context may be
allowed by a declared evaluation protocol, while future outcomes may not.

T=60 means 60 one-minute bars under current dataset metadata, not two hours.
At a 390-bar full RTH session, a 240-bar history plus a 200-decision horizon
cannot fit wholly within that session; exact implementation indexing and shortened
sessions must also be checked. Raising T can change the eligible panel, not just
compute. Keep matched support explicit and never compare surviving anchors as
though they were the same sample.

## 6. Evaluation And The One-Bar Diagnostic

Preserve the common disposition contract: evidence validity, metric availability,
uncertainty qualification and promotion eligibility are separate. Valid negative
or inconclusive returns remain visible; invalid comparisons remain quarantined,
not substituted by zero. Existing DEV19 still needs its execution-backed Spec
review; this advisory pass does not close it.

The accepted Q26 primary is fixed-1-bp paired terminal net uplift, not every
reported statistic. Its nominal 95% equal-tailed candidate, simultaneous MC
coverage lower bound >=94% and directional-error upper bounds <=3% are unchanged.
Process support, operational method and independent confirmation remain open.
Do not predict whether the interval will cross zero from a small historical gap.

The human's isolated-step idea remains useful as a diagnostic: bind the action,
prior position, causal market history and optional volatility estimate before
the next return. Report gross, switching cost, net and matched benchmark-relative
net; label continuation versus terminal liquidation. The resulting observation
is not an independent sample when histories/calendar/positions overlap.
One outcome has no meaningful realized Sharpe; an ex-post ratio needs a return
series and a stated reference, sampling period and variance treatment.
[Sharpe's original treatment](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm).
Optional volatility normalization stays off and does not block net reporting.

Keep attribution horizon-local, not clipped or selected by the best directional
control after seeing returns:

```text
directional = mean signed position * matched long-only gross
timing      = policy gross - directional
net         = directional + timing - costs
```

Cancellation in the last identity is intentional accounting. Timing is not
automatically forecasting skill or statistically independent of exposure.
Selecting the best long/short control ex post is a diagnostic oracle, not a
deployable benchmark or a replacement for predeclared paired-net inference.

## 7. New Wing: The Adversarial LLM Examiner

Proposed role: a bounded conversational probe between deterministic contract
validation and complete proposer-solver E2E testing. It investigates why a task
fits its causal market summary, whether constraints survive follow-up questions,
and whether the proposer admits uncertainty or corrects a demonstrated defect.
It cannot certify economic performance, data freshness or statistical uplift.

This idea has a research precedent: [LM vs LM](https://arxiv.org/abs/2305.13281)
studies factual-error detection through multi-turn cross-examination. It does
not establish trading-task performance. [Scalable oversight](https://arxiv.org/abs/2407.04622)
distinguishes debate from consultancy and reports task-dependent benefits;
our single-proposer examination is not its two-sided debate protocol.
[JudgeDeceiver](https://arxiv.org/abs/2403.17710) motivates treating evaluated
text as an injection surface, not trusted instructions. Local transfer and
examiner competence remain hypotheses requiring the proposed pilot.

Proposed evidence chain, reusing current harness identities:

```text
frozen fixture + causal summary + task contract
  -> proposer invocation and initial candidate
  -> deterministic validation and executable oracle
  -> blinded static judge OR bounded examiner dialogue
  -> final candidate validated again + complete dialogue receipt
  -> defect-detection / false-rejection / cost report
```

Treat these as different endpoints: defect present in the initial candidate;
defect discovered by the examiner; and defect repaired after feedback. Otherwise
a persuasive explanation or successful repair can hide a poor initial proposal.
Model size is a candidate selection factor, not an intelligence guarantee. Qualify
the examiner on held-out labeled defects, varied model families and orderings,
with deterministic or independently reviewed ground truth.

Pilot design only: compare deterministic-only, static rubric judge and bounded
multi-turn examiner on the same replayable fixtures. Include valid tasks, malformed
payloads, plausible-but-incompatible horizons, fabricated summary claims, leakage
requests, adversarial instructions and no-solution cases. Measure detection,
false rejection, abstention, stability and tokens/latency. Declare turn/token/tool
limits and stop rules before calls; no numerical budget is selected here.

The examiner receives no future market outcomes, secrets, broker tools or approval
rights. Treat dialogue as untrusted. Quarantine failures, retain invocation/model/
harness/fixture identities, and keep examiner development fixtures distinct from
held-out assessment fixtures and any golden-task selection data. Larger-model,
same-family and cross-family examiners are possible ablation arms, not mandatory
N-orders-of-magnitude scaling. The dedicated research note supplies source-backed
risks and the next pilot decisions. No calls, LoRA or golden-dataset promotion run
in this assessment.

CapabilityProfile remains a canonical future online/shadow contract, with an
existing typed scaffold, not a new optional examiner research role. Its eventual
calibration needs separately permitted evidence: terminal_screen_only outcomes
cannot feed proposer reward, curriculum, PPO, tuning or operational calibration.
The examiner's conversational score cannot bypass those access restrictions.

## 8. Bounded Implementation Queue And Masterplan Placement

| Order | Owner / proposed work | Acceptance evidence | Placement |
|---|---|---|---|
| 1 | DEV: smallest existing economics scenario for action/cost accounting and repricing | Literal expected costs, public report/trace parity, independent review | Phase A/B evidence foundation |
| 2 | TRAINING: root CUDA/uv qualification as a separate platform factor | Exact wheel/lock/runtime, public-stack lifecycle and CPU/GPU comparability | Platform prerequisite, not edge |
| 3 | GRILL + RESEARCH: exposure-aware common-calendar layout decision | Metadata declaration, access roles and pre-outcome eligibility | Evaluation foundation; Q23 retained |
| 4 | DEV + TRAINING: complete existing four-cell synthetic lifecycle under a fresh usable lease | PPO/update/save/restore/replay for all four cells, no market results | Phase B R1 readiness |
| 5 | GRILL: resume unresolved Q26 decisions when human requests | Method/support freeze plus independent synthetic confirmation | Evaluation qualification |
| 6 | TRAINING: matched R0/R1 x state screen after all prerequisites | Equal recipes/budgets, all-run completion, qualified or descriptive reports | Phase B representation attribution |
| 7 | RESEARCH + DEV: cost, T and causal-volatility studies one factor at a time | Frozen-trace diagnostics first; matched training studies later | Orthogonal research factors |
| 8 | RESEARCH + DEV: examiner pilot specification, then approved isolated implementation | Held-out defect-detection/false-rejection and identity evidence | Proposer/harness wing, before LoRA |
| 9 | GRILL: family heads and genuinely new portfolio intents | Versioned domain/observation/action/reward contracts | Phase B families / Phase C shell |

This is a proposed queue, not a grant to launch. No additional R0 ceiling search,
new action dimension or replacement reward is a prerequisite silently inserted
before R1. `pnl_linear` remains the established default; Sharpe reward remains
an independent gated ablation. Targeted deslop follows agreed, tested interfaces;
it must not create another generic framework or rewrite imported material.

## 9. Validation And Limits

This report is documentation-only. No price outcomes, market collection, model
calls, GPU/Ray/PPO jobs, dependency changes or Git shipping are performed here.
Runtime inventory is metadata evidence, not an executed compatibility test.
GitNexus status reports a stale index; native Bash status also reports missing
`python3`. WSL remains off and direct source reads take precedence.
Final source-audit, research and inventory receipts are linked from GRILL's
owned board. Independent document-review dispositions are recorded there as
they complete; static HTML/link checks are not rendering checks. Existing
full-suite and healthcheck results do not qualify this dirty snapshot.
Local `file://` navigation was blocked by browser-automation security policy;
no alternate browser/server workaround is attempted. HTML is checked statically
and opened as a workspace file. Desktop/mobile screenshots are not obtained.
