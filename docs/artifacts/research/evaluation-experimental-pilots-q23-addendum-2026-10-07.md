# Experimental evaluation pilots and Q23 route: DEV addendum

Date: 2026-10-07

Branch: `agent/Watson-dev`

Status: Q27 mechanics and the bounded diagnostic-pilot shell are implemented,
independently approved, and closed with the repository validation recorded in
the [Q27 report](evaluation-mechanical-repairs-2026-10-07.md). The separate-data
route, first-class terminal-screen-only eligible intervals, and separate
training/terminal dataset identities are approved but their owning Q23 seam is
still pending implementation/review. Metadata-first envelope planning is now
approved; an exact 2020-01-01 through 2021-12-31 request is recommended for the
next declaration decision, not authorized for collection. The volatility
estimator, numerical guard, Q26 method/numbers, and autoresearch objective remain
unapproved. No market outcome was opened, no bar data was acquired, and no
terminal run was authorized.

## Decision checkpoint

This addendum records the human decisions made after the
[evaluation methodology report](evaluation-quant-methodology-2026-10-06.md).
The direct human decision makes that report's **Q27-pending and
Q23-route/role-pending labels obsolete**; its derivations, limits,
primary-source evidence, Q24/Q25 authority, and unresolved Q26 choices remain in
force. This dated artifact records the coordination delta but does not supersede
active architecture/masterplan files. DEV has reconciled the active
[masterplan](../../plans/itrader_masterplan.md) and
[R1 readiness contract](../../project-state/r1-development-readiness.md) with
the completed Q27 evidence. Phase ordering is unchanged.

| Item | Decision on 2026-10-07 | Boundary |
|---|---|---|
| Q27 mechanics | Implemented; independent Sol and Luna reviews approve; repository validation is closed | Preserve the approved scope and evidence; this does not authorize a terminal panel |
| Episode attribution | Approved as a secondary experimental report | Not reward, a headline metric, or training feedback |
| One-step probe | Approved as a bounded diagnostic pilot | The specific volatility estimator and numerical guard below are recommendations for GRILL, not adopted defaults |
| Q23 route and role | Approved: a separately declared terminal-screen-only dataset, first-class certified eligible intervals, and separate training/terminal identities | Owning-seam code/review pending; complete exclusion from training, proposer/curriculum, tuning, calibration, checkpoint selection, and the old sealed final test |
| Metadata-first envelope | Planning sequence approved; exact 2020-01-01 through 2021-12-31 request recommended for declaration review | Retrospective clearance pending; no bars acquired and no outcome opened |
| Q26 | Synthetic-null calibration before outcomes is approved; method and all numerical choices remain pending | Null Type-I/coverage/stability is distinct from separately named simulated-effect power; no grouping, method, block length, interval, multiplicity, threshold, or seed interpretation is adopted |
| `CapabilityProfile` | Remains a future online/shadow contract | Offline work may produce transferable evidence, not an operational profile |
| Autoresearch | Staged only after the contracts and gates below are implemented and reviewed | No optimization loop is authorized now |

The four completed Q27 repairs are: reject remapped explicit starts; reject
ambiguous terminal selection; make physical-EOF/reset/evidence identities agree;
and require an explicit state-tail schema before state intervention. The dated
[code audit](evaluation-code-audit-2026-10-06.md) remains the historical
finding owner, while the [Q27 report](evaluation-mechanical-repairs-2026-10-07.md)
owns implementation, review, and validation evidence.

The dependency order is also explicit. The optional volatility-normalized pilot
does **not** block the headline Q24 terminal-net screen when disabled. Q27
mechanics and the Q23 qualified-metadata gate are required for reliable terminal
execution. Q26 design can proceed in parallel, but its final support/calibration
decision depends on the selected Q23 calendar layout. No real panel is open.

## Architecture boundary

The pilots are evaluation modules downstream of a frozen policy decision. They
must not change `TaskSpec`, observations, `RewardEngine`, PPO advantages, solver
actions, proposer rewards, curriculum selection, terminal checkpoint selection,
or Phase C broker/risk semantics.

```text
frozen offline episode/decision trace
              |
              +--> versioned experimental diagnostic sidecars
              |       - episode direction/timing/cost attribution
              |       - causal one-step action diagnostic
              |
              +--> existing Q24/Q25 reducers (unchanged)

qualified offline evidence
              |
              +--> future, separately versioned reducer/calibrator
                          |
                          v
                 online/shadow CapabilityProfile
```

`CapabilityProfile` therefore stays on the future online path described by the
[system architecture](../../architecture/itrader_system_architecture.md). An
offline evidence record may use compatible identities so it can be transferred
later, but it is not itself a profile, must not populate an uncalibrated
`estimated_p_solve`, and must not enter `r_prop`. Any later Phase B
capability-to-curriculum adapter needs separate approval, versioning, a declared
adaptive-validation source, and a hard exclusion of terminal-screen and final-
test evidence.

## Pilot A: episode direction, timing, and cost

### Exact estimand

For the exact `H` decision steps already used by `EpisodeEvidence`, let `p_k` be
the signed target position applied for step `k`, and let `r_k` be that step's
recorded simple close-to-close asset return. Do not reconstruct either series
from a different index convention.

```text
G            = sum_(k=0..H-1) p_k * r_k
p_bar        = (1 / H) * sum_(k=0..H-1) p_k
L            = sum_(k=0..H-1) r_k
directional  = p_bar * L
timing       = G - directional
C_actual     = c * (sum_(k=0..H-1) |p_k - p_(k-1)| + |p_(H-1)|)
net          = directional + timing - C_actual
```

Here `p_(-1)` is the episode's recorded starting position (zero in the current
reset contract) and `c = evidence_cost_bps * 1e-4`. `C_actual` includes the one
real terminal liquidation already included by
[`EpisodeEvidence`](../../../itrader/rl_lib/evaluation/episode_evidence.py); it
must not be added twice.

This is an **additive simple-return identity**, matching the current
[`ITraderEnv`](../../../itrader/envs/itrader_env.py) evidence arithmetic. It is
not a compounded wealth decomposition. `timing` is the observed-horizon
residual relative to constant signed exposure `p_bar`. It is not clipped and is
not automatically alpha, forecasting skill, causality, or a market-timing
estimate.

The identity cancels exactly: `directional + timing = G`, so attribution cannot
change net. For example, if `G = 30 bps`, `directional = 20 bps`,
`timing = 10 bps`, and cost is `4 bps`, net remains `26 bps`. `L` is the
unselected always-long gross return on the same bars; it is not a hindsight
"best direction," and neither `L` nor `directional` is clamped. Signed negative
exposure works unchanged. A constant long or short path has zero timing residual
because `G = p_bar * L`; flat from zero does too. None of this is a capability,
alpha, or causal-skill claim.

Compute the fields per complete episode, then apply the unchanged Q24 order:
four horizons within anchor, anchors within symbol, and twelve symbols equally.
Q25 continues to normalize each episode before aggregation. A pooled step mean
or the raw 128-task mean may be secondary only; neither replaces Q24.

### Required reconciliation tests

- Reconcile `G` exactly with stored gross return and reconcile both
  `net = G - C_actual` and `net = directional + timing - C_actual` within one
  declared numerical tolerance.
- Cover every prior/target position transition and charge exactly one actual
  terminal liquidation.
- Make constant long and constant short have zero timing residual; make an
  all-flat, zero-start episode have zero gross, direction, timing, and cost.
- Include a path where additive and compounded calculations differ, and assert
  that this pilot follows the additive result.
- Reject missing, duplicated, censored, mismatched-step, non-finite, or
  wrong-cost traces. Never repair them by clipping or partial aggregation.
- Prove row-order invariance and preserve the accepted Q24/Q25 reducers.

## Pilot B: causal one-step realized-return diagnostic

### Freeze first, score second

For a decision at source row `i`, persist the decision identity, policy action,
prior position, and the 60 visible close levels from rows `i-60` through `i-1`
before row `i` is exposed. Those levels produce exactly 59 trailing simple
returns. The subsequently attached outcome is the return from row `i-1` to row
`i`; it is not part of the scale window.

The following scale is a **recommended candidate for the next GRILL, not an
adopted estimator**. It is deliberately narrow and versioned:

```text
scale_id = lagged_close_std_60c_59r_ddof1_v0
u_j      = (C_j - C_(j-1)) / max(C_(j-1), 1e-8), j=i-59,...,i-1
sigma_i  = sqrt(sum_j (u_j - mean(u))^2 / 58)
```

Its unit is dimensionless one-minute return. If the candidate is approved, use
the same `sigma_i` for the policy and every reference action at that decision.
The proposed invalid-scale rule is: if the scale is non-finite or
`sigma_i <= 1e-12`, keep all raw fields, set normalized fields explicitly
unavailable, record the reason, and increment coverage counters. The estimator
and cutoff both require human approval. The cutoff is a proposed numerical-
degeneracy guard aligned with current evidence code, not a significance
threshold or a literature-derived permanent choice.

Only after the frozen record is durable may the next simple asset return
`r_next` be attached. For prior position `q`, candidate target position
`a in {-1, 0, +1}`, and cost rate `c`:

```text
gross(a)           = a * r_next
switch_cost(a)     = c * |a - q|
continuation_net(a)= gross(a) - switch_cost(a)
hypothetical_roundtrip_net_return(a) = continuation_net(a) - c * |a|
volatility_scaled_continuation_net_return(a)
                       = continuation_net(a) / sigma_i  # only when valid
```

`continuation_net` is the primary one-step diagnostic.
`hypothetical_roundtrip_net_return` is a separately named hypothetical forced
close. Neither may alter the episode's actual terminal liquidation or
`EpisodeEvidence`.

At every eligible decision, score the policy action and the named references
`hold_prior`, `flat`, `long`, and `short` with the same `q`, outcome, cost, and
scale. Preserve semantic labels even when two labels resolve to the same
action. A random draw adds no action coverage because all three actions are
already enumerated; the seeded-random terminal control remains unchanged.
`max_a continuation_net(a)` may be stored only as `oracle_diagnostic_only` and
must never be described as attainable policy PnL.

No output from this pilot may be named Sharpe. A single outcome does not supply
a return series or dispersion estimate; the primary-source rationale remains
in the [capability diagnostics evidence](evaluation-quant-methodology-2026-10-06-evidence/capability-diagnostics-primary-sources.md)
and [Sharpe's differential-return definition](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm).

### Candidate pilot output and tests

The opt-in sidecar should retain raw and normalized action vectors, scale value
and validity, selected action, references, cost convention, observation/trace
identity, and an explicit diagnostic-only/oracle marker. If summarized, first
form an episode-level result over common valid steps and record valid count and
coverage; only then may the existing Q24 hierarchy be applied. Do not pool all
steps into a new headline.

Minimum tests:

- 60 close levels produce 59 returns and sample standard deviation uses
  denominator 58 (`ddof=1`), verified against a hand calculation.
- Perturbing any future close cannot change the frozen action, prior position,
  trailing returns, or scale; it may change only the attached outcome fields.
- Exhaust all nine prior-position/target-action transition costs.
- Policy and references share one outcome and one scale at each decision.
- Zero, NaN, and infinite scales make normalized values unavailable while raw
  values survive; no epsilon substitution is allowed.
- Actual episode liquidation and hypothetical roundtrip liquidation remain
  separately identifiable and cannot be double-counted.
- Default-off execution produces no sidecar and no training/evaluation behavior
  change. Static checks should prevent any diagnostic field from entering
  rewards, observations, `TaskSpec`, proposer feedback, or checkpoint choice.

## Recommended module seams for DEV

These are interface recommendations, not adopted class names or implementation
authority. DEV may preserve the approved default-off pilot shell while the
estimator and reporting subdecisions remain unconfigured.

1. A small frozen-decision trace interface owns causal serialization and outcome
   attachment. Its invariant is that decision-side fields exist before an
   outcome can be attached.
2. A pure episode-attribution evaluator owns the additive identities,
   validation, and reconciliation. Callers should not duplicate formulas.
3. A pure one-step evaluator owns the lagged-scale version, all three target
   actions, reference labels, invalid-scale policy, and oracle marking.
4. An opt-in diagnostic reporter owns sidecar persistence and later aggregation.
   Prefer this to widening `EpisodeEvidence` before the pilot earns stability.

Each output needs a schema version and joins to dataset, source, config,
checkpoint, task, symbol/session/anchor, horizon, cost, policy/control, seed,
producer, and method identities. The modules should expose compact validated
records and hide their arithmetic and failure policy. This keeps the evaluation
seam deep without leaking experimental fields into the solver or online shell.

### Shared-tree implementation closeout

DEV's shared tree now contains
[`economic_diagnostics.py`](../../../itrader/rl_lib/evaluation/economic_diagnostics.py),
a default-false `capture_step_economics` flag, and an opt-in pre-action context
in `ITraderEnv`. Direct inspection confirms that the current draft freezes a
past-close tuple before `step`, keeps continuation, hypothetical roundtrip, and
actual terminal liquidation distinct, reconciles episode gross/cost/net before
attribution, emits common-scale `hold_prior`/`flat`/`long`/`short` references,
retains both gross and continuation-net normalization fields, and chooses no
default volatility estimator. Q27 and the default-off pilot shell received
independent approval and passed the final core suite, healthcheck, scoped
pre-commit rerun, and diff checks recorded in the Q27 report.

That is a sound foundation, not completion of the candidate contract. At this
checkpoint the evaluator accepts any finite positive supplied scale and does
not implement the named 60-close/59-return `ddof=1` candidate or the
`<= 1e-12` guard, which is correct until they receive human approval. Secondary
Q24 reduction and durable sidecar receipts also remain outstanding. The pilot's
current reconciliation tolerance is looser than the terminal reporter's, so it
must be aligned or explicitly versioned and justified before Q24 integration.
Those approved-pilot deltas and candidate subdecisions should be resolved
without changing the environment's reward or evidence arithmetic.

## Q23: separate terminal-screen-only dataset

### Chosen route and invariants

Historical recovery remains evidence: 512 recorded control rows reduce to 48
known starts (32 unpanelled, eight September, eight January), and zero candidate
anchors were certified fresh under the conservative 2,052-anchor exposure
bound. Those facts motivate the separate-data route; they do not select dates
or certify a new panel.

The new snapshot must have its own immutable dataset identity and the approved
first-class `terminal_screen_only` role with certified eligible intervals and
separate training/terminal dataset identities. Code and independent review are
pending. The snapshot is not appended to
dataset v2, passed through a new 70/15/15 split, used for training,
proposer/curriculum generation, tuning,
calibration, checkpoint selection, or treated as a replacement final test. The
existing training/validation source stays unchanged, and the existing final
test stays sealed.

Reuse the existing
[`HistoricalMarketDataSource`](../../../itrader/inputs/market_data/contracts.py),
[`AlpacaMarketDataSource`](../../../itrader/inputs/market_data/alpaca.py), and
immutable [snapshot builder](../../../itrader/inputs/market_data/snapshot.py).
They already provide a read-only acquisition adapter, split-adjusted one-minute
SIP requests, raw page/request retention, retrieval timestamps, UTC/session
normalization, duplicate rejection, stable ordering, hashes, and no forward
fill. Do not create a second provider backend for this route.

The data layer is reusable, but the current evaluation path cannot yet express
this route. `run_terminal_screen` requires the terminal dataset hash to equal
the completed training screen's dataset hash; `panel_qualification` requires
the candidate hash to equal the retained exposure inventory's hash and assumes
that inventory is non-empty. The qualifier also assumes
`data_split=validation`, 0.7/0.15/0.15, and a purge boundary.
`terminal_panel_report` also rejects episode rows not labeled `validation`;
`DataPartition`/`ITraderEnv` accept only the existing split vocabulary,
`EpisodeEvidence` copies that split, and the experimental diagnostic runner
currently admits only train/validation.

The approved role therefore needs an end-to-end bridge, not just a preflight
flag. Keep `terminal_screen_only` as a first-class role bound to the terminal
dataset identity and certified-interval hash; do not alias it to `validation`.
Certified intervals replace fractional splitting for eligibility, while the
existing `DataPartition` continuity/session checks remain reusable. Environment
reset/evidence, diagnostic collection, qualification, preflight, and terminal
reporting must all preserve and validate the role plus certificate. DEV has
been dispatched for scoped TDD and independent review. Every consumer must fail
closed until that seam lands; relabeling the new snapshot as `validation` would
misstate provenance.

### Metadata-only feasibility gate

Before any return, regime, solver action, or model outcome is inspected, bind
and review:

- provider/feed, adjustment, timeframe, timezone, exchange-calendar identity,
  session policy, missing-bar/no-forward-fill policy, retrieval time, raw-page
  hashes, manifest hash, and usage classification;
- the same twelve symbols, unless a separately approved scientific question
  explicitly changes the universe;
- enough continuous same-session support for a 60-bar past window and each
  96/128/160/200 future horizon at the declared 60/120-minute decision offsets;
- enough metadata support for 32 distinct symbol/session anchors, at least two
  per symbol, before crossing the four horizons;
- timestamp/session duplication, OHLC-schema presence, gap, early-close, and
  complete-final-bar checks without reading price values;
- calendar-component counts and breadth needed to evaluate Q26 candidates,
  without choosing a Q26 method; and
- an exposure registry covering training, curriculum/proposer generation,
  tuning, calibration, progress/terminal panels, diagnostics, plots, human
  inspection, and agent/tool access. Any uncertain interval is excluded.

Selection must be outcome-blind. Do not choose a provider, period, date, anchor,
or exclusion by return, volatility, regime, policy action, or solver result.
Date novelty alone is not non-exposure proof. The complete feasibility contract
is in the [Q23 evidence note](evaluation-experimental-pilots-2026-10-07-evidence/q23-evaluation-only-metadata-feasibility.md).

### The next GRILL question before acquisition

Same-stack parity is the recommendation: Alpaca/SIP, split adjustment,
one-minute bars, New York exchange calendar, RTH selection, the same twelve
symbols, raw-page retention, immutable manifests, and no forward fill. A
different provider/feed would be a separately approved acquisition-shift
factor, not a free freshness upgrade.

The metadata-first sequence is approved. The outcome-blind calendar/access
audit now recommends the exact request bounds 2020-01-01 through 2021-12-31;
the first actual exchange session is 2020-01-02, giving 505 sessions and 6,060
symbol/session slots before bar qualification. This choice is the nearest two
complete calendar years before dataset-v2, not a conclusion drawn from returns
or solver behavior. Two years is a reviewable design judgment, not proof that
the final layout can support Q26.

The immediate question is now: **approve
`bounded_retrospective_clearance_v1` and the exact
`q23_terminal_screen_only_20200101_20211231_v0` declaration, subject to a human
manual/external-access statement and every fail-closed stop rule; or require
prospective-only accumulation instead?** The complete standard, declaration,
access receipt, and stop rules are in the
[metadata-first envelope report](evaluation-metadata-first-envelope-2026-10-07.md).
Approval would authorize declaration preparation, not bypass the pending Q23
role/review gate or silently authorize outcome access.

Either answer must separately declare prospective-only clearance or a
retrospective interval cleared against the complete exposure ledger. Do not
presume that "post-January 2026" or merely "after dataset v2" is fresh. A long
historical slice can be eligible only because its retained access history proves
it unused, never because its date looks new.

### Metadata and local-access facts behind the recommendation

The retained request/manifest proves that the established path already acquired
Alpaca/SIP, split-adjusted, one-minute data for the fixed twelve symbols from
2022-01-03 through 2026-09-22. A 2026-09-24 authenticated gate also proved that
old explicit `feed=sip` requests worked for this account at that time. On
2026-10-07 the existing calendar-only adapter path succeeded with locally
configured credentials; no secret, bar, quote, trade, account, market outcome,
or model output was accessed. This proves a current credential/calendar path and
historical old-SIP access, **not** current SIP bar entitlement and not that SIP
is free today. No paid upgrade or provider change is recommended or authorized.

The old snapshot remains useful provenance but predates the current builder's
retrieval-time, usage-classification, and per-raw-file hash ledger. Any new
snapshot must use the current manifest contract. The retained calendar contains
1,185 sessions through 2026-09-23. The old metadata audit found 2,084 physically
eligible anchors and 2,052 after the 32 known explicit exclusions across 176
dates, but **zero certifiably fresh anchors** because the start-unbound random-
run exposure bound covers that whole dataset-v2 candidate domain. The 512 recovered rows
and 48 starts remain evidence; they do not make another old date fresh.

Accordingly, the exact historical request is proposed but retrospective
clearance remains pending. It is disjoint from retained v1/v2 date coverage,
yet retained metadata
cannot prove that deleted copies, external tools, or manual chart inspection
never exposed it. Any unresolved source that could overlap blocks clearance.
Current calendar-only access succeeds, but current SIP bar entitlement remains
unknown until the first post-approval sealed explicit-SIP request; no fallback
feed or paid upgrade is authorized. Prospective-only collection gives the
strongest freshness proof but cannot yet provide comparable calendar breadth.

Because dataset-v2 begins in 2022, the proposed 2020-2021 block is a backward-
time, cross-period retrospective representation screen. It is not prospective
forward-time validation or a live-economic promotion gate. The temporal shift,
all-twelve bar presence, minute continuity, gap/duplicate status, and early-close
eligibility all remain conditional until a declared sealed acquisition and
metadata-only qualification.

## Q26 and future autoresearch

Q26 method and numerical choices remain entirely unresolved: calendar grouping,
resampling method, block length or grid, null calibration/coverage tolerance,
interval construction, multiplicity family, failure policy, and the
interpretation of two training seeds still require outcome-blind analysis and
human approval. The approved research sequence first freezes the outcome-blind
layout, then measures Type-I error, interval coverage, numerical/support
failure, and stability on predeclared synthetic null dependence families.
Separately named simulated-effect power cannot rescue a method that fails null
calibration. Every failed null and method attempt remains recorded. Neither
pilot may open the terminal panel early or expand the confirmatory family.

Only after Q27 review closure, the two pilot contracts pass their remaining
tests, the Q23 snapshot/panel is metadata-sealed, and Q26 is accepted may the
`autoresearch-optimizer` loop be considered. One eligible first objective is:

```text
minimize median wall-clock seconds to produce both validated diagnostic
sidecars on one frozen non-terminal fixture
```

The baseline and challenger must consume identical bytes and emit semantically
identical validated records; the holdout stays frozen, one implementation factor
changes per challenge, and tests plus a fixed memory ceiling are hard gates.
This objective optimizes mechanics only. It may not change data, dates, rewards,
economic thresholds, Q24/Q25, Q26, pilot formulas, or any market outcome.

## DEV and GRILL completion checklist

The Q27 repairs and default-off diagnostic-pilot shell are implemented and
review-closed. The separate-data route, first-class terminal-screen-only
intervals, and dual dataset identities are approved work. The exact historical
envelope is a reviewed recommendation, not collection authority. Estimator,
numerical guard, Q26 method/numbers, and autoresearch decisions remain pending.

- [x] Land and independently review all four human-approved Q27 repairs.
- [ ] Complete the approved pilot integration and versioned sidecars behind the
      implemented default-off shell, using small evaluation-side interfaces;
      leave the estimator unset until GRILL approves one.
- [ ] Pass the reconciliation, edge-case, causality, reference-action, and
      no-training-coupling tests above on synthetic/non-terminal fixtures.
- [ ] Confirm that Q24/Q25 outputs are byte/semantically unchanged when pilots
      are off, and use the same reducer ordering when pilot summaries are on.
- [ ] Implement and independently review the approved first-class
      terminal-screen-only eligible-interval seam with dual dataset identities;
      carry role/certificate identity through env reset, `EpisodeEvidence`,
      diagnostics, qualification, preflight, and reporting; do not reuse the
      ordinary validation split by relabeling.
- [x] Approve metadata-first sequencing before outcome access; preserve the
      source-parity, no-fallback, and synthetic-before-real-outcome boundary.
- [ ] GRILL `bounded_retrospective_clearance_v1` and the exact
      `q23_terminal_screen_only_20200101_20211231_v0` declaration before any bar
      acquisition; otherwise require prospective-only accumulation.
- [ ] Run the Q23 metadata-only feasibility gate and independent review without
      reading returns or model outcomes; acquire no data from this document.
- [ ] Resolve and seal Q26 separately before any terminal-panel access.
- [ ] Keep offline evidence distinct from the future online/shadow
      `CapabilityProfile`; introduce no automatic `r_prop` path.
- [ ] Update active project-state/architecture documentation only when DEV has
      implementation and review evidence.

## Evidence and provenance

- [Pilot contract evidence](evaluation-experimental-pilots-2026-10-07-evidence/attribution-one-step-contract.md)
- [Q23 metadata feasibility](evaluation-experimental-pilots-2026-10-07-evidence/q23-evaluation-only-metadata-feasibility.md)
- [Metadata-first envelope, clearance standard, and draft declaration](evaluation-metadata-first-envelope-2026-10-07.md)
- [Primary evaluation methodology and source ledger](evaluation-quant-methodology-2026-10-06.md)
- [Evaluation code audit and historical recovery](evaluation-code-audit-2026-10-06.md)
- [Dataset-v2 acquisition contract](dataset-v2-r0-audit-2026-09-24.md)
- [Canonical masterplan](../../plans/itrader_masterplan.md)

The GitNexus index was stale and WSL remained off. Current source, tests, active
architecture, and the masterplan remain the repository's durable authority;
the direct human decision is the authority for the dispatched status delta, and
DEV owns its reconciliation into those active surfaces.
