# Q23 clearance and Q26 decision chain

Date: 2026-10-09, Europe/Madrid. Scope: `Q23-Q26-PLAN-01`.
Status: decision proposal; no acquisition, simulation, model run, or statistical
qualification. The human approved the 2020-01-01 through 2021-12-31 candidate
envelope and artifact-backed retrospective-clearance approach. That approval
does not supply the unresolved manual/external-access statement.

## 1. Minimal remaining clearance receipt

Reuse the [Oct7 envelope](evaluation-metadata-first-envelope-2026-10-07.md) and
[exposure inventory](evaluation-metadata-first-envelope-2026-10-07-evidence/retained-identities-and-exposure.md).
The first session is 2020-01-02; the calendar contains 505 sessions, 6,060
requested symbol/session slots, and three early closes. Actual minute coverage
and SIP entitlement remain conditional on a separately authorized acquisition.

| Exposure tier | What remains to record |
|---|---|
| Exact retained files | All 163 dataset-v2 records, including 42 unresolved starts, are date-disjoint; sixteen exact-hash legacy configs are also disjoint. Carry forward those hashes, without another outcome inspection. |
| 120 older path-only runs | Bind historical file identity or a credible provider/date bound using retained receipts, backup manifests, or explicit owner knowledge. Current file dates alone cannot bind historical bytes. |
| Three absent OneDrive-checkout references | Record whether the historical source contained any 2020-2021 values, with provenance or a signed bounded owner statement; otherwise retain unknown exposure. |
| Twelve task pools / six candidate ledgers | Bind the source of their opaque market summaries and task windows, including any external prompts/agent access. Static generation may be attested if actually known; schema capability alone proves neither access nor absence. |
| Manual/external access | Record charts, notebooks, exports, other providers, drives/checkouts, broker tools, calibration, diagnostics, and agent prompts that could expose the candidate interval. Give known windows and unknown scope explicitly. |

One append-only receipt should identify the owner, timestamp, audited roots and
their existing inventory hashes, each tier's disposition (`bounded-disjoint`,
`known-overlap`, or `unknown`), evidence source, semantic symbol/date bound,
timezone uncertainty, and the scope of the owner's knowledge. An attestation is
an accountable bounded statement; hashing it does not prove universal absence.
Any unknown scope that could cover the entire candidate blocks historical
clearance. A narrower bound excludes every intersecting full information/outcome
window, across providers; if no usable layout survives, report that result and
refer a replacement declaration to GRILL. No seed reconstruction, absence of a
year literal, or new snapshot hash resolves these gaps.

Minimal clearance question: **Can the owner bound the four unresolved tiers
above and attest that no remaining known or uncertain access overlaps eligible
2020-2021 windows; otherwise, what exact symbol/date bounds must be excluded?**

## 2. Scientific claim and calendar eligibility

Training dataset-v2 starts in 2022. Comparing its trained solvers on 2020-2021
therefore measures backward-time, cross-period representation robustness.
Later training information, retained symbol selection, and split adjustment
prevent a historical trading-deployment interpretation. This study does not
establish forward-time economic generalization. A later prospective validation
requires its own chronological source and protocol.

Preserve the approved 32 distinct symbol/session anchors, at least two per
symbol, plus eight assigned by the declared metadata-only algorithm/seed; cross
each anchor with horizons 96/128/160/200. The decision offset is 60 or 120
minutes after RTH open, with continuous same-session 60-bar lookback and full
200-bar future support. Exclude semantic `[start-60,start+200)` windows that
intersect adaptive or uncertain exposure, including the approved adjacent-boundary
purge. Keep UTC/session and calendar-close checks explicit.

Distinct symbol/session anchors do not make contemporaneous symbols independent.
The four horizons intentionally overlap; cross-symbol calendar overlap is
allowed and must enter dependence treatment. Same-date all-twelve availability
is a diagnostic, not an extra eligibility gate. Non-overlap means separation
from prohibited exposure; it does not promise independent market observations.

## 3. Recommended first confirmatory endpoint

Recommend one representation contrast under the **fixed 1 bp training arm**:
`R1 + R4-v1` minus `R0 + R4-v1`, using predetermined terminal checkpoints,
deterministic 1 bp evaluation, and the existing Q24 arithmetic terminal net
return including entry/switch/liquidation costs. This holds explicit state and
cost curriculum fixed while comparing the market encoder. It is a proposal;
the sixteen-cell study and its other comparisons remain available descriptively.

For training seed `k`, symbol `s`, eligible anchor `a`, horizon `h`, let
`d[k,s,a,h]` be the paired difference in episode net return. With `n_s` anchors:

```text
theta_k = (1/12) sum_s (1/n_s) sum_a (1/4) sum_h d[k,s,a,h]
theta_fixed_pair = (theta_seed1 + theta_seed2) / 2
```

Recommend `theta_fixed_pair` as the sole confirmatory endpoint, conditional on
these two frozen trained checkpoint pairs and the declared calendar design.
Display both `theta_k` estimates and their calendar uncertainty; opposite signs
must remain visible. The proposed superiority hypothesis is `theta_fixed_pair
<= 0` versus `> 0`. This is neither a claim about the population of training
seeds nor an absolute profitability/promotion gate. Q25 net-bps-per-bar, Sharpe,
horizon slices, clean R1/R0, state/curriculum interactions, all five behavioral
controls, turnover/drawdown, and attribution remain descriptive unless added to
an explicitly expanded confirmatory family before outcomes.

The [Q24 reporter](../../../itrader/rl_lib/evaluation/terminal_panel_report.py)
already uses horizon -> anchor -> symbol -> equal-symbol weighting; Q26 must
retain it. The selected stronger of long/short is a descriptive oracle choice:
confirmation against directional controls requires both contrasts in the
declared family, rather than choosing one after seeing its return.

## 4. Joint dependence and candidate methods

Use one shared calendar draw or multiplier for all paired policies, twelve
symbols, four horizons, both frozen training seeds, and any declared contrasts.
Keep horizons attached to their anchor. Build metadata components from overlaps
of full maximal windows on actual exchange dates; record chronological session
gaps as well as overlap. Separated components can still have serial dependence.
Do not compress widely separated sparse anchors into an apparently dense daily
series, insert zero outcomes for unobserved dates, independently resample
symbols, or omit missing symbols and silently renormalize Q24 weights.
Repeatedly rejecting incomplete draws until all symbols appear changes the
resampling law; any such conditioning must be declared and calibrated.

| Candidate | Proposed use and principal limitation |
|---|---|
| Synchronized calendar block multiplier | First candidate for the sparse fixed layout: keep observations/weights present and apply common correlated calendar weights to centered contributions. Covariance kernel, bandwidth, centering, studentization, and interval law still require definition and calibration. Preserved support alone does not prove validity. |
| Moving or stationary calendar-block bootstrap | Comparison candidate when the locked layout supplies valid chronological blocks and full weighted symbol support. Circular wrapping creates an artificial boundary and needs explicit justification. Do not resample twelve separate symbol series. |
| Fixed-b subsampling / block bootstrap | Sensitivity candidate with its own calibrated limiting/reference distribution. Subsamples can lose symbol coverage; fixed-b is not repaired by an ordinary IID percentile interval. |
| HAC covariance of paired contributions | Sensitivity only unless its sparse-calendar covariance construction and finite-sample calibration qualify; bandwidth and long-run variance assumptions remain substantive choices. |

Block construction must follow the declared trading-session geometry, not just
row order. Report unique dates, component count/size/gaps, symbol coverage,
maximum component share of fixed Q24 weight, admissible blocks, and whether
every replicate retains the target. No universal minimum block count or
bandwidth is adopted. Short blocks can miss dependence; long blocks can remove
effective replication. Candidate methods and reference distributions must be
registered before calibration. [Kunsch](https://doi.org/10.1214/aos/1176347265),
[Shao and Politis](https://publish.illinois.edu/xshao/files/2012/11/ShaoPolitisJRSS.pdf),
and [MacKinnon et al.](https://doi.org/10.1016/j.jeconom.2022.04.001) motivate
these limitations; the sparse twelve-symbol adaptation is our proposal.

## 5. Calibration and ordered decisions

After exposure clearance and separately authorized sealed acquisition, lock one
metadata-only candidate layout for calibration. Overlay synthetic paired returns
using the exact Q24 weights and calendar geometry. Include serial dependence,
common-factor/cross-symbol shocks, overlapping horizon outcomes, heterogeneous
scales, heavy tails/asymmetry, and combined stress families. State stationarity
assumptions and include a period-change sensitivity; no resampler automatically
solves distribution shift.

Before simulations, humans must declare alpha, interval coverage target,
Monte Carlo precision/replications, independent RNG streams, error tolerances,
support/numerical failure rules, and how performance is aggregated across null
families. A decision-ready candidate criterion is: every required null family
must satisfy the declared Type-I and coverage tolerances with Monte Carlo
uncertainty, preserve the estimand/support, and avoid declared numerical or
bandwidth-instability failures. Keep invalid replicates in the requested-count
ledger; a failed family cannot be dropped. Infinite-variance scenarios require
an explicit incompatibility disposition rather than pretending a mean/variance
procedure has passed. [Morris et al.](https://doi.org/10.1002/sim.8086) provide the
simulation reporting framework.
Report success-conditional error/coverage separately from total failure rates.
Mean-zero population truth must not be replaced by forcing every realized
synthetic sample's Q24 estimate to zero, which would suppress sampling variation.

Method development on a calibration battery requires a versioned change log
and a predeclared independent confirmation battery. A method that fails cannot
be rescued by attractive simulated-effect power; report power separately under
predeclared nonzero effects. If no registered method qualifies, retain
`Q26_UNRESOLVED_FOR_LAYOUT`, report descriptive effects, and return the next
design decision to GRILL. Do not redraw dates or optimize tolerances to pass.

| Order | Decision or receipt |
|---|---|
| 1 | Resolve the bounded exposure receipt or conservative exclusions. |
| 2 | Adopt the endpoint, contrast, directional hypothesis, and conditional seed interpretation above, or specify the desired alternative. |
| 3 | Define calendar unit/adjacency, full-support block or multiplier construction, candidate method registry, and interval/reference distribution. |
| 4 | Adopt support, Type-I/coverage, numerical failure, simulation precision, RNG, and stability criteria; no numbers are supplied before endpoint choice. |
| 5 | Under separate authority, acquire sealed data, qualify metadata, lock candidate layout, and calibrate/confirm on synthetic outcomes. |
| 6 | Human selects qualified method or descriptive status; seal final panel + Q26 + complete contrast family before any sixteen-cell training run. |
| 7 | Open terminal outcomes only after all sixteen cells complete and every source/config/checkpoint/control/producer gate passes. |

For the single proposed confirmatory endpoint there is no additional
across-contrast multiplicity family; multiple method/bandwidth searches still
require the declared calibration/confirmation rule. If other seeds, curricula,
representations, metrics, or behavioral-control comparisons become confirmatory,
declare the complete family and prefer simultaneous intervals/stepdown testing
under the same shared calendar law. [Romano and Wolf](https://doi.org/10.1111/j.1468-0262.2005.00615.x)
support that candidate family treatment; their method does not make an invalid
calendar resampler valid. Two seeds remain a conditional mechanics/attribution
screen; [Agarwal et al.](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)
motivate explicit training-replication uncertainty rather than pseudo-replication.
For an expanded family, calibrate familywise error and simultaneous coverage,
including mixed true/false nulls; marginal coverage or complete-null checks alone
do not establish strong familywise control.

First minimal Q26 question after clearance: **Should the sole confirmatory
endpoint be the Q24 equal-symbol terminal-net improvement of `R1 + R4-v1` over
`R0 + R4-v1` in the fixed-1-bp arm, averaged over the two frozen seed pairs with
calendar inference conditional on those checkpoints, keeping other outputs
descriptive?**

## 6. Boundaries, provenance, and validation

Volatility normalization remains off and nonblocking. Direction/timing/cost
decomposition is accounting; its timing residual is not demonstrated alpha and
must not be clamped to a long-only oracle. `CapabilityProfile` stays a future
online/shadow contract; terminal evidence cannot enter proposer reward,
curriculum, PPO, tuning, or operational calibration.

The [coordination board](../../coordination/research.md) records branch/HEAD,
input hashes, ownership, planning budget, and final receipt. This note supersedes
the Oct7 pending-envelope approval label only to the extent of the Oct9 human
decision; it does not rewrite historical evidence or DEV's active plan.
No-Mistakes remains an experimental sidecar: methodological review cannot
replace independent evidence, quantify absent model telemetry, or certify
statistical calibration. Its runtime pilot is outside this scope.

Sources: the primary links above and the existing
[statistical source ledger](evaluation-quant-methodology-2026-10-06-evidence/statistical-protocol-primary-sources.md)
and [synthetic boundary](evaluation-metadata-first-envelope-2026-10-07-evidence/q26-synthetic-calibration-boundary.md).
Verification is limited to documentation/link/companion checks and source
inspection of the existing reducer; no new performance evidence is produced.
