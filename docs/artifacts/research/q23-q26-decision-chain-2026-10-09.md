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

## 7. Legacy-source bound: owner statement and corroboration

Follow-up `Q23-V1-BOUND-02`, 2026-10-09. Branch `agent/Watson-dev`, HEAD
`8f9fcf94e028b51d883a1a484c2c9466a3d6b547`, externally advanced by the human;
this research pass made no Git mutation. This section supersedes the request to
reattest already bounded legacy-training windows in section 1. All Q26 choices
remain proposals.

The owner answered GRILL's exposure query asynchronously on 2026-10-09. As
relayed in the finite dispatch, the owner says older training runs did not use
the candidate old data; legacy collection covered roughly two years relative
to the July-2025 project start, recalled as approximately 2022-2024, with AMZN,
a Microsoft symbol, and one other symbol. Record this as an accountable bounded
owner statement that legacy training did not use 2020-2021. The exact response
timestamp is not supplied. The two recalled date descriptions differ and are
not an exact acquisition range or immutable historical file identity.

Metadata corroboration:

- Git commit `6b6ba79`, dated 2025-07-05, introduced the ingestion mechanism;
  `b780691` modified it later that day. The original backfill documented AAPL
  and defaulted to `--days 730`, with request start computed as execution-day
  minus that duration and adjusted Polygon aggregates. This is consistent with
  July-2025 trailing collection starting around July-2023, rather than an exact
  2022-2024 range. Git/code dates and defaults do not prove an executed request.
- The retained exact v1 identity already establishes AAPL stored dates
  2023-07-05 through 2025-07-03 and AMZN/MSFT 2024-07-05 through 2025-07-03.
  Therefore source metadata supports normalizing the recalled Microsoft symbol
  to MSFT and identifies AAPL as the retained third symbol. This describes the
  retained file, not every historical copy or the owner's original vocabulary.
- The existing inventory connects 120 old run configs to the canonical legacy
  source location (19 absolute, 101 repository-relative), across `OLD_RESULTS`,
  `Phase_A_PBT_overnight_20260216`, and `Phase_A_PBT_followup_20260216`.
  Their historical hashes are absent. The owner's legacy-training statement,
  supported by acquisition chronology and retained-source dates, supplies a
  bounded date-disjoint account for these runs; historical byte identity stays
  unproved. No fresh provider request receipt was recovered in this limited
  Git/config review.

| Artifact group | Updated clearance disposition |
|---|---|
| 120 canonical-path legacy runs | Bounded-disjoint from 2020-2021 by the owner's legacy-training statement plus the corroboration above. Carry the statement provenance; do not label them exact-hash-bound or ask again about the same legacy-training period. |
| Three OneDrive-checkout run references | Still unknown: a shared basename and older-run label do not establish that this external checkout used the same source. The statement is not automatically an attestation of that external file. |
| Twelve pools / six candidate ledgers | Still unknown: no dataset/path/symbol/time binding. Generator defaults to the legacy source in a value-reading mode, but defaults and later comparison configs do not establish which source produced these retained summaries. |
| Manual/external access | Still unknown: the reply addresses legacy training, not chart/notebook/export/prompt/agent/calibration access across providers. |
| Exact v1/v2 files and v2's 42 unresolved starts | Already date-disjoint by retained identities; this follow-up adds no new uncertainty or inspection. |

The remaining clearance question is narrower: **Were the three external-checkout
sources and opaque task/candidate summaries derived solely from the already
bounded legacy/v2 data, and was there any separate manual/external/agent access
to 2020-2021 values; if uncertain, what symbol/date bounds must be excluded?**
Do not request the legacy-training period again. Unknowns that cannot be bounded
remain exclusions and may block the candidate. The statement does not establish
current SIP entitlement, authorize collection, or accept the Q26 endpoint.

Provenance: owner answer relayed by GRILL scope `Q23-V1-BOUND-02`;
[retained inventory](evaluation-metadata-first-envelope-2026-10-07-evidence/retained-identities-and-exposure.md);
[backfill source](../../../itrader/inputs/backfill.py) and
[ETL source](../../../itrader/inputs/etl_job.py); `git log --all` for those paths
and a metadata-only projection of `git show 6b6ba79:itrader/inputs/backfill.py`.
No credentials, provider values, OHLCV outcomes, or new run artifacts were opened.
Relevant input hashes and validation receipt remain on the research board.

## 8. Q26 calendar proposal after primary-endpoint acceptance

Scope: Q26-CALENDAR-PROPOSAL-04, 2026-10-09. This section supersedes the
pending-endpoint wording in sections 3, 5 and 7, **only for the accepted primary
contrast and fixed-pair interpretation**. The
[canonical decision](../../project-state/decisions.md) accepts fixed-1-bp
R1+R4-v1 minus R0+R4-v1, Q24 terminal net averaged over the two frozen checkpoint
pairs. Preserve all four cells, both curricula and all sixteen completed runs
before opening the fresh panel; an R4-pair-first execution order does not waive
that gate. The human has now accepted synchronized calendar multipliers as the **first
metadata/synthetic qualification candidate**, not a qualified inference method.
No kernel, centering, bandwidth, interval policy, alpha, tolerance or coverage
qualification is adopted. Calendar-block resampling remains a later comparator.

### 8.1 Target and source boundary

The realized fixed-panel weighted effect is descriptive and exactly computable.
An interval concerns its proposed repeated-market/calendar sampling counterpart,
conditional on the checkpoint pairs and locked selection design; it is not
randomization over frozen market paths, training-seed inference, or an unbiased
estimate over every 2020-2021 session. Generalization beyond selected windows
requires a stated stochastic/design assumption. Metadata selection depending
on availability/exposure can itself restrict that target.

[Shao (2010), sections 2, 4 and 5](https://publish.illinois.edu/xshao/files/2012/11/JASA-DWB.pdf)
motivates mean-zero, unit-variance multipliers correlated at actual time
distances, with a positive-semidefinite covariance kernel. Its regular-series
results assume stationarity, moments and weak dependence with growing samples
and bandwidth restrictions. Section 4 studies expanding lattice regions;
section 5 uses random locations sampled independently from the process under
density/domain conditions. Arbitrary-layout implementability is not a
coverage theorem for our stratified sparse fixed panel. The adaptation below
is **our qualification proposal**, not an application certified by that paper.

### 8.2 Fixed weights and centering candidate

For paired net differences d[k,s,a,h] from section 3, collapse dependent
horizons and the two fixed pairs before primary uncertainty:

```text
x[s,a] = (1/2) sum_k (1/4) sum_h d[k,s,a,h]
q[s,a] = 1 / (12*n_s)
theta_hat = sum_s sum_a q[s,a]*x[s,a]

m_hat[s] = (1/n_s) sum_a x[s,a]
e[s,a] = x[s,a] - m_hat[s]
c[u] = sum_{observed (s,a) on session u} q[s,a]*e[s,a]
```

Use exactly these Q24 weights, not pooled episode counts, per-bar weights or
inverse-volatility weighting. No per-draw renormalization. Symbols with extra
anchors retain total weight 1/12. Mean-centering does not change theta_hat;
it changes the variance estimator and therefore requires explicit acceptance.

**First centering candidate:** symbol-specific anchor means, intended to retain
fixed symbol heterogeneity while modeling within-symbol calendar variation.
Some symbols have only two anchors: fitting those means can seriously
underestimate variance, particularly with serial dependence. Applying n/(n-1),
leave-one-out or another leverage correction is a separate candidate, not a
universal repair. Pooled centering x-theta_hat instead can inject differences
between fixed symbol means into sampling uncertainty. Register that alternative
as a diagnostic, not select between them using terminal results.

If time-varying symbol means are allowed, neither stationary symbol centering
nor arbitrary detrending is enough. With two observations per symbol, we cannot
reliably infer that mean path from this panel. Synthetic oracle-known means may
diagnose centering error but cannot become an operational estimator. No proposed
centering can qualify arbitrary heterogeneous/nonstationary processes from these
counts.

For pair-specific descriptive uncertainty, repeat the same reducer and centering
separately for each k, but use the identical calendar multiplier draw. Do not
bootstrap k or claim twice the independent calendar sample size.

### 8.3 Joint calendar law and PSD construction

Let u be the ordinal of each **actual exchange session in the approved full
calendar**, retaining gaps even when no anchor is selected. Thus Monday after
Friday is one session apart, while selected dates ten exchange sessions apart
remain ten apart, not adjacent rows. UTC session identity, holidays, early closes
and decision offsets remain metadata. This geometry is not outcome-derived
regime labeling, and W varies over dates: it is not one constant multiplier
for the entire panel. An elapsed-day alternative is a different
distance law requiring separate declaration/calibration.

First kernel proposal, for a positive bandwidth b measured in exchange sessions:

```text
K[u,v] = max(1 - abs(u-v)/b, 0)       # Bartlett
W_observed ~ Normal(0, K_observed)
Delta_star = sum_{observed sessions u} c[u]*W[u]
theta_star = theta_hat + Delta_star
V_hat = c' K_observed c
```

This kernel is PSD on session coordinates; restricting it to observed dates
preserves PSD. Use a symmetric eigendecomposition square root so legitimate
singular PSD matrices do not require Cholesky jitter. Predeclare floating-point
symmetry/eigenvalue tolerances before implementation; roundoff-only correction
must be bounded/logged, while materially negative eigenvalues fail. Do not
silently project an invalid matrix to PSD or change b after seeing outcomes.
The displayed zero outside the kernel's support is **multiplier covariance**,
not a fabricated market return.

One W[u] is shared across all symbols, paired policies, horizons, checkpoint
pairs and any registered contrasts on that session. This does not posit perfect
correlation of returns: their joint residual contributions enter the same
weighted sum. It avoids independently resampling contemporaneous market shocks.
Do not add independent symbol/seed multipliers that break this law.

Retain each full maximal information/outcome interval [start-60,start+200).
Overlapping horizons within an anchor remain attached. Cross-symbol intervals
can share shocks despite different offsets; all same-session anchors receive
the same multiplier, a deliberately coarse cluster approximation. Nonoverlapping
intraday windows are not presumed independent, and disjoint sessions can remain
serially dependent. If qualified metadata reveals cross-session windows, this
session-only mapping must be redesigned before calibration. Sparse cross-date
symbol overlap may leave common/idiosyncratic dependence poorly identifiable;
a shared multiplier alone does not recover unobserved covariance.

No zero outcomes, imputed residuals or synthetic symbols on unobserved dates.
Generating latent W on a full calendar is permissible, but only observed
contributions enter the statistic. Missing outcomes invalidate the panel, not
just that draw.

### 8.4 Bandwidth/support and interval proposals

Do not set b from 32 anchors, 128 anchor-horizon cases or two seeds as though these were
independent sample counts. On qualified metadata, register a **finite** grid:
a short-session candidate, candidates bridging occupied-date gaps, and a
longer-persistence stress candidate. Exact values, selection rule and numerical
tolerances must be accepted before simulation; no terminal-outcome bandwidth
fit or significance-maximizing scan. Compact kernel support asserts negligible
dependence beyond b, which must be challenged by longer-lived synthetic factors.

Report occupied-session count/span/gaps; symbol-session incidence; full-window
overlap components; fixed Q24 weight per date/component and its concentration;
kernel rank/spectrum; and the number/weight of temporal neighborhoods actually
supported at each b. Kernel rank or many multiplier draws is not an effective
sample-size certificate. Isolated dates make dependence unestimable; near-common
multipliers can produce V_hat near zero because centered contributions sum to
zero. Declare support/concentration/stability failure criteria before calibration.
Do not erase dates, redraw the layout or enlarge b until a desired verdict appears.

**First interval candidate:** unstudentized basic multiplier interval in original
net-return units; no invented sqrt(128) scaling. For conditional quantiles Q_p
of Delta_star:

```text
two-sided interval = [theta_hat-Q_(1-alpha/2), theta_hat-Q_(alpha/2)]
one-sided lower bound L = theta_hat-Q_(1-alpha)
primary superiority: H0 theta <= 0 versus theta > 0; reject only if L > 0
```

Alpha and coverage tolerances remain pending. Test the boundary null theta=0
and negative-null configurations. Gaussian linear multipliers yield a symmetric
conditional root; they do not reproduce arbitrary finite-sample skew or heavy
tails. Require nondegenerate V_hat; no zero-width interval on a constant or
unsupported panel. Studentization requires a fully specified repeatable variance
estimator and its own calibration, not dividing by an arbitrary SD.
Fixed-b changes cannot borrow ordinary small-bandwidth reference laws:
[Shao and Politis](https://publish.illinois.edu/xshao/files/2012/11/ShaoPolitisJRSS.pdf).
Calendar-block resampling remains a separately registered comparator if selected;
repeat-until-all-symbols-appear rejection is not an innocuous implementation.

### 8.5 Decision and independent qualification sequence

| Gate | Proposed next decision/evidence |
|---|---|
| First method choice | Accepted: synchronized calendar multipliers first, holding panel and Q24 weights fixed; acceptance does not certify coverage or the parameterization below. |
| Design/target | Resolve Q23 unknown external/pool/manual access; qualify actual coverage/calendar metadata and freeze one layout. Accept session distance, centering, kernel, finite b grid and inference target. |
| Acceptance policy | Predeclare alpha, support/numerical rules, coverage/Type-I tolerances, replication/Monte Carlo precision, RNG separation, family aggregation, bandwidth selection and independent confirmation policy. |
| Calibration development | Under a separate lease, synthetic DGPs on exact weighted layout; keep every attempted family/version and failure. No market outcomes. |
| Independent confirmation | Freeze chosen method/version/rules, then use untouched RNG streams and predeclared DGP variants, including stresses not used for tuning. No iterative retuning against confirmation results. |
| Seal/open | Accept only a justified qualified method or Q26_UNRESOLVED_FOR_LAYOUT; seal its receipt before screen execution/opening gates. Other outputs stay descriptive. |

Synthetic global nulls must allow different symbol means whose **Q24-weighted
population aggregate** is zero, unequal scales and pair correlations. Do not
force each realized sample mean to zero. Include shared market and symbol factors,
actual session lags, unequal gaps, overlapping horizons, long persistence,
skew/heavy tails, leverage/concentration and smooth/abrupt mean/variance changes.
Generate a joint process on the full calendar, then observe only selected sites;
do not invent IID episode returns. Nonstationary DWB results exist for particular
other statistics, not this panel:
[Rho and Shao's unit-root study](https://arxiv.org/abs/1802.05333).

Assess every required null family's Type-I error and coverage with Monte Carlo
uncertainty; predeclare how simultaneous uncertainty across many DGP/bandwidth
checks is handled. Effect/power runs are separate and cannot rescue null failure.
Finite synthetic success qualifies only the declared stress class, not all market
processes. Infinite-variance/unsupported classes receive an explicit incompatible
disposition. Follow an explicit estimand/design/performance reporting framework:
[Morris et al.](https://doi.org/10.1002/sim.8086).

Retain requested/attempted/valid/failed counts at both dataset and multiplier-draw
levels. Missing support, degenerate covariance, failed factorization and timeout
are recorded failures, never omitted or endlessly redrawn. Report conditional-on-
success coverage/Type-I plus unconditional procedure failure rates and a
predeclared conservative treatment of failures. Increasing the bootstrap-draw
count reduces conditional Monte Carlo noise, not sparse-calendar information.

**Next human question:** use a one-sided lower confidence bound for the accepted
superiority hypothesis, with a companion two-sided descriptive interval, or
another interval/test policy? No numerical alpha/coverage level is proposed as
accepted. This decision precedes numerical calibration thresholds.

**Decision-ready recommendation:** test the session-synchronized Bartlett
Gaussian multiplier with symbol-specific centering first, under the now accepted multiplier-first development direction. Treat sparse centering/support as likely limiting
gates, not details to smooth away. If no registered method survives independent
confirmation, keep descriptive Q24 effects and Q26_UNRESOLVED_FOR_LAYOUT; return
a layout/claim redesign decision to GRILL without opening outcomes.

This append is planning only. Q23 unknown access remains unresolved. CUDA planning
does not broaden TRAINING's separately bounded bare-LSTM compute lease.

## 9. Accepted 95% interval, metric meanings and next calibration decision

Scope Q26-INTERVAL-TERMS-05, 2026-10-09. The
[canonical decision](../../project-state/decisions.md) now accepts a **two-sided
95% confidence interval** for the fixed-1-bp R1+R4-v1 minus R0+R4-v1 Q24
terminal-net contrast, averaged over two frozen checkpoint pairs.
This supersedes section 8's pending interval-policy question. It does not adopt
the proposed basic/equal-tailed construction, centering, kernel, bandwidth,
coverage tolerances or qualification. Do not replace it with a one-sided 5% test.
All four cells and all sixteen runs remain required before fresh-panel opening.

### 9.1 What each measurement actually says

| Measurement | Current meaning and inference boundary |
|---|---|
| Primary paired Q24 net difference | A positive estimate means R1+R4 lost less or earned more than R0+R4. After method qualification, a primary CI wholly above zero supports this conditional representation uplift, not absolute profit. |
| Absolute episode/Q24 net | Arithmetic gross strategy return minus switching and final liquidation costs; report both policies and five controls descriptively. Positive absolute net and beating controls are distinct questions, not conclusions from the contrast's CI. |
| net_sharpe_per_sqrt_bar | Within each episode, mean cost-adjusted strategy-bar return divided by its sample SD (ddof=1). Final liquidation is deducted from the final bar. The reporter hierarchically averages episode ratios; that is not Sharpe of a pooled chronological portfolio. |
| Isolated one-step return | Realized next-close asset return times target position, with cost accounting below. One scalar outcome has no sample Sharpe; a positive bar or positive_net_bar_fraction is not risk-adjusted significance. |
| Optional lagged-volatility scaling | Gross or continuation net divided by a named positive pre-action scale from past closes. It is a diagnostic, not realized Sharpe, primary Q24, reward or promotion authority; estimator/epsilon/window remain unchosen and diagnostic stays off. |

Source:
[EpisodeEvidence](../../../itrader/rl_lib/evaluation/episode_evidence.py),
[Q24 reducer](../../../itrader/rl_lib/evaluation/terminal_panel_report.py),
[economic diagnostics](../../../itrader/rl_lib/evaluation/economic_diagnostics.py)
and [environment](../../../itrader/envs/itrader_env.py).
EpisodeEvidence returns a numerical zero when fewer than two observations or
SD <=1e-12: this fallback does not establish an estimable one-bar Sharpe.
Annualized output, if enabled, uses sqrt(configured periods), not a dependence-
corrected portfolio estimate.

For previous position p, selected target q, next-bar asset return r and
c=cost_bps*1e-4, current source implements:

```text
gross = q*r
continuation_net = gross - c*abs(q-p)
actual_net = continuation_net - c*abs(q) only at actual episode termination
hypothetical_exit_after_bar_net = continuation_net - c*abs(q)
```

Holding the existing position has no switch cost; switching -1 to +1 has
turnover 2. The hypothetical expression retains the actual prior position:
it is a conditional exit-after-this-bar counterfactual, not necessarily a fresh
flat-entry round trip. Do not charge hypothetical exit on every continuing bar.
The optional estimator receives only closes through the decision time, excluding
the next close; invalid scales are unavailable, not rescued by an adopted floor.

Sharing calendar draws does **not** give Sharpe, per-bar net or isolated-step
statistics the primary net CI. Separate confirmatory claims require their own
estimands, dependence-aware uncertainty, calibration and predeclared multiplicity.
Keep them descriptive here; do not use terminal diagnostics to tune the solver.

### 9.2 One proposed confirmation gate, not adopted thresholds

**Next human question:** should we use the following tolerance-qualified,
independent synthetic confirmation gate for the nominal two-sided 95% interval,
with a central equal-tailed construction as the first interval candidate?

The equal-tailed construction is a proposal: nominal two-sided 95% alone does
not require balanced tails. For that candidate, target total noncoverage is 5%,
with 2.5% in each tail. Under theta=0, declaring positive uplift from L>0 is one
of those tails, not a new one-sided 5% test.

| Proposed criterion | Candidate rule for each predeclared required DGP family |
|---|---|
| Coverage | Simultaneous Monte Carlo lower confidence bound >=94%; permits at most one percentage point undercoverage relative to nominal 95%. |
| Tail errors | Simultaneous upper bounds <=3% for each directional miss; permits 0.5 percentage point inflation above each nominal 2.5% tail. Also check positive-null declarations for predeclared theta<0 cases. |
| Procedure failure | Simultaneous upper bound <=0.1%, with no systematic schema/support/PSD defect. Invalid outer panels count as noncoverage and as possible errors in both worst-case tail bounds. |
| Outer Monte Carlo | Fixed 10,000 independent synthetic panels per family on the qualified fixed calendar layout; never treat inner multiplier draws as independent outer panels. |
| Simultaneous precision | With K frozen families, use exact one-sided binomial bounds at error allocation 0.05/(4K) for coverage, two tails and failure. Include any additional required checks in the allocation count before running. |
| Confirmation isolation | Freeze method, parameter-selection rules and family list after development; independent RNG streams and predeclared unseen stress variants. Failure/inconclusive evidence does not license extending runs until passing. |

These are **our engineering acceptance proposals**, not literature-prescribed
thresholds or proof of exact 95% coverage. Bounds that cannot meet a gate at the
fixed budget leave it inconclusive/unqualified. If exact nominal control without
these tolerances is required, that needs a different predeclared criterion and
precision budget. For reference, 10,000 outer panels give approximate unadjusted
Monte Carlo SE of 0.218 percentage points at 95% coverage and 0.156 points at a
2.5% tail; simultaneous bounds are wider.

[Morris et al.](https://doi.org/10.1002/sim.8086) grounds explicit simulation aims,
DGPs, estimands, performance measures and Monte Carlo uncertainty.
[NIST's exact binomial bounds](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
grounds finite-count bounds, including rare failures.
Neither source supplies our chosen tolerances. Independent outer panels make
this binomial treatment possible; shared randomness across method comparisons
does not turn those comparisons into extra independent panels.

Track all attempted/valid/invalid outer panels and inner draws. Report success-
conditional performance separately, but qualify on conservative all-attempt
accounting. Inner quantile approximation has separate error: the proposed linear
Gaussian multiplier root has known conditional variance c'Kc, so analytic normal
quantiles could avoid inner Monte Carlo error; that computational choice is also
unaccepted. Other multiplier/studentized choices need a separately frozen inner
budget and quantile-stability check. Report interval widths, bias and effect-power
descriptively; an arbitrarily wide interval is not a useful performance result.

The sparse-date/two-anchor centering risks in section 8 remain blocking scientific
questions, not resolved by 10,000 repetitions. Include heterogeneous symbol means
with zero global weighted population effect, cross-symbol/session shocks, long
dependence, sparse gaps and mean/variance changes. No family may be dropped for
failure; no realized sample is forcibly re-centered to a zero endpoint. If the
method cannot qualify on the declared supported class, retain
Q26_UNRESOLVED_FOR_LAYOUT. Synthetic qualification does not attest Q23 exposure,
all market processes, absolute profitability or training-seed populations.

CUDA install planning is superseded: the human requires reuse of existing
torch2.9dev/cu128 for TRAINING's bounded tensor-only CPU/GPU comparison. Keep the
root CPU/Ray2.58 lifecycle separate; no new environment or platform migration.
This append performed no simulation, install, probe or market-outcome inspection.

## 10. Sparse support: a new centering boundary, not a layout verdict

Scope Q26-SUPPORT-CENTERING-10; paired-document closure Q26-PAIR-CLOSE14,
2026-10-09. Current
[canonical decisions](../../project-state/decisions.md) supersede section 9's
pending labels for **central equal-tailed nominal 95%** and the **simultaneous
Monte Carlo lower coverage bound >=94% in every required scenario**.
The human delegated the directional-error trade-off to GRILL, which accepted
the **simultaneous Monte Carlo upper bound <=3% for each directional miss in
every required predeclared scenario**, including positive-uplift declarations
under theta<=0. The nominal target remains 2.5% per tail. These are calibration
acceptance tolerances, not changes to interval level/width or evidence of exact
95% coverage or exact 2.5% directional-error control. Only after passing may
the method be labelled nominal 95%, tolerance-qualified on its declared class.
Monte Carlo family confidence/allocation, failure tolerance, outer budget,
scenario registry and centering/kernel/bandwidth remain unaccepted. No method
has been statistically qualified, and gates must not be relaxed after failure.

The proposed 32 anchors across twelve symbols with at least two anchors each
leave eight extra anchors: **at least four symbols necessarily have only two
anchors**. Possible symbol counts range from two to ten; occupied dates can
range from three to 32 under the structural contract, not 505. Actual counts,
dates, gaps, shared dates and eligible full windows require qualified Q23
metadata. The 505 sessions are the candidate calendar, not 505 observed returns.

After the prescribed horizon/pair collapse, symbol-specific centering removes
twelve fitted symbol means from 32 scalar anchor outcomes. Its linear residual
space has dimension at most **20**, before date aggregation can reduce it
further. Twenty is an algebraic rank bound, **not degrees of freedom for a
t-test, independent calendar replication or effective sample size**. Four
overlapping horizons and two frozen pairs do not increase that calendar support.

A boundary example makes the unidentified assumption explicit:
x[s,a]=mu[s]+Z+epsilon[s,a], where Z is a random shock persistent across the
whole selected panel. Symbol centering cancels Z exactly, although the
repeated-market variance of the weighted primary estimate includes Var(Z).
No PSD multiplier construction or additional draws recovers that missing
component. This is a counterexample to unrestricted covariance identification,
not evidence our market actually follows this model. A permanently common
level violates the weak-dependence class underlying
[Shao (2010)](https://publish.illinois.edu/xshao/files/2012/11/JASA-DWB.pdf);
it does not refute that paper. Finite long-lived-factor stresses and explicit
persistence assumptions are therefore substantive, not optional smoothing.

| Can be declared now | Cannot be certified from current counts |
|---|---|
| Fixed Q24 target/weights and conditional checkpoint interpretation; proposed stationary symbol-mean nuisance model and supported dependence/moment class | True stationarity, dependence decay, variance components or future market representativeness |
| Metadata audit recipe: symbol/session incidence, actual session gaps, full-window overlap, date-weight concentration and candidate-kernel rank | Actual eligible layout/support until exposure clearance, retained source identity and minute/session coverage qualify |
| Fail-closed treatment and finite candidate/stress registry | Adequate coverage from rank, calendar span or nominal bootstrap repetitions |

Qualified metadata can reject a design as structurally unusable; it cannot
prove market covariance assumptions. Later independent synthetic confirmation
can qualify only the declared process/layout class. Two anchors per symbol
provide very weak nuisance-mean/variance information; neither pooled centering,
n/(n-1), cross-fitting nor calendar-block resampling is an automatic cure.
No bandwidth or universal minimum occupied-date count is selected here.

**Next smallest support decision:**
if the unchanged metadata-qualified layout cannot support the declared
centering/dependence class and independent calibration, should we retain its
Q24 results as descriptive with Q26_UNRESOLVED_FOR_LAYOUT, and require a
separate human-approved redesign before making an inferential uplift claim?

Recommended fallback is that descriptive disposition. If inference remains
essential, the alternative is a separately declared exposure-cleared layout
with more within-symbol temporal replication and suitable shared-time coverage,
or a narrower justified process/claim class; neither has been selected or is
guaranteed to succeed. Redesign must precede outcomes, version the selection/
target/weights and calibration, preserve all-cell opening gates, and never
silently enlarge or replace the accepted Q23 sample.

## 11. Common evaluation contract: retain evidence, qualify claims separately

Scope EVAL-FRAMEWORK-RESEARCH18, 2026-10-09. The human's latest
[decision](../../project-state/decisions.md) supersedes automatic larger-layout
prioritization. Strengthen the existing pipeline, not a second framework.
Preserve the accepted Q23 sample/weights, fixed primary and all-cell opening
gates; historical exposure remains unresolved. No acquisition, outcome access,
simulation, method adoption, code change or qualification occurred in this review.

### Evidence disposition is not a single pass/fail bit

Proposed reporting vocabulary below is a contract candidate, not a new accepted
schema. Retain original artifacts and rejection reasons in every case.

| Evidence validity | Inferential status | Permitted interpretation |
|---|---|---|
| Complete, bound, reconciled matched evidence | Not attempted, insufficient support, method failure or unqualified method | Valid descriptive net estimate, positive/zero/negative; no inferential uplift claim |
| Same valid evidence | Qualified interval includes zero | Valid descriptive estimate; no demonstrated positive uplift under that procedure, not proof of zero effect |
| Same valid evidence | Qualified interval excludes zero in the declared direction | Conditional inferential statement only for the frozen target/process class; absolute profitability and deployment gates remain separate |
| Wrong/unbound provenance, broken pairing, incomplete primary or inconsistent accounting | Not eligible for certification | Retained raw/quarantined evidence; not a valid matched comparison, not an invented zero |

Keep evidence validity, metric availability, inference qualification and economic
promotion as separate statuses. A launcher failure is not a solver loss; a valid
negative return is not an invalid run; an inconclusive interval is not a failed
accounting check. Retention does not permit opening the still-unqualified Q23
panel or silently averaging only surviving episodes.

### Primary-source lessons worth transferring

| Primary source | Concrete lesson for the current stack |
|---|---|
| [Lo, Statistics of Sharpe Ratios](https://traders.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf) | Serial dependence affects Sharpe inference and annualization. Keep the current per-bar statistic descriptive; averaging episode Sharpes is not portfolio Sharpe, and square-root frequency scaling is not automatically justified. |
| [White, Reality Check](https://onlinelibrary.wiley.com/doi/10.1111/1468-0262.00152) and [Hansen, SPA](https://www.tandfonline.com/doi/abs/10.1198/073500105000000063) | Search over alternatives changes the inferential problem. Retain candidate/checkpoint/control search history and aligned observations; a paired interval around a selected winner is not itself selection adjustment. Neither test is adopted here. |
| [Bailey and Lopez de Prado, Deflated Sharpe](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) | Trial multiplicity, non-normality and sample definitions matter. Preserve failed/negative attempts and selection lineage. DSR is not a drop-in correction for the Q24 weighted net-return endpoint. |
| [FinRL-Meta paper](https://arxiv.org/abs/2112.06753) and [FinRL stock environment source](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_stock_trading/env_stocktrading.py) | Separate data processing from strategy design; retain action/account-value/reward histories and explicit buy/sell costs. Transfer evidence discipline, not its continuous share actions, portfolio accounting or trading constraints. |
| [TensorTrade environment source](https://github.com/tensortrade-org/tensortrade/blob/master/tensortrade/env/generic/environment.py) | Separate observation, action, reward, stopping, information and rendering; reset component state. Our browser consumes evidence rather than certifying it. No TensorTrade migration or reward replacement follows. |
| [Gym-Trading-Env documentation](https://gym-trading-env.readthedocs.io/en/latest/environment_desc.html) and [Gymnasium time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) | Dataset exhaustion/external time limits differ from task-defined termination. Causal warmup and complete outcomes need explicit checks. A finite task horizon is not automatically an external truncation; training bootstrap semantics and fixed-panel censoring are distinct questions. |
| [gym-anytrading StocksEnv](https://github.com/AminHP/gym-anytrading/blob/master/gym_anytrading/envs/stocks_env.py) and [base environment](https://github.com/AminHP/gym-anytrading/blob/master/gym_anytrading/envs/trading_env.py) | Reward and fee-adjusted profit are separate, with endpoint-specific profit updates. Its two actions and long-trade accounting are not our short/flat/long target-position convention; test our endpoint independently. |

External environments differ in timing, compounding, initial position, borrowing,
costs and liquidation. Their benchmark numbers cannot be substituted for iTrader
economics. Record the modelled limitations: fixed turnover cost is not spread,
slippage, borrow, capacity or broker fills. Keep RewardEngine ownership and the
pnl_linear default unchanged; evaluation arithmetic is not a reward redesign.

### Boundary and gap map: use current owners

Source snapshot: agent/Watson-dev, HEAD 8f9fcf9 plus inherited dirty files.
These are source findings, not newly executed test receipts.

| Boundary / edge case | Current owner and source-backed behavior | Proposed gap or qualification rule |
|---|---|---|
| Gross / net / optimization reward; switches versus liquidation | [EpisodeEvidence](../../../itrader/rl_lib/evaluation/episode_evidence.py) adds final liquidation once; [economic diagnostics](../../../itrader/rl_lib/evaluation/economic_diagnostics.py) distinguishes continuation, hypothetical roundtrip and actual terminal costs and reconciles the trace | Keep additive target-position economics explicit. Turnover is absolute position change, not trade count; exposure is not action count. Existing max_drawdown is gross-return-space drawdown, not net account drawdown. Do not relabel it. |
| Missing minutes, duplicate/non-increasing timestamps, session crossing, insufficient warmup, early close or dataset end | [DataPartition](../../../itrader/rl_lib/evaluation/data_partition.py) owns eligible chronological starts/segments; [panel qualification](../../../itrader/rl_lib/evaluation/panel_qualification.py) checks Q23 minute/session/calendar windows and historical exclusion | Continuity depends on declared timestamp/interval/boundary configuration, not row count. Reject incomplete fixed-horizon primary; retain censored trace separately, never rescale it into a complete result. |
| Wrong source/checkpoint/task/control pairing; duplicate/reordered trace | [Replay evidence](../../../itrader/rl_lib/evaluation/replay_evidence.py) binds contract/artifact/producer and independent launch receipts; attribution verifies order and reset-flat position continuity | Arithmetic agreement is not provenance. Terminal report callers must bind checkpoint and model-matched controls; missing identities remain unknown/quarantined. |
| Flat, one bar, constant net returns / zero SD | EpisodeEvidence stores 0.0 for fewer than two returns or SD<=1e-12; net-step costs include final liquidation | Stored zero alone cannot distinguish unavailable Sharpe from measured zero. A future versioned adapter needs return count/dispersion and reason; do not infer availability from the scalar or insert null into an unchanged finite-number schema. Net accounting remains usable. |
| Unequal symbol counts and crossed horizons | [Terminal panel report](../../../itrader/rl_lib/evaluation/terminal_panel_report.py) computes horizon -> anchor -> symbol -> equal-symbol primary; bps/bar normalizes before averaging | Preserve raw episode means as separately labelled diagnostics. Per-horizon current means are explicitly raw, not automatically equal-symbol. No accidental weighting substitution for the primary. |
| Episode controls versus local bar alternatives | Whole-panel flat/long/short/random/model-matched-untrained contrasts belong to terminal reporting; economic diagnostics supplies hold-prior/flat/long/short at the same prior position | Local counterfactual actions are not complete policies with their own state histories. Stronger constant direction is selected once over the whole panel, not opportunistically per episode/bar; inferential control selection still needs its declared family. |
| Thin dates, overlapping horizons, common shocks, two frozen checkpoint pairs | [Legacy block uncertainty](../../../itrader/rl_lib/evaluation/block_uncertainty.py) is development diagnostics; terminal reporting leaves Q26 uncertainty unscored | Independent-symbol row blocks miss shared-date shocks; anchor resampling does not certify sparse-calendar dependence. Seed-then-anchor resampling is not the fixed-two-pair target. Its cross-seed count check is not equality of anchor identities. More bootstrap draws do not add support. |
| Selection, multiplicity, method failure and retained negative outcomes | Existing provenance/registry/replay retain identities; block intervals explicitly disclaim selection adjustment | Separate calibration-scenario simultaneous confidence from policy/checkpoint/endpoint multiplicity. Preserve all attempts, denominator/reasons and selection lineage; qualified fixed-checkpoint evidence is not a population guarantee over training seeds. |
| Preflight versus actual evaluation versus browser presentation | [Terminal runner](../../../itrader/rl_lib/evaluation/run_terminal_screen.py) currently performs read-only preflight and emits evaluation_authorized=false; [browser renderer](../../../itrader/scripts/experiments/render_experiment_browser.py) consumes registered evidence | A reviewed execution/report/disposition bridge is still needed. Do not call preflight a completed terminal evaluator or expect the browser to repair missing evidence. No new parallel runner/ledger is proposed. |

The accepted primary is fixed-1-bp R1+R4-v1 minus R0+R4-v1 Q24 terminal net,
averaged over two frozen checkpoint pairs with fixed equal-symbol hierarchy.
Keep per-pair results visible, including negative ones. These are conditional
checkpoint comparisons, not enough replication to certify the training-seed
population. The 505-session calendar, 32 anchors and 128 crossed episodes remain
different units; no claim of actual layout failure follows from counts alone.

### Four named public integration oracles, not a unit-per-field ladder

Extend existing tests where they already cover these scenarios. Expected values
must be literal/hand-derived, not produced by the same accounting/reducer helpers
under test. No learned solver or market outcomes are needed for these proposed
synthetic fixtures; none was run in this investigation.

| Named scenario | Independent expected outcome and public path |
|---|---|
| AccountingLifecycle | Public reset -> step -> terminal evidence -> attribution on prices 100 -> 110 -> 99, targets long then short: gross=0.1+0.1=0.2, entry/reversal/liquidation turnover=1+2+1=4, 1bp cost=0.0004, net=0.1996; no extra terminal bar. The existing pilot already checks this and reward=199.6 only for its explicitly selected net-return x1000 reward. Add flat (all economics zero) and held-long (gross=0, turnover=2, net=-0.0002) variants. |
| CausalCompleteWindow | Public reset/replay on literal timestamped rows: authorized past-only warmup is unchanged by a future-price perturbation; missing minute/session/warmup/full-horizon failure is rejected. A 210-minute early-close session cannot supply offset60 + 200 future bars, even if a shorter horizon fits. Keep failed/censored raw evidence, not a shortened primary. |
| WeightedMatchedPanel | Existing complete fixture: one two-anchor symbol has net=0.012 and eleven symbols zero; equal-symbol net=0.001 versus raw episode mean=0.00075. Reordering rows leaves the result unchanged; duplicate/wrong seed/control identity fails. Two literal pair effects +0.002 and -0.001 average to +0.0005 without discarding the negative pair. No CI is implied. |
| EvidenceDisposition | Public receipt -> report -> registered browser fixture retains valid positive/zero/negative net with inference-unresolved status; qualified interval crossing zero remains inconclusive. Wrong provenance/accounting stays raw/quarantined, not zero. Flat/one-bar/zero-SD Sharpe displays unavailable with a supported reason despite legacy scalar fallback. Opening without all16 completion/Q23 qualification must fail before evaluation. |

Existing [economic diagnostics tests](../../../itrader/tests/test_economic_diagnostics.py)
and [terminal panel tests](../../../itrader/tests/test_terminal_panel_report.py)
already establish several arithmetic and weighting invariants. Retain them and
the focused provenance/partition tests; the gap is their public lifecycle and
disposition integration, not a target of fewer tests or a happy-path replacement.

### Next smallest decision

**Approve a common evidence-disposition contract through the existing
runner/report/browser seams, plus these four named public integration scenarios,
before choosing an uncertainty implementation or revisiting layout size?**

Recommended priority: status/identity/accounting integration first; then a
separately frozen Q26 support/process/method/calibration protocol. This does not
relax all-cell or exposure gates. Retain the accepted central nominal95%,
nominal2.5% per-tail target, simultaneous MC coverage LOWER>=94% and each
directional-miss UPPER<=3%, including theta<=0 false-uplift. Family confidence/
allocation, failure cap, outer budget, scenarios, centering/kernel/bandwidth and
any redesign remain unaccepted. No method is qualified. Future labels must say
nominal95%, tolerance-qualified only after passing; never exact95%/2.5% control.

## 12. Remaining calibration terms: ordered decisions, not qualification

Scope Q26-CALIBRATION-TERMS20, 2026-10-09. The latest
[canonical decision](../../project-state/decisions.md) **accepts the shared
disposition contract and four public integration oracles**, superseding section
11's pending-approval wording. It does not adopt a Q26 method, process class,
calibration confidence/budget or enlarged layout. Primary/fixed checkpoint
pairs/Q24 weights and nominal95% / coverage LOWER>=94% / directional
UPPER<=3% remain fixed. Q23 exposure/layout qualification remains unresolved.

### 12.1 First decide what must pass, what must reject, and what is outside the claim

The following is a proposed role registry to freeze before outcomes or method
tuning, with exact generator parameters still undecided.

| Role | Candidate contents | Required disposition |
|---|---|---|
| Required coverage: elementary baseline | Stationary fixed heterogeneous symbol means, independent calendar innovations and nonzero finite variance; zero and nonzero known weighted population effects | Meet every accepted coverage/directional gate. Set the population effect in the generator; never recenter realized panels to force it. |
| Required coverage: relevant dependence and class boundaries | Stationary shared calendar factors, finite-memory/stable autoregressive dependence, unequal variances, positive/negative cross-symbol dependence, actual horizon overlap, and sufficiently integrable volatility clustering/heavy tails | Declare moment conditions and bounded persistence/parameter ranges first. Strong but finite dependence within that class remains required even if the method fails. No postfailure exclusion. |
| Structural expected-rejection fixtures | Bad identity/pairing, incomplete windows, malformed kernel, nonfinite root or an explicitly predeclared unsupported-layout condition | Fail closed with retained reason/evidence, not an invented interval or zero. These are boundary-oracle checks, not required-coverage successes. |
| Out-of-class scientific diagnostics | Permanently random common panel level, changing symbol means/regime breaks, unit-root/excluded long-memory behavior or insufficient moments | Retain limitations, widths and errors descriptively. Passing cannot broaden the claim; failing cannot silently change a predeclared required scenario into this role. |

**Out-of-class is not synonymous with detectable.** A sparse real panel cannot
reliably expose every assumption violation, including persistent common shocks.
Require rejection for observable contract/operational defects; do not promise an
assumption detector that cannot identify them. A finite synthetic scenario grid
qualifies only its declared tested design/process points, not uniform validity
for every process in a broad class or for future markets.

[Shao (2010)](https://publish.illinois.edu/xshao/files/2012/11/JASA-DWB.pdf)
grounds stationary weak-dependence/moment and bandwidth restrictions, with
expanding-domain and sampling conditions in its irregular-location analysis.
It does not certify our finite stratified layout. The registry above is our
proposed adaptation, not a theorem inherited from that paper. Narrower supported
assumptions may permit honest qualification but reduce the scope of the claim.

### 12.2 Metadata and method freeze: what is still unknowable

An outcome-free qualified-layout audit must bind source/calendar/exposure
identities, eligible complete 60-bar warmup and 200-bar outcome windows, actual
symbol/session incidence, session distances/gaps, shared dates, horizon-window
overlap and fixed date-weight concentration. Review missingness/selection
assumptions for inference conditional on that layout. Metadata establishes design
facts, not stationarity, covariance decay or representative market sampling.

Also audit residual/date-aggregation ranks and each candidate covariance matrix.
The 505-session envelope is not 505 observations; the proposed 32 anchors leave
at most twenty residual coordinates after twelve symbol means, not twenty
independent observations. The actual eligible support is not yet qualified.
Consequently we cannot now freeze layout-specific bandwidths, operational support
rules, calibration cost or the complete confirmation scenario family. No numeric
support cutoff or sample enlargement is selected.

Symbol versus pooled centering has the opposing risks in sections 8 and 10.
Oracle-known generator means can diagnose centering error, not become a deployable
estimator. One additional exact boundary follows from the existing fixed weights:

```text
sum_u c[u] = sum_s,a q[s,a] * (x[s,a] - m_hat[s]) = 0
For Bartlett bandwidth much larger than occupied calendar span:
K -> all-ones, so c'Kc -> (sum_u c[u])^2 = 0.
```

Thus very short bandwidth may omit dependence, while very long bandwidth can
collapse centred variance; increasing it is not an automatic conservative rescue.
This is algebra, not an actual-layout failure finding. Actual exchange-session
distance, PSD/rank checks and finite candidate rules are necessary construction
checks, not sufficient coverage evidence. Freeze centering, kernel,
bandwidth-selection rule, interval construction, inner precision and any fixed
fallback as one operational method before confirmation. Parameter values remain
unaccepted; arbitrary wide intervals do not rescue identification. Report
width/power descriptively and define any usefulness criterion before confirmation.

### 12.3 Enumerate Monte Carlo checks before selecting precision

Let J count all one-sided qualification bounds across required
method x scenario x metric checks. A proposed auditable option is **95% Monte
Carlo family confidence with exact one-sided binomial limits and equal
Bonferroni allocation delta=0.05/J**. This confidence choice and limit/allocation
procedure are not yet accepted. Unequal predeclared allocations can also sum to
the selected family error budget. The
[NIST Bonferroni inequality](https://itl.nist.gov/div898/handbook/prc/section4/prc473.htm)
requires no independence between checks; independent identically distributed
outer attempts within each fixed scenario underpin each binomial bound.

The earlier 0.05/(4K) is valid only for exactly four checks per each of K
scenarios: coverage LOWER, two directional UPPER bounds, and a procedure-failure
UPPER bound, at the proposed MC family confidence. Additional methods, metrics,
separately certified declarations or confirmation attempts change J.

For interval [L,U] and known theta, define positive-direction miss as L>theta,
negative-direction miss as U<theta. Under theta=0, false positive uplift L>0
equals the first miss; under theta<0 it is a subset. That bound can therefore
cover the accepted no-uplift declaration requirement without duplicating a
check. If a separate false-uplift bound is certified, include it in J.
MC simultaneous confidence is neither the nominal market CI level nor correction
for selecting trading policies/checkpoints/endpoints.

### 12.4 Finite budgets and failure accounting

The proposed 0.1% procedure-failure ceiling and 10,000 outer attempts per scenario
remain **unaccepted**. For zero observed failures and one-sided allocation
delta, direct binomial inversion gives:

```text
U_failure = 1 - delta^(1/N)
For a proposed failure cap f: N >= ceil(log(delta) / log(1-f))
```

These are planning arithmetic, not simulation results. The
[NIST exact-binomial construction](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
supports inversion; positive failure counts require the full binomial CDF.
Zero observed failures never prove zero probability.

| Illustrative J, with MC family error 0.05 | Zero-failure upper at N=10,000 | Minimum N for upper<=0.1%, if zero failures |
|---|---|---|
| 4 | 0.04381% | 4,380 |
| 40 | 0.06682% | 6,682 |
| 400 | 0.08983% | 8,983 |

For illustrative J=40, zero-failure bounds at N=1,000 / 5,000 / 10,000 / 25,000
are 0.66623% / 0.13360% / 0.06682% / 0.02673%. Larger families or nonzero failures
demand more precision. This does not establish a feasible runtime budget.

At assumed true rates 0.95 and 0.025, pointwise Monte Carlo standard errors
sqrt(p*(1-p)/N) are, respectively:

| Outer N per scenario | Coverage MCSE, percentage points | Each nominal-tail MCSE, percentage points |
|---|---|---|
| 1,000 | 0.689 | 0.494 |
| 5,000 | 0.308 | 0.221 |
| 10,000 | 0.218 | 0.156 |
| 25,000 | 0.138 | 0.099 |

These are not simultaneous bound widths or qualification-pass probabilities.
Ten thousand is a planning candidate, not universally adequate. Choose fixed N
using the finalized J, rates/margins, failure cap, desired probability of correctly
qualifying a satisfactory procedure and separately authorized resource budget.

Count every attempted required-scenario panel. Invalid/missing intervals are
noncoverage; a conservative directional gate counts invalid OR directional miss
over all attempts. Report valid-interval conditional performance separately,
never as the only gate. Retain numerical failures and handled rejection reasons.
A malformed identity/accounting/kernel is a hard contract failure, not acceptable
merely because its frequency is below a numerical failure ceiling. Known
out-of-class and structural fixtures keep their frozen roles and denominators;
expected rejection there must not be pooled into supported-family coverage.

### 12.5 Ordered closure packet and next smallest question

[Morris, White and Crowther's ADEMP guidance](https://arxiv.org/abs/1712.03198)
supports explicit aims, generators, estimands, methods, performance measures and
Monte Carlo precision. Apply that discipline in the current evaluation owners:

| Order | Decision or prerequisite |
|---|---|
| 1 | Accept a supported-process statement and required-coverage / structural-rejection / out-of-class registry, including null/negative/positive effects. Exact ranges need review before any experiment. |
| 2 | Obtain qualified outcome-free Q23 metadata and declare operational layout-support checks without altering the sample automatically. |
| 3 | Freeze a finite development protocol and its complete operational candidate methods; develop on separately declared streams only. |
| 4 | Freeze the chosen method, confirmation registry, J, MC confidence/allocation, failure cap, outer/inner precision and manifests before independent confirmation. |
| 5 | Predeclare PRNG/version and disjoint development/confirmation stream identities, replication indices and stopping rule; confirm once under a separate run lease. |

No stream/seed cherry-picking, scenario dropping or relabelling after failure.
No fixed-N extension until green; any staged/sequential design needs its own
predeclared valid error accounting. Multiple fresh confirmations consume a
declared family/error budget rather than resetting confidence opportunistically.
Preserve failed confirmation and valid descriptive net results under the
accepted disposition contract; freeze a new protocol before a changed method.
Calibration does not clear exposure or authorize market outcomes.

**Next human question: shall we first freeze a bounded stationary,
fixed-symbol-mean repeated-market class with shared calendar dependence, and
separate required coverage, observable structural rejection and out-of-class
diagnostic roles, before choosing centering/bandwidth and confirmation precision?**

**Recommended answer: yes.** Include strong finite dependence inside the chosen
range as required, not a removable nuisance. Trade-off: an explicit narrower
conditional claim rather than an unsupported promise over arbitrary regime
changes/persistence. No exact parameter range is selected by this recommendation.
Next thereafter: MC family confidence/allocation, then failure ceiling and N
against the actual scenario/layout manifest. Accepted 95/94/3 stays unchanged;
no method or statistical qualification is claimed.

### 12.6 CONTINUATION22: outcome-free support and process checklist

2026-10-10 continuation. The human resumed Q26; approval of the bounded
stationary fixed-symbol-mean class and required-coverage / observable-rejection /
out-of-class roles is **pending**. This checklist extends the existing packet;
it selects no process range, centering, kernel, bandwidth, support cutoff or MC
term. Existing primary, Q24 weights and nominal95 / coverage LOWER94 /
directional UPPER3 remain unchanged.

**A needed distinction before ranges:** exact finite-lag dependence and bounded
stable persistence are not identical. An m-dependent construction has independent
collections separated beyond its declared lag; stable AR dependence generally
decays without vanishing at a finite lag. Section 12.1 lists both as candidates,
not as equivalent classes. A compact multiplier kernel does not imply that
market innovations have finite dependence. The pending process statement must
disambiguate those alternatives; neither is adopted by this continuation.

| Checklist item | Verifiable from separately permitted, source-bound metadata | Not established by that check |
|---|---|---|
| Source and calendar identity | Dataset/panel/calendar digests, provider provenance, symbols, resolution, adjustment policy, date bounds and exposure-certificate identities | Access permission for another source or absence of undocumented manual/external exposure |
| Calendar geometry | Reviewed exchange-session ordinals, occupied dates, gaps, shared-date symbol incidence and local decision offsets | Stationarity, covariance decay, representative sampling or independent dates |
| Complete eligibility | Current canonical `[start-60,start+200)` span; continuous minute timestamps; one symbol/session; RTH open/close, early-close, partition and historical-window exclusion checks | Adequacy from row counts, purge size or a nominal session label alone |
| Fixed Q24 support | Actual anchor counts per symbol, fixed weights/date contributions, interval overlap, concentration and later distance-law support neighborhoods | Effective sample size from anchors, horizons, seed pairs or 505 calendar sessions |
| Exposure versus initialization | Retained training/tuning/validation/manual exposure history separately from checkpoint ancestry/reset identity | Fresh weights erase historical research exposure or chronology alone makes a panel fresh |
| Missingness and selection | Outcome-free frozen selection/exclusion rules and retained reasons for unavailable sites or unknown exposure | Ignorable missingness, representativeness or permission to replace missing outcomes by zero |

These are **audit prerequisites, not findings that the actual layout passed**.
[Panel qualification](../../../itrader/rl_lib/evaluation/panel_qualification.py)
is designed to project `symbol/ts/session/session_id` rather than economic
columns. Its retained-proof checks do not independently establish the truth of
every historical source/start assertion. No loader, dataset projection or
qualification routine was executed in this research scope. Use existing
partition/qualification/report seams, not another parallel evaluator.

The [metadata inventory](saluki-runtime-inventory-2026-10-10.md) reports
dataset-v2 beginning 2022-01-03. The approved 2020-2021 candidate envelope is
outside that source. A different source needs its own identity, authorized access
and exposure/metadata qualification. Relabelling a split, restarting a model or
using dataset-v2's history does not supply those receipts. No acquisition or new
data access is authorized. `terminal_screen_only` evidence remains excluded from
proposer reward, curriculum, PPO, tuning and operational capability calibration.

**Conditional range-manifest proposal, not values:** only after process/role
approval, specify fixed symbol means and known null/negative/positive population
effects; positive innovation scales and heterogeneity; shared/idiosyncratic
factor loadings and valid joint covariance; exact dependence lag **or** bounded
decay/persistence assumptions; tail/moment and volatility-dynamics restrictions;
within-anchor horizon/pair dependence; and frozen layout/missingness incidence.
Declare finite stress points including strong dependence inside approved bounds.
Never recenter realized panels to force the population target or relabel a
failed required point as out-of-class. Finite parameter bounds do not mean
bounded observed returns, and a finite tested grid is not uniform qualification
over every parameter between its points. No numeric ranges are chosen here.

**What centering/kernel/bandwidth decisions actually need next:**

1. Source/exposure-qualified frozen layout, actual counts/gaps/overlap and Q24
   weights, not merely a calendar envelope.
2. Approved process interpretation, moment assumptions and three-role registry,
   followed by the finite scenario manifest.
3. Explicit deployable centering candidates and correction rules; oracle-known
   generator means remain diagnostic, never available market estimators.
4. Declared calendar-distance law/units, finite kernel/bandwidth candidates,
   support/numerical checks, selection rule and fixed fallback/disposition.
5. Design-only projection/date-aggregation ranks and candidate-kernel PSD/support
   checks. These can be derived without returns; realized residuals, empirical
   covariance, positive estimated variance and coverage cannot. No such matrix
   calculations were run here.
6. Separately authorized synthetic development, then a frozen operational method
   and independent confirmation protocol. MC family/allocation, failure cap and
   inner/outer precision remain open; no simulation is authorized by this note.

The section 12.2 centering algebra still applies: an almost-common multiplier can
collapse variance because centred contributions sum to zero. Larger bandwidth
or higher rank is not automatically conservative or an effective-sample-size
certificate. Sparse metadata cannot prove the selected covariance class.

**Next decision remains the pending process-class/three-role question**, with
exact finite-lag versus bounded decaying persistence made explicit. If approved,
the next packet is the qualified support and finite scenario manifest, before
operational centering/kernel/bandwidth and confirmation terms. If actual support
cannot qualify the chosen claim, retain valid descriptive results and unresolved
inference; any redesign needs separate approval. Fresh chronological training
would be a new protocol, not automatic freshness clearance or permission to
change the accepted panel.

### 12.7 CLOCK27: exchange-session versus elapsed-day distance

2026-10-11 Europe/Madrid; source/document investigation only.
**Decision reconciliation:** the [canonical Oct11 decision](../../project-state/decisions.md)
supersedes sections 12.1/12.5/12.6's pending process/role/subtype labels.
The bounded fixed-symbol-mean claim and required-coverage / observable-rejection /
out-of-class roles are accepted; **both exact finite-lag and bounded stable-decay
families are mandatory**. Neither may be dropped or relabelled after failure.
Central equal-tailed nominal95%, simultaneous MC coverage LOWER>=94% and each
directional-miss UPPER<=3% remain fixed. Investigation of both clocks is approved,
not either clock or a complete method. Historical proposals above are preserved.

#### Target and four separate choices

**Source fact:** the current target is session-safe RTH paired terminal net,
including liquidation, under fixed Q24 horizon -> anchor -> symbol ->
equal-symbol weights and the two frozen checkpoint pairs. After averaging
horizons/pairs, each anchor weight is `q[s,a]=1/(12*n_s)`. Changing distance
must not change this point estimate, create observations or reweight symbols.
[Q24 reporter](../../../itrader/rl_lib/evaluation/terminal_panel_report.py),
[panel qualification](../../../itrader/rl_lib/evaluation/panel_qualification.py).

Keep four manifest choices distinct: **DGP clock** indexes synthetic innovations;
**inference distance** compares observed sites; **multiplier correlation kernel**
maps that distance to a valid covariance; **bandwidth** sets its scale/support
in declared units. DGP persistence is not automatically estimator bandwidth.
A compact multiplier kernel does not imply finite-lag market dependence.
Calendar coordinates are not regime, news or outcome labels.

Let `S(d)` count every actual session in the complete reviewed exchange calendar,
not only selected anchors; let `D(d)` be the ordinal of the exchange-local civil
trading date. Candidate distances are `|S(d_i)-S(d_j)|` and
`|D(d_i)-D(d_j)|`. Civil-day distance is not elapsed UTC hours divided by 24;
timezone/DST conventions would make that a separate clock.

| Issue | Full exchange-session ordinal | Elapsed civil-calendar-day ordinal |
|---|---|---|
| Ordinary Friday -> Monday | One session step | Three days |
| Holiday/closure | No extra open-session step | Closed days increase distance |
| Unselected open sessions | Preserve every skipped session; never compress selected anchors | Preserve literal date gaps |
| Interpretation / advantage | Dependence per trading opportunity; matches current RTH sampling seam | Dependence can decay during closures; distinguishes long closed gaps |
| Assumption / limitation | Does not prove dependence pauses when the exchange closes | Does not prove daily decay or measure closed-day strategy returns |
| Required identities | Complete reviewed exchange calendar, date mapping and units | Canonical local dates/units; exchange calendar still needed for eligibility |

**Cited fact:** NYSE documents core hours, holidays and early closes.
[NYSE calendar documentation](https://www.nyse.com/trade/hours-calendars).
An early-close session still occupies one session ordinal; whether a complete
window fits is a separate eligibility question. These public rules do not bind
the actual historical calendar or certify panel support; no calendar is acquired.

#### Symbolic constructions under both clocks, not adopted generators

The following are **mathematical examples/assumptions**, not findings about market
dependence or numerical scenario choices. Innovations are independent and
mean-zero with declared variance; stationary initialization and nonzero finite
scales are necessary; the MA coefficient vector must be nonzero. Evaluate
processes only at selected coordinates, without compressing their gaps.

| Mandatory family | Session-clock example | Calendar-day example |
|---|---|---|
| Exact finite-lag | `F_s=sum(j=0..m) a_j*epsilon_(s-j)`: innovation sets are disjoint beyond m session steps. | `F(t)=[B(t)-B(t-L)]/sqrt(L)`: Brownian increments give covariance `(1-abs(t-u)/L)_+` and independence beyond L days. Sample at `t=D(d)`; start the stationary-increment construction before the first required coordinate. |
| Bounded stable decay | Stationary AR(1), `F_s=rho*F_(s-1)+epsilon_s`, with `abs(rho)<=rho_max<1`; correlation `rho^abs(delta S)`. | Stationary OU covariance `sigma^2*exp(-lambda*abs(delta D))`, with `lambda>=lambda_min>0`; decay continues over closed days by assumption. |

[MIT Brownian-motion notes](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/3b97c6b0c282dd9dc024c4c7ffe3fba8_MIT18_S096F13_lecnote17.pdf)
establish independent disjoint increments; the triangular covariance above is
our overlap-length derivation.
[GPML chapter4](https://gaussianprocess.org/gpml/chapters/RW4.pdf) supplies
exponential/OU covariance and the positive-semidefinite requirement.
Latent closed-day innovations are not imputed trading outcomes. Gaussian examples
bound persistence, not realized values; tails/moments and richer negative or
oscillatory dependence remain open. Zero covariance alone would not establish
finite independence for an arbitrary non-Gaussian process.

A shared factor with declared symbol loadings and joint idiosyncratic terms can
preserve same-date coupling without making all symbols identical. Retain
within-anchor horizon/pair dependence. A valid DGP covariance and a valid
multiplier covariance are separate requirements; arbitrary hard truncation of
an exponential need not be PSD. Neither these examples nor a finite grid
qualifies the whole supported class uniformly.
These examples vary construction as well as clock; an isolated clock comparison
must hold the latent construction fixed and change only its coordinates.

#### Support, calibration and existing software owners

**Cited support / limit:** [Shao's DWB paper](https://publish.illinois.edu/xshao/files/2012/11/JASA-DWB.pdf)
addresses stationary weak dependence and irregular sampling under stated
asymptotic conditions. It motivates distance-aware treatment, not a theorem
selecting our clock or validating this finite stratified panel.

**Local inference:** same-date symbols have distance zero under both clocks;
the proposed interface keeps one synchronized date multiplier across symbols,
horizons and fixed pairs, not an already implemented or qualified procedure.
Resolve canonical trading date rather than grouping symbol-bearing
`session_id` values as different dates. Sparse anchors change neighborhoods under
each clock; unselected sessions are coordinate gaps, not zero observations.
Neither a 505-session envelope, extra latent days nor more draws adds observed
independent support. Qualified metadata must later bind date incidence, gaps,
overlaps, fixed weight concentration, units and kernel PSD/rank. No layout
matrix or support audit is computed here.

Keep the proposed interface inside existing evaluation owners: qualification /
provenance binds reviewed coordinates/calendar identities; an uncertainty
adapter receives frozen paired contributions/layout; the reporter retains Q24
reduction; disposition preserves valid descriptive estimates if inference cannot
qualify. [Legacy block uncertainty](../../../itrader/rl_lib/evaluation/block_uncertainty.py)
uses per-symbol positional-row blocks and is explicitly exploratory; relabelling
it as a calendar method would be incorrect. No new evaluator, loader or source
implementation is introduced.

A candidate `K_ij=k(distance(i,j)/b)` needs declared units and PSD checks.
The same numeric b in sessions and days is not an equal-persistence comparison.
Changing the clock changes neighborhoods and the method identity; later
cross-clock DGP/inference cases must have frozen roles and units. Both mandatory
families retain all required coverage/miss gates, including strong cases inside
declared bounds. More clock/method checks affect the eventual simultaneous MC
allocation; confidence, ranges/moments, bandwidths and budgets remain open.
Section12.2's almost-common-multiplier centering collapse applies under either
clock: a larger bandwidth is not automatically conservative.

**Limitation:** these windows end intraday and liquidate; they do not observe
a held position's close-to-next-open return. Weekend news may influence later
intraday outcomes, but its strength and decay clock are unverified assumptions.
Neither elapsed days nor session distance can establish overnight/weekend
portfolio risk from this endpoint. No prices/outcomes are inspected.

Proposed public oracles, not executed tests: **ClockGapIdentity** checks literal
holiday/unselected-session distances and ordering invariance; **SameDateCoupling**
maps distinct symbol sessions to one date coordinate; **FixedTargetInvariant**
retains Q24 weights/point estimate under both clocks; **ClosureAndDisposition**
keeps latent closed days out of RTH observations and rejects missing identities,
unit conflicts or invalid covariance without deleting valid descriptive evidence.

#### One next decision for GRILL

**Shall full exchange-session ordinals be the first inference-clock candidate,
with both mandatory dependence families tested under each DGP clock in the
later frozen calibration registry?**

**Recommendation: conditional yes.** Session distance matches the current
intraday opportunity seam; elapsed-day DGP cases explicitly challenge that
assumption rather than declare closures irrelevant. Freeze their required roles
before development/confirmation; feasibility depends on separately qualified
support and finite ranges. If both inference clocks are retained as candidate
methods, predeclare their selection/comparison and confirmation accounting too;
do not select whichever is green after confirmation. This recommendation adopts
no clock, numerical range, centering, kernel, bandwidth, MC term or method.
Unresolved inference remains descriptive, not a reason to discard failed required
cases or broaden the claim. No loaders, acquisition, simulation or heavy work.
