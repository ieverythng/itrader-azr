# Q26 synthetic-calibration boundary

Date: 2026-10-07, Europe/Madrid

Scope: outcome-blind candidate calibration after metadata-first envelope design

Status: **candidate research contract only; no Q26 method, calendar unit, block
length or grid, interval construction, alpha, coverage target, simulation seed,
draw count, support threshold, multiplicity family, or acceptance tolerance is
selected here**

## Outcome

Q26 can be investigated before any terminal market outcome is opened by applying
candidate inference procedures to synthetic paired outcomes laid onto the exact
metadata-only Q23 calendar geometry. The null study has three distinct jobs:

1. estimate empirical Type-I error for a predeclared test;
2. estimate interval coverage for a predeclared confidence procedure; and
3. expose numerical, support, and conclusion instability under predeclared
   serial, cross-symbol, within-anchor, and heavy-tail dependence families.

A separate nonzero-effect study may estimate simulated power. It must be named
and reported separately. Synthetic power is **not** evidence that a solver is
profitable, that the chosen effect size is realistic, or that market outcomes
will resemble the generator.

The calibration is a falsification exercise for candidate Q26 procedures, not a
search for a route to significance. Every predeclared null family and every
failure remains in the record. A procedure that fails a null-family gate cannot
be rescued by deleting that family, shortening blocks until it passes, changing
the panel dates, or pointing to attractive simulated power.

This boundary implements the approved order:

```text
metadata/access/exposure evidence
              |
              v
explicit acquisition-envelope declaration and review
              |
              v
outcome-blind calendar/layout object
              |
              v
synthetic null calibration ----> separate simulated-effect power
              |
              v
human Q26 decision and sealed analysis contract
              |
              v
only then: terminal outcome access
```

No OHLCV value, realized return, policy action, checkpoint score, or terminal
outcome was read or generated from market data for this note.

## Authority and current decision boundary

The accepted Q24 point reducer is already fixed: four horizons within each
anchor, anchors within each symbol, then twelve symbols equally. Candidate and
control are joined by immutable task identity and subtracted before uncertainty
is assessed. The current
[`terminal_panel_report.py`](../../../../itrader/rl_lib/evaluation/terminal_panel_report.py)
implements the descriptive reducer and deliberately leaves uncertainty
unscored.

The existing
[primary-source statistical note](../evaluation-quant-methodology-2026-10-06-evidence/statistical-protocol-primary-sources.md)
establishes that:

- the four horizons are overlapping measurements of one anchor, not four
  independent paths;
- nearby anchors carry serial calendar dependence;
- symbols exposed to the same calendar shocks cannot be resampled independently;
- every valid replicate must recompute the exact Q24 hierarchy; and
- two training seeds remain two training seeds, regardless of the number of
  episodes or synthetic/bootstrap draws.

This note narrows only the synthetic-calibration seam. It does not select which
Q24 contrasts are confirmatory, select a Q26 method, qualify the eventual Q23
layout, or authorize acquisition or terminal evaluation.

## Information boundary

### Permitted inputs

The calibration harness may consume only:

- an immutable metadata-only layout identity;
- the declared twelve-symbol universe and four-horizon set;
- anchor/session identities and complete `[start-60, start+200)` UTC/session
  windows;
- merged or alternative candidate calendar-unit assignments derived without
  market values;
- the exact fixed Q24 weights and candidate-control pairing schema;
- certified role, dataset, calendar, eligibility, and exposure-ledger hashes;
- a predeclared synthetic scenario registry; and
- a predeclared candidate-method registry.

The layout object may describe session gaps, early closes, window intersections,
symbol coverage, component influence, and complete-support feasibility. It must
not contain price fields, realized returns, model outputs, action traces,
human-inspected charts, or any statistic derived from them.

### Forbidden inputs and consumers

The harness must fail closed if supplied with real market outcomes or model
results. Its outputs must not enter:

- solver observations, policy/value loss, `RewardEngine`, or PPO rewards;
- task generation, proposer/curriculum selection, `r_prop`, or checkpoint
  selection;
- panel-date or acquisition-envelope optimization;
- online `MarketFramer`, `DecisionGate`, `RiskEngine`, or `Executor` decisions;
- an operational or calibrated `CapabilityProfile`; or
- the old sealed final test.

Passing synthetic calibration is statistical-method evidence only. It is not a
capability, promotion, profitability, freshness, entitlement, or execution-safety
certificate.

## Exact estimand carried through every simulation

For a fixed checkpoint and one predeclared candidate-control contrast, let the
synthetic paired terminal-net difference at symbol `s`, anchor `a`, and horizon
`h` be `d[s,a,h]`. With the accepted horizon set `H`, frozen anchors `A_s`, and
twelve symbols `S`, every candidate procedure must target:

```text
anchor_delta[s,a] = (1 / |H|) * sum_h d[s,a,h]

symbol_delta[s] = (1 / |A_s|) * sum_a anchor_delta[s,a]

theta = (1 / 12) * sum_s symbol_delta[s]
```

The synthetic generator may construct paired candidate/control rows and derive
`d`, or generate `d` directly for a narrow statistical unit test. The
end-to-end calibration path must exercise the immutable join, complete support,
and exact Q24 reducer. It may not replace `theta` with a raw episode mean,
silently reweight symbols, treat horizons as independent, or renormalize an
incomplete replicate.

Q25 net-bps-per-bar and any Sharpe, horizon, regime, or action diagnostic remain
outside the Q26 confirmatory target unless a later human decision explicitly
adds them to the multiplicity family. A Q26 procedure for Q24 cannot be
validated by showing calibration for a different statistic.

## Synthetic data-generating contract

The simulation study should follow an explicit aims, data-generating mechanism,
estimand, methods, and performance-measures plan. Morris, White, and Crowther
recommend this structured approach and require Monte Carlo uncertainty to be
reported for simulation performance estimates
([Morris, White, and Crowther 2019](https://doi.org/10.1002/sim.8086)).

### Null truth

Every null scenario must have a declared true Q24 effect `theta = 0`, either by
construction in each synthetic population or in expectation under a documented
generator. The truth declaration belongs in the scenario record. The harness
must not infer the truth from its realized synthetic sample.

Where the generator uses unequal symbol, anchor, horizon, or component loadings,
the centering step must be expressed on the exact Q24 scale above. A zero raw-row
mean is not sufficient when the Q24 weights are unequal.

### Minimum predeclared dependence axes

The locked null registry must cover, separately and in combinations, synthetic
families capable of expressing:

| Axis | Required property | Failure the axis is intended to expose |
|---|---|---|
| Serial calendar dependence | Persistent common and/or idiosyncratic component shocks over ordered sessions/components | Blocks too short, invalid IID anchors, fragile HAC bandwidths |
| Cross-symbol dependence | Same-calendar common-factor shocks plus symbol-specific loadings and residual dependence | Independent-per-symbol resampling and false effective sample size |
| Within-anchor horizon dependence | All four horizons inherit shared anchor shocks with declared horizon loadings/covariance | Treating 128 rows as independent rather than 32 anchors |
| Heavy tails and asymmetry | Non-Gaussian innovations with predeclared tail/skew families | Non-finite behavior, undercoverage, unstable studentization |
| Scale heterogeneity | Symbol/component volatility differs without changing the Q24 null | Dominant components and invalid homoskedastic approximations |
| Combined stress | Serial, common-factor, horizon-overlap, heavy-tail, and scale features coexist | Procedures that pass only isolated, easy nulls |

The exact stochastic families, parameter grids, covariance/loadings, persistence,
tail parameters, and scale ratios are **not chosen here**. They must be declared
and reviewed before a calibration batch starts. Parameters must come from a
scientific stress rationale or metadata geometry, never from terminal returns.
No generator may be fitted to the unopened panel's outcomes.

Block methods were developed to preserve adjacent weak dependence rather than
pretend observations are IID
([Kunsch 1989](https://doi.org/10.1214/aos/1176347265);
[Politis and Romano 1994](https://doi.org/10.1080/01621459.1994.10476870)).
Their validity remains conditional on assumptions that synthetic success cannot
prove for the market. Fixed-block and subsampling performance can depend
materially on bandwidth; fixed-b calibration makes that dependence explicit
([Shao and Politis 2013](https://publish.illinois.edu/xshao/files/2012/11/ShaoPolitisJRSS.pdf)).

### Layout fidelity

Synthetic values are overlaid onto the actual metadata geometry rather than an
idealized balanced time series. The generator must preserve:

- exact anchor counts and unequal per-symbol anchor counts, if present;
- exact session order, gaps, early-close labels, and window intersections;
- all four horizons as one anchor-owned group;
- the selected candidate calendar components without moving their dates;
- twelve-symbol support and fixed Q24 weights; and
- the distinction between market-sample dependence and training-seed identity.

Simulation may reveal that no predeclared candidate procedure qualifies on a
frozen layout. It may not optimize individual anchor dates or shrink the
acquisition envelope to make a method look stable. The neutral permitted result
is `Q26_UNRESOLVED_FOR_LAYOUT`; method-specific failures stay attached to those
methods and do not by themselves prove the layout inadequate. Any broader or
restructured envelope requires a new explicit declaration and review before the
complete simulation protocol is rerun.

## Candidate method registry: comparison, not adoption

A later locked registry may compare some or all of the already documented
families:

- paired synchronized moving, circular, or stationary calendar-block bootstrap;
- a fixed-support synchronized block multiplier;
- calendar-block subsampling or fixed-b calibration;
- a positive-semidefinite HAC sensitivity analysis; and
- chronology-group t inference only if metadata supports non-overlapping,
  sufficiently separated, complete-support groups.

Listing a family does not adopt it. Each registry entry must bind a versioned
implementation, interval/test construction, calendar unit, block-selection rule
or grid, studentization and centering rule, edge-case policy, and applicable
contrast/multiplicity scope. No entry may silently fall back to IID anchors or
independent symbol resampling.

If a multiplicity procedure is later requested, it requires the complete
predeclared candidate/metric/control family and the same valid synchronized
dependence law. White's Reality Check, Hansen's SPA, and Romano-Wolf StepM solve
different familywise questions; none repairs missing trials or an invalid
resampling unit
([White 2000](https://doi.org/10.1111/1468-0262.00152);
[Hansen 2005](https://doi.org/10.1198/073500105000000063);
[Romano and Wolf 2005](https://doi.org/10.1111/j.1468-0262.2005.00615.x)).

## Null-calibration outputs

For every method x null-family x parameter-cell combination, retain at least:

- total requested synthetic replications;
- analyzable replications;
- empirical Type-I error, where rejection means the predeclared test rejects
  `theta = 0` (or the predeclared confidence interval excludes zero);
- empirical confidence-interval coverage of the true `theta = 0`;
- Monte Carlo standard errors for Type-I error and coverage estimates;
- interval width and endpoint summaries, labelled secondary;
- bias and dispersion of the point/standard-error estimates where defined;
- numerical-failure, support-failure, and invalid-replicate counts by reason;
- sensitivity across every predeclared block/bandwidth/parameter cell;
- repeated-battery stability under the predeclared independent RNG streams; and
- the complete method/scenario/layout/code identities.

For `M` analyzable replications and an observed proportion `p_hat`, the basic
binomial Monte Carlo standard-error diagnostic is:

```text
MCSE(p_hat) = sqrt(p_hat * (1 - p_hat) / M)
```

This diagnostic does not choose `M`, the desired precision, alpha, or the target
coverage. Those values require prospective human approval. A small reported
MCSE also does not cure a misspecified synthetic family.

Results must be reported at the individual null-family and parameter-cell level.
An average across scenarios may be supplemental but cannot hide one failing
family. Worst-case and failure tables stay visible.

## Separate simulated-effect power study

Power uses the same frozen layout, method implementation, and dependence
families, but replaces the null truth with one or more predeclared nonzero Q24
effects. The effect grid, heterogeneity pattern, and sidedness are not selected
here and must not be derived from terminal outcomes.

Required labels and limits:

- call the output `simulated_effect_power`, never solver power, expected alpha,
  expected return, or probability of profitability;
- report rejection probability and Monte Carlo uncertainty by effect/scenario;
- keep null Type-I error, null coverage, and failure results in separate tables;
- never use power to override failed null calibration; and
- do not select a method solely because it rejects more often under a convenient
  simulated effect.

A power study can expose that a well-calibrated method is too insensitive for a
particular hypothetical effect. It cannot establish that the hypothetical
effect exists or that collecting a terminal panel is economically worthwhile.

## Anti-tuning and retention rules

The following are hard research-integrity rules:

1. Freeze and hash the layout, scenario registry, method registry, performance
   measures, RNG-stream plan, and acceptance/failure rules before a batch.
2. Retain every attempted method, scenario, replication failure, and null-family
   result. A failure may be marked as caused by a confirmed implementation bug,
   but the original receipt is superseded, not deleted.
3. Do not stop when a desired rejection rate, coverage result, interval width,
   or significance pattern appears.
4. Do not alter calendar dates, drop difficult symbols/components, change the
   null family, or shorten blocks after seeing calibration results.
5. Do not choose the narrowest interval or highest-power method without first
   satisfying all predeclared null-calibration and support gates.
6. Any method change after inspecting a batch creates a new version. It cannot
   be called confirmed on the same battery; a later human-approved, locked
   confirmation battery is required.
7. Autoresearch may later optimize implementation mechanics such as runtime or
   memory on fixed synthetic fixtures. It may not optimize dates, alpha,
   thresholds, block choices, scenario membership, or statistical conclusions.

These rules prevent the synthetic stage from becoming another adaptive search
whose failures disappear from the denominator.

## Fail-closed boundary

The eventual human-approved contract must define numerical tolerances. Until it
does, no candidate passes. Once those values are sealed, a candidate remains
unqualified if any of the following occurs:

- any predeclared null-family Type-I or coverage requirement fails;
- Monte Carlo uncertainty is too large to adjudicate the declared tolerance;
- technical or support failures exceed the declared policy;
- a replicate loses one of the twelve symbols or four horizons, silently
  renormalizes weights, or breaks candidate-control pairing;
- same-calendar/overlapping cross-symbol windows do not share the required draw
  or multiplier under a synchronized method;
- results depend qualitatively on plausible predeclared block specifications and
  the contract supplies no outcome-blind selection rule;
- a component dominates enough to break the declared calibration criterion;
- a null family, failed method, or invalid replication is omitted from the
  retained ledger;
- the layout, method, code, scenario, or RNG identity differs from the sealed
  manifest; or
- real outcomes, checkpoint scores, or observed policy behavior influenced any
  design choice.

If no candidate qualifies, the honest result is that Q26 remains unresolved.
Q24 may then be reported only at the descriptive level permitted by the later
human decision; no method should be loosened until it produces a favorable
terminal claim.

## Required receipts

The future harness should produce immutable, machine-readable receipts with at
least these identities:

| Receipt | Minimum contents |
|---|---|
| `layout_manifest` | role/dataset/certificate/calendar/exposure hashes; symbols; anchors; horizons; UTC/session windows; calendar components; exact Q24 weights |
| `scenario_registry` | null/effect status; generator family and parameters; true Q24 effect; dependence axes; scenario version/hash |
| `method_registry` | implementation/code hash; method family; interval/test definition; block/bandwidth rule; multiplicity scope; method version/hash |
| `calibration_plan` | performance measures; alpha/coverage target; tolerances; requested replications; RNG scheme/seeds; stop/failure rules; plan hash |
| `replication_ledger` | one row per requested replication, including failed/invalid status and reason; no success-only ledger |
| `calibration_summary` | scenario-level Type-I error, coverage, MCSE, stability, widths, failures, support diagnostics, and pass/unqualified state |
| `power_summary` | separately named effect/scenario power with MCSE; never merged into null qualification |

Hashing records identity and later drift; it does not make a weak design valid,
prove non-exposure, or certify that an older panel is fresh.

## Decisions that remain explicitly open

Before any calibration run can qualify a method, humans must still approve:

1. the sole confirmatory Q26 estimand and complete contrast/multiplicity family;
2. the calendar unit derived from the eventual metadata-only layout;
3. candidate method families and exact versioned implementations;
4. null generators, parameter grids, combined stresses, and any independent
   confirmation battery;
5. block length/selector/grid, bandwidths, studentization, and interval/test
   construction;
6. alpha, target coverage, Type-I/coverage tolerance, stability rule, and
   technical/support failure policy;
7. simulation replication count, Monte Carlo precision, RNG algorithm/streams,
   and fixed seeds;
8. nonzero effect grid and the limited purpose of the power study; and
9. the explicit statement that two training seeds remain a mechanics/attribution
   screen rather than population-level algorithm inference.

The immediate next decision should remain upstream and singular: approve the
historical exposure-clearance/support standard for the proposed metadata-only
acquisition envelope. Q26 numerical choices should follow the resulting
calendar-layout evidence, not be guessed in advance.

## Primary sources

- Agarwal, Schwarzer, Castro, Courville, and Bellemare (2021),
  [*Deep Reinforcement Learning at the Edge of the Statistical Precipice*](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf).
- Hansen (2005),
  [*A Test for Superior Predictive Ability*](https://doi.org/10.1198/073500105000000063).
- Kunsch (1989),
  [*The Jackknife and the Bootstrap for General Stationary Observations*](https://doi.org/10.1214/aos/1176347265).
- MacKinnon, Nielsen, and Webb (2023),
  [*Cluster-Robust Inference: A Guide to Empirical Practice*](https://doi.org/10.1016/j.jeconom.2022.04.001).
- Morris, White, and Crowther (2019),
  [*Using simulation studies to evaluate statistical methods*](https://doi.org/10.1002/sim.8086).
- Politis and Romano (1994),
  [*The Stationary Bootstrap*](https://doi.org/10.1080/01621459.1994.10476870).
- Romano and Wolf (2005),
  [*Stepwise Multiple Testing as Formalized Data Snooping*](https://doi.org/10.1111/j.1468-0262.2005.00615.x).
- Shao and Politis (2013),
  [*Fixed-b Subsampling and the Block Bootstrap: Improved Confidence Sets Based on P-value Calibration*](https://publish.illinois.edu/xshao/files/2012/11/ShaoPolitisJRSS.pdf).
- White (2000),
  [*A Reality Check for Data Snooping*](https://doi.org/10.1111/1468-0262.00152).

## Fact / inference / decision ledger

- **Verified local facts:** the exact Q24 reducer is implemented; Q26 remains
  unscored; the planned screen has only two training seeds; and current Q23/Q26
  documents require outcome-blind metadata design before terminal access.
- **Primary-source facts:** block methods address dependent sequences under
  assumptions; bandwidth affects finite-sample inference; simulation performance
  estimates have Monte Carlo uncertainty; and multiplicity procedures require a
  declared family and valid dependence law.
- **Research recommendations:** overlay synthetic data on the exact calendar
  geometry, require serial/cross-symbol/horizon/heavy-tail null families, retain
  every failure, and separate null calibration from simulated-effect power.
- **Human decisions still required:** every numerical setting and acceptance
  gate listed above, plus the final Q26 target, method, and multiplicity family.
- **Not established:** that any candidate method calibrates on the eventual
  layout, that the synthetic families match real markets, that two seeds support
  algorithm-level inference, or that any solver is profitable.
