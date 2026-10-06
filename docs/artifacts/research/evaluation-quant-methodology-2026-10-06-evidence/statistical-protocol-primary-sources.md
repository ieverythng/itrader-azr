# Statistical protocol for paired R0/R4/R1 evaluation

Date: 2026-10-06
Scope: primary-source research note for Q21/Q23/Q26
Status: evidence and candidate designs only; **no Q26 method, block length,
confidence level, alpha, or promotion threshold is adopted here**

## Executive conclusion

The Q24 point estimand is already well defined and should not be reopened:
average the four crossed horizons within each market anchor, anchors within
each symbol, and then the twelve symbols equally. The corresponding paired
candidate-minus-control contrast uses the same weights. Q25's
`net_bps_per_bar` is a separately labelled diagnostic that normalizes each
episode before applying that same hierarchy. The current implementation
correctly leaves Q26 uncertainty unscored.

The most defensible Q26 candidate is a **paired, synchronized
calendar-block resampling procedure that recomputes the exact Q24 hierarchy in
every draw**. All four horizons stay attached to their anchor; all symbols whose
input/outcome windows occupy the same calendar block receive the same block
draw or multiplier. Resampling symbols independently would destroy market-wide
co-movement. This is a recommendation, not an adopted method: the procedure
must be calibrated against the exact outcome-independent Q23 layout, and it
must fail closed when the layout yields too few or excessively influential
calendar blocks.

Two training seeds cannot support an inferential claim about the population of
training runs. Report both seeds, their paired effects, their sign agreement,
and a descriptive mean/range. Calendar-block uncertainty may quantify market-
sample uncertainty **conditional on each trained checkpoint**, but thousands of
bootstrap draws cannot manufacture training-seed replication. Deep-RL
evaluation research likewise stresses that finite-run aggregate metrics carry
material uncertainty; it does not turn two runs into an adequate seed sample
([Agarwal et al., 2021](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf);
[Henderson et al., 2018](https://ojs.aaai.org/index.php/AAAI/article/download/11694/11553)).

PSR/DSR, HAC, block bootstrap, Reality Check, and SPA solve different problems:

- PSR and DSR are Sharpe-ratio procedures. They do not provide uncertainty for
  Q24 terminal net return, and current crossed, overlapping episodes cannot be
  stitched into a valid portfolio return series.
- HAC estimates a long-run variance under weak-dependence asymptotics. It is a
  useful analytic sensitivity check, but with roughly 32 anchors it is highly
  bandwidth- and small-sample-sensitive and does not correct model search.
- Moving/stationary block bootstrap can target the actual nonlinear hierarchy
  while retaining dependence, but block construction and effective support are
  first-class design choices.
- White's Reality Check and Hansen's SPA test whether any member of a declared
  candidate family beats a common benchmark after data snooping. They require a
  complete candidate family and the same dependence-preserving resampling law;
  they neither create seed replication nor repair an exposed test panel.

## 1. Authority and verified local facts

These statements are verified from current source and active documentation,
not inferred from the literature:

1. [`terminal_panel_report.py`](../../../../itrader/rl_lib/evaluation/terminal_panel_report.py)
   implements Q24 as four horizons within anchor, anchors within symbol, and
   twelve symbols equally. It preserves the raw 128-episode mean separately.
2. The same reporter computes Q25 as
   `10000 * episode_net_return / actual_steps` **before** hierarchical
   aggregation. It rejects censored/missing/duplicate horizons rather than
   renormalizing incomplete evidence.
3. Its output explicitly says
   `uncertainty: not_scored_requires_equal_symbol_calendar_block_method`.
4. [`block_uncertainty.py`](../../../../itrader/rl_lib/evaluation/block_uncertainty.py)
   contains exploratory development-panel utilities, not Q26:
   `paired_block_interval` resamples row buckets independently inside each
   symbol and averages episodes globally; `paired_anchor_interval` treats
   anchors as exchangeable; and `pooled_paired_anchor_interval` resamples the
   observed training seeds before anchors. None reproduces Q24 equal-symbol
   weighting or synchronized cross-symbol calendar dependence.
5. The proposed screen is four representations (`R0`, `R0 + R4-v1`, `R1`,
   `R1 + R4-v1`) by two training seeds by two curricula. It uses predetermined
   terminal checkpoints, a common 128-task validation panel, five controls, and
   keeps final test sealed. R1 is not implemented and no panel has been frozen
   or opened.
6. Q23 specifies 32 distinct symbol/session anchors over twelve symbols, at
   least two per symbol, crossed with horizons 96/128/160/200. Metadata-only
   selection must precede outcomes. Historical non-exposure remains unresolved;
   this statistical note cannot certify freshness.

The local contracts are in
[`r1-development-readiness.md`](../../../project-state/r1-development-readiness.md),
[`r1-panel-qualification-2026-10-06.md`](../r1-panel-qualification-2026-10-06.md),
and the [masterplan](../../../plans/itrader_masterplan.md).

## 2. Exact target estimands

### 2.1 Q24 primary

For a fixed trained policy `p`, matched control `c`, symbol `s`, anchor `a`, and
horizon `h`, let

```text
d[p,c,s,a,h] = terminal_net_return[p,s,a,h]
               - terminal_net_return[c,s,a,h].
```

With `H = {96,128,160,200}`, `A_s` the frozen anchors for symbol `s`, and
`S = 12`, the accepted paired macro contrast is

```text
anchor_delta[p,c,s,a] = (1 / 4) * sum_h d[p,c,s,a,h]

symbol_delta[p,c,s] = (1 / |A_s|) * sum_a anchor_delta[p,c,s,a]

theta[p,c] = (1 / 12) * sum_s symbol_delta[p,c,s].
```

This order matters. A raw task mean overweights symbols with three anchors
relative to symbols with two. Treating the four horizons as four independent
observations multiplies apparent support without adding four independent market
paths. The horizons are repeated measurements of the same anchor and share
most of their bars.

**Recommendation:** Q26 must recompute `theta[p,c]` in every resample. Applying
an interval to a differently weighted episode mean would answer a different
question.

### 2.2 Q25 duration diagnostic

For every complete episode, compute first:

```text
q[p,s,a,h] = 10000 * terminal_net_return[p,s,a,h] / actual_decision_steps.
```

Then apply the same horizon -> anchor -> symbol hierarchy to paired `q`
differences. This estimates an average arithmetic net-bps-per-bar diagnostic.
It is not a compounded return, annualized return, or replacement for Q24.

### 2.3 Training-run target is separate

There are two scientifically different targets:

- **Conditional market-sample effect:** `theta[p,c]` for one already-trained
  checkpoint, with uncertainty over eligible calendar blocks.
- **Algorithm/training-seed effect:** expectation of `theta` over independent
  training runs from the declared algorithm.

The 128 terminal tasks inform the first target. They do not increase the sample
size for the second. With two training seeds, the second target remains a
screening summary; do not pool 256 episode rows and call them independent
algorithm replications.

## 3. Dependence structure that Q26 must preserve

### 3.1 Four distinct dependence channels

1. **Within-anchor horizon overlap.** The 96/128/160/200 episodes share the
   lookback, early actions, prices, and most outcomes.
2. **Within-symbol calendar overlap.** Nearby anchors can have overlapping
   lookback/outcome intervals and common serial market regimes.
3. **Cross-symbol market dependence.** Equity and ETF episodes on the same or
   nearby sessions share market-wide shocks. Independent per-symbol resampling
   erases this covariance.
4. **Training-run dependence.** R0/R4/R1 cells are matched by seed/curriculum,
   but separately trained seeds are the units for learning stochasticity.

Long-horizon asset-return research shows that conventional approximations can
be badly misleading when returns overlap
([Richardson and Stock, 1989](https://www.nber.org/papers/w3335)). Block methods
were introduced precisely to approximate sampling distributions for dependent
stationary sequences by resampling adjacent observations rather than pretending
they are IID ([Kunsch, 1989](https://doi.org/10.1214/aos/1176347265);
[Politis and Romano, 1994](https://users.ssc.wisc.edu/~behansen/718/Politis%20Romano.pdf)).

### 3.2 Calendar components, not 128 tasks

**Recommended metadata construction:** for each anchor, form its maximal
information/outcome interval from the beginning of the 60-bar lookback through
the end of the 200-bar outcome. On the exchange calendar, merge intersecting or
adjacent intervals into ordered calendar components. Every symbol/anchor in a
component and all four of its horizons stay together. A block resampler then
draws contiguous sequences of these ordered components and applies the same
draw to the complete cross-symbol record.

This construction is an iTrader inference, not a theorem from the cited papers.
Its purpose is to make the empirical resampling unit match the actual shared
information set. Exact adjacency rules remain a Q26 decision and must be
predeclared from metadata, never chosen after returns are visible.

## 4. Candidate Q26 designs

### Candidate A: paired synchronized calendar block bootstrap

**Question answered:** uncertainty in the exact Q24 candidate-control macro
contrast over eligible market calendar blocks, conditional on fixed trained
checkpoints.

**Procedure candidate:**

1. Join candidate and control by the immutable task identity. Subtract first.
2. Keep the four horizon deltas inside each anchor.
3. Map anchors to ordered, outcome-independent calendar components.
4. Draw moving, circular, or stationary blocks of components **jointly across
   symbols**.
5. Rebuild a complete replicate and recompute horizons -> anchors -> symbols ->
   equal-symbol `theta`.
6. Studentize if the chosen finite-sample calibration supports it; report the
   point estimate, interval, component count, block specification, and stability
   across a predeclared plausible block grid.

Moving blocks use a fixed length; the stationary bootstrap uses random lengths
and yields a stationary pseudo-series. Both require weak-dependence/stationarity
conditions. Kunsch requires block length to grow while remaining small relative
to sample length for consistency; Politis and Romano extend inference to
general parameters. Block choice is not a cosmetic parameter: optimal order
depends on whether the target is variance, a one-sided distribution, or a
two-sided distribution ([Hall, Horowitz, and Jing, 1995](https://academic.oup.com/biomet/article-abstract/82/3/561/260651)).
Automatic selectors exist, but the original Politis-White result has a published
correction, so implementation must use the corrected formula
([Politis and White, 2004](https://www.math.ucsd.edu/~politis/SBblock-revER.pdf);
[Patton, Politis, and White, 2009](https://www.tandfonline.com/doi/abs/10.1080/07474930802459016)).

**Constraints:**

- Ordinary resampling with replacement may omit one of the twelve symbols in a
  replicate when the panel is sparse. Silently renormalizing eleven symbols
  violates Q24. Q23 should therefore demonstrate outcome-independent block-by-
  symbol support, or Q26 should use a fixed-support block multiplier/studentized
  construction whose validity is separately calibrated.
- A block multiplier avoids dropping symbols by assigning a common random
  weight to every observation in a calendar component. It still needs many
  non-dominant components and a justified multiplier law; it is an alternative,
  not an automatic solution.
- Resampling anchors IID, or resampling each symbol independently, is not this
  design.

### Candidate B: calendar-block subsampling / fixed-b sensitivity

**Question answered:** the same conditional Q24 contrast, using the empirical
distribution of the statistic on smaller contiguous calendar subsamples.

Subsampling can be valid under weaker conditions than a model-specific
bootstrap, provided the normalized statistic has a limiting distribution
([Politis, Romano, and Wolf, 2001](https://e-archivo.uc3m.es/entities/publication/61678ad2-b1ca-4ee3-ad5a-3ac789581c96)).
Fixed-b calibration explicitly acknowledges that block length remains a
non-negligible fraction of a small sample; simulations in Shao and Politis show
that block choice can materially change finite-sample inference
([Shao and Politis, 2013](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2012.01037.x)).

**Use:** a strong sensitivity companion when moving/stationary-bootstrap results
are block-length fragile. **Limit:** it still requires enough distinct
contiguous subsamples with all required symbol/horizon support; it cannot repair
a panel with only a handful of effective calendar components.

### Candidate C: HAC long-run variance sensitivity

Aggregate paired, fixed-weight anchor contributions to an ordered calendar
series and estimate the long-run covariance with a positive semidefinite HAC
estimator. Newey and West establish consistency under regularity conditions
([Newey and West, 1987](https://www.nber.org/papers/t0055)).

**Use:** transparent analytic sensitivity against the block interval. **Limits:**
asymptotic approximation, kernel/lag sensitivity, irregular sparse support,
cross-symbol aggregation choices, and small effective calendar sample. HAC
does not correct checkpoint/model selection or quantify training-seed variance.
It should not be the sole Q26 authority without layout-specific simulation.

### Candidate D: chronology-group t inference

Partition the eligible chronology into a small number of non-overlapping,
sufficiently separated groups; compute the full Q24 paired statistic inside
each group; then apply the group-estimate t method. Ibragimov and Muller show
conservative inference under independent, approximately Gaussian group
estimators with heterogeneous variances
([Ibragimov and Muller, 2010](https://www.princeton.edu/~umueller/tstat.pdf)).

**Use:** a robustness check if groups can be made genuinely independent and each
contains all twelve-symbol support. **Invalid when:** groups overlap, group
estimates are badly biased/non-Gaussian, or only a few outcome-bearing groups
can be formed. It is not a way to bless three calendar clusters.

## 5. Method comparison

| Method | Primary target | What it handles | What it does **not** handle | iTrader role |
|---|---|---|---|---|
| Paired synchronized block bootstrap | Q24/Q25 paired macro delta | nonlinear hierarchy, serial dependence, within-anchor grouping, cross-symbol dependence if drawn jointly | incomplete panel, unknown trial family, two-seed uncertainty | leading Q26 candidate, pending calibration |
| Calendar-block subsampling/fixed-b | same | weaker distributional modelling; exposes bandwidth sensitivity | sparse complete-support blocks; seed uncertainty | sensitivity / possible fallback |
| Newey-West HAC | mean/linearized statistic | heteroskedasticity and autocorrelation under large-sample regularity | nonlinear selection, multiplicity, small support | analytic sensitivity only |
| Ledoit-Wolf time-series bootstrap | difference of two genuine Sharpe ratios | nonnormal, serially dependent return streams via studentized time-series bootstrap | Q24 terminal return; stitched overlapping episodes | future use only if a genuine paired non-overlapping return stream exists |
| PSR | probability a true Sharpe exceeds a user benchmark | sample length, skewness, kurtosis under its return-series assumptions | Q24; model search; arbitrary episode-Sharpe vectors | secondary Sharpe audit only |
| DSR | PSR with selection-adjusted benchmark | nonnormality plus number/dispersion of tried Sharpe strategies | missing trial ledger; dependence misspecification; Q24 | research-search audit only, if trial inventory is complete |
| White Reality Check | global null: no candidate beats benchmark | multiple candidate strategies and dependence via bootstrap | identification of each winner; omitted trials; seed scarcity | optional predeclared familywise screen |
| Hansen SPA | same global null with studentization/recentering | less sensitivity to poor/irrelevant alternatives than RC | same inventory/dependence limits; no individual winner CI | preferable global screen if assumptions hold |
| Romano-Wolf StepM | individual benchmark comparisons with FWER control | joint dependence, studentization, stepdown power | incomplete family; invalid resampling law | optional if the goal is to name cells, not merely reject global null |

Ledoit and Wolf specifically show that the classic normal-IID difference-in-
Sharpe test is invalid for heavy-tailed or time-series returns and propose a
studentized time-series bootstrap interval
([Ledoit and Wolf, 2008](https://www.ledoit.net/Robust_Sharpe_2008.pdf)).
That is relevant only after defining a legitimate return time series. Lo likewise
shows that Sharpe inference depends on the return-generating process and serial
correlation ([Lo, 2002](https://ideas.repec.org/a/taf/ufajxx/v58y2002i4p36-52.html)).

## 6. Why PSR/DSR are not Q26

### Verified literature facts

PSR estimates the probability that a strategy's Sharpe exceeds a chosen
benchmark using track length plus estimated skewness and kurtosis. The authors
explicitly advise using the highest sampling frequency that does not violate
the IID assumption
([Bailey and Lopez de Prado, 2012](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID3594389_code434076.pdf?abstractid=1821643&type=2)).

DSR replaces PSR's user threshold with an expected maximum Sharpe that depends
on the number of independent trials and the cross-trial variance of estimated
Sharpes. It therefore addresses selection bias and nonnormality, but requires
information about unselected trials
([Bailey and Lopez de Prado, 2014](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)).

### iTrader inference

- Q24's primary outcome is terminal arithmetic net return, not Sharpe.
- The 128 tasks are four overlapping horizons on 32 anchors, not a return track
  sampled at a frequency for which IID is plausible.
- `net_sharpe_per_sqrt_bar` is computed inside an episode. Averaging 128 such
  ratios is not the same as the Sharpe of a portfolio return series.
- DSR's effective trial count is unknowable if failed configurations,
  checkpoints, reward variants, panels, and informal design iterations are not
  comprehensively logged. Guessing `N` would turn a correction into false
  precision.

Accordingly, PSR/DSR may accompany a future non-overlapping live/shadow equity
curve or a complete research-search ledger. They must not be used to convert
the current Q24/Q25 panel into a significance claim.

## 7. Multiplicity and selection protocol

White defines data snooping as reusing a dataset for model selection or
inference and constructs a bootstrap Reality Check over the candidate family
([White, 2000](https://onlinelibrary.wiley.com/doi/10.1111/1468-0262.00152)).
Hansen's SPA modifies the Reality Check with studentization and a sample-
dependent null distribution, improving power and reducing sensitivity to poor
alternatives
([Hansen, 2005](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=264569)).
Romano-Wolf StepM controls familywise error while identifying individual
strategies and exploits dependence among statistics
([Romano and Wolf, 2005](https://onlinelibrary.wiley.com/doi/10.1111/j.1468-0262.2005.00615.x)).

### Required research-family ledger

Before a multiplicity-adjusted claim, record at least:

- every representation/curriculum cell included in the family;
- all training seeds and whether they are replicates or selected candidates;
- every control and whether the benchmark was fixed or selected;
- every metric, horizon, cost scenario, and subgroup used for a claim;
- every checkpoint-selection rule and every panel previously inspected;
- dropped/failed cells and the reason for exclusion;
- the point at which the terminal panel identity was frozen.

The finance literature demonstrates why ordinary single-test criteria are not
credible after broad strategy/factor searches
([Harvey, Liu, and Zhu, 2016](https://academic.oup.com/rfs/article-abstract/29/1/5/1843824)).
This does **not** license transplanting that paper's numerical hurdle to
iTrader; the candidate family and dependence structure differ.

### Application to the current screen

1. Predetermining the terminal checkpoint before the fresh panel opens removes
   within-run best-checkpoint selection **for this terminal screen**.
2. Opening only after all sixteen cells complete prevents adapting later cells
   to early terminal outcomes.
3. R0/R4/R1 representation and curriculum comparisons still form a family if
   any one is to be named a winner. Predeclare whether Q26 asks one global SPA
   question, a small set of confirmatory paired contrasts, or Romano-Wolf
   adjusted individual questions.
4. Long and short controls must both be shown. Declaring the stronger direction
   after seeing the panel is a two-benchmark selection. A robust superiority
   claim can require the candidate to beat both with simultaneous inference,
   rather than treating the selected maximum as if it had been fixed ex ante.
5. Historical P0/checkpoint search and the fresh Q21 screen must remain distinct.
   A method applied only to the predeclared terminal family cannot retrospectively
   correct an incompletely logged historical search.
6. Failure to certify Q23 non-exposure is upstream of statistics. RC/SPA/DSR do
   not make a previously exposed panel fresh.

## 8. Effective support and fail-closed conditions

There is no defensible universal count of "enough blocks." Cluster-inference
research shows that reliability depends not just on the raw number of clusters
but on imbalance and influence; even wild bootstrap procedures can fail when
effective support is concentrated
([MacKinnon, Nielsen, and Webb, 2023](https://doi.org/10.1016/j.jeconom.2022.04.001)).
Block-bootstrap theory also establishes a bias/variance tradeoff: blocks too
short miss dependence, while blocks too long leave too few resampling units
([Shao and Politis, 2013](https://publish.illinois.edu/xshao/files/2012/11/ShaoPolitisJRSS.pdf)).

### Report these support diagnostics before outcomes

- number of anchors, unique sessions, unique trading dates, and merged maximal-
  horizon calendar components;
- counts by symbol, decision offset, chronological region, and component;
- component size distribution and each component's share of the fixed Q24
  weight;
- number of available moving/circular blocks over a declared candidate block
  grid;
- whether every resample/subsample can retain all twelve symbols and all four
  horizons without renormalization;
- calendar span and gaps between components;
- maximum component influence and a labelled descriptive effective-cluster
  measure, with its formula;
- interval/standard-error sensitivity across the predeclared block grid;
- simulation coverage and type-I error under DGPs matched to the frozen layout,
  including common-factor cross-symbol shocks, serial dependence, heavy tails,
  and a null paired effect.

### Fail closed when any of these occurs

- fewer than two chronological components (the resampling distribution is
  degenerate);
- a candidate method cannot preserve complete twelve-symbol/four-horizon Q24
  support;
- one component dominates the estimator so heavily that the planned calibration
  fails its predeclared coverage/error tolerance;
- plausible block specifications lead to qualitatively contradictory
  conclusions and no design-level reason selects one;
- the layout-specific null simulation shows unacceptable false-positive control;
- checkpoint, task, cost, dataset, source, or candidate-control pairing is
  incomplete;
- terminal-panel non-exposure is unproved;
- a claim about training-run generalization relies on only the two screen seeds.

The third through fifth items deliberately avoid inventing a numerical cutoff.
The human must approve simulation tolerances and the block family before outcomes
are visible. If none is credible, Q21 remains a descriptive attribution screen.

## 9. Candidate practical protocol for Q21/Q23/Q26

This is the recommended decision sequence, not an implementation authorization:

1. **Q23 metadata design:** after provenance/non-exposure is resolved, construct
   candidate panels from calendar, symbol, offset, split, purge, and continuity
   only. Add the support diagnostics above before reading returns.
2. **Q26 simulation calibration:** instantiate the exact 32-anchor sparse layout
   with synthetic paired outcomes under several dependence/heavy-tail/common-
   factor nulls. Compare synchronized moving/circular/stationary block bootstrap,
   block multiplier, fixed-b subsampling, and HAC sensitivity over a predeclared
   block grid.
3. **Human method decision:** select the uncertainty family, block selector or
   grid, interval type, error target, multiplicity family, and fail rule. Do not
   select whichever method gives the narrowest interval.
4. **Seal:** hash panel metadata, calendar components, weights, selected method,
   code/source, bootstrap seed, draws, and candidate family before terminal
   outcomes open.
5. **Q21 execution:** require all sixteen cells and five controls; evaluate the
   same panel deterministically at 1 bp; join candidate-control pairs exactly.
6. **Report per training seed first:** Q24/Q25 point estimates and calendar-
   block intervals conditional on that seed's terminal checkpoint.
7. **Across-seed screen:** show both effects, mean, range, sign consistency, and
   interaction contrasts. State explicitly that two seeds do not estimate the
   training-run sampling distribution.
8. **Multiplicity:** run the predeclared global or stepdown procedure only on
   its complete candidate family, using the same synchronized calendar
   resampling law. Preserve raw unadjusted estimates alongside adjusted results.
9. **No automatic promotion:** economic controls, costs, exposure/turnover,
   drawdown, critic/PPO diagnostics, and replay/provenance gates remain separate.

## 10. Minimum implementation tests for a future Q26 owner

1. Hand-computed unequal-anchor example reproduces exact horizon -> anchor ->
   symbol -> equal-symbol weights in every resample.
2. Row permutation leaves estimates and seeded resamples invariant after
   canonical ordering.
3. Missing/duplicate/wrong horizon, symbol, seed, start/end, cost, or checkpoint
   identity fails closed.
4. Candidate identical to control gives exactly zero paired deltas and a
   degenerate zero interval where mathematically expected.
5. All four horizons of one anchor always share the same resampling weight.
6. Same-calendar/overlapping-window anchors across symbols always share the same
   block draw or multiplier.
7. A deliberate independent-per-symbol implementation produces a failing test.
8. Q24 and Q25 remain distinct; Q25 normalizes every episode before aggregation.
9. Bootstrap/subsampling replicates never silently omit a symbol or renormalize
   incomplete support.
10. Synthetic common-factor data show that synchronized resampling retains
    cross-symbol covariance.
11. Null simulations record empirical type-I error and coverage for every
    candidate block rule; too-few-block layouts are rejected.
12. The seed-level API rejects language implying population seed significance
    when only two training seeds are supplied.
13. Multiplicity tests require the complete predeclared candidate list and fail
    if a cell/metric/control is missing.
14. Selected stronger-direction comparisons either test both long and short
    simultaneously or are labelled descriptive.

## 11. Ordered one-question-at-a-time grilling agenda

These questions should be asked in order; each answer constrains the next.

1. **Q23 authority:** Will the terminal panel use source-bound recovered starts,
   or a separately declared evaluation-only dataset/split with independently
   proven non-exposure?
2. **Q23 calendar support:** Within that source, can 32 outcome-independent
   anchors satisfy twelve-symbol coverage while yielding enough distinct,
   non-dominant maximal-horizon calendar components for Q26?
3. **Q26 target:** Is confirmatory inference required only for the Q24 terminal-
   net contrast, with Q25/horizon/Sharpe outputs descriptive, or are additional
   confirmatory metrics part of the multiplicity family?
4. **Q26 contrast family:** Which paired comparisons are confirmatory: R1 vs R0,
   `R1+R4` vs `R0+R4`, R4 interaction, curriculum interaction, all five controls,
   or a smaller ordered family?
5. **Q26 dependence family:** Should layout simulation compare moving/circular/
   stationary block bootstrap, fixed-support block multiplier, fixed-b
   subsampling, and HAC, or exclude any on scientific grounds before outcomes?
6. **Q26 calibration:** What simulation error/coverage tolerance and instability
   rule should make the design fail closed? No literature default is silently
   adopted.
7. **Q26 block selection:** Should one corrected automatic selector be sealed,
   or should a predeclared block grid be reported as sensitivity with an explicit
   decision rule?
8. **Q26 multiplicity:** Is the desired output a single global RC/SPA-style
   family test, adjusted individual Romano-Wolf contrasts, or descriptive
   attribution only for this two-seed screen?
9. **Seed scope:** Is the explicit decision that two seeds remain a mechanics/
   attribution screen, with a later independently powered seed replication
   required before economic promotion?
10. **Failure outcome:** If no uncertainty design calibrates credibly on the
    sparse panel, do we enlarge/restructure the still-unopened metadata-only
    panel with human approval, or keep Q21 descriptive and defer confirmation?

## 12. Primary sources

- Agarwal, Schwarzer, Castro, Courville, and Bellemare (2021),
  [*Deep Reinforcement Learning at the Edge of the Statistical Precipice*](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf).
- Bailey and Lopez de Prado (2012),
  [*The Sharpe Ratio Efficient Frontier*](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID3594389_code434076.pdf?abstractid=1821643&type=2).
- Bailey and Lopez de Prado (2014),
  [*The Deflated Sharpe Ratio*](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).
- Hall, Horowitz, and Jing (1995),
  [*On Blocking Rules for the Bootstrap with Dependent Data*](https://academic.oup.com/biomet/article-abstract/82/3/561/260651).
- Hansen (2005),
  [*A Test for Superior Predictive Ability*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=264569).
- Harvey, Liu, and Zhu (2016),
  [*... and the Cross-Section of Expected Returns*](https://academic.oup.com/rfs/article-abstract/29/1/5/1843824).
- Henderson et al. (2018),
  [*Deep Reinforcement Learning that Matters*](https://ojs.aaai.org/index.php/AAAI/article/download/11694/11553).
- Ibragimov and Muller (2010),
  [*t-Statistic Based Correlation and Heterogeneity Robust Inference*](https://www.princeton.edu/~umueller/tstat.pdf).
- Kunsch (1989),
  [*The Jackknife and the Bootstrap for General Stationary Observations*](https://doi.org/10.1214/aos/1176347265).
- Ledoit and Wolf (2008),
  [*Robust Performance Hypothesis Testing with the Sharpe Ratio*](https://www.ledoit.net/Robust_Sharpe_2008.pdf).
- Lo (2002),
  [*The Statistics of Sharpe Ratios*](https://ideas.repec.org/a/taf/ufajxx/v58y2002i4p36-52.html).
- MacKinnon, Nielsen, and Webb (2023),
  [*Cluster-Robust Inference: A Guide to Empirical Practice*](https://doi.org/10.1016/j.jeconom.2022.04.001).
- Newey and West (1987),
  [*A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*](https://www.nber.org/papers/t0055).
- Patton, Politis, and White (2009),
  [correction to *Automatic Block-Length Selection for the Dependent Bootstrap*](https://www.tandfonline.com/doi/abs/10.1080/07474930802459016).
- Politis and Romano (1994),
  [*The Stationary Bootstrap*](https://users.ssc.wisc.edu/~behansen/718/Politis%20Romano.pdf).
- Politis, Romano, and Wolf (2001),
  [*On the Asymptotic Theory of Subsampling*](https://e-archivo.uc3m.es/entities/publication/61678ad2-b1ca-4ee3-ad5a-3ac789581c96).
- Richardson and Stock (1989),
  [*Drawing Inferences from Statistics Based on Multi-Year Asset Returns*](https://www.nber.org/papers/w3335).
- Romano and Wolf (2005),
  [*Stepwise Multiple Testing as Formalized Data Snooping*](https://onlinelibrary.wiley.com/doi/10.1111/j.1468-0262.2005.00615.x).
- Shao and Politis (2013),
  [*Fixed-b Subsampling and the Block Bootstrap*](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2012.01037.x).
- White (2000),
  [*A Reality Check for Data Snooping*](https://onlinelibrary.wiley.com/doi/10.1111/1468-0262.00152).

## 13. Fact / inference / hypothesis ledger

- **Verified facts:** the Q24/Q25 hierarchy, current reporter's unscored Q26
  field, two-seed screen design, overlapping horizon construction, current
  exploratory bootstrap behavior, and unresolved Q23 freshness are source/doc
  facts.
- **Literature facts:** method targets and assumptions described with direct
  primary citations above.
- **Recommendations/inferences:** synchronized cross-symbol calendar blocks,
  maximal-window component construction, seed-conditional reporting, layout-
  specific null simulation, and fail-closed support diagnostics.
- **Open hypotheses:** that a 32-anchor panel can simultaneously deliver the
  accepted symbol coverage, strict freshness, and enough non-dominant calendar
  components for calibrated Q26 inference; and that any R1 effect will be stable
  across independently trained seeds. Neither is established by this note.
