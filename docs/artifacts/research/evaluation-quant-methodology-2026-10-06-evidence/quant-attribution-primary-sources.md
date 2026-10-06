# Quantitative performance attribution for R0/R4/R1 evaluation

Date: 2026-10-06
Scope: quantitative/algotrading attribution only
Status: supporting research note; not an adopted metric, threshold, Q26 method,
panel approval, or promotion decision

No market outcomes, candidate panels, checkpoints, or datasets were opened for
this note. It does not alter the active architecture, masterplan, source, config,
or evaluation code. The fresh-panel gate remains blocked for the reasons in the
[R1 panel-qualification audit](../r1-panel-qualification-2026-10-06.md).

## Claim labels

- **Verified** means directly supported by current iTrader source or a linked
  primary paper/official first-party source.
- **Inference** means an algebraic or design conclusion derived from those
  verified facts.
- **Hypothesis** means a candidate diagnostic whose value for iTrader still
  requires prospective evidence.

## Executive conclusion

No single return, Sharpe, alpha, or composite score can answer whether a solver
representation is economically better. The defensible attribution stack is:

1. preserve the arithmetic bridge `gross - cost = net`;
2. split gross return into average directional exposure and a within-episode
   timing residual on the identical price path;
3. compare every representation with flat, long, short, random, and matched
   untrained controls, while keeping the four R cells paired by seed,
   curriculum, anchor, and horizon;
4. report risk normalization as a *vector* of companion diagnostics rather than
   a replacement objective; and
5. expose symbol, calendar, horizon, time-of-day, and ex-post regime
   concentration before interpreting an aggregate.

The current terminal evidence path -- `EpisodeEvidence`, evaluator-added action
fractions, and the matched long control -- is already sufficient for the first
layer and most of the second. `EpisodeEvidence` by itself is not sufficient for
step-level market-beta regression, differential-return Sharpe, or execution
shortfall.
Metric construction and uncertainty are separate: none of the decompositions
below supplies a confidence interval or significance claim by definition.

## 1. Current iTrader accounting contract

### Verified implementation facts

Current [`EpisodeEvidence`](../../../../itrader/rl_lib/evaluation/episode_evidence.py)
records optimization reward separately from economic evidence. Its economic
identity is:

```text
total_turnover = in_episode_turnover + terminal_liquidation_turnover
cost_return    = total_turnover * evidence_cost_bps * 1e-4
net_return     = gross_return - cost_return
```

The environment maps actions to target positions `{-1, 0, +1}`, realizes the
new target position over the next price return, charges turnover as
`abs(new_position - previous_position)`, and adds `abs(final_position)` at
termination. A direct flip therefore contributes two turnover units, while a
terminal non-flat position contributes one liquidation unit. See the current
[`ITraderEnv.step` and terminal evidence path](../../../../itrader/envs/itrader_env.py).

`EpisodeEvidence` also records gross maximum drawdown, gross and net
per-square-root-bar Sharpe, optional annualized net Sharpe, gross exposure
fraction, positive-net-bar fraction, position-change count, symbol, split,
start/end rows, and horizon. Important interpretation limits follow directly
from the implementation:

- `max_drawdown` is computed from gross additive episode PnL before evaluation
  costs; the terminal reporter therefore correctly renames it
  `gross_max_drawdown`. It is not a net drawdown.
- `net_sharpe_per_sqrt_bar` is the mean divided by sample standard deviation of
  within-episode net additive bar returns. It returns zero when fewer than two
  observations exist or when the standard deviation is effectively zero.
- `annualized_net_sharpe`, when enabled, is a square-root-of-periods scaling of
  the within-episode statistic. It is not an autocorrelation-adjusted estimate.
- `gross_exposure_fraction` is mean absolute position. The terminal control
  [`evaluate_policy_control`](../../../../itrader/rl_lib/evaluation/policy_baselines.py)
  path adds action shares to replay rows; those shares recover signed mean
  position as `long_fraction - short_fraction`.
- `positive_net_bar_fraction` counts bars whose modeled net increment is
  strictly positive, including the final liquidation charge in the last bar.
  It is not a trade win rate or an episode success probability.
- `position_changes` counts changed target decisions, whereas turnover measures
  their magnitude. A short-to-long flip is one position change but two turnover
  units, so turnover is the relevant base for the current cost arithmetic.

The current
[`terminal_panel_report`](../../../../itrader/rl_lib/evaluation/terminal_panel_report.py)
requires complete, uncensored, exact-1-bp evidence for 32 anchors crossed with
all four horizons (96/128/160/200), and it reports flat, long, short, random, and
untrained controls on the identical panel. It explicitly labels its comparison
as descriptive and leaves uncertainty unscored.

### What the 1 bp result does and does not mean

**Inference.** The current 1 bp calculation is a transparent *linear cost
scenario*, not a broker-fill model. It omits spread state, order type, queueing,
latency, partial fill, borrow/funding, capacity, and nonlinear market impact.
This is consistent with the architecture boundary that `EpisodeEvidence` does
not encode broker semantics.

That limit matters economically. Frazzini, Israel, and Moskowitz measure live
institutional trades using implementation shortfall and market impact; their
execution-cost definition includes bid/ask spread, market impact, and
commissions, while implementation shortfall also captures opportunity cost.
They also find that proportional cost models do not match their trade data and
that price impact is convex in trade size. Their setting and scale do not
estimate iTrader's costs, but the paper demonstrates why a fixed bp model is a
scenario rather than execution truth
([primary working paper, pp. 12--18](https://pages.stern.nyu.edu/~afrazzin/pdf/Trading%20Cost%20of%20Asset%20Pricing%20Anomalies%20-%20Frazzini,%20Israel%20and%20Moskowitz.pdf)).

Consequently, 0.5/1/2 bp repricing can answer how a policy responds to a
*linear* turnover penalty. It cannot establish capacity or live implementability.
The existing
[`cost_sensitivity`](../../../../itrader/rl_lib/evaluation/block_uncertainty.py)
correctly limits arithmetic repricing to gross return and turnover and notes
that other per-step metrics require replay.

## 2. Exact economic attribution on a matched episode

Let `r_t` be the symbol's one-bar return and `p_t` the position selected before
that return. Under the current environment:

```text
G = sum_t p_t r_t
C = k * (sum_t |p_t - p_(t-1)| + |p_T|),  k = cost_bps * 1e-4
N = G - C
```

For an episode of `T` bars, define average signed exposure

```text
p_bar = (1/T) * sum_t p_t = long_fraction - short_fraction
L     = sum_t r_t                         # matched long-control gross return
G_direction = p_bar * L
G_timing    = G - G_direction
```

Then the identity

```text
N = G_direction + G_timing - C
```

holds exactly when the candidate and long control share the same symbol,
anchor, horizon, and price-return sequence.

### Interpretation

- **Verified algebra:** `G_direction` is the gross return of a fractional static
  position equal to the policy's realized average signed exposure.
- **Inference:** `G_timing` is the part of gross return not explained by that
  average directional loading. It is a useful timing/rotation residual, not
  proof of forecasting skill or a causal alpha.
- **Inference:** for any paired candidate/control contrast,
  `delta_net = delta_direction + delta_timing - delta_cost`. This directly shows
  whether an apparent net uplift came from greater gross directional loading,
  state-dependent timing, or reduced trading cost.

This decomposition is preferable to immediately fitting a market-beta model:
it is exact, uses the policy's actual action semantics, and requires no factor
model. It also exposes the hard-flat basin. A policy with near-zero absolute
exposure, zero gross return, zero timing residual, and zero cost has not
demonstrated edge merely because it beats a losing active policy.

### Required cautions

1. `p_bar` is a realized outcome-window exposure selected by the policy; it is
   not an ex-ante allocation.
2. `G_timing` can contain noise, path dependence, and checkpoint-selection
   effects. It has no built-in sampling correction.
3. The fractional static counterfactual is an attribution device, not one of
   the executable `Discrete(3)` policies.
4. Exact equality requires matched steps and the same return stream. Aggregate
   means from unmatched panels cannot be substituted.
5. Because returns are accumulated additively in the current environment, this
   is an additive episode attribution, not a compounded wealth or leveraged
   capital-accounting statement.

## 3. Benchmark ladder

William Sharpe defines performance through *differential return* relative to a
chosen benchmark and stresses that the benchmark determines what the ratio
means. He also describes a passive portfolio with similar style as a way to
separate style exposure from selection return
([Sharpe 1994](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm),
[Sharpe 1992](https://web.stanford.edu/~wfsharpe/art/sa/sa.htm)). For iTrader,
the benchmark should therefore be a ladder, not a single universal baseline.

| Comparator | Question it answers | Correct interpretation | Main failure mode |
|---|---|---|---|
| Flat | Did the policy add positive net value over no exposure under the modeled costs? | Operational cash/abstention control inside the solver action set | A flat policy can look relatively good when all active policies lose |
| Long and short, reported separately | Did the policy beat either constant direction on the identical path? | Transparent directional controls | One side may be favored by the sampled market regime |
| `max(long, short)` on the panel | Did the policy beat the best ex-post constant direction? | Strict descriptive hurdle | Choosing the stronger side from the same outcomes is selection; it is not an unadjusted inferential benchmark |
| Seeded random | Did structured behavior beat a known non-learning action process? | Behavioral control | A single random stream is not the distribution of random policies |
| Matched untrained architecture | Did training improve that initialized policy on the frozen panel? | Training-value control when model, source, and producer identity are verified | Unverified or differently initialized artifacts invalidate the label |
| Exposure-matched static return | Is gross return more than average signed direction alone? | Exact `G_direction`/`G_timing` attribution above | Fractional position is analytical, not executable |
| Market/factor benchmark | Is return left after modeled systematic exposure? | Model-dependent beta/alpha diagnostic | Frequency mismatch, unstable beta, omitted factors, and insufficient blocks |

Jensen's original performance regression explicitly adjusts portfolio return for
market systematic risk and interprets the intercept as return beyond that risk;
the paper also states the CAPM assumptions on which that interpretation rests
([Jensen 1968](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1968.tb00815.x)).
Fama and French show that stock returns share variation beyond a single market
factor, including size and value factors
([Fama and French 1993](https://doi.org/10.1016/0304-405X(93)90023-5)).

**Inference for iTrader.** A step-level regression such as
`policy_return_t = alpha + beta * market_return_t + error_t` could describe
directional loading, but `EpisodeEvidence` does not persist the needed aligned
per-step market and strategy-return series. Moreover, the official Fama/French
library offers daily (and lower-frequency) factors, not one-minute factors; its
daily data must not be repeated across minute rows as if they were intraday
observations
([official Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/Faculty/ken.french/data_library.html)).
Until an aligned, predeclared benchmark exists, matched long/short controls and
the exact exposure/timing split are the more transparent diagnostics.

## 4. Risk normalization: useful companions, not substitutes

Sharpe's formulation is average *differential* return divided by its standard
deviation, and he notes that mean/variance alone can be insufficient when other
moments or state-dependent outcomes matter
([Sharpe 1994](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)). Lo derives the
sampling behavior under iid and stationary returns and shows that simple
square-root annualization is valid only under special conditions; serial
correlation can materially change both values and rankings
([Lo 2002, author publication record](https://alo.mit.edu/publications/page/18/)).

These results support the following hierarchy:

| Candidate diagnostic | Economic role | iTrader interpretation and constraint |
|---|---|---|
| Paired terminal net return | Primary arithmetic outcome | Keep gross/cost/net beside it; do not replace with a ratio |
| Net bps per actual bar | Duration-normalized companion | Compute per episode before aggregation, as current Q25 does; not a compounded or capital-weighted return |
| Net Sharpe per sqrt bar | Zero/cash-benchmark risk efficiency | Descriptive at short horizons; avoid claiming annual Sharpe or significance from the value alone |
| Differential-return Sharpe versus a named control | Active-return efficiency | Requires aligned per-step candidate-minus-control returns; benchmark name and construction are part of the metric |
| Lagged-volatility-standardized net or active return | Cross-symbol/risk-state comparability | Volatility estimator must use only pre-anchor information and account for time of day; outcome-window volatility cannot become a decision-time feature |
| Jensen-style alpha/beta | Systematic-risk diagnostic | Model-dependent, data-hungry, and frequency-sensitive; never synonymous with solver capability |
| Downside deviation / Sortino | Penalize returns below a declared target | Target and sampling unit must be predeclared; unstable or undefined under sparse downside observations ([Sortino and Price 1994](https://doi.org/10.3905/joi.3.3.59)) |
| Expected shortfall | Tail-loss companion | Requires enough independent loss units; small panels make tail estimates fragile ([Rockafellar and Uryasev 2000](https://doi.org/10.21314/JOR.2000.038)) |
| Maximum drawdown | Path-loss companion | Current evidence is *gross* additive drawdown; do not label it net drawdown ([Magdon-Ismail et al.](https://authors.library.caltech.edu/records/2aprk-1ee98)) |

### Lagged volatility, specifically

Moreira and Muir construct volatility-managed portfolios with exposure scaled by
the inverse of a previous period's realized variance, not the current outcome's
realized variance
([author-hosted published paper](https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf)).
This is relevant evidence for the *causal timing* of a volatility normalizer,
but their monthly factor-portfolio result does not validate any particular
one-minute iTrader estimator.

**Hypothesis.** A fixed pre-anchor volatility estimate may make cross-symbol
return magnitudes more comparable. It should remain a secondary view alongside
unscaled net return because scaling changes the estimand and can overweight
quiet symbols. A zero/near-zero volatility rule must be explicit and fail closed;
adding an epsilon without sensitivity analysis can manufacture extreme scores.

### Ratios to avoid as promotion metrics

- `net_return / turnover` can reward a nearly flat policy and is undefined at
  zero turnover.
- `net_return / gross_exposure` has the same zero-denominator problem and can
  reward tiny accidental exposure.
- `net_return / max_drawdown` is unstable when drawdown is zero and mixes a net
  numerator with the current gross-drawdown denominator.
- Averaging per-episode Sharpes is not the same estimand as computing Sharpe from
  one portfolio return stream.

These may be shown descriptively with explicit `NA` behavior, but they should
not become a magic scalar or automatic gate.

## 5. What current panel aggregation means

The hierarchical Q24 reporter first averages the four horizons within each
anchor, then anchors within each symbol, then the twelve symbols equally.
Therefore:

- **Inference:** `equal_symbol.net_return` is the mean terminal task return for
  a design that chooses a symbol uniformly, an available anchor uniformly within
  that symbol, and one of the four horizons uniformly. It is not the realized
  return of a simultaneously traded twelve-asset portfolio.
- **Inference:** `raw_episode_mean` weights a symbol with three anchors 1.5 times
  as much as a symbol with two anchors. It is a task-average diagnostic, not the
  equal-symbol estimand.
- **Inference:** `net_bps_per_bar` first divides by each episode's actual
  duration, so it describes average per-bar efficiency under the same hierarchy.
  It need not rank policies identically to terminal net return.
- **Inference:** the four crossed horizons at one anchor are alternative
  counterfactual tasks sharing market history. They are not four independent
  trades and must not be treated as four independent market events.

Equal-symbol weighting is appropriate if the target is a typical symbol in the
declared twelve-symbol universe. It is not automatically appropriate for a
capital-weighted deployment or a typical calendar day. TLT receives the same
top-level weight as each equity or equity ETF; AAPL, QQQ, and SPY can also share
substantial market variation. Fama and French's common-factor evidence makes
clear why more tickers do not imply the same number of independent risk sources
([Fama and French 1993](https://doi.org/10.1016/0304-405X(93)90023-5)).

## 6. Symbol, regime, calendar, and horizon dependence

### Symbol concentration

Always preserve `by_symbol` and paired symbol deltas beside the equal-symbol
summary. A positive aggregate driven by one instrument has a different scope
from a broad improvement. Avoid percentage contribution shares when total net
is near zero; signed group returns and paired deltas remain interpretable.

### Regime concentration

The current
[`regime_conditioned_summary`](../../../../itrader/rl_lib/evaluation/block_uncertainty.py)
uses the matched long control's realized gross episode return to label an
episode bullish, bearish, or sideways. The implementation correctly calls this
an exploratory ex-post diagnostic, not a predictive regime detector.

**Inference.** This view can reveal whether a policy is just a disguised static
direction, but it cannot be fed into training or described as decision-time
information. The sideways threshold must not be tuned after inspecting candidate
outcomes. Report long/short/flat shares and the exposure/timing/cost bridge
inside each regime rather than only regime net return.

### Calendar and time of day

Andersen and Bollerslev document pervasive intraday periodicity in return
volatility and show that it materially affects high-frequency volatility
dynamics
([primary publisher record, 1997](https://www.sciencedirect.com/science/article/pii/S0927539897000042)).
The present panel's permitted decisions at fixed offsets after the open therefore
do not make clock time irrelevant.

**Inference.** A result concentrated in particular sessions or decision offsets
can reflect calendar or intraday-volatility composition. Equal-symbol averaging
does not remove that. Attribution should show synchronized calendar-session and
decision-offset slices, with any volatility normalizer conditioned only on
past data and a predeclared intraday-seasonality treatment. The statistical
method for uncertainty over those calendar blocks is a separate Q26 decision.

### Horizon and liquidation

Every non-flat episode pays terminal liquidation once, so shorter horizons have
greater liquidation cost per bar, all else equal. Raw terminal net and net bps
per bar answer different questions. Report both, plus liquidation turnover's
share of total turnover, at every horizon. Do not infer that a long-horizon win
is horizon-robust unless the four horizon contrasts support it on matched
anchors.

## 7. Attribution matrix for the planned four-representation screen

Within each curriculum and training seed, the representation cells form a
useful 2x2 descriptive grid:

```text
                    R4 state absent       R4 state present
feed-forward        R0                    R0 + R4-v1
window encoder      R1                    R1 + R4-v1
```

The valid named contrasts are:

```text
R4 effect on feed-forward = (R0 + R4-v1) - R0
window-encoder effect     = R1 - R0
R4 effect on R1           = (R1 + R4-v1) - R1
interaction               = [(R1 + R4-v1) - R1]
                            - [(R0 + R4-v1) - R0]
```

Call `R1 - R0` a **window-encoder/representation effect**, not a causal
recurrence or carried-memory effect: R1 resets its window encoder every step.
This follows the active architecture's attribution boundary.

For every contrast, report the following vector before any inferential layer:

| Attribution layer | Paired quantities | Question |
|---|---|---|
| Economic | terminal gross, cost, net; net bps/bar | Did the representation earn or preserve modeled net value? |
| Gross source | `G_direction`, `G_timing` | Was gross change average direction or within-episode timing? |
| Activity | gross exposure, signed exposure, short/flat/long shares, position changes | Was the result active, directional, or flat? |
| Friction | total turnover, liquidation turnover, cost at declared scenarios | Was uplift mostly cost avoidance? |
| Path risk | gross max drawdown, per-bar net Sharpe, positive-net-bar fraction | What path did the return require? |
| Benchmark | paired deltas to flat/long/short/random/untrained | Did it beat no action, static direction, and training controls? |
| Concentration | symbol, calendar/session, horizon, time offset, ex-post regime | Where did the result occur? |

Two curricula are not nuisance rows to collapse silently. Report representation
contrasts within exact-cost and staged-cost cells first, then describe whether
the sign and mechanism agree. Likewise, two training seeds are a bounded screen,
not evidence that seed variation has been estimated precisely.

## 8. Practical recommended protocol (attribution layer only)

1. **Lock identities before outcomes:** representation, curriculum, seed,
   checkpoint rule, symbol, session, anchor, horizon, cost scenario, and control
   provenance.
2. **Validate each episode:** complete horizon; exact gross/cost/net identity;
   turnover includes one terminal liquidation; finite returns; matched price
   path across policies.
3. **Produce the mechanical bridge:** gross, cost, net, exposure, timing
   residual, total/liquidation turnover, and activity.
4. **Compare the full benchmark ladder:** report flat, long, short, random, and
   matched untrained separately. If the stronger static direction is summarized,
   label it explicitly as ex-post descriptive selection.
5. **Aggregate transparently:** show per-horizon, per-anchor/session, per-symbol,
   equal-symbol, and raw-task views. State the estimand for each.
6. **Add risk companions:** current per-bar Sharpe and gross drawdown first;
   differential Sharpe, lagged-vol normalization, alpha/beta, or downside/tail
   metrics only when their required data and assumptions are satisfied.
7. **Keep attribution separate from uncertainty:** pass the same paired,
   calendar-aware units to the separately approved Q26 procedure. Do not let a
   ratio's construction stand in for a statistical interval.
8. **Do not collapse to one capability scalar:** a future `CapabilityProfile`
   may retain the vector, scope, benchmark, and uncertainty metadata.

This protocol proposes no block length, confidence level, significance cutoff,
promotion threshold, or new default acceptance rule.

## 9. Minimum future implementation tests

These are test requirements for any later attribution implementation, not code
changes made by this research pass.

1. Exact property test for `net = direction + timing - cost` across long, short,
   flat, flips, and terminal liquidation.
2. Long-only must have `p_bar=1`, `G_direction=long_gross`, and zero timing
   residual; short-only must mirror it; flat must have all components zero.
3. Candidate/control joins must reject missing, duplicated, censored, or
   mismatched symbol/session/start/horizon rows.
4. Cost repricing must change only arithmetic cost/net outputs; net Sharpe,
   positive-net-bar fraction, and net drawdown require step-level recomputation.
5. Zero turnover, zero exposure, and zero volatility must yield an explicit
   undefined/NA diagnostic where a denominator is required, never infinity or an
   arbitrary winning value.
6. Equal-symbol aggregation must be row-order invariant and must not silently
   become raw-task weighting.
7. Any beta, differential Sharpe, or timing computation must verify timestamped
   per-step alignment and benchmark frequency.
8. Any pre-anchor volatility estimate must fail if it touches the outcome
   interval and must specify treatment of intraday seasonality.
9. Ex-post regime labels must remain evaluation metadata and must not appear in
   decision-time observations or task embeddings.

## 10. Assumptions and failure modes

- The exact decomposition assumes current additive return accounting and target
  positions `{-1,0,+1}`.
- The 1 bp cost is a scenario, not a live execution estimate.
- Split-adjusted prices omit dividend cashflows under the active architecture;
  this matters most for longer holding interpretations, even though the current
  horizons are intraday.
- Equal-symbol results describe the declared universe, not a market-cap or
  deployable multi-asset portfolio.
- Correlated equity names and index ETFs reduce effective diversification; a
  symbol count is not an independence count.
- Ex-post directional regimes and the best static direction are descriptive
  selectors and cannot be treated as untouched benchmarks.
- Per-episode Sharpe can be unstable over 96--200 bars, and annualization does
  not repair serial dependence or intraday seasonality.
- Beta/alpha depends on the chosen benchmark and model; omitted factors and
  frequency mismatch can move the intercept.
- Tail ratios and exposure/turnover efficiency ratios are especially fragile
  near zero denominators.
- Checkpoint search, panel selection, training-seed uncertainty, overlapping
  horizons, and calendar-block dependence are not solved by attribution. They
  belong to the separate statistical protocol.

## Primary sources

1. William F. Sharpe, [The Sharpe Ratio](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm),
   1994 -- benchmark-relative differential return, risk normalization, and
   limitations of mean/variance summaries.
2. William F. Sharpe,
   [Asset Allocation: Management Style and Performance Measurement](https://web.stanford.edu/~wfsharpe/art/sa/sa.htm),
   1992 -- passive style exposure and tracking-return interpretation.
3. Andrew W. Lo,
   [The Statistics of Sharpe Ratios](https://alo.mit.edu/publications/page/18/),
   2002 -- estimation error, serial correlation, and time aggregation.
4. Michael C. Jensen,
   [The Performance of Mutual Funds in the Period 1945--1964](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1968.tb00815.x),
   1968 -- market-risk-adjusted performance and its assumptions.
5. Eugene F. Fama and Kenneth R. French,
   [Common Risk Factors in the Returns on Stocks and Bonds](https://doi.org/10.1016/0304-405X(93)90023-5),
   1993, plus the
   [official factor data library](https://mba.tuck.dartmouth.edu/pages/Faculty/ken.french/data_library.html).
6. Andrea Frazzini, Ronen Israel, and Tobias J. Moskowitz,
   [Trading Costs of Asset Pricing Anomalies](https://pages.stern.nyu.edu/~afrazzin/pdf/Trading%20Cost%20of%20Asset%20Pricing%20Anomalies%20-%20Frazzini,%20Israel%20and%20Moskowitz.pdf),
   2012 working paper -- implementation shortfall, impact, turnover, and
   non-proportional cost evidence.
7. Torben G. Andersen and Tim Bollerslev,
   [Intraday Periodicity and Volatility Persistence in Financial Markets](https://www.sciencedirect.com/science/article/pii/S0927539897000042),
   1997 -- intraday volatility periodicity.
8. Alan Moreira and Tyler Muir,
   [Volatility-Managed Portfolios](https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf),
   2017 -- lagged, rather than contemporaneous, variance-based exposure scaling.
9. Frank A. Sortino and Lee N. Price,
   [Performance Measurement in a Downside Risk Framework](https://doi.org/10.3905/joi.3.3.59),
   1994 -- target-relative downside-risk measurement.
10. R. Tyrrell Rockafellar and Stanislav Uryasev,
    [Optimization of Conditional Value-at-Risk](https://doi.org/10.21314/JOR.2000.038),
    2000 -- expected-tail-loss/CVaR formulation.
11. Malik Magdon-Ismail, Amir Atiya, Amrit Pratap, and Yaser Abu-Mostafa,
    [The Maximum Drawdown of the Brownian Motion](https://authors.library.caltech.edu/records/2aprk-1ee98),
    2003/2004 -- maximum-drawdown definition and path-risk motivation.
