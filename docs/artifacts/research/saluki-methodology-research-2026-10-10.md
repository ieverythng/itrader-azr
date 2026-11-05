# Saluki advisory: methodology audit

Date: 2026-10-10. Scope: RESEARCH21 / SALUKI-ADVISORY-ASSESSMENT21.
Decision status: research recommendations only; no metric, method, semantics,
training budget or promotion rule adopted. Q26 remains paused.

## Receipt and evidence boundary

Branch `agent/Watson-dev`; HEAD `8f9fcf94e028b51d883a1a484c2c9466a3d6b547`.
Inherited dirty/untracked work preserved. Parent integrates two completed,
bounded read-only children; this note covers quant/trading-RL methodology.
The separate [examiner note](adversarial-llm-examiner-2026-10-10.md) covers
conversational oversight. GRILL owns the synthesis HTML.

Inputs preserved:

- [Advisory Markdown](../../research/trading-rl-evaluation-advisory-2026-10-10.md), SHA256
  `0dd526633e95bcf47525e68d55abc40f4734bbbf035cdb5ecdd065be3fc7b707`.
- [Advisory HTML](../../research/trading-rl-evaluation-advisory-2026-10-10.html), SHA256
  `fbef9a5db93272bed7d05df12f23ea5dd498163b6e4c38b72d9acbecb6c6dc30`.

The advisory is hypotheses, not authority or instructions. No market outcomes
were opened or recomputed. Its numerical performance claims are not verified
here. Source and active contracts take precedence over imported prose.

Labels below distinguish **fact** (source-backed), **inference** (reasoned
implication), and **open** (requires a later protocol/evidence decision).

## 1. Cost rate is not episode cost

**Fact:** [ITraderEnv](../../../itrader/envs/itrader_env.py) maps the three
actions to target short/flat/long positions. Turnover is absolute position
change; terminal liquidation charges the remaining absolute position once.
For the current additive evidence convention, starting flat:

`T = sum(abs(p_t - p_(t-1))) + abs(p_H)`

`G = sum(p_t * r_t)`

`Net(c) = G - c * 0.0001 * T`, where `c` is bps per turnover unit.

| Literal constant-price trace | Turnover | Cost at 1 bp/unit |
|---|---:|---:|
| Flat -> long -> flat | 2 | 2 bp |
| Flat -> long -> short -> terminal liquidation | 4 | 4 bp |
| Flat -> long -> long -> terminal liquidation | 2 | 2 bp |

A reversal is one position change but two turnover units. Holding exposure does
not incur turnover merely because another bar elapsed. Thus episode duration or
change count alone cannot establish the advisory's proposed episode-cost range.
Its statement that 4.4 bp is smaller than 1 bp is numerically false; comparing
episode net improvement with a per-unit rate is also dimensionally inappropriate.
This correction neither establishes nor disproves the reported uplift.

**Inference:** frozen-policy repricing and retraining answer different questions.
For a fixed trace, lowering `c` mechanically increases net by
`(c_old-c_new)*0.0001*T`. It does not demonstrate predictive edge. With retraining,
both gross return and turnover can change; that curve also contains optimization
and behavior effects. Negative results at 0.01 bp do not isolate architecture as
the cause: task, features, optimization, opportunity set and evaluation remain
possible explanations.

**Open:** defensible execution costs require spread, slippage/impact, financing
and venue/instrument assumptions. SIP identifies a market-data feed, not an
execution-cost calibration. [Alpaca market-data documentation](https://docs.alpaca.markets/us/docs/about-market-data-api)
does not justify calling 1 bp, 0.1 bp or 0.01 bp realistic execution rates.
Cost sensitivity may be diagnostic; the accepted fixed-1-bp primary is unchanged.

## 2. Turnover penalties and action aliases

**Fact:** repeated target-long means hold long; target-flat closes an existing
position. Appended WAIT and EXIT therefore alias existing transitions under
current mechanics unless given genuinely different meanings. An online
`WAIT/NO_ACTION` gate outcome is not offline flat: flat can order a liquidation.
Position-target actions are also an explicit design in the
[official Gym-Trading-Env tutorial](https://gym-trading-env.readthedocs.io/en/latest/rl_tutorial.html).
That is a design precedent, not evidence of local performance.

**Inference:** five labels instead of three change policy probabilities, entropy
and learning even if transitions alias. Sequential decisions already exist with
Discrete(3); sequential dependence does not require more labels. Adding a fixed
penalty above ten changes creates a new objective, not an accounting correction.
Trade count alone cannot diagnose overtrading without horizon, turnover,
exposure and gross/net benefits. Keep optimization reward separate from economic
evidence; do not double-count costs.

**Open:** size/stop/hybrid outputs need new contracts and restore identities;
broker semantics and safety remain with deterministic RiskEngine/Executor, not
the solver. No such changes are adopted here.

## 3. Chronology is necessary, not a freshness certificate

**Fact:** [DataPartition](../../../itrader/rl_lib/evaluation/data_partition.py)
owns eligible starts and contiguous segments; split proportions are per-symbol
rows, not necessarily shared calendar fractions. Qualified eligibility must
cover warmup and full outcome horizon, timestamp continuity and session rules.
[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
uses ordered, successive training sets and notes equal sample spacing for
comparable fold durations. One chronological replay is not repeated refitting.

**Inference:** a chronological split cannot erase prior use in training,
selection, prompts, diagnostics or transform fitting. A newly run replay is
fresh execution, not fresh information. Refitting on the first 85% after tuning
on 70/15 produces a different checkpoint and consumes validation into training;
an old interval cannot be attributed to that refit model. A prespecified
refit-and-final-test protocol can be legitimate, but is not automatically the
current protocol.

**Open:** purge must follow the information intervals and causal support of
outcomes, lookback and derived features, not a universal 500/1000-row rule.
Pre-boundary context already available at prediction time is not automatically
leakage, though dependence and reuse still need disclosure. Verify symbol,
missing-minute and session boundaries rather than assuming row counts suffice.
Under a same-session full-window rule, 240 lookback bars plus 200 outcome bars
cannot fit a 390-minute regular session; early-close eligibility is tighter.
Changing window size can change the eligible panel and confound comparisons.

## 4. Training exposure does not determine statistical significance

**Fact:** forty times 4,800 is a nominal sampled-transition budget, not a count of
unique market bars, independent episodes or SGD updates. PPO alternates sampling
and minibatch optimization; GAE is a temporally weighted value-residual
estimator, not an environment-owned complete-episode-only calculation.
[PPO paper](https://arxiv.org/abs/1707.06347),
[GAE paper](https://arxiv.org/abs/1506.02438).

**Inference:** neither 192k nor 1.9-19M steps establishes convergence or an honest
universal budget. More training does not enlarge the held-out calendar or fix
thin dependence support. Two frozen checkpoint pairs support a conditional
comparison, not a population-wide assertion about seeds. An interval cannot be
predicted to cross zero from cost rhetoric alone. Nominal cell count is not a
resource estimate or statistical qualification.

**Fact:** serial dependence affects Sharpe inference and annualization;
selection among alternatives changes inferential requirements.
[Lo, Statistics of Sharpe Ratios](https://traders.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf),
[White, Reality Check](https://onlinelibrary.wiley.com/doi/10.1111/1468-0262.00152),
[Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).
These sources motivate preserving dependence and search history; none adopts a
replacement method for Q26 or makes DSR a drop-in correction for its endpoint.

**Open:** later learning-curve comparisons need matched controls, prespecified
checkpoints/resource budgets, and complete trial history. A 4-window x 3-cost x
2-seed grid is 24 cells for one architecture, 48 for two before other factors;
selection and runtime budgets must be declared. Do not substitute an arbitrary
R0 ceiling sweep for the accepted staged matched-R1 plan.

## 5. Normalization and capability boundaries

**Fact:** current market-relative preprocessing preserves OHLC relative price
amplitudes; it does not z-score every price column. Volume uses log1p and local
normalization. The advisory's all-feature-z-score description is inaccurate.
Uniform positive price rescaling cancels in the relative-price expression,
apart from numerical floors/clipping; volatility amplitude is not wholly lost.

**Inference/open:** causal rolling volatility context is plausible as a separate
feature ablation, not an established rescue. A fitted constant does not encode
current regime; precomputed features can still leak. 252 minute bars are not a
year. Added input width does not guarantee checkpoint compatibility.

The [Q23-Q26 boundary](q23-q26-decision-chain-2026-10-09.md) remains binding:
`terminal_screen_only` evidence must not feed proposer reward, curriculum, PPO,
tuning or operational capability calibration. CapabilityProfile is an
online/shadow-facing scaffold; offline contract preparation does not clear
evidence for operational use. No capability promotion follows this audit.

## 6. Decision-ready disposition

| Disposition | Recommendation |
|---|---|
| Adopt now as interpretation/reporting discipline, not new behavior | Correct units; distinguish gross/net/reward and repricing/retraining; retain all attempts, provenance, checkpoint/panel identities and negative results. |
| Ablate only after separate authorization | Matched cost sensitivity, bounded training curves, causal volatility features and genuine rolling refits; keep diagnostic factors separate from the fixed primary. |
| Defer | Broad grids, larger action spaces, arbitrary turnover floors/penalties, larger purges, new final-test promotion gates and operational capability calibration. |
| Reject as conclusions | Low costs prove edge; low-cost losses prove architecture bottleneck; 1 bp is SIP execution cost; 10-100x is inherently honest training; chronological split/purge guarantees freshness; additional WAIT/EXIT labels are free expressivity. |

Next decision: retain these reporting corrections and choose, if desired, one
separately authorized matched diagnostic after existing clearance gates. No
automatic experiment dispatch, Q26 resumption or plan replacement.

Validation receipt and output hashes are on the
[owned RESEARCH board](../../coordination/research.md). No market outcomes,
acquisition, model calls, simulation, imports/jobs, installations, WSL, code
changes, tests, heavy lease or Git shipping were used.

## 7. RESEARCH23: two chronological protocol routes, not a selected split

2026-10-10 compact follow-up. This is a proposal for a separately approved
protocol, not a change to current partitions or reserved evaluation. No raw
data, new metadata projection, calendar split or actual eligibility was accessed
or chosen. Q26's existing question window remains with GRILL.

### Common calendar and complete information support

**Source fact:** [DataPartition](../../../itrader/rl_lib/evaluation/data_partition.py)
currently forms percentage boundaries separately from each symbol's row count.
It does not thereby give all symbols the same calendar cutoffs. Its eligible
outcomes lie inside the assigned split while causal observation context may
precede that split boundary. That context is not automatically leakage.

**Protocol proposal:** freeze common calendar boundaries across symbols before
selection, with timezone, session policy and endpoint inclusivity declared.
Then determine eligibility per symbol from complete information intervals;
shared dates do not guarantee shared eligible starts. Record missing-bar/session
attrition, symbol coverage and matched-panel rules rather than silently comparing
different opportunities. Such common-calendar behavior would require later
implementation and validation; it is not claimed available now.

Use `W` for observation lookback (the requested `T`), since this note already
uses `T` for turnover. For each decision/task, bind the earliest observation or
derived-feature dependency and the latest reward/outcome/terminal-valuation
dependency. Include rolling reward/statistic support, successor-price references
where required, causal warmup and transformation-fitting lineage. Purge and
eligibility follow those actual intervals, not a universal `W+H` or fixed-row
buffer. The existing 60-bar context / up-to-200-bar outcome convention is a
known contract, not a chosen new split.

A future strict information-disjoint protocol would purge cross-boundary
overlap of the complete observation/reward footprint. If already-available
pre-boundary context is instead permitted, declare that explicitly, prohibit
forward outcome/fit leakage and disclose shared-context dependence. The choice
remains open; neither silently reinterprets the current split or certifies
historical freshness. No purge size or interval-selection method is adopted.

### Conditional ways forward

| Route | Intended use | Prerequisites and limitations |
|---|---|---|
| Historical chronological redevelopment | Development/validation on explicitly designated retained historical evidence, with matched fresh-initialization controls. | Record all known training, prompts, diagnostics and human/model selection exposure. Previously inspected evidence stays development evidence; label results descriptive, not a fresh confirmatory test. Dates and access remain unselected. |
| Separately cleared confirmation | A prospective or demonstrably exposure-cleared interval/source, with choices frozen before permitted access. | Separate source/access/exposure clearance, common-calendar eligibility, frozen protocol and run authority are necessary. A date label alone cannot establish untouched evidence. No acquisition or confirmation is authorized here. |

Fresh solver initialization excludes checkpoint, optimizer and donor carryover
under a declared initialization contract. It does **not** reset researcher/agent
knowledge, selection history, proposer/model training exposure or transform
fitting. Track initialization and information exposure as separate receipts.
2025-2026 is not declared fresh by being later; the reserved 2020-2021 Q23
envelope cannot become training/development data without a new explicit decision.
Dataset-v2's 2022 start also cannot provide that earlier reserved envelope.

**Smallest later recommendation for GRILL:** distinguish historical development
from separately cleared confirmation before approving actual calendar bounds
and information-span rules. Preserve existing Q23/Q26 primary, weights, gates
and terminal-screen-only restrictions. No direct human question, split selection,
new method, training or reserved-evidence consumption follows this note.

## 8. RESEARCH25: selection exposure and a small causal-feature pilot

2026-10-10; research-only extension, not a new split or a repeat of section 7.
CONTINUATION25 authorizes this investigation, not market access, training,
Q26 calibration, feature implementation or metric/method adoption. Source
identities and output hashes are recorded on the owned RESEARCH board.

### Selection inflation is not a finite reservoir of dataset gain

**Cited facts:** optimizing a noisy selection criterion can itself overfit;
adaptive reuse can also overfit a holdout. More candidate searches can increase
the maximum observed score without corresponding skill. These findings support
selection controls, not a guarantee that each extra trial worsens true performance.
[Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html),
[Dwork et al.](https://arxiv.org/abs/1506.02629),
[DSR](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).

**Local inference:** distinguish three budgets: sampled optimizer experience;
unique chronological market/economic support with overlap and dependence; and
candidate-selection attempts/exposure. More passes through the same bars can
improve fitting without adding independent evidence. Different symbols sharing
calendar shocks, overlapping horizons and nearby sessions are not automatically
independent observations. Neither raw bar count nor raw trial count determines
a universal effective sample size; the relevant dependence is estimand-specific.

Reject the metaphor of a measurable stock of already-extracted versus remaining
dataset gain. A plateau does not establish exhausted predictive information;
an improved backtest does not establish newly discovered generalizable signal.
Representation, optimization, reward, selection noise and regime change can all
affect observed scores. Discoverable edge, achievable capacity and stability are
open empirical questions, not quantities recoverable from a learning curve.

**Proposed discipline:** retain a trial/exposure ledger at existing experiment
and provenance owners, not a parallel framework. Bind candidate and parent IDs,
architecture/feature/reward/cost/window choices, seeds/checkpoints/budgets,
source/panel/calendar identities, transform fit lineage, inspected metrics and
prompt/agent/human choices. Preserve failures, abandoned trials and selected
checkpoints. An exposure record is broader than whatever candidate grouping a
later statistical diagnostic requires; do not count every SGD update as an
independent strategy trial or infer complete history from the winner alone.

### Conditional evaluation and the limited role of PBO/DSR

**Proposal:** within separately authorized historical development evidence,
inner earlier-train/later-validation folds select from a small frozen feature
menu; outer later folds assess the complete selection procedure, with all fits
redone inside the corresponding earlier training fold. Nesting separates
selection and assessment; chronological eligibility and information support
still follow section 7. This is a local adaptation, not direct validation of a
financial protocol by an IID classification example.
[Official nested-CV guidance](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
If inspected outer results drive redesign, they become development exposure.
Nesting neither erases old exposure nor grants a fresh-test certificate.

**Cited fact / transfer limit:** PBO describes selection instability, including
an in-sample winner ranking below the out-of-sample median of considered
candidates. Its CSCV implementation needs comparable candidate performance
series; it is not a posterior probability that a particular strategy has no
edge. DSR adjusts Sharpe inference for selection and non-normality using inputs
including return-series length/moments and independent-trial information.
[PBO](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf),
[DSR](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).
Both are deferred diagnostic candidates, not drop-in corrections for Q26's
paired fixed-symbol-mean net endpoint, thin calendar support or incomplete trial
history. Neither certifies freshness, repairs incorrect costs/leakage or selects
Q26 dependence ranges. No PBO/DSR statistic is computed or adopted here.

**When new data helps, conditionally:** genuinely later/unexposed, separately
authorized support can assess frozen choices and broaden regime coverage.
More highly correlated rows or another vendor's duplicate history need not add
equivalent information. Extra history does not fix broken accounting or leakage;
more old training data does not make already-inspected evaluation fresh.
Access, exposure clearance, comparability and representativeness remain open.
No acquisition, date selection or reservation release follows this reasoning.

### Verified current transform and candidate features

**Source facts:** [ITraderEnv](../../../itrader/envs/itrader_env.py),
`_get_obs` / `_market_relative_window` (lines 342-370), uses the trailing visible
buffer. OHLC is divided by the last visible close, minus one. Volume uses
`log1p(max(volume,0))` and visible-window mean/std normalization. Other non-OHLC
columns receive that same window-standardization; sanitization and clipping
apply. This is fitted-state-free, not a global train-fitted z-score. The
[tagged-observation integration test](../../../itrader/tests/test_semantic_observation_contract.py)
checks chronology and literal OHLC/volume reconstruction; no test is run here.

**Small proposed pilot after the R1 representation/platform/evidence contract
is frozen:** reuse C0/current, C1/causal and shuffled-C1 identities rather than
inventing another research series. Begin with one fixed compact market-summary
bundle: last completed close return; completed-bar high/low range relative to
close; trailing return volatility; and trailing volume context. These are
proposals, not implemented inputs or evidence of predictive value. Preserve the
same visible lookback initially: deterministic summaries of those bars add
representation, not new information. Longer support is a later explicit
information/warmup factor, not attributed to feature encoding alone.

| Candidate | Causal availability / scope |
|---|---|
| Returns and range | Only completed bars available before the next action; no next-close return, future high/low or centered window. Validate price floors and session continuity. |
| Trailing volatility and volume | Fixed past-only support; no future volatility target, full-session statistic before close, or backward fill from later bars. Distinct scale summaries may complement current window-normalized volume, but benefit is unknown. |
| Session context | Known timestamp/session calendar only; later separately approved addition, not another initial grid axis. No future session outcome or hindsight regime label. |
| Endogenous solver state | Keep the existing position/remaining-horizon R4-v1 contract fixed in this pilot. PnL, holding age, turnover and drawdown remain separate state factors with reset/accounting contracts; do not silently merge them into C1 or alter R4-v2 joint fusion. |

The shuffled control must be frozen independently of outcomes and respect the
allowed evidence split; its exact construction is open and it is not proof of
causality by itself. Matching dimensions, optimizer budget and compute, with
explicit inert channels/capacity controls where necessary, prevents extra model
capacity masquerading as feature benefit. No numerical run budget is chosen.
Split-local shuffling alone does not establish causal availability or preserve
temporal dependence; declare control timing and any deliberately non-deployable
diagnostic separately from the real causal feature path.

### Owners, oracles and decision

The later training preprocessing/runner owns learned scales, thresholds or
feature-selection fits: fit on training only, serialize fit/source identities,
then transform validation/test without refitting. A visible-window statistic
is a different, causal fitted-state-free operation. Validation may choose a
recipe, but must not supply its fitted constants. [Official leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
`ITraderEnv` owns causal observation assembly; model/config owners must version
width, order, timing and restore identity. Appending columns under the current
generic window-standardization can destroy intended session/summary semantics;
a reviewed explicit feature interface is needed, not silent column addition.
Evaluation/provenance owns matched panels and assessment, with GRILL selecting
the protocol and DEV/TRAINING performing only later authorized implementation
and execution. Terminal-screen-only evidence stays outside fits/tuning.

Proposed public integration oracles: **PrefixCausality** (change future bars,
earlier features unchanged); **TrainFitIsolation** (alter validation/test,
training fit unchanged); **LiteralFeatureTiming** (hand-computable completed-bar
fixtures, gaps and warmup); **MatchedIdentityRestore** (same eligible panel and
declared schema/checkpoint, incompatible restore fails). These are designs,
not new tests or claims of passing them.

**Recommend now as research controls:** complete exposure/fit lineage and the
selection-versus-information distinction. **Ablate later:** the small matched
C0/C1/shuffled-C1 pilot and, if separately approved, nested chronological
development. **Defer:** PBO/DSR implementation, additional feature grids/session/
state factors and stress curricula. **Reject:** exhausted-gain estimates from
plateaus, features as guaranteed edge, all-data fitting and a new test label as
freshness. Actual menu/support, shuffled control, fit protocol, development
envelope, budgets and statistical criteria remain open. The
[separate market-stress note](market-scenario-stress-2026-10-10.md) changes
neither this feature pilot nor Q26.
