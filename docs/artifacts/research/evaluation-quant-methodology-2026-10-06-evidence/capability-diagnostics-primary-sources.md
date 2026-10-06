# Causal capability diagnostics: primary-source research note

Date: 2026-10-06
Scope: research evidence for the proposed "60 past bars + one immediate step
uplift/PnL/Sharpe normalized by volatility" diagnostic. This note does **not**
change code, configuration, rewards, model inputs, evaluation panels, promotion
criteria, or active architecture.

## Executive conclusion

The useful core of the proposal is **not** a one-step Sharpe ratio. It is an
out-of-sample, cost-aware **action sensitivity vector** evaluated after the
solver has committed to one of its three target positions:

1. use only information available through decision time to produce the action,
   a volatility scale, and any benchmark loading;
2. use the next realized bar only as the scored outcome;
3. report raw gross return, switch cost, net return, exposure, and net advantage
   versus predeclared controls;
4. optionally divide those return-space quantities by one common, strictly
   lagged volatility estimate for cross-regime comparability;
5. keep the hindsight-best action and its gap to the policy explicitly labelled
   as an **oracle diagnostic**, never as achieved return or capability; and
6. measure uncertainty, calibration, and statistical significance separately
   over a sufficiently supported evaluation panel.

This vector is a causal **diagnostic of local decision quality**, not a complete
estimate of terminal trading capability. Terminal capability still requires the
predeclared multi-step policy, exact costs including genuine terminal
liquidation, and the existing episode evidence contract.

## Epistemic labels

- **Verified local fact** means the statement follows from current iTrader
  source or active architecture.
- **External fact** means the statement follows from a cited primary paper,
  author page, publisher page, or official source.
- **Inference** is the recommended interpretation of those facts for iTrader.
- **Hypothesis** is a design candidate that still needs prospective validation.

## 1. Current iTrader contracts that constrain the diagnostic

### Verified local facts

- `ITraderEnv` has `Discrete(3)` actions mapped to target positions
  `{-1, 0, +1}`. At each step it first sets the target position, then obtains the
  next market return, computes gross mark-to-market return as position times
  return, and records turnover as the absolute change from the prior position.
  The market row transition itself does not depend on the action. See
  [`itrader_env.py`](../../../../itrader/envs/itrader_env.py#L613).
- `EpisodeEvidence` calculates economic evidence separately from the optimized
  reward. It adds liquidation turnover to total turnover, subtracts the
  liquidation cost from the **last** net-return observation, and returns Sharpe
  as `0.0` whenever fewer than two returns are present. See
  [`episode_evidence.py`](../../../../itrader/rl_lib/evaluation/episode_evidence.py#L15).
- The registered `net_return` reward charges position-change cost each step and
  liquidation cost only when the environment says the step is terminal. See
  [`rewards.py`](../../../../itrader/rl_lib/tasks/rewards.py#L266).
- `CapabilityProfile` is currently a storage dataclass with identity fields,
  `estimated_p_solve`, `expected_net_utility`, `calibration_error`,
  `sample_count`, and metadata. It does not define how any of those quantities
  are estimated or calibrated. See
  [`capability_profile.py`](../../../../itrader/rl_lib/tasks/capability_profile.py).
- The active architecture places `WAIT` beside candidate-task generation and
  upstream of solver scoring; it is not a fourth solver action. It also says
  PPO logits/value outputs are not automatically calibrated confidence or
  expected net utility. See
  [`itrader_system_architecture.md`](../../../architecture/itrader_system_architecture.md#L360).

### Immediate consequences

1. A one-bar iTrader `EpisodeEvidence.net_sharpe_per_sqrt_bar == 0` would be a
   **code convention**, not evidence of zero risk or a statistically meaningful
   Sharpe estimate.
2. Applying the existing terminal liquidation charge to every one-step probe
   changes the estimand into a forced one-bar round trip. Omitting it describes
   a continuation decision. These are different economic questions and must not
   share a label.
3. Because current historical prices are exogenous to actions in the simulator,
   all three immediate action payoffs can be reconstructed exactly for a fixed
   anchor under the simulator's price-taker assumptions. That does not show that
   the ex-post best action could have been selected at decision time, and it does
   not validate the same counterfactual under live market impact.
4. Any new diagnostic should be an evaluation-side artifact. It must not
   silently alter `RewardEngine`, `pnl_linear`, observations, or the policy's
   three-action contract.

## 2. Why "one-step Sharpe" must be rejected

### External facts

William Sharpe defines the ex-post ratio from a **series** of differential
returns: its historical average divided by their historical standard deviation.
He also distinguishes ex-ante from ex-post use, notes that the ratio is
time-period dependent, and warns that correlation and other distributional
features may need to supplement it ([Sharpe 1994, author-hosted
copy](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)). Andrew Lo derives the
sampling distribution under different return processes and shows that ordinary
square-root-of-time annualization only holds under special conditions; serial
correlation can materially alter both estimated ratios and rankings ([Lo 2002,
author publication page](https://alo.mit.edu/publications/page/18/)).

Moody and Saffell's differential Sharpe ratio is an incremental objective for a
sequential trading system that accounts for transaction costs; their paper does
not redefine the Sharpe ratio as the realized score of a single return
([Moody and Saffell 2001, IEEE record via
PubMed](https://pubmed.ncbi.nlm.nih.gov/18249919/)).

### Inference for iTrader

- A single outcome has no estimable sample standard deviation. Calling
  `one_step_return / lagged_volatility` "Sharpe" conflates a realized
  standardized return with an estimated mean-to-standard-deviation ratio.
- Dividing an already computed episode Sharpe by another volatility estimate is
  also not a standard Sharpe construction and has no clear economic units.
- A panel can later summarize the **distribution** of one-step differential
  returns. Its mean, dispersion, and uncertainty remain panel-level statistics
  subject to dependence and support analysis; they are not "one-step Sharpe."
- Therefore the diagnostic should use names such as
  `one_step_net_return`, `one_step_active_return`, and
  `one_step_net_return_per_lagged_vol`. Statistical uncertainty belongs in a
  separate layer and must not be asserted by the metric's name or formula.

## 3. A cost-correct immediate action-value contract

Let:

- `q_t` be the position immediately before the decision;
- `a` be a candidate target position in `A = {-1, 0, +1}`;
- `r_(t+1)` be the realized next-bar asset return; and
- `c` be the predeclared cost per unit turnover in return units.

For a **continuation** probe, define:

```text
gross_1(a)       = a * r_(t+1)
switch_cost_1(a) = c * abs(a - q_t)
net_continue_1(a)= gross_1(a) - switch_cost_1(a)
```

For a genuinely terminal, forced one-bar round trip, define a separate score:

```text
net_roundtrip_1(a) = net_continue_1(a) - c * abs(a)
```

The second expression includes liquidation; the first intentionally does not.
The diagnostic must record which estimand it used. A short-to-long or
long-to-short switch has turnover `2`, while entering or exiting a unit position
has turnover `1`, matching the current environment's arithmetic.

### Recommended per-anchor outputs

Record the vector below before aggregation:

- prior position `q_t`, selected action `a_policy`, and the three candidate
  target positions;
- gross return, switch turnover/cost, optional true-terminal liquidation cost,
  and net return for each action;
- policy-selected net return;
- net advantages versus each predeclared baseline;
- hindsight-best action, hindsight-best value, and oracle gap, all explicitly
  marked `diagnostic_only`;
- lagged volatility value, estimator identity, source bars, and availability
  flag;
- symbol, timestamp/session, horizon/task identity, cost identity, solver and
  checkpoint identity; and
- exposure and action-change flags.

This preserves enough information to detect a policy that appears profitable
only because it is nearly always short/long, rarely trades, or benefits from a
particular cost convention.

## 4. Lagged volatility normalization: useful, but not self-validating

### External facts

Andersen, Bollerslev, Diebold, and Labys formally connect realized volatility
constructed from high-frequency intraday returns to quadratic variation and use
it for volatility and return-distribution forecasting ([Econometrica publisher
record](https://onlinelibrary.wiley.com/doi/10.1111/1468-0262.00418)). Moreira
and Muir scale monthly factor returns by the inverse of the **previous** month's
realized variance; their evidence relies on lagged volatility being strongly
related to subsequent volatility while being weakly related to subsequent mean
returns in their studied assets and horizons ([published author-hosted
paper](https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf)).

### Inference for iTrader

A volatility denominator can be causal when it is calculated entirely from bars
available through `t` and frozen before `r_(t+1)` is revealed. The paper evidence
supports lagging the estimate; it does **not** establish that exactly 60 one-minute
bars, a particular estimator, or the same persistence relationship is valid for
iTrader's symbols and intraday horizons.

### Hypothesis to evaluate, not silently adopt

One candidate per-bar scale is a 60-return trailing RMS:

```text
sigma_hat_t = sqrt(mean(r_j^2 for j in the predeclared trailing window ending at t))
scaled_net_1(a) = net_continue_1(a) / sigma_hat_t
```

Alternatives include a trailing sample standard deviation, an EWMA estimate, or
a session/time-of-day-conditioned scale. The choice is a named factor and must
be made without examining the scored future outcomes.

The following constraints are more important than the estimator choice:

- use the **same** `sigma_hat_t` for every candidate action and every compared
  solver at the same anchor;
- divide the complete net-return quantity, including cost, if the result is
  called net return per risk unit;
- always retain raw gross, cost, and net values alongside the normalized value;
- record the estimator and exact causal source interval;
- treat non-finite or zero scale as `normalized_value_unavailable` with an
  explicit zero-volatility flag; do not convert it into a win, zero-risk claim,
  infinity, or an outcome-tuned epsilon; and
- never interpret the scaled realization as a confidence score, z-statistic,
  Sharpe ratio, or proof of significance.

## 5. Benchmark-relative scoring

Sharpe's formulation is explicitly based on **differential return** against a
benchmark. He describes active/selection return as the difference from a
similar passive alternative and notes that a benchmark can be chosen to match
factor loadings or investment style ([Sharpe 1994](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)).

No single benchmark answers every iTrader question. Preserve several named,
non-interchangeable contrasts:

| Baseline | Immediate question | Important caveat |
|---|---|---|
| `hold_prior_position` (`a=q_t`) | Did the new decision improve on doing nothing at this step? | Best local action-change attribution; "hold" is not the same as flat. |
| `flat_target` (`a=0`) | Did the selected exposure beat cash/zero exposure after transition costs? | If `q_t != 0`, flattening itself incurs a close cost. |
| matched `long_only` / `short_only` | Is the solver more than a directional exposure rule on the exact same anchor? | Must retain the same cost and terminal convention. |
| predeclared passive/factor benchmark | Is return incremental to market/style exposure? | Any beta/loading must be estimated only with information through `t`; the future benchmark return is an outcome, not a decision feature. |

For a predeclared baseline action `b_t`, the immediate active return is:

```text
active_net_1 = net_continue_1(a_policy) - net_continue_1(b_t)
```

For an optional market-residual diagnostic, a lagged-only loading could define:

```text
residual_gross_1(a) = a * (r_asset_(t+1) - beta_hat_t * r_market_(t+1))
residual_net_1(a)   = residual_gross_1(a) - switch_cost_1(a)
```

This is permissible outcome evaluation only when `beta_hat_t` and the selected
action were frozen before both future returns. A 60-bar intraday beta may be too
unstable to be useful; that is an empirical hypothesis, not a default. Exact
same-anchor flat/long/short controls are simpler and should remain visible even
if a residual measure is later added.

## 6. Decision-time causality versus hindsight oracle leakage

### External facts

In contextual-bandit evaluation, the policy receives context and selects an
action; only then is its reward observed. Replay evaluation is unbiased under
the paper's randomized-logging assumptions because it retains events where the
logged action agrees with the target policy, not because it lets the target
policy inspect every realized reward before choosing ([Li et al. 2011,
Microsoft-hosted paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Published-3.pdf)).
Doubly robust evaluation likewise estimates a **fixed target policy** from
historical context/action/reward data and makes explicit assumptions about the
reward model or past-policy propensities ([Dudik, Langford, and Li 2011,
official Microsoft Research page](https://www.microsoft.com/en-us/research/publication/doubly-robust-policy-evaluation-and-learning-2/)).
Online-learning regret compares the action actually selected with a reference
available only in hindsight; the agent still chooses before receiving the
payoff ([Auer et al. 2002, author-hosted paper](https://homepages.math.uic.edu/~lreyzin/papers/auer02.pdf)).

### Inference for iTrader

The exogenous replay path gives iTrader richer **full-information diagnostic
feedback** than a standard logged bandit: it can mechanically score all three
one-step positions. However:

```text
oracle_value_t = max_a net_continue_1(a)
oracle_gap_t   = oracle_value_t - net_continue_1(a_policy)
```

is a hindsight label. It is useful for measuring action separability, regret,
or an upper bound under the simulator. Reporting `oracle_value_t` as the policy's
PnL, using the future-best action as though the solver chose it, or selecting a
metric/threshold because it maximizes these same outcomes is look-ahead leakage.

The action must be materialized and frozen from the causal observation before
loading the outcome. A future return may then be used to **score** that decision;
it may not be fed back into that decision or its ex-ante volatility/benchmark
features.

## 7. One-step versus multi-step and terminal capability

| Evaluation object | What it identifies | What it cannot establish |
|---|---|---|
| One-step continuation vector | Local action sensitivity, switch-cost awareness, directional correctness, and immediate active return | Long-horizon state use, later turnover, drawdown, liquidation, or full task success |
| One-step forced round trip | Profitability of enter/change-and-liquidate within exactly one bar | Continuation value; it structurally favors flat when costs dominate |
| Fixed-action hold for `H` bars | Controlled directional exposure response over a named horizon | Quality of a solver that can revise actions each bar |
| Branch one action, then follow policy | Consequence of the first action under a defined continuation policy | Not valid unless environment, inventory, model/recurrent state, and randomness are cloned consistently |
| Full terminal replay | Actual policy economics under the task horizon and true terminal liquidation | Still requires panel-level uncertainty and comparison with controls |

For R4 state, the first action changes position and therefore changes future
policy observations. For recurrent solvers, it may also alter later state via
action/observation trajectories. A multi-step counterfactual cannot reuse one
shared mutable state for all branches. If branch evaluation is later attempted,
each branch needs an independent clone and a predeclared continuation policy.

The 60-past/one-next diagnostic should therefore be treated as a **microscope**
for decision quality, not as a replacement for 96/128/160/200-bar terminal
comparisons.

## 8. Offline evidence first; CapabilityProfile is a future online scaffold

### External facts

Proper scoring rules evaluate a predictive distribution against the event that
materializes; strict propriety rewards reporting the forecaster's actual belief.
Gneiting and Raftery also distinguish calibration (statistical consistency of
forecasts and outcomes) from sharpness and discuss Brier, logarithmic, and
continuous ranked probability scores ([Gneiting and Raftery 2007,
author-hosted paper](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)).

### Required two-stage boundary

```text
held-out offline capability evidence bundle
  = metric vector + uncertainty + calibration + coverage + provenance
                |
                | separately specified, versioned, held-out transformation
                v
future online/shadow CapabilityProfile
  = eligibility and task/capability matching evidence for the deterministic shell
```

The first object is the scientific authority for what was measured. The second
is a compact operational projection for future replay/shadow eligibility and
matching. The transformation needs its own version, source-panel identity,
freshness/staleness rules, and validation; fields in the current dataclass do
not create those semantics by themselves.

`CapabilityProfile` is therefore a scaffold, not a current estimator, solver
reward, training input, unapproved proposer feedback signal, or Phase-A/R1
promotion authority. This restriction is scoped to the present R-series
evaluation and unapproved diagnostic-to-training feedback. It does not prohibit
a separately approved Phase B adapter from transforming calibrated held-out
validation or rolling capability evidence into versioned `r_prop` or curriculum
feedback. Terminal-screen and sealed final-test evidence must remain isolated
from adaptive proposer updates. The offline bundle should remain inspectable
even after a future profile is derived from it, so operational compression
cannot erase benchmark, horizon, cost, coverage, or uncertainty qualifications.

### Inference for iTrader

The present fields should not force premature scalar compression:

- `estimated_p_solve` needs a defined event, horizon, baseline, and estimator.
  Examples might later include the probability of positive **terminal** net
  advantage against a named control or task success under the existing contract.
  The fraction of hindsight-correct one-step labels is not automatically
  `p_solve`.
- `expected_net_utility` needs a predeclared utility/return estimand, costs,
  horizon, and evaluation population. It must use the policy's frozen action,
  not `max_a` hindsight return.
- `calibration_error` is meaningful only if a model produced a probability or
  predictive distribution for a defined outcome. Without that, the truthful
  value is unavailable, not `0.0`.
- `sample_count` is not a substitute for effective support. Symbol, session,
  calendar block, horizon, action, exposure, regime, and zero-volatility coverage
  should accompany the estimates; dependence-aware uncertainty is a separate
  statistical-protocol concern.

A research sidecar can first hold the versioned offline evidence bundle without
changing the dataclass. A future validated reducer may map selected estimates
into an online/shadow `CapabilityProfile`, but no aggregate should claim
statistical significance by definition. Metric construction answers "what is
measured?"; uncertainty answers "how precisely is it known?"; calibration
answers "do stated probabilities match frequencies?"; operational projection
answers "what compact evidence may the future shell consume?" These are
separate layers.

The [implementation specification](../../../iTRADER_Codex_Implementation_Spec.md)
also anticipates offline and online capability consumers, while the
[masterplan](../../../plans/itrader_masterplan.md) retains a later
solver-feedback-to-`r_prop` proposer seam. A future offline consumer therefore
needs its own versioned adapter, declared validation/rolling source, and leakage
tests. It is not authorized by the current dataclass, and terminal/final
evaluation evidence must not be recycled into adaptive proposer training.

## 9. Required null and edge controls

1. **Flat with zero prior exposure:** `q_t=0, a=0` has zero gross return and zero
   switch cost. This is absence of exposure, not demonstrated skill.
2. **Flatten an open position:** `q_t!=0, a=0` has zero gross next-bar exposure
   but nonzero close turnover/cost. Treating every flat action as zero PnL would
   be wrong.
3. **Zero or unusable lagged volatility:** preserve raw values, set normalized
   values unavailable, and count the case. Do not hide it behind an epsilon.
4. **No-action/hold versus flat:** `hold_prior_position` and `flat_target` are
   distinct whenever `q_t!=0`.
5. **One-action policies:** always-short, always-flat, and always-long controls
   on the exact same anchors expose directional beta and near-flat basins.
6. **Oracle-only positivity:** a positive mean of `max_a net_1(a)` can arise
   mechanically from choosing after seeing each outcome. It is not policy edge.
7. **WAIT separation:** WAIT/NO_ACTION remains a deterministic online-shell
   outcome. Do not encode it as solver action `3`, merge it with target-flat, or
   infer its value from a three-action one-step oracle.

## 10. Minimum implementation tests for a later engineering pass

These are contract tests, not acceptance thresholds.

### Causality and leakage

- Freeze observation, policy action, lagged volatility, and lagged benchmark
  loading at `t`; perturb every row after `t`; assert all decision-time values
  remain byte/equality stable while outcome scores may change.
- Assert the policy action is persisted before the outcome is accessed.
- Assert any `argmax` over realized action payoffs is named and serialized only
  under `oracle`/`hindsight`, never under policy return or utility.
- Time-shift or permute outcomes as a negative control: policy advantage should
  not inherit oracle selection, while the explicitly marked hindsight maximum
  may remain optimistically positive.

### Economic arithmetic

- Exhaustively test all nine `(q_t, a)` pairs for target-position mapping,
  turnover `0/1/2`, gross return sign, switch cost, and net continuation return.
- Test continuation and forced-round-trip semantics separately. Liquidation is
  charged only for the latter or at the actual terminal step.
- For a full action trace, assert the evaluation-side arithmetic reproduces
  `EpisodeEvidence` gross return, turnover including liquidation, cost return,
  and net return without calling or changing the optimized reward.
- Assert the same predeclared cost identity is used for policy and baselines.

### Volatility and benchmark construction

- Assert every volatility source index is `<= t`, its estimator identity is
  recorded, and the same denominator is used across actions/solvers at an
  anchor.
- Assert zero/non-finite scale produces an unavailable normalized field and a
  visible coverage count while raw fields remain intact.
- If a market loading is used, assert it is fitted only through `t`; future
  market return may enter only the outcome expression.
- Retain exact-anchor flat/long/short results even when a residual benchmark is
  present.

### Horizon, state, and ownership

- Assert one-step continuation results contain no liquidation cost and no
  "Sharpe" field.
- Assert terminal results match the declared horizon and include exactly one
  liquidation charge.
- For any later branch evaluator, prove independent environment, inventory,
  recurrent/model state, and random-state ownership per action branch.
- Assert no current diagnostic field enters solver observations, task encoding,
  `RewardEngine`, or unapproved proposer-controlled reward parameters.
- Assert solver action width remains three and WAIT remains outside the solver.

### Capability and calibration

- Require an explicit target/outcome definition before populating
  `estimated_p_solve` or `calibration_error`.
- When probabilities exist, test a named proper score and reliability output on
  held-out records; absent probabilities yield unavailable calibration, not
  perfect calibration.
- Serialize raw metric definitions, panel/checkpoint/cost identities, coverage,
  and separately produced uncertainty references so a scalar cannot conceal its
  construction.
- Assert that any future evidence-bundle-to-profile reducer has a distinct
  version and cannot feed the solver reward, training observations, unapproved
  proposer feedback, or current representation-promotion gates.
- If a later Phase B capability-to-curriculum adapter is approved, require its
  own version, declared validation/rolling source, and a hard prohibition on
  terminal-screen or sealed final-test evidence entering adaptive updates.

## 11. Recommended research protocol, without adopting thresholds

1. Define two explicit one-step estimands: continuation and forced round trip.
   Use continuation as the main local decision diagnostic; keep the round-trip
   result only when that is the intended economic question.
2. Predeclare the lagged volatility candidate(s), exact source window, and
   zero-volatility handling. Treat "60 bars" as a candidate aligned with the
   existing observation window, not as validated by convenience.
3. Freeze the solver action before reading the future bar. Compute the full
   three-action payoff vector only as an evaluation artifact.
4. Report raw policy return and active return versus hold, flat, long, and short
   controls. Add a market-residual diagnostic only if its lagged loading has
   adequate support and a clear benchmark identity.
5. Report oracle action/gap separately to answer whether action discrimination
   is possible at an anchor; never substitute the oracle action for the policy.
6. Cross one-step diagnostics with the predeclared full terminal horizons. A
   model can look locally directionally useful yet fail on switching, state,
   drawdown, costs, or continuation.
7. Keep the metric vector intact through the first research round. Hand its
   per-anchor records to the separately chosen dependence-aware uncertainty
   protocol; do not embed confidence into a reward-like score.
8. Consider the separately versioned offline-bundle-to-`CapabilityProfile`
   transformation only after the outcome event, probability source, calibration
   method, support dimensions, and estimator are fixed and prospectively
   evaluated. Its immediate consumer in this note is the future online/shadow
   eligibility and matching layer, not the solver's training or current
   promotion loop. A separately approved Phase B adapter may later consume
   qualified validation/rolling evidence for curriculum feedback, while
   terminal-screen and sealed final-test evidence remain isolated.

## 12. Open decisions for the one-question-at-a-time grill

No item below implies a chosen answer or threshold.

1. Is the immediate probe meant to estimate continuation value or a forced
   one-bar round trip?
2. Which primary baseline answers the intended question: hold prior position,
   target-flat, long/short directional controls, or a passive market benchmark?
3. Is 60-bar trailing RMS, trailing sample standard deviation, EWMA, or a
   time-of-day-conditioned estimator the first predeclared volatility factor?
4. Should the first multi-step companion hold the initial action fixed, or
   branch once and then follow a cloned policy state?
5. What precisely is the future binary/continuous target for any claimed
   `estimated_p_solve` or calibration field?
6. Which metric vector components are descriptive diagnostics versus eventual
   inputs to the deterministic DecisionGate? The latter require held-out
   calibration evidence and remain outside this research note.

## Source ledger

Primary external sources:

- William F. Sharpe, "The Sharpe Ratio," *Journal of Portfolio Management*,
  1994: <https://web.stanford.edu/~wfsharpe/art/sr/sr.htm>
- Andrew W. Lo, "The Statistics of Sharpe Ratios," *Financial Analysts Journal*,
  2002: <https://alo.mit.edu/publications/page/18/>
- Torben G. Andersen, Tim Bollerslev, Francis X. Diebold, and Paul Labys,
  "Modeling and Forecasting Realized Volatility," *Econometrica*, 2003:
  <https://onlinelibrary.wiley.com/doi/10.1111/1468-0262.00418>
- Alan Moreira and Tyler Muir, "Volatility-Managed Portfolios," *Journal of
  Finance*, 2017: <https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf>
- Tilmann Gneiting and Adrian E. Raftery, "Strictly Proper Scoring Rules,
  Prediction, and Estimation," *JASA*, 2007:
  <https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf>
- Lihong Li, Wei Chu, John Langford, and Xuanhui Wang, "Unbiased Offline
  Evaluation of Contextual-bandit-based News Article Recommendation
  Algorithms," WSDM 2011:
  <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Published-3.pdf>
- Miroslav Dudik, John Langford, and Lihong Li, "Doubly Robust Policy Evaluation
  and Learning," ICML 2011:
  <https://www.microsoft.com/en-us/research/publication/doubly-robust-policy-evaluation-and-learning-2/>
- Peter Auer, Nicolo Cesa-Bianchi, Yoav Freund, and Robert E. Schapire, "The
  Nonstochastic Multiarmed Bandit Problem," *SIAM Journal on Computing*, 2002:
  <https://homepages.math.uic.edu/~lreyzin/papers/auer02.pdf>
- John Moody and Matthew Saffell, "Learning to Trade via Direct Reinforcement,"
  *IEEE Transactions on Neural Networks*, 2001:
  <https://pubmed.ncbi.nlm.nih.gov/18249919/>

Local primary sources are the linked current iTrader files in Section 1. They
take precedence over this dated research note if the implementation changes.
