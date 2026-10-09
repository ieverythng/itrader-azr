# Experimental attribution and one-step normalization contract

Date: 2026-10-07
Status: the default-off diagnostic pilot is approved; the exact volatility
estimator and `<= 1e-12` guard below are candidates for human decision, **not**
adopted defaults, metrics, rewards, headlines, promotion rules, capability
estimators, or training signals

## Decision

Implement the approved shells for both views as opt-in evaluation-side reporting
under a new versioned secondary namespace, while leaving volatility estimation
unset until its method and invalid-scale rule are approved. Preserve the current `EpisodeEvidence`
economics and the Q24 `q24_equal_symbol_terminal_v1` result byte-for-byte at its
existing keys. Neither pilot may enter `RewardEngine`, solver observations,
task encoding, PPO/PBT selection, proposer feedback, `CapabilityProfile`, or a
current promotion gate.

The two views answer different questions:

- **A — episode attribution:** where an already reported *additive* terminal net
  return came from: average signed direction, within-episode timing residual,
  and modeled costs.
- **B — one-step normalization:** how large one causally selected, realized
  next-bar return was relative to one strictly lagged volatility scale. It is a
  local action diagnostic, not a Sharpe ratio or terminal-capability estimate.

Sharpe's ex-post ratio requires a *series* of differential returns and their
historical dispersion; one realized return divided by an earlier volatility
estimate does not meet that definition ([Sharpe 1994](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)).
Moreira and Muir support the causal principle of using lagged rather than
outcome-period volatility, but their monthly result does not validate a
59-return, one-minute estimator for iTrader
([Moreira and Muir 2017](https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf)).

## Shared time and accounting alignment

Let an episode start at dataframe row `s`. Immediately before decision `k`,
`i = s + k` and the 60-bar observation contains rows `i-60, ..., i-1`, oldest
to newest. The action is selected before row `i` is exposed. The current
environment then applies target position `p_k in {-1, 0, +1}` to

```text
r_k = (C_i - C_(i-1)) / max(C_(i-1), 1e-8)
g_k = p_k * r_k
u_k = |p_k - p_(k-1)|, with p_(-1) = 0 at reset
```

and advances `idx` to `i+1`. Thus an `H`-step complete episode has `H` scored
returns, `end_idx = s + H`, and one final liquidation turnover
`u_liq = |p_(H-1)|` but no extra decision or return bar. These are the source
semantics in [`ITraderEnv`](../../../../itrader/envs/itrader_env.py), and
the tagged-market test proves that row 60 is unseen when rows 0--59 form the
observation ([test](../../../../itrader/tests/test_semantic_observation_contract.py#L22)).

All return and cost fields below use decimal return units. With
`c = evidence_cost_bps * 1e-4`, 1 bp is `0.0001` per turnover unit. This remains
a linear price-taker cost scenario, not broker execution semantics.

## A. Additive episode direction/timing/cost attribution

For a complete candidate episode, use the exact recorded step trace and existing
terminal evidence. `L` is the gross return of an **unselected always-long
reference on those same realized bars**, derived as the sum of the trace's asset
returns; it is not a hindsight choice between long and short. A separately run
long-only control may cross-check `L` when its full identity matches, but is not
an input required to compute the decomposition:

```text
G             = sum(k=0..H-1) g_k                    # existing gross_return
p_bar         = long_fraction - short_fraction
G_direction   = p_bar * L
G_timing      = G - G_direction
C_in_episode  = c * (turnover - liquidation_turnover)
C_liquidation = c * liquidation_turnover
C             = C_in_episode + C_liquidation         # existing cost_return
N             = G_direction + G_timing - C            # existing net_return
```

This is an exact algebraic decomposition of the environment's **additive**
return accounting. It is not compounded wealth: do not replace any sum with
`prod(1 + g_k) - 1`, and do not label Q24 as a portfolio return. `G_timing` is
a residual after realized average directional loading; it is not automatically
forecasting skill, causal alpha, or an executable fractional-position policy.
Compute `p_bar * L` separately for each episode before aggregation; multiplying
an aggregated exposure by an aggregated long return is a different quantity.
The current source already defines `net = gross - cost`, total turnover
including liquidation, and sample-Sharpe behavior
([`EpisodeEvidence`](../../../../itrader/rl_lib/evaluation/episode_evidence.py#L52));
the adjacent primary-source review derives the direction/timing identity and
its limits
([quant attribution note](../evaluation-quant-methodology-2026-10-06-evidence/quant-attribution-primary-sources.md#2-exact-economic-attribution-on-a-matched-episode)).

The minimum per-episode secondary payload, aligned with the current pilot
source, is:

```text
version = "episode_direction_timing_v1"
steps = H
mean_signed_exposure = p_bar
matched_long_gross_return = L
gross_return = G
directional_gross_return = G_direction
timing_gross_return = G_timing
cost_return = C
net_return = N
volatility_available_steps
volatility_total_steps
trace_identity_consistent
producer_provenance_verified = false  # until a gated runner verifies receipts
```

Despite the retained field name `matched_long_gross_return`, `L` is derived from
the same trace; it is not selected by comparing directional controls. Per-step
records retain switching and actual-liquidation costs separately.

Fail closed unless all trace/evidence inputs are finite; action fractions sum to
one; turnover is at least liquidation turnover; step and episode identities
match; and both `G = direction + timing` and
`N = direction + timing - cost` agree with the existing evidence under a
declared, tested tolerance. The current pilot uses a looser tolerance than the
terminal reporter; Q24 integration must align or explicitly version and justify
that difference before acceptance. Long-only must
have `p_bar=1` and zero timing; short-only `p_bar=-1` and zero timing; flat from
zero position must have zero gross, direction, timing, turnover, and cost.

### Q24 preservation

Do not create a new reducer. After the existing complete-panel validation,
carry each component through the exact Q24 hierarchy: mean over four horizons
inside anchor, anchors inside symbol, then equal mean over 12 symbols. Use the
same weights for every policy, control, component, and paired delta. Preserve
the current raw-episode and per-horizon views. At every reported level, assert
that reconstructed net equals the already reported net; the existing
`equal_symbol.net_return` remains the headline
([reporter](../../../../itrader/rl_lib/evaluation/terminal_panel_report.py#L74),
[tests](../../../../itrader/tests/test_terminal_panel_report.py#L40)).

Keep flat, long, short, seeded-random, and provenance-verified matched-untrained
policies as separate Q24 controls. Select the stronger constant direction only
once with whole-panel Q24 weights, never per anchor. The exposure-matched
`p_bar * L` term is an attribution counterfactual, not another executable
control.

## B. One-step return per lagged-volatility unit

### Provisional scale

At decision row `i`, the 60 visible closes `C_(i-60), ..., C_(i-1)` contain
exactly **59**, not 60, adjacent close-to-close returns:

```text
x_j       = (C_j - C_(j-1)) / max(C_(j-1), 1e-8), j = i-59, ..., i-1
n         = 59
x_bar     = sum(x_j) / n
sigma_i   = sqrt(sum((x_j - x_bar)^2) / (n - 1))
```

The unadopted candidate for the next GRILL is therefore
`lagged_close_std_60c_59r_ddof1_v0`: trailing sample standard deviation,
`ddof=1`, denominator `58`. Its units are decimal simple return per one-minute
bar; it is not annualized. A 60-return estimator would require 61 closes and
must not silently reach outside the declared observation window. This choice is
provisional because realized-volatility research supports causal lagging, not
this exact window or estimator; RMS, EWMA, or time-of-day-conditioned scales
remain separately named future factors. The current evidence review reaches the
same research boundary
([capability note](../evaluation-quant-methodology-2026-10-06-evidence/capability-diagnostics-primary-sources.md#4-lagged-volatility-normalization-useful-but-not-self-validating)).

Use only finite, positive source closes from the same symbol, session, and
contiguous one-minute history. Freeze `sigma_i` before reading `C_i`. The scored
outcome is the next return `r_k` above, from `C_(i-1)` to `C_i`; it is excluded
from the 59-return scale window.

### Cost-aware action values and controls

For prior position `q`, target action `a`, and the same `c` used by terminal
evidence:

```text
gross_continue_1(a) = a * r_k
switch_turnover_1(a)= |a - q|
switch_cost_1(a)    = c * switch_turnover_1(a)
net_continue_1(a)   = gross_continue_1(a) - switch_cost_1(a)
volatility_scaled_continuation_net_return(a) = net_continue_1(a) / sigma_i
```

The main pilot is the **continuation** estimand and has no synthetic liquidation.
If the scored step is the episode's true terminal, report the existing terminal
quantity separately as
`net_terminal_1(a) = net_continue_1(a) - c * |a|`. A deliberately forced
one-bar round trip uses that same formula and must use the current schema name
`hypothetical_roundtrip_net_return`; never charge it on every continuation
probe. Full traces must still reconcile to `EpisodeEvidence`, including exactly
one genuine terminal liquidation. The registered reward's terminal behavior is
independently tested
([reward test](../../../../itrader/tests/test_rewards.py#L8)), but these
diagnostics never call or modify the reward.

Freeze the policy action from the causal observation, then score it and these
canonical named references with the **same** `q`, `r_k`, `c`, `sigma_i`, and
terminal convention:

- `hold_prior` (`a=q`; zero switch cost);
- `flat` (`a=0`; closing an open `q` costs one unit);
- `long` (`a=+1`); and
- `short` (`a=-1`).

Report raw gross, turnover, cost, and net beside every normalized value, plus
`policy_net - reference_net` and its same-scale normalized counterpart. A
seeded-random or matched-untrained policy may be an additional policy row, but
must retain its own frozen causal action and provenance. The post-outcome
`max_a net_continue_1(a)` and gap may be serialized only under
`oracle_diagnostic_only`; neither is achieved policy return. This
ordering follows the fixed-policy-before-reward principle used in offline
policy evaluation ([Li et al. 2011](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Published-3.pdf)).

### Invalid scale behavior

If this scale contract is approved, set `volatility_status="unavailable"` and
every normalized field to JSON `null` when the return count is not 59, any
lagged input is non-finite, or `sigma_i` is non-finite or `<= 1e-12`. Record
`volatility_unavailable_reason` and include the row in availability/coverage
counts. Retain finite raw outcome, gross, cost, and net fields. Do not substitute
zero, infinity, or an epsilon-clamped denominator. Reject the entire row if its
raw outcome or cost arithmetic is non-finite. The `1e-12` cutoff is a versioned
pilot convention aligned with current episode Sharpe's effective-zero guard,
not a literature-derived permanent threshold.

Call the normalized field `volatility_scaled_continuation_net_return`, with unit label
`lagged_one_minute_volatility_units`. Do not call it Sharpe, a z-score,
confidence, probability, calibration, expected utility, or statistical
significance. If an episode summary is produced, construct it over common valid
steps with explicit coverage and carry it through the existing Q24 hierarchy
only in the secondary namespace. It must never replace terminal net; also retain
the raw distribution and coverage.

## Minimum acceptance tests

1. **Alignment:** a tagged 60-close window ending at `i-1` yields exactly 59
   lagged returns; action `a_i` scores only `C_(i-1) -> C_i`; an `H`-step trace
   ends at `s+H`.
2. **Episode identity:** property-test
   `net = direction + timing - in_episode_cost - liquidation_cost` for holds,
   entries, exits, both direct flips, and each terminal position; reproduce
   existing `EpisodeEvidence` exactly.
3. **Control identities:** long and short have zero timing residual, flat from
   zero has all-zero economics, and all one-step named controls share one
   denominator and cost identity at an anchor. Exhaust all nine `(q,a)` pairs.
4. **Q24 invariance:** opt-in off and on produce identical existing Q24 keys and
   values. Secondary components use the same complete joins and hierarchy,
   remain row-order invariant, and reconstruct Q24 net after aggregation.
5. **Future perturbation:** materialize observation, prior position, policy
   action, 59 lagged returns, scale, and reference definitions before outcome
   access. Perturb `C_i` and all later rows: all frozen decision-time values
   remain bit/equality stable while only outcome-derived scores may change.
   Perturb rows after `C_i`: the complete one-step record must not change.
   For the episode bridge, perturb one matched realized return by `delta` under
   fixed actions and assert changes of `p_k*delta` in gross,
   `p_bar*delta` in direction, and `(p_k-p_bar)*delta` in timing, with costs
   unchanged.
6. **No oracle leakage:** future-best action can change after outcome
   perturbation, but the frozen policy action cannot; oracle values appear only
   under the diagnostic-only namespace.
7. **Invalid scales:** zero, near-zero, non-finite, incomplete, cross-symbol,
   cross-session, or non-contiguous scale windows yield `null` normalized
   values and visible coverage reasons, never a favorable numeric score.
8. **Isolation:** tests prove the new fields cannot reach observations,
   `TaskSpec`/task encoding, `RewardEngine`, PPO/PBT metrics, Q24 headline,
   adaptive proposer input, `CapabilityProfile`, or sealed-test feedback.

## Local evidence inspected

- [`ITraderEnv`](../../../../itrader/envs/itrader_env.py) and
  [`EpisodeEvidence`](../../../../itrader/rl_lib/evaluation/episode_evidence.py)
- [Q24 terminal reporter](../../../../itrader/rl_lib/evaluation/terminal_panel_report.py)
  and its [tests](../../../../itrader/tests/test_terminal_panel_report.py)
- [environment economics tests](../../../../itrader/tests/test_env_core.py) and
  [semantic-window tests](../../../../itrader/tests/test_semantic_observation_contract.py)
- [quantitative attribution primary-source note](../evaluation-quant-methodology-2026-10-06-evidence/quant-attribution-primary-sources.md)
  and [capability-diagnostics primary-source note](../evaluation-quant-methodology-2026-10-06-evidence/capability-diagnostics-primary-sources.md)
