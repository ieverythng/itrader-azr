# Sharpe-Aware Reward Design

Last updated: 2026-09-12
Status: **experimental challengers; `pnl_linear` remains the only proven/default reward**

## Decision summary

The reward seam is independent of recurrent architecture. Keep `pnl_linear` as
the family `00` default. `sharpe_only` and `pnl_sharpe` are challengers that
require matched held-out evidence. A recurrence implementation does not require
changing reward, and a longer horizon is not automatically better.

The reward seam is environment-owned by `RewardEngine`. These challenger formulas
are reward calculations, not learned reward heads, and recurrence/head selection
must remain an independent decision.


```text
score_t = tanh((mean(strategy_returns) / std(strategy_returns)) / target)
sharpe_delta_t = w_sharpe * (score_t - score_(t-1))
pnl_sharpe_t = pnl_linear_t + sharpe_delta_t
```

Warmup and effectively zero variance produce a finite zero contribution. The
exact staged source change in `itrader/rl_lib/tasks/rewards.py` is pre-existing,
staged, and outside this docs-only pass; do not infer that its current behavior
has been reimplemented or revalidated here.

## Source-grounded family map

| ID | Roadmap family label | Current state | Reward position |
|---|---|---|---|
| `00` | `directional_predictor` | Enabled | `pnl_linear` sole proven/default; challengers experimental |
| `01` | `breakout_expansion` | Disabled bootstrap contract | Defer assignment |
| `02` | `mean_reversion` | Disabled bootstrap contract | Defer assignment |
| `03` | `trade_management` | Future/disabled | Needs explicit position/action contract |
| `04` | `abstain_unsuitable_regime` | Future/disabled | WAIT semantics, not a Sharpe family |

Families `00`–`04` are roadmap labels, not a settled shared-head architecture.
Even if `00`–`02` use `Discrete(3)`, their inputs, task semantics, reward
assignments, and state ownership may differ and must be recorded explicitly.


Before clipping and with an undiscounted sum, deltas telescope:

```text
sum_t(score_t - score_(t-1)) = score_T - score_0
```

That algebra does not make the shaped reward policy-invariant under discounted
PPO (`gamma < 1`), clipping, finite warmup, changing episode boundaries, or
other implementation details. Report exactly which sequence, discount, clipping,
and terminal conventions were used. Do not call incremental Sharpe a guaranteed
policy-preserving potential term.

No annualization is needed for optimization by default; minute-scale annualizing
can saturate a bounded signal. Annualized values may be reported separately with
the sampling convention stated. Sharpe remains sensitive to serial correlation,
regime duration, overlap, estimator warmup, and episode boundaries. It is not
automatically horizon-invariant.

Phase B includes both offline/curriculum work and online replay/shadow validation;
Phase C is later operationalization. Proposer and solver/reward studies remain
separate until the user's grill-me review resolves open architecture choices.


The local AZR v2 paper is not evidence that Sharpe shaping is required. Its Eq. 4
proposer reward is `0` at observed solve rate `p=0` or `p=1`, otherwise `1-p`.
That is asymmetric and not a symmetric 50% peak or a measured learning-progress
uplift. The paper's coding/math ablations are not iTrader returns.

iTrader uses historical OHLCV, a language-model proposer, and a numerical PPO
solver. `train_AZR.py` currently performs task-pool/bootstrap handoff rather than
a co-adaptive AZR loop. Proposer reference-buffer and frozen/adaptive studies
belong in a separate experiment matrix, with validation feedback and an untouched
final chronological test. Use a cost-aware verifier and report rejected tasks,
generation calls, tokens/latency, compute, costs, turnover, and drawdown.

## Matched reward study

Hold data eligibility, task identity, window, horizon, seeds, solver head, PPO
budget, and evaluation split constant. Compare:

- `pnl_linear`;
- `sharpe_only`;
- `pnl_sharpe`.

Start with R0. Recurrent comparisons follow the same reward cell rather than
combining reward and architecture changes in the first pass. Track held-out
return, Sharpe under a stated convention, max drawdown, turnover, costs, action
entropy, reward-component saturation, finite losses, throughput, and confidence
intervals across seeds. Family/regime claims require emitted task metadata; an
aggregate episode-return metric is insufficient.

Keep the input window, episode horizon, BPTT/unroll (`max_seq_len`), and
reward-statistics window distinct; Sharpe history is the last of these, not the
episode horizon itself.


Recurrence asks whether a solver benefits from temporal input encoding and carried
state. Sharpe shaping asks whether optimization values a bounded path statistic.
Neither proves the other is necessary. Keep reward controls and R-series controls
orthogonal until each has evidence.

Do not claim that a recurrent hidden state automatically tracks PnL, position,
action history, or episode progress from OHLCV alone. If those are causal inputs,
add them as explicit, normalized fields to both feed-forward and recurrent cells.
Do not call logits/value outputs calibrated confidence or expected net utility
without separate calibration and cost-aware economic evaluation.

## Promotion gate

A challenger is promoted only when compatible-environment tests, at least three
matched seeds, held-out chronological evaluation, finite behavior, action
diversity, drawdown/turnover/cost criteria, and horizon/regime slices support it.
Record the selected parameters in experiment artifacts; do not silently change a
family default. No reward conclusion is reported by this documentation pass.
