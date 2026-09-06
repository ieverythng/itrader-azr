# Sharpe-Aware Reward Design

Last updated: 2026-09-03
Status: **Two experimental challengers implemented; `pnl_linear` remains the only proven default**

## Decision Summary

Sharpe-aware reward shaping is a strong candidate for iTrader, especially when
task horizons become long enough for return dispersion to be meaningful. It is
not yet evidence that every longer-horizon family should use a Sharpe-dominant
objective, and it is not a reason to redefine the task-family registry.

The implementation therefore keeps `pnl_linear` as the sole family `00` default
and adds two selectable challengers. `sharpe_only` uses only the change in a
rolling Sharpe score. `pnl_sharpe` combines the same change with the existing
normalized PnL, drawdown, and turnover components:

```text
score_t = tanh((mean(strategy_returns) / std(strategy_returns)) / target)
sharpe_only_t = w_sharpe * (score_t - score_(t-1))
pnl_sharpe_t = pnl_linear_t + sharpe_only_t
```

The score is unannualized, uses a configurable rolling window, and remains zero
during warmup or when variance is effectively zero.

## Source-Grounded Family Map

The canonical names and ownership come from
`itrader/rl_lib/tasks/families.py`:

| ID | Canonical family | Current state | Reward position |
|---|---|---|---|
| `00` | `directional_predictor` | Enabled baseline | `pnl_linear` sole default; both Sharpe challengers are experimental |
| `01` | `breakout_expansion` | Bootstrap contract, disabled | Keep reward choice open until family behavior exists |
| `02` | `mean_reversion` | Bootstrap contract, disabled | Keep reward choice open until family behavior exists |
| `03` | `trade_management` | Future contract, disabled | Requires position-aware state and a distinct action contract review |
| `04` | `abstain_unsuitable_regime` | Future contract, disabled | Online `WAIT`/abstention semantics, not a Sharpe family |

Earlier versions of this research artifact swapped families `01` and `02` and
treated family `03` as trend following. Those labels were propositions, not the
implemented registry, and are no longer used here.

## Why Sharpe Is Compelling

Pure cumulative PnL can reward volatile trajectories, repeated exposure, and
longer episodes simply because they contain more opportunities. A
risk-adjusted statistic can distinguish a smoother return path from an equally
profitable but unstable one.

The argument is strongest as a hypothesis:

- medium and long episodes provide more observations for dispersion estimates;
- a bounded score prevents near-zero volatility from exploding the reward;
- blending with dense PnL preserves an early learning signal;
- reporting Sharpe separately improves evaluation even if it does not improve
  optimization.

Sharpe is not automatically horizon-invariant in this environment. Sampling
error, serial correlation, market-regime duration, and the episode boundary all
change the estimate. It must be evaluated by fixed horizon and held-out regime,
not inferred from aggregate training return.

## Why Incremental Sharpe

Adding the full rolling Sharpe level at every step repeatedly pays the solver
for an already-achieved trajectory and makes reward magnitude depend on episode
length. The incremental form telescopes over an episode:

```text
sum(score_t - score_(t-1)) = score_T - score_0
```

This leaves dense PnL as the step-level signal while rewarding improvement in
the path-level statistic. It also makes the `sharpe_delta` component directly
inspectable in existing reward-component telemetry.

No annualization is used during optimization. At one-minute resolution,
annualization can produce very large intermediate values that immediately
saturate `tanh`, erasing useful differences between candidates. Annualized
Sharpe may still be reported as an evaluation metric with an explicitly stated
sampling convention.

## Implemented Contract

`ITraderEnv` now owns the episode return history and passes
`strategy_returns` into `RewardEngine`, preserving the existing ownership
boundary. A strategy return is the current mark-to-market return after applying
the chosen position:

```text
strategy_return_t = position_t * market_return_t
```

Both challengers share the following Sharpe parameters. The `pnl_sharpe` blend
also reuses all normalized `pnl_linear` parameters; `sharpe_only` deliberately
does not include PnL, drawdown, or turnover components.

| Parameter | Default | Bound / meaning |
|---|---:|---|
| `w_sharpe` | `0.5` | `[0, 2]`, incremental score weight |
| `sharpe_window` | `32` | `[2, 512]` strategy returns |
| `sharpe_min_samples` | `8` | Warmup floor, capped by window |
| `sharpe_target` | `1.0` | Positive scale before `tanh` |

The implementation is finite for missing history, short history, non-finite
inputs, and zero variance. Unit tests cover warmup, boundedness, component
accounting, and environment plumbing.

## Reward Assignment Hypotheses

These assignments must be tested rather than wired as defaults:

| Family | First comparison | Rationale |
|---|---|---|
| `00` directional | `pnl_linear` vs `sharpe_only` vs `pnl_sharpe` | Separate pure risk-adjusted shaping from a dense PnL blend |
| `01` breakout | Same three-way comparison | Breakouts may have sparse asymmetric returns; Sharpe could either stabilize or suppress them |
| `02` mean reversion | Same three-way comparison | Frequent smaller returns may benefit from consistency shaping |
| `03` trade management | Defer | Needs explicit position age, entry state, and action semantics before reward design |

Do not introduce additional volatility or trend-persistence bonuses until one
of the Sharpe challengers demonstrates incremental value.
Those additions create new objectives and environment state, so they belong in
separate TDD slices.

## Evaluation Matrix

Run matched experiments with identical dataset slices, task pools, seeds, PPO
budget, and solver head:

The executable first pair is
`itrader/scripts/experiments/configs/sharpe_reward_ablation.json`. Reward
parameter sampling is isolated from task identity sampling so switching the
reward does not alter the horizon or task-seed sequence.

| Axis | Required cells |
|---|---|
| Reward | `pnl_linear`, `sharpe_only`, `pnl_sharpe` |
| Horizon | `96`, `128`, `160`, `200` or explicitly paired subsets |
| Sampling | random episode, contiguous stream |
| Solver | Start with `SolverMLPFiLM`; recurrent comparisons come later |

Primary measurements:

- held-out strategy Sharpe and episode return;
- max drawdown and turnover;
- `sharpe_delta` magnitude and saturation rate;
- action collapse and entropy;
- bootstrap confidence intervals across seeds;
- capability slices by family, regime, and horizon once metadata reaches the
  metrics records.

The reward is promoted only if it improves held-out risk-adjusted behavior
without unacceptable loss of absolute return, action diversity, or stability.

## Relationship To Recurrence

Sharpe-aware optimization and recurrent FiLM are complementary but independent
variables. A recurrent policy may represent path-dependent context, while a
Sharpe-aware reward may value smoother paths. Neither proves the other is
necessary.

The staged design is:

1. validate `sharpe_only` and `pnl_sharpe` against `pnl_linear` on `SolverMLPFiLM`;
2. establish the best episode/horizon sampling mode;
3. compare MLP-FiLM, corrected window-LSTM-FiLM, and true recurrent FiLM under
   the same reward;
4. test family-specific defaults only after families `01` and `02` have real
   samplers and evaluation slices;
5. leave family `03` reward work behind its position-aware task contract.

## Open Questions

1. Should `sharpe_window` scale with task horizon or remain fixed?
2. Is differential Sharpe less learnable than a terminal-only Sharpe bonus?
3. Does zero-variance handling create a flat-policy loophole? The base PnL term
   should discourage it, but the ablation must verify this.
4. Should downside deviation or Sortino replace standard deviation for
   asymmetric breakout returns?
5. How should serial correlation and overlapping windows affect reported,
   annualized evaluation Sharpe?

## Promotion Gate

Both Sharpe challengers remain experimental until:

- focused and full tests pass in the pinned environment;
- at least three matched seeds complete without NaN or action collapse;
- held-out evaluation beats or matches `pnl_linear` on predefined return,
  drawdown, turnover, and Sharpe criteria;
- results are sliced by horizon and market regime;
- the winning parameters are recorded in experiment artifacts rather than
  silently becoming family defaults.
