# Sharpe-Aware Reward Design — Task Families 00-03 & Recurrent Solver Alignment

Last updated: 2026-09-03
Status: **Design complete, implementation pending** (after alignment ablation runs)

## Context: Where We Are

The current reward system has two registered functions:

| Reward ID | Family | Formula | Status |
|-----------|--------|---------|--------|
| `pnl_linear` | 00 (primary) | `w_pnl * sign(pnl) * min(abs(pnl), cap) / ret_unit` | Active, default |
| `pnl_sharpe` | 01 (secondary) | PnL term + Sharpe bonus/penalty on rolling returns | Registered, NOT wired to any family |

Family registry:

| Family ID | Name | Default Reward | Status |
|-----------|------|----------------|--------|
| `primary_00` | Primary / baseline | `pnl_linear` | Active |
| `secondary_01` | Secondary / mean-reversion | `pnl_linear` (should be `pnl_sharpe`) | **Misconfigured** |
| `tertiary_02` | Tertiary / vol-breakout | None registered | Stub |
| `quaternary_03` | Quaternary / trend-following | None registered | Stub |

The gap: `pnl_sharpe` exists in code but no family uses it. Families 02 and 03 have no reward functions at all. This doc designs the full reward system for all four families, with Sharpe as the unifying risk-adjusted metric.

## Why Sharpe Is The Right Surrogate For Trading RL

### The Problem With Pure PnL Reward

`pnl_linear` rewards *absolute profit*. This creates three failure modes:

1. **Risk-blindness**: A strategy that makes 5% in low-vol conditions and -20% in high-vol conditions gets the same reward signal as one that makes 3% consistently. The solver learns to be aggressive when it shouldn't.

2. **Horizon mismatch**: PnL accumulates linearly with time. A 256-step episode naturally produces larger PnL than a 32-step episode, even with identical per-step skill. The solver is implicitly rewarded for *longer* episodes, not *better* trading.

3. **No drawdown awareness**: PnL doesn't distinguish between "steady +1% per step" and "+5% then -3% then +4%." Both end at +6%, but the second has much higher risk. The solver learns nothing about path quality.

### What Sharpe Fixes

The Sharpe ratio (excess return / volatility of returns) addresses all three:

- **Risk-normalized**: High returns in high-vol conditions are penalized; consistent returns in low-vol are rewarded.
- **Horizon-invariant**: Sharpe is a *rate*, not a *total*. A 256-step episode and a 32-step episode can have identical Sharpe if per-step skill is the same.
- **Path-aware**: Volatility of returns captures drawdowns, whipsaws, and inconsistency. A smooth equity curve scores higher than a bumpy one with the same terminal PnL.

### The Catch: Sharpe Is Noisy At Short Horizons

Sharpe ratio requires enough return observations to estimate volatility reliably. With <20 steps, the vol estimate is dominated by 2-3 outlier bars. This is why:

- Family 00 (short horizon, 32-64 steps): `pnl_linear` is still appropriate. Not enough data for stable Sharpe.
- Family 01 (medium horizon, 64-128 steps): `pnl_sharpe` with a *blended* formula. PnL provides the signal; Sharpe regularizes risk.
- Family 02 (long horizon, 128-256 steps): Full `sharpe_dominant` reward. Enough data for stable vol estimates.
- Family 03 (very long horizon, 256+ steps): `sharpe_trend` reward. Sharpe + trend persistence bonus.

## Reward Design Per Family

### Family 00: Primary / Baseline (Short Horizon, 32-64 Steps)

**Current**: `pnl_linear` — correct as-is.

**Proposed addition**: Optional Sharpe *regularization* term (not dominant, just a penalty for excessive vol):

```
reward = pnl_term - lambda_vol * max(0, rolling_std - vol_target)
```

Where:
- `pnl_term` = existing `w_pnl * sign(pnl) * min(abs(pnl), cap) / ret_unit`
- `rolling_std` = std of per-step returns over last `win` steps
- `vol_target` = task-specified target volatility (e.g., 0.02 per step)
- `lambda_vol` = small penalty weight (0.1-0.3), NOT dominant

This keeps PnL as the primary signal but penalizes *excessive* risk-taking. The solver learns "make money, but don't be a maniac."

**Why not full Sharpe here?** At 32-64 steps, the vol window is too short for stable estimation. A single outlier bar dominates the std. The penalty term is robust to this (it only fires when vol *exceeds* target), while full Sharpe would be noise-dominated.

**Solver fit**: FiLM MLP (R0) is sufficient. Short episodes don't benefit much from recurrence — there's not enough history to compound. The alignment ablation (v0-v3) runs on this family.

### Family 01: Secondary / Mean-Reversion (Medium Horizon, 64-128 Steps)

**Current**: `pnl_linear` (misconfigured — should use Sharpe).

**Proposed**: `pnl_sharpe` with tuned parameters. The existing implementation is close but needs refinement:

```python
def reward_pnl_sharpe_v2(params, ctx):
    """PnL + Sharpe blend for medium-horizon mean-reversion tasks.
    
    The solver should fade deviations from the mean and capture reversion.
    Sharpe regularizes risk; PnL provides the primary signal.
    """
    w_pnl = params.get("w_pnl", 1.0)
    w_sharpe = params.get("w_sharpe", 0.5)  # < w_pnl: PnL dominates
    sharpe_window = params.get("sharpe_window", min(32, len(ctx["returns"])))
    sharpe_target = params.get("sharpe_target", 1.0)
    
    pnl_term = w_pnl * sign(pnl) * min(abs(pnl), cap) / ret_unit
    
    # Sharpe on rolling returns (risk-adjusted performance)
    if len(returns) >= sharpe_window:
        recent = returns[-sharpe_window:]
        mu = mean(recent)
        sigma = std(recent) + 1e-8
        sharpe = sqrt(n_steps_per_year) * (mu / sigma) if sigma > 1e-6 else 0.0
        # Reward for hitting target Sharpe, penalize for missing it
        sharpe_term = w_sharpe * tanh(sharpe / sharpe_target)
    else:
        sharpe_term = 0.0  # not enough data yet
    
    return pnl_term + sharpe_term
```

**Key differences from current `pnl_sharpe`**:
- `w_sharpe < w_pnl` (0.5 vs 1.0): PnL is still the primary signal. Sharpe is a *regularizer*, not the objective.
- `tanh(sharpe / target)` instead of raw Sharpe: bounded reward, no explosion from outlier vol estimates.
- `sharpe_window = min(32, len(returns))`: adaptive window, never exceeds available data.

**Why this fits mean-reversion**: Mean-reversion strategies are inherently low-Sharpe (they fade moves, so returns are small and frequent). The PnL term captures "did I profit from the reversion?" while Sharpe captures "was my risk reasonable for a fading strategy?" A solver that over-fades (enters too early, gets stopped out) has high PnL variance → low Sharpe → penalized.

**Solver fit**: **Recurrent LSTM (R2c) is the target.** Mean-reversion requires tracking:
- How far price has deviated from mean (needs 20-50 bars of context)
- Whether the deviation is reverting or persisting (needs trend awareness)
- Position P&L trajectory (needs episode progress memory)

This is exactly what persistent hidden state provides. The FiLM task embedding tells the solver "you're solving mean-reversion" and the recurrence tracks the deviation/reversion cycle across steps.

### Family 02: Tertiary / Volatility Breakout (Long Horizon, 128-256 Steps)

**Current**: No reward function registered.

**Proposed**: `sharpe_dominant` — Sharpe is the primary signal, PnL is secondary.

```python
def reward_sharpe_dominant(params, ctx):
    """Sharpe-dominant reward for long-horizon vol-breakout tasks.
    
    Vol-breakout strategies enter on volatility expansion and ride the move.
    They need: (1) patience to wait for the breakout, (2) conviction to hold
    through the move, (3) risk management to cut losers fast.
    
    Sharpe captures all three: patience → consistent small returns while waiting,
    conviction → large positive returns during the move, risk management → low
    vol on losing trades.
    """
    w_sharpe = params.get("w_sharpe", 1.0)  # dominant
    w_pnl = params.get("w_pnl", 0.3)        # secondary
    sharpe_window = params.get("sharpe_window", 64)
    sharpe_target = params.get("sharpe_target", 1.5)  # higher bar for breakout
    
    # Sharpe term (primary)
    if len(returns) >= sharpe_window:
        recent = returns[-sharpe_window:]
        mu, sigma = mean(recent), std(recent) + 1e-8
        sharpe = sqrt(n_steps_per_year) * (mu / sigma)
        sharpe_term = w_sharpe * tanh(sharpe / sharpe_target)
    else:
        # Warmup: use PnL until enough data for Sharpe
        sharpe_term = 0.0
    
    # PnL term (secondary, captures terminal wealth)
    pnl_term = w_pnl * sign(pnl) * min(abs(pnl), cap) / ret_unit
    
    # Vol-expansion bonus: reward entering when vol is expanding
    vol_expansion_bonus = _compute_vol_expansion_bonus(ctx, params)
    
    return sharpe_term + pnl_term + vol_expansion_bonus
```

**Vol-expansion bonus** (new component):
```python
def _compute_vol_expansion_bonus(ctx, params):
    """Bonus for entering positions when volatility is expanding.
    
    Vol-breakout strategies make money by being *early* to the breakout.
    This bonus rewards the solver for taking position when rolling vol
    crosses above its recent mean (expansion signal).
    """
    w_vol = params.get("w_vol_expansion", 0.2)
    vol_window = params.get("vol_window", 50)
    
    if len(ctx["returns"]) < vol_window * 2:
        return 0.0
    
    recent_vol = std(returns[-vol_window:])
    prior_vol = std(returns[-2*vol_window:-vol_window])
    
    # Vol expansion ratio: >1 means vol is increasing
    expansion = recent_vol / (prior_vol + 1e-8)
    
    # Bonus if solver has position AND vol is expanding
    position_size = abs(ctx.get("position", 0))
    if position_size > 0.1 and expansion > 1.2:
        return w_vol * min(expansion - 1.0, 2.0)  # capped bonus
    
    return 0.0
```

**Why Sharpe-dominant here**: Vol-breakout strategies have a specific risk profile — they accept small losses while waiting for the breakout, then capture large moves. Pure PnL reward would penalize the waiting period (small negative returns). Sharpe normalizes this: consistent small losses while waiting + one large winning move = high Sharpe. The solver learns to be *patient* because patience improves Sharpe even if it hurts short-term PnL.

**Solver fit**: **Recurrent LSTM (R2c) is essential.** Vol-breakout requires:
- Tracking vol regime over 50-100 bars (is vol expanding or contracting?)
- Patience: staying flat while waiting for the signal (needs memory of "I've been waiting for 30 bars")
- Conviction: holding through the move (needs memory of "I entered at bar 40, I'm at bar 60, the move is still going")
- Asymmetry: cutting losers fast but letting winners run (needs P&L trajectory memory)

A feed-forward window encoder *cannot* do this. It sees the last 30 bars and decides "vol looks elevated, enter." But it doesn't know if it's been waiting for 5 bars or 50 bars, or if it already entered at bar 40 and should hold. The recurrent state encodes all of this.

### Family 03: Quaternary / Trend-Following (Very Long Horizon, 256+ Steps)

**Current**: No reward function registered.

**Proposed**: `sharpe_trend` — Sharpe + trend persistence bonus.

```python
def reward_sharpe_trend(params, ctx):
    """Sharpe + trend persistence for very long-horizon trend-following.
    
    Trend-following strategies identify and ride trends over 100+ bars.
    They need: (1) trend detection, (2) entry timing, (3) trailing stops,
    (4) the ability to STAY IN the trade for the full trend duration.
    
    The key challenge: the solver must not exit prematurely. A feed-forward
    network sees "price is still going up" and might exit at bar 80 of a
    200-bar trend. A recurrent solver knows "I've been in this trend for
    80 bars, trends typically last 150-200 bars, I should hold."
    """
    w_sharpe = params.get("w_sharpe", 1.0)
    w_pnl = params.get("w_pnl", 0.3)
    w_trend = params.get("w_trend_persistence", 0.3)
    sharpe_window = params.get("sharpe_window", 64)
    
    # Sharpe (primary, same as family 02)
    sharpe_term = _compute_sharpe_term(ctx, params)
    
    # PnL (secondary)
    pnl_term = w_pnl * sign(pnl) * min(abs(pnl), cap) / ret_unit
    
    # Trend persistence bonus: reward for staying in winning trends
    trend_bonus = _compute_trend_persistence_bonus(ctx, params)
    
    return sharpe_term + pnl_term + trend_bonus

def _compute_trend_persistence_bonus(ctx, params):
    """Bonus for maintaining position through trending moves.
    
    If the solver has been in a position for N bars and the price has moved
    favorably by M%, award a bonus proportional to N * M. This directly
    rewards the "stay in the trade" behavior that trend-following requires.
    """
    w_trend = params.get("w_trend_persistence", 0.3)
    min_trend_bars = params.get("min_trend_bars", 20)
    
    position = ctx.get("position", 0)
    if abs(position) < 0.1:
        return 0.0
    
    # How long has the solver been in this position?
    # (Approximated from ctx["position_history"] if available,
    #  otherwise from the recurrent state's implicit memory)
    bars_in_position = ctx.get("bars_in_position", 0)
    
    if bars_in_position < min_trend_bars:
        return 0.0
    
    # Favorable move since entry
    entry_price = ctx.get("entry_price", None)
    if entry_price is None:
        return 0.0
    
    current_price = ctx["close"][-1]
    if position > 0:  # long
        favorable_move = (current_price - entry_price) / entry_price
    else:  # short
        favorable_move = (entry_price - current_price) / entry_price
    
    if favorable_move <= 0:
        return 0.0  # only reward winning trends
    
    # Bonus scales with duration * magnitude, capped
    raw_bonus = bars_in_position * favorable_move
    return w_trend * min(raw_bonus, 5.0)  # cap at 5.0
```

**Why trend persistence bonus**: This is the component that *cannot* be captured by Sharpe alone. Sharpe measures risk-adjusted returns, but it doesn't specifically reward *duration in a winning trade*. A solver that enters at bar 100, exits at bar 120 (capturing 2% of a 5% trend), and re-enters at bar 130 has the same Sharpe as one that stays in from bar 100 to bar 200. But the latter is *better* trend-following — less turnover, less slippage, more consistent. The persistence bonus explicitly rewards this.

**Solver fit**: **Recurrent LSTM (R2c) is non-negotiable.** Trend-following is the *ultimate* test of recurrence:
- Trend detection requires 50-100 bars of context (is this a trend or noise?)
- Entry timing requires knowing where in the trend you are (early = more upside, late = less)
- Trailing stops require tracking the peak since entry (needs memory of "highest price since I entered was X")
- Duration awareness requires counting bars in position (needs episode progress tracking)

A feed-forward network is *fundamentally incapable* of trend-following at this horizon. It has no concept of "I've been in this trade for 80 bars." The recurrent state IS the trade tracker. This is the family where the R2c architecture shows its maximum advantage over R0.

## Reward Architecture Summary

| Family | Horizon | Primary Signal | Secondary Signal | Bonus Term | Solver Fit |
|--------|---------|---------------|-----------------|------------|------------|
| 00 (Primary) | 32-64 | PnL | Vol penalty (small) | None | R0 (FiLM MLP) |
| 01 (Mean-Reversion) | 64-128 | PnL | Sharpe (tanh-bounded) | None | **R2c (Recurrent FiLM LSTM)** |
| 02 (Vol-Breakout) | 128-256 | Sharpe | PnL | Vol-expansion bonus | **R2c (Recurrent FiLM LSTM)** |
| 03 (Trend-Following) | 256+ | Sharpe | PnL | Trend-persistence bonus | **R2c (Recurrent FiLM LSTM)** |

**Pattern**: As horizon increases, the reward shifts from PnL-dominant to Sharpe-dominant, and the solver shifts from feed-forward to recurrent. This is not a coincidence — it's a direct consequence of the information requirements:

- Short horizon → limited history → PnL is sufficient (you don't need risk adjustment when you only have 32 data points) → feed-forward is sufficient (you don't need memory when the episode is short)
- Long horizon → rich history → Sharpe provides meaningful risk normalization → recurrence provides the memory to exploit long-horizon patterns

## Implementation Roadmap

### Phase 1: Wire `pnl_sharpe` to Family 01 (Quick Win)

1. Update `families.py`: change `secondary_01` default_reward_id from `pnl_linear` to `pnl_sharpe`
2. Tune `pnl_sharpe` params: `w_sharpe=0.5`, `sharpe_window=32`, `sharpe_target=1.0`
3. Add unit test: verify family 01 tasks get `pnl_sharpe` reward
4. Run smoke training on family 01 with R0 (FiLM) to verify no NaN

**Effort**: ~2 hours. **Risk**: Low. **Value**: Validates the Sharpe reward mechanism before building families 02/03.

### Phase 2: Implement `sharpe_dominant` for Family 02

1. Add `reward_sharpe_dominant` to `rewards.py`
2. Add `_compute_vol_expansion_bonus` helper
3. Register in `REWARD_REGISTRY`
4. Update `families.py`: `tertiary_02` default_reward_id = `sharpe_dominant`
5. Unit tests: Sharpe computation, vol-expansion bonus, edge cases (empty returns, zero vol)
6. Smoke training on family 02 with R0 (baseline) and R2c (recurrent, if available)

**Effort**: ~1 day. **Risk**: Medium (new reward components need careful tuning). **Value**: First family where Sharpe is the primary signal.

### Phase 3: Implement `sharpe_trend` for Family 03

1. Add `reward_sharpe_trend` to `rewards.py`
2. Add `_compute_trend_persistence_bonus` helper
3. Requires `ctx["bars_in_position"]` and `ctx["entry_price"]` — check if env provides these, add if not
4. Register in `REWARD_REGISTRY`
5. Update `families.py`: `quaternary_03` default_reward_id = `sharpe_trend`
6. Unit tests + smoke training

**Effort**: ~1-2 days. **Risk**: Medium-High (trend persistence bonus is novel, needs tuning). **Value**: Completes the 4-family reward system.

### Phase 4: Recurrent Solver (R2c) Integration

1. Implement `SolverRecurrentLSTMTFiLM` (design in `recurrent-solver-ablation.md`)
2. Wire into RLlib recurrent policy interface
3. Run R0 vs R2c comparison on families 01, 02, 03
4. Validate: R2c should show larger gains on 02/03 (long horizon) than 01 (medium horizon)

**Effort**: ~3-5 days. **Risk**: High (new architecture, RLlib recurrent API). **Value**: The core architectural contribution.

### Phase 5: Proposer Model Selection + Loop Closure

Per the Notion log (23/08/26), Phase B closure requires:
1. **Proposer model locked**: Qwen3.8-27B (heavily quantized, ~4-bit) or Gemma-class with hardware LoRA support. OOM constraint: must run alongside llama.cpp server OR replace it.
2. **Proposer hooked to task generation loop**: Proposer generates candidate tasks → validator filters → solver trains on accepted tasks → capability report feeds back.
3. **Solver head 00 fully developed**: FiLM (R0) for family 00, recurrent LSTM (R2c) for families 01-03.

**The OOM problem**: Qwen3.8-27B at 4-bit is ~14GB. The RTX 5070 Ti has 16GB VRAM. This leaves ~2GB for the solver training + environment. **Tight but feasible** if:
- Proposer runs in inference-only mode (no gradients)
- Solver training uses CPU offloading for optimizer states
- Batch sizes are reduced during proposer-active periods

**Alternative**: Run proposer on a separate process with its own VRAM allocation, communicate via API (the existing `openai_compat` proposer path). This avoids VRAM contention entirely.

## Connection To The Recurrent Solver Ablation

The reward design and the solver architecture are **co-dependent**:

| | R0 (FiLM MLP) | R2c (Recurrent FiLM LSTM) |
|---|---|---|
| Family 00 (PnL) | ✓ Sufficient | Overkill (short episodes) |
| Family 01 (PnL+Sharpe) | Baseline | **Target** (mean-reversion tracking) |
| Family 02 (Sharpe+Vol) | Inadequate | **Essential** (patience + conviction) |
| Family 03 (Sharpe+Trend) | Inadequate | **Non-negotiable** (trade duration memory) |

The ablation design in `recurrent-solver-ablation.md` should be extended: run R0 vs R2c on each family separately, not just a pooled comparison. The hypothesis is that the R2c advantage *grows with horizon*:

- Family 00: R2c ≈ R0 (no significant difference)
- Family 01: R2c > R0 (moderate improvement, mean-reversion tracking)
- Family 02: R2c >> R0 (large improvement, vol-breakout patience)
- Family 03: R2c >>> R0 (transformative improvement, trend-following memory)

If this gradient holds, it's strong evidence that the recurrent architecture is not just "better" but *necessary* for long-horizon trading. That's a publishable result.

## Open Questions

1. **Sharpe window length**: Is 32/64/64 optimal? Should it scale with horizon (e.g., `min(0.25 * horizon, 64)`)?
2. **Tanh vs linear Sharpe**: Tanh bounds the reward but compresses high-Sharpe signals. Is this desirable or does it cap the solver's incentive to excel?
3. **Vol-expansion bonus tuning**: The `expansion > 1.2` threshold is arbitrary. Should it be task-specified?
4. **Trend persistence vs turnover penalty**: The current design rewards staying in trades. But excessive holding also has costs (opportunity cost, drawdown risk). Should there be a *penalty* for holding too long past the trend peak?
5. **Proposer model**: Qwen3.8-27B 4-bit (~14GB) vs Gemma-2-9B (~6GB) vs CLIProxyAPI GPT-5 (no local VRAM). The Notion log mentions "hardware LoRA like Gemma" — does this mean fine-tuning the proposer on iTrader task data? If so, that's a separate workstream.

## Success Criteria For The Full System

The 4-family reward + recurrent solver system is "complete" when:

1. All four families have distinct, tested reward functions
2. R2c outperforms R0 on families 01-03 (held-out evaluation)
3. The performance gradient holds: R2c advantage grows with horizon
4. Capability reports show family-specific improvements (not just aggregate)
5. Proposer generates tasks for all four families, validator accepts >60%
6. End-to-end loop: proposer → validator → solver training → capability report → (feedback to proposer)

This is the Phase B closure criterion from the Notion log.
