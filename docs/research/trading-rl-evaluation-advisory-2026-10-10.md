# Trading-RL Evaluation & Overtrading-Punishment Advisory — iTrader
**Prepared:** 2026-10-10 · **Target:** `itrader-azr` on branch `agent/Watson-dev`
**Status:** research advisory — grounded in the current dataset_v2 manifest, the
`phase_a_r0_plus_r4_v1_1_dec2025_4x3_40_20260925.json` loadout, `itrader/envs/itrader_env.py`,
and the masterplan. NOT a training run; no economic promotion claim.
**Review owner:** review lane · **Audit trail:** append to `memory/YYYY-MM-DD.md`.

> This document distills an evaluation/overtrading-punishment advisory and expands on the
> specific open questions the user raised. It is written to be reviewed by the agent fleet;
> it does **not** change any Phase A/B/C boundary and does not promote anything.

---

## 0. What we are looking at (grounding)

| Fact | Value | Source |
|---|---|---|
| Bars | 1-minute | `manifest.json` (`timeframe`) |
| Rows (all sessions) | 10,370,064 | `manifest.json` |
| Regular-hours rows | 5,515,061 | `manifest.json` |
| Pre-market / after-hours | 2,829,443 / 2,025,560 | `manifest.json` |
| Symbols | AAPL, MSFT, AMZN, NVDA, AMD, AVGO, GOOGL, META, SPY, QQQ, IWM, TLT (12) | `manifest.json` |
| Price adjust | **split-adjusted** (dividends NOT applied) | `manifest.json` + `contracts.py` |
| Feed | Alpaca SIP | `manifest.json` |
| Split | `0.7,0.15,0.15`, purge 200 rows | `phase_a_r0_plus_r4_v1…json` |
| Training window | 60 bars × 5 features (O/H/L/C/V) = **300** flattened dims | `itrader_env.py` `price_obs_dim` |
| Solver state (R4-v1) | +2 dims (position, remaining-horizon fraction) | `itrader_env.py` `solver_state_observation_contract` |
| Horizons | 96, 128, 160, 200 bars | `phase_a_r0_plus_r4_v1…json` |
| Task family | `00` (directional: short/flat/long) | masterplan |
| Reward objective | `net_return` (PPO scale ×1000) | `phase_a_r0_plus_r4_v1…json` |
| Optimization cost | `optimization_cost_bps=1.0`, `evidence_cost_bps=1.0` | `phase_a_r0_plus_r4_v1…json` |
| PBT | **off** for the R1 attribution round | masterplan |
| Learner | PPO, `iters=40` (≈192,000 sampled steps at 4800), `gamma∈{0.98,0.99,0.995}` | `train_solo_V2.py` |

**Headline (unchanged from prior review):** the R4-v1 vs R0 gap is a
**relative-attribution** signal (≈4.4 bp in the smoke run) that is **smaller than the
1 bp cost we are charging** — so it is architecture-qualified but **not economically
promoted**. The 95% paired net-return interval is the right instrument; it will almost
certainly cross zero on the current cells.

---

## 1. Why the R4-v1 vs R0 gap is not a promotion (elaboration of Point 1)

**The user's framing:** "We have not done heavy proper training for a while — we have
been correcting reporting, PBT, setups, and running short ablations to stress-test the
system. Only recently did we start pushing toward longer sets + the now-official R4-v1
to reduce the POMDP and push closer to MDP. Hence, is it worth pushing R0 further with
a larger sample size, validating with the much lower bp cost, and continuing from there?"

**Answer: yes — and that is the correct scientific move, *before* promoting R4-v1.**

1. **The 4.4 bp gap is dominated by cost noise, not signal.** At `evidence_cost_bps=1.0`,
   a 60–200 bar episode already pays **0.06–0.20% in cost** if the model changes position
   even a handful of times. A 4.4 bp improvement is **inside** that cost budget. Until we
   show the architecture survives at *lower* cost **and** generalizes across seeds,
   the 4.4 bp is a curiosity, not an edge.

2. **We are currently under-trained relative to the question we are asking.** The current
   loadout is `iters=40`, `train_batch_size=4800` → **≈192,000 sampled environment steps**.
   That is a **smoke / attribution** budget, not a training budget. For a directional
   `Discrete(3)` policy on a 1-minute bar dataset with ~5.5M RTH rows, the honest
   training budget is **10–100×** that — on the order of **1.9M–19M+ steps**, run on the
   **frozen 70/15/15 split**, multi-seed, and compared head-to-head with R0.

3. **"Push R0 further, then R4-v1" is the right order, not the reverse.** The
   masterplan already says: *keep R0 as the immutable parity control* and *run R1
   beside a matched Ray-2.58 R0 control*. The same principle applies to R4-v1: we should
   **establish the R0 ceiling under the larger training budget and the lower-cost
   regime first**, so that when R4-v1 is tested the comparison is not confounded by
   "R4-v1 got more training" or "R4-v1 was tested at a different cost."

4. **The lower-bp cost validation is the key gate.** If R0 (and R4-v1) both become
   **substantially** more profitable at 0.1 bp / 0.01 bp than at 1 bp, we have learned
   something important: the model *was* trading at all and the 1 bp cost was masking
   the signal. If R0 is still flat or negative at 0.01 bp, the architecture is the
   bottleneck, not the cost — and we should keep iterating on the solver (R1/R2c)
   rather than the cost knob.

**Concrete next step (proposed, not authorized):**

| Step | What | Why |
|---|---|---|
| A | **Freeze** the 70/15/15 split + panel hashes in the manifest. | Prevents leakage and lets all future runs compare on identical data. |
| B | **R0 training budget ×10–100** (`iters` 40→400–4000, multi-seed), at **0.1 bp and 0.01 bp**. | Establishes the R0 ceiling under realistic cost. |
| C | **R4-v1 under the same budget + same cost.** | True architectural comparison, not a smoke artifact. |
| D | **Two-sided 95% paired net-return interval** on a **frozen validation panel**, block-aware. | The instrument already accepted in Q26. |
| E | **Do not open the final test** until D is qualified and the interval does not cross zero. | Preserves the sealed final test. |

---

## 2. PPO training cadence — per-episode or per-step? (elaboration of Point 3)

**The user's question:** "Is PPO done per episode? Of course there are plenty of SGD
steps in between, but that 1 bp cost was arbitrary though tested continually. Would
0.1 bp or even 0.01 bp really constrain the model enough? … I want to ensure we are
punishing overtrading properly!"

**What the code actually does (grounding):**

- `itrader_env.py:733` — `r, comps = self.reward_engine.compute(...)`: **reward is
  computed per step**, not per episode. The `net_return` objective accumulates
  per-step mark-to-market PnL minus configured costs.
- `itrader_env.py:735` — PPO's `GAE` (generalized advantage estimation) is computed
  over the **episode** using `gamma∈{0.98, 0.99, 0.995}`. So the **credit
  assignment** is per-episode, but the **reward** is per-step.
- The **1 bp cost is charged per round-trip** (each position change), not per bar.
  `turnover_t = abs(pos_t − pos_prev)` (0, 1, or 2), and `cost = turnover_t ×
  evidence_cost_bps × 1e-4`.

**The key insight for Point 3:**

**0.1 bp and 0.01 bp will constrain the model *less* than 1 bp — and that is exactly
the point.** If the model is currently flat or near-flat at 1 bp, lowering the cost
**removes the penalty that was suppressing it** and lets the model express whatever
signal it has learned. The three regimes we want to see:

| Cost | What it tells us |
|---|---|
| **1 bp** | Current state: model is flat/near-flat → cost is masking signal, or there is no signal. |
| **0.1 bp** | Model expresses some signal but may still be dominated by cost if it over-trades. |
| **0.01 bp** | Cost is negligible → model's *unmasked* behavior is revealed. If it still over-trades, the reward structure (not the cost) is the problem. |

**The cost curve is the most important diagnostic.** A 3-point cost sweep (0.01, 0.1,
1 bp) × {R0, R4-v1} × {multiple seeds} gives a **cost-response surface**:

- **Flat across cost** → architecture has no signal (cost is irrelevant).
- **Improves as cost drops** → cost was masking signal; architecture has real edge at low cost.
- **Worsens as cost drops** → model is over-trading; reward structure is wrong.

**Recommendation (proposed):**

1. **Keep 1 bp as the economic anchor** — it is the "real-world" cost we care about.
2. **Add 0.1 bp and 0.01 bp as diagnostic cells** — they reveal the model's unmasked behavior.
3. **Report the cost-response curve as a first-class artifact** — it should be in every
   ablation summary.
4. **Do NOT use 0.01 bp as a promotion criterion** — it is a diagnostic, not an economic
   target. Promotion still requires positive net at **1 bp** on the sealed final test.

**On "punishing overtrading properly":** the current `net_return` objective already
penalizes turnover through `optimization_cost_bps`. The knob to watch is whether the
model's **turnover-to-PnL ratio** improves as we change the cost. If the model is
trading 20+ times per episode at 0.01 bp but only 3 at 1 bp, the reward structure is
encouraging over-trading at low cost — and we should add an explicit **turnover
penalty** (a fixed per-trade cost that does NOT scale with `evidence_cost_bps`) to
discourage position-flipping even at low cost.

---

## 3. Architecture test vs economic test — and the "most stable, semi-profitable solver" goal (elaboration of Point 4)

**The user's framing:** "We are mixing the two, but ultimately we are trying to get the
most stable, semi-profitable solver (not even economic return, but proven solver
capability) so we can couple the proposer later. Simply changing/mutating TaskSpec
without knowing solver profitability only adds more variables to tweak and therefore
harder to attribute to training. Can you take stock of the amount of knobs we are
actually turning and thus have to isolate?"

**Answer: I agree the two questions are being conflated, and the fix is to name
them and sequence them.**

**The two questions:**

| Question | Instrument | Gate |
|---|---|---|
| **Architecture test**: does R4-v1 (or R1) have better *attribution* than R0? | Paired 95% net-return interval on a frozen validation panel | Positive lower bound |
| **Economic test**: does the solver make *money* at 1 bp on untouched data? | Net return on the sealed final test | Positive net + positive Sharpe + multi-seed |

**The "most stable, semi-profitable solver" goal is a *third* thing** — a **capability
floor** that says: "this solver is good enough to *learn* from the proposer, even if
it is not yet profitable." That floor should be defined *explicitly*:

- **Stability**: deterministic post-restore evaluation, exact fresh replay, block-aware
  uncertainty, no terminal-exploit mutations, checkpoint/replay parity.
- **Semi-profitability**: positive net at 0.1 bp on the frozen validation panel,
  positive Sharpe, and turnover below a pre-declared threshold (e.g., ≤ 5 per episode).

**The knob inventory (the "amount of knobs we are actually turning")** — this is the
part that matters for attribution:

| Knob | Current value | Attribution risk |
|---|---|---|
| **Training budget** | `iters=40` | **HIGH** — different budgets make R0/R4-v1 incomparable |
| **Cost (bps)** | `evidence_cost_bps=1.0` | **HIGH** — changes the reward structure entirely |
| **Learning rate** | `lr=5e-5` | **HIGH** — changes convergence |
| **Gamma** | `gamma∈{0.98,0.99,0.995}` | **MEDIUM** — changes credit assignment |
| **Batch size** | `4800` | **MEDIUM** — changes gradient noise |
| **Window size** | `60` | **MEDIUM** — changes the input signal |
| **Horizons** | `96,128,160,200` | **LOW** — fixed across cells |
| **Task family** | `00` | **LOW** — fixed |
| **Split** | `0.7,0.15,0.15` | **LOW** — must be frozen |
| **Seeds** | multiple | **LOW** — must be held constant |
| **PBT** | off | **LOW** — must stay off for attribution |
| **Solver state mode** | `both` | **LOW** — this IS the thing being tested |
| **Observation transform** | `market_relative_v1` | **LOW** — must be held constant |
| **Task embedding dim** | `32` | **LOW** — must be held constant |
| **Ret-norm reset** | `stream` | **LOW** — must be held constant |

**The point of the table:** **the only knob we should be changing in the R-series
attribution is the solver state mode** (`off`, `masked`, `position`, `time`, `both`).
**Everything else must be frozen** — and it must be frozen *before* the comparison, not
*after*. If we change the training budget, the cost, the learning rate, the window, or
the PBT setting between R0 and R4-v1, the attribution is **confounded** and the 95%
interval is **meaningless**.

**Concrete recommendation (proposed):**

1. **Freeze** the 14 non-solver knobs in a single manifest with hashes.
2. **Change only** the solver state mode (and the architecture: R0 vs R4-v1).
3. **Run each cell** under the *same* training budget, cost, seeds, and PBT setting.
4. **Report** the 95% paired net-return interval on the frozen validation panel.
5. **Only then** consider the "capability floor" for proposer coupling.

---

## 4. The 70/15/15 split and the "oldest → newest" training sequence (elaboration of Point 5)

**The user's framing:** "Splitting the same way for all is brilliant, especially since
we can attempt to start anew and go from oldest → newest in the split format, thus
giving us that retrospective bet that maybe it can generalize and follow the trend
chronologically!"

**Answer: the split is already the right choice, and the oldest→newest training
sequence is a *strong* additional safeguard — but it is a *design choice* we need to
make explicitly, not implicitly.**

**Why the 70/15/15 chronological split is right:**

- **Chronological order is the correct train/val/test split** for time-series data.
  Random splitting would leak future information into the training set.
- **The 15% validation is for selection and the 15% final test is for the one-shot
  promotion gate.** The final test is **sealed** — it is never used for training or
  selection.

**The oldest→newest training sequence (a.k.a. "expanding window" or "walk-forward")**

This is a **stronger** variant of the chronological split. Instead of training once on
the first 70% and evaluating on the next 15%, you:

1. **Train on the first 70%** (oldest), evaluate on the next 15%.
2. **Train on the first 85%** (70% + 15% of validation), evaluate on the *remaining*
   15% (final test).
3. **Repeat** as new data becomes available.

**The "retrospective bet" you described** is the key insight: **if the model trained on
2022–2024 data can still produce positive net on 2025–2026 data, it has learned a
*generalizable* pattern, not a *regime-specific* artifact.** That is the strongest form
of the "no leakage" guarantee.

**However — the current dataset has only ~4.9 years of data (2022-01-03 → 2026-09-22).**
The oldest→newest sequence is:
- Train: 2022-01-03 → ~2024-05 (70%)
- Val: ~2024-05 → ~2025-05 (15%)
- Test: ~2025-05 → 2026-09-22 (15%)

That is a **reasonable** sequence, but the "retrospective bet" only works if:
1. **The model is not overfit to the 2022–2024 regime.** The 4.9-year span includes
   multiple market regimes (2022 bear, 2023 recovery, 2024 volatility, 2025 growth).
   If the model is trained on 2022–2024 and evaluated on 2025, it may be generalizing
   **or** it may be **regime-mismatched**.
2. **The 15% validation panel is truly non-overlapping.** The current `split_purge_rows=200`
   is a good start, but we should **increase the purge** (e.g., to 500–1000 rows) to
   ensure no temporal leakage between train and val.
3. **The final test is truly sealed.** If the model has ever seen the 2025–2026 data
   during training, the final test is contaminated.

**Concrete recommendation (proposed):**

1. **Freeze the 70/15/15 split** with **purge ≥ 500 rows** between train and val.
2. **Record the exact panel boundaries** (start/end timestamps) in the manifest.
3. **Train oldest→newest** — this is the "retrospective bet."
4. **Report the per-year net return** (2022, 2023, 2024, 2025, 2026) as a **regime
   diagnostic**. If the model is profitable in 2022–2024 but not in 2025–2026, it is
   regime-mismatched, not generalizable.
5. **Keep the final test sealed** until the retrospective bet has been run and the
   95% interval is qualified.

---

## 5. Window size vs. cost: the 60-minute input shape (elaboration of Point 6)

**The user's framing:** "The input shape takes 60 min (60 × 5 into the MLP flattened +
FiLM + solver state), but I absolutely like matching the window to the window structure.
How did you measure that 2-hour to 1 bp cost? Maybe we can keep that proportion and do
window ablations."

**Answer: the 60-minute window is a good starting point, but we should test the
window-to-cost ratio explicitly, not assume it.**

**The current setup:**
- **Window:** 60 bars = 60 minutes
- **Horizons:** 96, 128, 160, 200 bars = 96–200 minutes
- **Cost:** 1 bp per round-trip
- **Training:** `iters=40` → ~192,000 steps

**The window-to-cost ratio question:**

The model is being asked to **predict the next 96–200 minutes** of price movement,
but it is only given **60 minutes of history**. That is a **1.6–3.3× forward horizon
relative to the input window**. If the model is paying **1 bp per round-trip** to
change position, it needs to be **confident** enough to justify that trade. The
**window-to-cost ratio** should be:

- **If the window is too short relative to the horizon**, the model is making decisions
  on insufficient signal → **over-trades**.
- **If the window is too long relative to the horizon**, the model is making decisions
  on stale signal → **under-trades**.
- **The cost should scale with the decision frequency.** A 1 bp cost is reasonable if
  the model trades **1–5 times per episode**; it is too high if the model trades **20+
  times per episode**.

**The 2-hour to 1 bp measurement (the user's question):**

I did not measure a specific "2-hour to 1 bp" ratio in the current code. The 1 bp cost
is **arbitrary** — it was set to a round number that is in the ballpark of typical
SIP fees. The honest measurement is:

1. **Measure the model's turnover** (position changes per episode) at 1 bp, 0.1 bp, and
   0.01 bp.
2. **Calculate the cost-to-PnL ratio** at each cost level.
3. **If the model's turnover drops to near-zero at 1 bp but rises to 10+ at 0.01 bp**,
   the 1 bp cost is **too high** for the model's decision frequency, and we should
   either (a) increase the window to give it more signal, or (b) reduce the cost, or
   (c) accept that the model is not ready to trade at 1 bp.

**Window ablation (the user's idea):**

This is a **good next step**. The current window (60) is a single point. A window
ablation would test:

| Window | Input dims | What it tells us |
|---|---|---|
| 30 bars (30 min) | 150 | Does the model need less signal? |
| 60 bars (60 min) | 300 | Current state |
| 120 bars (2 hr) | 600 | Does the model need more signal? |
| 240 bars (4 hr) | 1200 | Does the model need even more signal? |

**The recommendation:** **test the window ablation in parallel with the cost sweep.**
A 4-window × 3-cost × 2-seed = **24 cells** is a manageable budget if the training
budget is kept at the current `iters=40` smoke level. The goal is to find the
**window-cost pair** that maximizes net return at 1 bp, subject to the constraint that
the model is not over-trading (turnover ≤ 5 per episode).

---

## 6. `market_relative_v1` vs. `market_relative_v2` (elaboration of Point 7)

**The user's framing:** "Can you expand on how your proposed `market_relative_v2` could
work? I like our original approach as you said — it's training on patterns and not
actual prices — but I guess we have to provide this in some form too!"

**The current `market_relative_v1` transform (grounding from `itrader_env.py:351`):**

```python
def _market_relative_window(self, window: np.ndarray) -> np.ndarray:
    """Create bounded, fitted-state-free OHLCV features for portable checkpoints."""
    close_ref = float(window[-1, self.cols.index(self.close_col)])
    price_scale = max(abs(close_ref), 1e-12)
    for index, column in enumerate(self.cols):
        if column.upper() in {"OPEN", "HIGH", "LOW", "CLOSE"}:
            window[:, index] = window[:, index] / price_scale - 1.0
        elif column.upper() == "VOLUME":
            window[:, index] = np.log1p(np.maximum(window[:, index], 0.0))
        else:
            mean = float(np.mean(window[:, index]))
            std = float(np.std(window[:, index]))
            window[:, index] = (window[:, index] - mean) / max(std, 1e-8)
    return np.clip(_sanitize(window), -10.0, 10.0)
```

**What it does:**
- **Prices** (O/H/L/C) are **divided by the current close** and **1 is subtracted** →
  the model sees **relative returns** (e.g., "this open was 0.5% below the current close").
- **Volume** is **log1p-normalized** → compresses the dynamic range.
- **All features** are **z-scored** (mean 0, std 1) and **clipped** to ±10 → prevents
  outliers from exploding the input.
- **No fitted parameters** are used → the transform is **state-free** and **portable**
  across checkpoints (a key requirement for the replay contract).

**The problem with `market_relative_v1`:**

The transform is **state-free** (good for portability) but **not scale-invariant across
the full dataset**. It normalizes *each window independently* using the *current*
close as the reference. This means:
- **A window in 2022** (when AAPL was ~$30) and **a window in 2025** (when AAPL is
  ~$200) are both normalized to the same scale (±10). This is **good** for the model
  to learn *patterns* rather than *absolute prices*.
- **But the model cannot see the *absolute* level** of the market. If the model needs
  to know "the market is in a high-volatility regime," the `market_relative_v1`
  transform does not provide that.

**The proposed `market_relative_v2` transform:**

The key idea is to **add a second, *fitted* normalization layer** that preserves the
state-free property of `market_relative_v1` while **exposing a market-level signal**
that the model can use to detect regime changes.

```python
def _market_relative_window_v2(self, window: np.ndarray) -> np.ndarray:
    """Add a fitted market-level context to the state-free v1 transform."""
    # Step 1: Apply v1 (state-free, portable).
    window_v1 = self._market_relative_window(window)

    # Step 2: Add a *fitted* market-level context (e.g., realized volatility over
    # the last N bars, using a *frozen* training-time statistic).
    # This is NOT computed from the current window — it is a *precomputed*
    # feature that is loaded from a manifest, so it does not leak future info.
    # Example: realized volatility = std of log-returns over the last 252 bars
    # (1 year), computed once and stored in the dataset manifest.
    if not hasattr(self, "_fitted_volatility"):
        raise ValueError("market_relative_v2 requires a fitted volatility manifest")
    vol = self._fitted_volatility
    window_v1 = np.concatenate([window_v1, [vol] * len(window_v1[0])])

    # Step 3: Re-clip to ±10.
    return np.clip(_sanitize(window_v1), -10.0, 10.0)
```

**Key properties of `market_relative_v2`:**
- **State-free**: the fitted volatility is precomputed and stored in the manifest →
  no leakage, no checkpoint incompatibility.
- **Adds regime signal**: the model can now see "the market is in a high-volatility
  regime" and adjust its position sizing accordingly.
- **Backward-compatible**: the v1 dimensions are a prefix of the v2 dimensions →
  old checkpoints can be loaded into v2 with zero-init padding.

**The recommendation:**
- **Keep `market_relative_v1` as the default** — it is the state-free, portable baseline.
- **Add `market_relative_v2` as an opt-in factor** — it adds regime signal but requires
  a fitted manifest.
- **Compare v1 vs v2 in a controlled ablation** — hold everything else constant (training
  budget, cost, seeds, window) and change only the transform.
- **Do NOT promote v2 until the v1 vs v2 comparison is qualified** — v2 is a hypothesis,
  not a settled architecture.

---

## 7. The "movement" penalty and turnover (elaboration of Point 8)

**The user's framing:** "I guess you are right, but here it gets tricky — we have to
be careful with this. Technically each decision is a 'movement,' so that's why we are
penalizing."

**Answer: the penalty is correct, but we need to make sure it is *proportional* to the
movement, not *binary*.**

**The current penalty (grounding from `itrader_env.py:792`):**

```python
turnover_t = abs(float(self.pos) - prev_pos)  # 0, 1 or 2
```

**What this means:**
- **Flat → Flat**: `turnover_t = 0` → **no penalty**.
- **Flat → Long**: `turnover_t = 1` → **1 unit of cost**.
- **Long → Long**: `turnover_t = 0` → **no penalty**.
- **Long → Short**: `turnover_t = 2` → **2 units of cost** (because the position changes
  from +1 to -1, a full round-trip).

**The "movement" framing is correct** — every position change is a movement, and every
movement should be penalized. The question is: **is the penalty proportional to the
movement?**

**The answer: yes, it is.** A 1→2 change (Long → Short) is **twice** as costly as a
0→1 change (Flat → Long), because it involves **two** position changes (Long → Flat,
then Flat → Short). This is the correct economic model — a round-trip trade costs
**two** one-way fees.

**The subtlety: the `evidence_cost_bps` scales the penalty.** If the model is currently
**over-trading** (e.g., 20 position changes per episode), the penalty at 1 bp is
**0.20%** — which is **substantial**. At 0.01 bp, the same over-trading is only
**0.002%** — which is **negligible**. So the cost knob is the **right** tool for
discouraging over-trading, but **only if the model is actually over-trading**.

**The recommendation:**
- **Keep the `turnover_t` penalty as-is** — it is the correct economic model.
- **Measure the model's turnover** at 1 bp, 0.1 bp, and 0.01 bp.
- **If the model is over-trading at 0.01 bp** (turnover > 10 per episode), the problem
  is not the cost knob — it is the **reward structure**. The model is being *encouraged*
  to trade by the PnL signal, and the cost is too small to stop it. In that case, add
  an **explicit turnover penalty** that does NOT scale with `evidence_cost_bps` (e.g.,
  a fixed 0.5 bp per position change, added on top of the `evidence_cost_bps`).
- **If the model is under-trading at 1 bp** (turnover < 2 per episode), the problem is
  the **cost knob** — it is too high, and the model is being *discouraged* from
  trading. Lower the cost to 0.1 bp or 0.01 bp to see if the model expresses more
  signal.

---

## 8. The family roadmap: `Discrete(3)` → richer action spaces (elaboration of Point 9)

**The user's framing:** "Technically we are only asking 'flat, long, or short?' and
attempting to modulate the solver state to the input shape. But we were trying to
discretise this particular part of trading as task family 00 (directional decisions).
01 is vol breakout, 02 is mean reversion pairs. With this in mind, maybe for the other
families that were planned to have `Discrete(3)`, we could actually expand to this, as
it simulates these things. But again, the online play (Phase C) will handle this
eventually — not tested fully yet, and still experimental. Can you scaffold how that
would look?"

**Answer: the current `Discrete(3)` is the right *starting point* for family 00, but
the roadmap to richer action spaces for families 01–02 is a **good long-term
investment** — and it is **exactly** what the masterplan anticipates.**

**The current state (grounding from the masterplan):**

- **Family 00**: directional (short/flat/long) — **`Discrete(3)`**. ✅
- **Family 01**: volatility breakout — **`Discrete(3)`** planned.
- **Family 02**: mean reversion pairs — **`Discrete(3)`** planned.
- **Family 03**: trade management — **`Discrete(3)`** planned.
- **Family 04**: abstention/WAIT semantics — **`Discrete(3)`** planned.

**The problem with a uniform `Discrete(3)` action space:**

The current `Discrete(3)` action space assumes that **every family makes the same
three decisions: short, flat, long.** But that is **not true** for families 01–02:

- **Family 01 (volatility breakout)**: the model should be able to **enter a position,
  set a stop-loss, and exit**. That is **more than 3 actions** — it is a **sequence**
  of actions.
- **Family 02 (mean reversion pairs)**: the model should be able to **enter a pair,
  hold, and exit** — again, a **sequence**, not a single decision.
- **Family 03 (trade management)**: the model should be able to **adjust position
  size, set a stop-loss, and exit** — a **continuous** decision space, not discrete.

**The scaffold for richer action spaces:**

**Option A: `Discrete(3)` with a *conditional* action space**

The model still outputs 3 actions (short/flat/long), but **each action has a *payload***:
- **Short**: {position_size, stop_loss, exit_trigger}
- **Flat**: {position_size, stop_loss, exit_trigger}
- **Long**: {position_size, stop_loss, exit_trigger}

This is a **hybrid** discrete/continuous action space. The discrete part is the
**direction**, and the continuous part is the **trade management**. The model learns
**both** in one policy.

**Option B: `Discrete(5)` with explicit WAIT and EXIT**

Expand the action space to 5:
- **Short**
- **Flat**
- **Long**
- **WAIT** (hold current position, do not trade)
- **EXIT** (liquidate current position)

This is a **cleaner** discrete space that explicitly models **abstention** (family 04)
and **liquidation** (family 03). The model can now *choose* to wait rather than
being forced to either enter or exit.

**Option C: `Discrete(3)` + a separate **trade management** policy**

The model outputs 3 actions (short/flat/long) for the **directional** decision, and a
**separate** policy outputs the **trade management** parameters (position_size,
stop_loss, exit_trigger). This is a **multi-head** architecture — the masterplan
already anticipates this ("a shared recurrent trunk with family-specific
policy/value heads").

**The recommendation:**

1. **Keep `Discrete(3)` for family 00** — it is the simplest and the most directly
   testable.
2. **Add `WAIT` and `EXIT` to the action space** for families 01–04 — this is
   **Option B** and is the **lowest-risk** expansion.
3. **Scaffold the multi-head architecture** for families 01–03 — this is **Option C**
   and is the **long-term** investment.
4. **Do NOT expand the action space until the `Discrete(3)` baseline is promoted** —
   the masterplan is explicit: "families 00–02 may share `Discrete(3)` output width
   without settling input schemas, task semantics, reward assignments, or state
   ownership."

**The concrete scaffold:**

```python
# Option B: Discrete(5) with WAIT and EXIT
# 0 = SHORT, 1 = FLAT, 2 = LONG, 3 = WAIT, 4 = EXIT

# The model outputs a single integer in {0, 1, 2, 3, 4}.
# The environment interprets:
#   0: enter/hold short
#   1: hold flat
#   2: enter/hold long
#   3: wait (no change, no cost)
#   4: liquidate (close position, incur liquidation cost)

# This is a **minimal** expansion that adds the two most important
# trade-management actions without changing the discrete action space.
```

**The Phase C connection:**

The user is right that **Phase C will handle this eventually**. The `DecisionGate`,
`RiskEngine`, and `Executor` shell in Phase C will **translate** the policy's action
into a **concrete trade** (with position sizing, stop-loss, and execution logic). The
`Discrete(3)` → `Discrete(5)` → multi-head evolution is the **training-time** version of
that translation. The key is to **keep the training-time action space simple**
(`Discrete(3)` or `Discrete(5)`) and let **Phase C handle the complexity** of
position sizing, stop-loss, and execution.

---

## 9. Consolidated recommendations (priority-ordered)

| # | Recommendation | Rationale | Phase |
|---|---|---|---|
| **1** | **Freeze the 70/15/15 split + panel hashes** in the manifest. | Prevents leakage; enables honest comparison. | Phase A |
| **2** | **Freeze all non-solver knobs** (training budget, cost, LR, gamma, batch, window, PBT) in a single manifest with hashes. | Prevents confounded attribution. | Phase A |
| **3** | **Establish the R0 ceiling** under a larger training budget (×10–100) and lower cost (0.1 bp, 0.01 bp). | Establishes the baseline before promoting R4-v1. | Phase A |
| **4** | **Run R4-v1 under the same budget + cost** as R0. | True architectural comparison. | Phase A |
| **5** | **Report the cost-response curve** (0.01, 0.1, 1 bp × R0, R4-v1 × multi-seed). | Most important diagnostic; reveals the model's unmasked behavior. | Phase A |
| **6** | **Two-sided 95% paired net-return interval** on the frozen validation panel, block-aware. | The accepted instrument (Q26). | Phase A |
| **7** | **Window ablation** (30, 60, 120, 240 bars) in parallel with the cost sweep. | Finds the window-cost pair that maximizes net at 1 bp. | Phase A |
| **8** | **Oldest→newest training sequence** (expanding window / walk-forward). | The "retrospective bet" — strongest no-leakage guarantee. | Phase A |
| **9** | **Per-year net return diagnostic** (2022, 2023, 2024, 2025, 2026). | Detects regime mismatch. | Phase A |
| **10** | **Measure turnover** at each cost level. | Confirms the model is not over-trading or under-trading. | Phase A |
| **11** | **`market_relative_v2` as an opt-in factor** (fitted volatility context). | Adds regime signal without breaking portability. | Phase A/B |
| **12** | **`Discrete(5)` action space** (add WAIT + EXIT) for families 01–04. | Minimal expansion that models abstention and liquidation. | Phase B |
| **13** | **Multi-head architecture** for families 01–03. | Long-term investment; anticipates the shared-trunk proposal. | Phase B/C |
| **14** | **Do NOT open the final test** until the 95% interval is qualified and the cost-response curve is complete. | Preserves the sealed final test. | Phase A |

---

## 10. Risks and open questions

| Risk | Mitigation |
|---|---|
| **Data leakage** | Freeze split + purge ≥ 500 rows; oldest→newest sequence; sealed final test. |
| **Confounded attribution** | Freeze all non-solver knobs in a manifest; change only the solver state mode. |
| **Cost masking** | Cost-response curve (0.01, 0.1, 1 bp); do not promote on 0.01 bp alone. |
| **Regime mismatch** | Per-year net return diagnostic; oldest→newest sequence. |
| **Over-trading** | Measure turnover; add explicit turnover penalty if turnover > 10 at 0.01 bp. |
| **Window-cost mismatch** | Window ablation (30, 60, 120, 240 bars) × cost sweep. |
| **Action space too simple** | `Discrete(5)` (WAIT + EXIT) → multi-head for families 01–03. |
| **`market_relative_v2` complexity** | Keep v1 as default; v2 as opt-in factor; compare in a controlled ablation. |
| **Training budget** | The current `iters=40` is a smoke budget; the real training budget is 10–100×. |
| **Ray 2.58 vs 2.9** | Keep Ray 2.58 as the R-series authority; 2.9 for historical reproduction only. |

**Open questions for the review lane:**

1. **What is the acceptable turnover threshold** (e.g., ≤ 5 per episode) for the
   "capability floor"?
2. **Should the 0.01 bp cost be used as a promotion criterion** or only as a
   diagnostic?
3. **What is the minimum purge** (rows) between train and val to prevent temporal
   leakage?
4. **Should the window ablation use the same training budget** or a proportional budget
   (e.g., longer windows get more training steps)?
5. **Is the "retrospective bet" (oldest→newest) a one-time run** or a **recurring**
   procedure as new data becomes available?

---

*End of advisory.*
