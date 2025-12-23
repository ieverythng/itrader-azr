# Phase B Performance Analysis & Planning

## Current Run Analysis (window_size=30, WITH start_idx changes)

### Key Observations from Metrics

#### 1. **Reward Signal Learnability** ✅ PARTIALLY LEARNABLE

**Evidence:**
- **Episode Return Mean**: Shows upward trend from -1 to ~1.5 until iteration ~30, then declines
- **Policy Entropy**: Decreasing (1.05 → 0.80), indicating policy is becoming more confident
- **PnL Delta Mean**: Generally positive (peak ~0.01), though highly variable

**Concerns:**
- High variability in episode return (fluctuates ±1.0 around mean)
- Decline after iteration 30 suggests either:
  - Overfitting to early tasks
  - Reward signal becoming too noisy as task diversity increases
  - Insufficient capacity to generalize across diverse start_idx positions

#### 2. **Action Behavior** ⚠️ EXPLORATORY BUT SHIFTING

**Patterns:**
- Initial phase (0-20): Short/flat dominant (~40%/35%)
- Middle phase (20-35): Long becomes dominant (~45%+)
- Late phase (35-50): Returns to more balanced distribution

**Interpretation:**
- Agent is exploring different strategies (good)
- Preference shifts suggest learning, but high variability indicates uncertainty
- With start_idx diversity, different market regimes may require different strategies

#### 3. **Reward Component Analysis** 📊 SIGNAL STRENGTH ISSUES

**Critical Finding:**
- **PnL Delta**: ~0.01 mean (dominant signal)
- **Drawdown**: ~-0.006 (very small, 60% of PnL magnitude)
- **Turnover**: ~0.001 (tiny, 10% of PnL magnitude)

**Problem:** Drawdown and turnover penalties are TOO SMALL relative to PnL signal
- Current config: `dd_w=0.04`, `turnover_w=0.001`
- With normalization (`norm_mode="all"`), these get scaled down further
- Result: Agent primarily optimizes PnL, ignoring risk management

#### 4. **Episode Length Stability** ✅ GOOD

- Stable at 140-152 steps (horizon enforcement working)
- No premature terminations
- Suggests environment is functioning correctly

---

## Phase B Requirements Assessment

### ✅ Requirements Met:
1. **Task Diversity**: start_idx changes expose agent to different market regimes
2. **Episode Stability**: No crashes, NaNs, or Infs
3. **Learning Signal**: Agent improves initially (upward trend)
4. **Generalization Attempt**: Action preferences shift (exploring strategies)

### ⚠️ Requirements at Risk:
1. **Reward Signal Clarity**: High variability may mask learnable signal
2. **Risk Management**: Drawdown/turnover penalties too weak
3. **Generalization Stability**: Decline after iteration 30 suggests overfitting

---

## Root Cause Analysis

### Primary Drivers of High Variability:

#### 1. **Reward Scale Imbalance** 🔴 CRITICAL
```
Current: PnL (~0.01) >> DD (~-0.006) >> Turnover (~0.001)
Ideal:   PnL ~ DD ~ Turnover (balanced scales)
```

**Why it matters:**
- Agent optimizes primarily for PnL
- Risk penalties (DD) are 60% of PnL signal → insufficient constraint
- Turnover penalty is negligible → no trading cost awareness

**Fix:** Increase `dd_w` and `turnover_w` in PBT search space, or disable normalization for penalties

#### 2. **start_idx Diversity Impact** 🟡 MODERATE
- Each start_idx exposes different market regime
- Agent sees completely different price patterns each episode
- High variance in "learnability" of each regime
- Some regimes may be easier/harder to trade profitably

**Mitigation:**
- PBT should find optimal balance
- Proposer should learn which regimes are most informative
- Consider curriculum: start with narrower start_idx range, expand

#### 3. **Normalization Side Effects** 🟡 MODERATE
- `norm_mode="all"` scales ALL terms by `ret_unit`
- This can dampen penalty signals if volatility is high
- Current `norm_clip=5.0` may be too permissive

**Options:**
- Use `norm_mode="penalties"` (only scale penalties, not PnL)
- Increase `norm_clip` to allow larger penalty signals
- Make `norm_mode` a PBT hyperparameter

#### 4. **Task Diversity vs. Learning Capacity** 🟢 LIKELY OK
- With varying horizons (96-200) + start_idx diversity, agent sees many task variants
- Current model capacity (256x256 MLP) may be sufficient
- FiLM conditioning should help distinguish tasks

---

## Comparison: window_size=50 (WITHOUT start_idx changes)

**Expected Differences:**
1. **Lower Variability**: Episodes always start from same position → more consistent signal
2. **Higher Initial Performance**: Agent can memorize early data patterns
3. **Lower Generalization**: Won't work well on unseen data positions
4. **Faster Convergence**: Less diversity = easier to overfit

**This run (window_size=30, WITH start_idx):**
- Higher variability (expected with diversity)
- Should generalize better (exposed to multiple regimes)
- Lower peak performance (harder learning problem)
- More "honest" evaluation for Phase B

---

## Actionable Recommendations

### Immediate (Before PBT):

#### 1. **Reward Balance Tuning** 🔴 PRIORITY
```python
# In train_solo_V2.py, expand PBT search space:
"env_config/task_sampler_overrides": {
    "dd_w": tune.loguniform(0.02, 0.20),  # Current: 0.04, increase max
    "turnover_w": tune.loguniform(0.001, 0.01),  # Current: 0.001, increase max
    # Consider adding:
    "norm_mode": tune.choice(["all", "penalties", "off"]),  # Test normalization modes
}
```

#### 2. **Add Reward Signal Diagnostics**
- Track per-component magnitudes in callbacks
- Log `reward_scale` and `ret_unit` distributions
- Identify when penalties are being clipped/damped

#### 3. **start_idx Distribution Analysis**
- Log histogram of sampled start_idx across workers
- Verify uniform distribution across valid range
- Check if certain start_idx ranges correlate with better returns

### Short-term (PBT Optimization):

#### 1. **Expanded Hyperparameter Space**
```python
hyperparam_mutations = {
    # Reward shaping (critical for Phase B)
    "env_config/task_sampler_overrides": {
        "dd_w": tune.loguniform(0.02, 0.20),
        "turnover_w": tune.loguniform(0.001, 0.01),
        "dd_T_weight": tune.loguniform(0.0, 0.2),  # Test terminal DD
        "norm_mode": tune.choice(["all", "penalties"]),  # Exclude "off" for stability
    },
    # Model capacity (may need more for diverse tasks)
    "model/custom_model_config/fcnet_hiddens": tune.choice([
        [256, 256], [512, 256], [256, 256, 128]  # Test capacity
    ]),
    # Learning rate (critical for diverse tasks)
    "lr": tune.loguniform(1e-5, 5e-4),  # Current: 5e-5
}
```

#### 2. **Curriculum Learning Strategy**
- Option A: Start with narrower start_idx range, expand over time
- Option B: Weight sampling toward "easier" regimes initially
- Option C: Let PBT/proposer discover optimal distribution

### Medium-term (Proposer Integration):

#### 1. **Proposer Reward Configuration Search**
- Proposer should generate tasks with diverse reward params
- Focus on balancing PnL/DD/Turnover scales
- Track which reward configs lead to better learning

#### 2. **Task Difficulty Scoring**
- Add metric: "task_difficulty" based on:
  - Market regime volatility at start_idx
  - Historical Sharpe ratio for that window
  - Agent's current performance on similar tasks
- Use difficulty to guide proposer sampling

#### 3. **Regime-Aware Task Generation**
- Proposer learns: "Bull markets need high DD penalty"
- Proposer learns: "High volatility needs lower turnover_w"
- Adapt reward params based on start_idx market characteristics

---

## Success Criteria for Phase B

### Phase B Readiness Checklist:

#### ✅ Signal Quality:
- [x] Episode return shows learning trend (met, but noisy)
- [ ] Reward components are balanced (PnL ~ DD ~ Turnover)
- [ ] Variability decreases over time (not yet)

#### ✅ Generalization:
- [x] Agent works across diverse start_idx (met)
- [ ] Performance doesn't collapse with task diversity (at risk)
- [ ] Action preferences adapt to market regimes (partial)

#### ✅ Stability:
- [x] No NaNs/Infs (met)
- [x] Episode lengths stable (met)
- [ ] Return variance decreases with training (not yet)

#### ⚠️ Trading Readiness:
- [ ] Positive mean PnL across diverse regimes
- [ ] Drawdown control (current penalties too weak)
- [ ] Reasonable turnover (current penalty too weak)
- [ ] Consistent strategy (high action preference variance is concerning)

---

## Next Steps Priority Order

1. **🔴 CRITICAL: Fix Reward Balance**
   - Increase `dd_w` max in PBT to 0.20
   - Increase `turnover_w` max to 0.01
   - Test `norm_mode="penalties"` vs `"all"`

2. **🟡 HIGH: Run PBT with Expanded Search**
   - Let PBT find optimal reward weights
   - Test different normalization modes
   - Monitor component balance

3. **🟡 HIGH: Add Diagnostic Metrics**
   - Track reward component magnitudes
   - Log start_idx distribution
   - Monitor per-regime performance

4. **🟢 MEDIUM: Proposer Integration**
   - Generate tasks with balanced reward configs
   - Learn which reward params work best
   - Adapt based on agent performance

5. **🟢 LOW: Curriculum Learning**
   - Start with easier regimes, expand
   - Or: Let proposer discover optimal curriculum

---

## Expected Outcomes After Fixes

### With Balanced Rewards:
- **Lower Return Variance**: More consistent signal
- **Better Risk Control**: Lower drawdowns, controlled turnover
- **Stable Learning**: Continuous improvement, less decline
- **Regime Adaptation**: Different strategies for different market conditions

### For Phase B Proposer:
- **Informative Reward Signal**: Clear feedback on task quality
- **Learnable Task Space**: Proposer can identify good/bad reward configs
- **Generalization Baseline**: Agent that works across diverse tasks

---

## Conclusion

**Current Status**: ✅ Phase B infrastructure ready, ⚠️ reward signal needs tuning

The agent IS learning, but high variability suggests:
1. **Reward imbalance** (primary issue)
2. **Task diversity** (expected, but may be too aggressive)
3. **Normalization effects** (may be damping penalties)

**Recommendation**: Fix reward balance FIRST (critical blocker), then run PBT to find optimal hyperparameters. Proposer integration can proceed in parallel but will benefit from stable reward signal.

**Timeline Estimate:**
- Reward balance fix: 1-2 days (quick PBT run to validate)
- PBT optimization: 1 week (full sweep)
- Proposer integration: 2 weeks (after stable baseline)

The foundation is solid - just needs tuning! 🚀

