# Recurrent Solver Ablation — Design & Rationale

Last updated: 2026-09-03
Status: **Design complete, implementation pending** (after alignment ablation runs)

## The Core Insight

The real power of an RNN solver is not "encoding a window." Any feed-forward network can encode a window. The power is **persistent hidden state that compounds across environment steps within an episode.**

In trading terms:

- Step 1: Market opens, solver sees first bar. Hidden state initialized (task-conditioned).
- Step 2: New bar arrives. LSTM updates `(h, c)` using previous state + new input. The network now "remembers" step 1's context.
- Step 3: Another bar. State accumulates. If the solver has been losing for 3 steps, the hidden state encodes that drawdown trajectory without needing an explicit `drawdown_so_far` feature.
- ...
- Step N (episode end): State resets. New episode starts fresh.

This is **compounding**: each step's gradient update flows through the entire recurrent path back to the episode start. The value function learns "what happens over the next 20 steps if I'm in *this* state," not just "what does this window look like."

### Why FiLM MLP Can't Do This

`SolverMLPFiLM` is stateless. Every forward pass is independent: `f(obs_t) → action_t`. There is no memory of `obs_{t-1}`, `obs_{t-2}`, etc. The only "temporal" information is whatever the environment packs into the rolling observation window — and that's a fixed-size snapshot, not a learned accumulation.

### Why Window-Encoder LSTM Is Not Enough Either

The current `SolverLSTMModel` reshapes the flat obs into `(B, win, F)` and runs an LSTM over those `win` timesteps. But it **resets every step**. The hidden state from processing bar 1..30 is discarded when the next observation arrives. It's a fancy window encoder, not a recurrent policy.

The difference:

| | Window Encoder (current) | True Recurrent (target) |
|---|---|---|
| Input per step | Full flat window (150 dims) | New bar(s) only (5-10 dims) |
| Hidden state | Computed, discarded each step | Persists across steps |
| Memory | Limited to `win` bars | Unlimited within episode |
| Gradient flow | Within window only | Through entire episode |
| Task conditioning | Baked into flat vector (breaks reshape) | FiLM on initial state + per-step injection |

## Why FiLM + True Recurrence Is The Right Architecture For iTrader

A trading episode has two fundamentally different kinds of information:

1. **Static task identity** — "You are solving a mean-reversion task on EURUSD, horizon 256, target Sharpe 1.2." This never changes within an episode. It's a *condition*, not a *signal*.
2. **Dynamic market state** — Prices, volumes, volatility, position P&L. This evolves every step. It's the *input stream*.

FiLM is the perfect interface between these two because it's literally "modulate the representation based on a condition." The task embedding doesn't get *added* to the observation (which would conflate identity with signal) — it *reshapes how the network processes* the signal. Different scale, different shift, per feature channel.

Make that recurrent and you get **task-aware compounding**: by step 50, the hidden state has accumulated 50 steps of market context *filtered through the task lens*. A mean-reversion solver's hidden state at step 50 looks fundamentally different from a momentum solver's hidden state at step 50 — not because the inputs are different (same market data), but because FiLM shaped every single recurrent update differently.

### What FiLM Buys Over "Just Concat Task To Input"

If you concat `[price, task_embed]` and feed into LSTM, the task embedding gets **diluted** over 256 steps. The LSTM's effective attention drifts toward recent prices. FiLM avoids this because it's applied *after* the recurrent update — the task condition is reasserted at every step, not just at initialization. It's a constant reminder: "You're still solving mean-reversion, don't forget."

### Attention To The Moving Window: Short-Term vs Long-Term Pattern Derivation

This is where the architecture becomes genuinely interesting for RL trading. A true recurrent FiLM solver gets **two timescales of pattern detection for free**:

**Short-term patterns (1-20 bars):**
The hidden state's *fast-moving components* (the parts that change most between consecutive steps) capture recent price action, momentum bursts, and microstructure signals. These are the "what just happened" features. The LSTM's input-to-hidden weights (`W_ih`) dominate here — new information flows in directly.

**Long-term patterns (50-256+ bars):**
The hidden state's *slow-moving components* (the parts that persist across many steps) encode regime context: "vol has been elevated for 40 bars," "this is a trending market, not mean-reverting," "I've been in drawdown for 15 steps." These are the "what's been happening" features. The hidden-to-hidden weights (`Whh`) dominate here — information persists and compounds.

**The key insight**: A single recurrent state naturally separates these timescales through its internal dynamics. Fast components decay (forgetting gate), slow components persist (input + forget gates aligned). This is *exactly* what a trader needs: react quickly to new information while maintaining awareness of the broader regime. No explicit multi-scale feature engineering required — the LSTM learns the decomposition.

**FiLM's role in timescale separation**: The task embedding modulates *which* timescale matters for the current task. A mean-reversion task (short horizon) should weight fast components more heavily; a trend-following task (long horizon) should weight slow components. FiLM scale/shift effectively reweights the hidden state's channels per-task, so the same recurrent architecture adapts its effective "memory horizon" based on what it's solving.

**Overarching goal within the episode**: The solver isn't just predicting the next bar — it's optimizing a *trajectory*. The hidden state at step t encodes not just "where am I in market state space" but "where am I in my *plan*." If the task is "buy EURUSD, hold 100 bars, target +2%," then at step 50 the hidden state should encode: "I'm halfway through my holding period, position is +0.8%, vol has been calm, I should hold." The recurrence *is* the plan tracker. This is something a feed-forward window encoder fundamentally cannot do — it has no concept of "progress toward goal."

### Practical Implications For Pattern Discovery

| Pattern Type | Timescale | How Recurrent FiLM Captures It | Example |
|---|---|---|---|
| Momentum burst | 1-5 bars | Fast hidden components, input-dominated | Price gaps up 3σ in 2 bars → immediate buy signal |
| Mean reversion | 5-20 bars | Medium persistence, forget gate tuned | Price deviates from 20-bar MA → fade the move |
| Regime shift | 20-100 bars | Slow components accumulate vol/trend context | Vol regime transitions from low to high over 30 bars |
| Trend persistence | 50-256 bars | Long-term hidden state, slow decay | Uptrend continues for 80+ bars → hold and trail stop |
| Episode progress | Full episode | State encodes "how far along am I" | Step 120/256: time to start reducing position |

The last row is the killer feature. **Episode progress tracking** emerges naturally from recurrence: the hidden state at step t is a function of all t previous inputs, so it implicitly knows "how much has happened." A feed-forward network would need an explicit `t/T` feature; the recurrent solver learns it automatically.

## Ablation Design

### Variants

| ID | Name | Head | State | Purpose |
|----|------|------|-------|---------|
| R0 | FiLM Baseline | `SolverMLPFiLM` | None | Control. Current default. |
| R1 | Window LSTM Encoder | Corrected `SolverWindowLSTMFiLM` | Reset every step | Isolate within-window temporal bias. Same compute as R0-ish, but with LSTM structure over the window. |
| R2 | True Recurrent LSTM | `SolverRecurrentLSTM` | Persists across steps, reset at episode end | **The main event.** Full recurrence. Hidden state carries reward bias, position memory, regime awareness through the episode. |
| R3 | True Recurrent GRU | `SolverRecurrentGRU` | Persists across steps, reset at episode end | Compare LSTM vs GRU. GRU has fewer params (no separate cell state), may train faster. Tests if full LSTM capacity is needed. |
| R4 | Recurrent + Task-FiLM State | `SolverRecurrentLSTMTaskFiLM` | Persists, task-conditioned initial state | Isolate the benefit of task conditioning on the recurrent state itself. Does knowing *which task* you're solving at initialization improve long-horizon performance? |

### R2 Sub-Variants: Isolating FiLM's Role In Recurrence

The FiLM + recurrence combo is the novel contribution. Break it down:

| ID | Variant | Task Conditioning | What It Isolates |
|----|---------|-------------------|------------------|
| R2a | Recurrent LSTM, task concat to input | Concat `[price, task]` → LSTM | Baseline recurrence (task as just another input) |
| R2b | Recurrent LSTM + FiLM init only | Task → initial `(h₀, c₀)` via FiLM | Does task-conditioned *start* help? |
| R2c | Recurrent LSTM + FiLM per-step | Task → FiLM modulation after each LSTM step | Does *reasserting* task every step help? (The "constant reminder" effect) |
| R2d | Recurrent LSTM + FiLM init + per-step on value head only | Task-FiLM on value network's hidden state, not policy | Does the *value function* benefit more from task conditioning than the policy? |

**Hypothesis**: For short episodes (<100 steps), R2b ≈ R2c (init-only is enough, task identity doesn't fade). For long episodes (256+), R2c > R2b (task identity fades from hidden state without per-step reassertion). R2d tests whether the value function (which estimates "expected future reward") needs task awareness more than the policy (which just outputs actions).

### What Each Pair Isolates

- **R0 → R1**: Does LSTM structure over the window help vs MLP? (Temporal inductive bias within a fixed window.)
- **R1 → R2**: Does persistent state help vs reset-every-step? (**The compounding effect.** This is the key comparison.)
- **R2a → R2b**: Does FiLM init beat concat for task conditioning?
- **R2b → R2c**: Does per-step FiLM beat init-only? (The "constant reminder" hypothesis.)
- **R2c → R2d**: Does the value head benefit more from task-FiLM than the policy head?
- **R2 → R3**: Is LSTM's extra capacity (cell state) worth it vs GRU? (Practical: training speed, overfitting risk.)

### Shared Configuration

All variants:
- Same task pool (generated once, tagged with regimes)
- Same dataset slice, same seed set
- Same train budget (total timesteps)
- Same PPO hyperparameters (lr, clip range, gamma, GAE lambda) — EXCEPT:
  - R2-R4: `shuffle_sequences=False`, `recurrent_policy=True`
  - R2-R4: `minibatch_size` must divide `max_seq_length` cleanly
  - R2-R4: `max_seq_length` = episode length (or capped at 512)
- Evaluation: held-out rolling windows, capability report by family/regime/horizon

### New Metric: Memory Utilization

For recurrent variants only, track:
- **Long/short horizon ratio**: Performance on horizon≥256 tasks vs horizon≤64 tasks. A truly recurrent solver should show *larger* gains on long-horizon tasks (more steps = more compounding).
- **State norm trajectory**: L2 norm of `(h, c)` over the episode. Should grow then stabilize, not explode or collapse.
- **Forgetting probe**: Inject a regime shift mid-episode (e.g., vol spike at step 50/100). Does the solver's risk adjust faster than R0/R1?
- **Timescale separation score** (new): Correlation between fast hidden components (high step-to-step variance) and short-horizon task performance, vs slow components (low variance, high persistence) and long-horizon task performance. Validates that the LSTM is actually decomposing timescales as theorized.

## Implementation Notes

### Obs-Shape Fix (Prerequisite for All)

```python
# In model __init__:
price_dim = win * n_feat  # e.g., 30 * 5 = 150
task_dim = 32
assert obs_space.shape[0] == price_dim + task_dim

# In forward:
obs = input_dict["obs"].float()
price_flat = obs[:, :price_dim]       # (B, 150)
task_embed = obs[:, price_dim:]       # (B, 32)
```

No more `flat_dim // win`. Explicit, testable, no silent breakage.

### R2c: True Recurrent LSTM + Per-Step FiLM (The Novel Variant)

```python
class SolverRecurrentLSTMTFiLM(nn.Module):
    """True recurrent LSTM with per-step task-FiLM modulation.
    
    The hidden state persists across environment steps within an episode.
    Task embedding modulates the state at initialization AND after every
    recurrent update, keeping task identity "alive" throughout the episode.
    """
    def __init__(self, n_feat=5, hidden_dim=128, task_dim=32):
        super().__init__()
        self.n_feat = n_feat
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim

        # Core recurrent cell: processes the NEW BAR only (not full window)
        self.lstm_cell = nn.LSTMCell(n_feat, hidden_dim)

        # FiLM on initial state: task sets the "personality" at episode start
        self.film_init_h = nn.Linear(task_dim, hidden_dim)
        self.film_init_c = nn.Linear(task_dim, hidden_dim)

        # FiLM per-step: task reasserted after every recurrent update
        # Scale centered at 1 (identity) so bad embeddings can't zero the state
        self.film_step_scale = nn.Linear(task_dim, hidden_dim)
        self.film_step_shift = nn.Linear(task_dim, hidden_dim)

        # Action head
        self.pi_head = nn.Linear(hidden_dim, action_dim)
        self.vf_head = nn.Linear(hidden_dim, 1)

    def get_initial_state(self):
        return [torch.zeros(1, self.hidden_dim),
                torch.zeros(1, self.hidden_dim)]

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        # CRITICAL: only the new bar goes into the LSTM (5 dims)
        # The recurrence IS the window — hidden state replaces rolling buffer
        price = obs[:, :self.n_feat]       # (B, 5)
        task  = obs[:, self.n_feat:]      # (B, 32)

        B = price.size(0)

        if state[0].abs().sum() == 0:     # episode start
            h = torch.tanh(self.film_init_h(task))  # bounded init
            c = torch.tanh(self.film_init_c(task))
        else:
            h, c = state                   # carry forward from previous step

        # Recurrent update: new bar + previous state → new state
        h, c = self.lstm_cell(price, (h, c))

        # Per-step FiLM: task modulates the post-LSTM state
        scale = 1.0 + self.film_step_scale(task)   # centered at identity
        shift = self.film_step_shift(task)
        h = h * scale + shift

        logits = self.pi_head(h)
        value = self.vf_head(h)
        return logits, [h, c]  # NEW state carried to next environment step
```

Key design decisions:
- **Only the new bar** (5 dims) enters the LSTM, not the full 150-dim window. The hidden state *is* the learned window — no more `win` parameter, no reshape assumptions, no `flat_dim // win` bugs. This is a massive simplification.
- **`scale = 1 + ...`** — FiLM scale centered at identity. A zero-initialized or corrupted task embedding degrades to "no modulation" rather than "zero out the state."
- **`tanh` on init** — keeps initial state bounded, prevents explosion from step 1.
- **`return logits, [h, c]`** — RLlib carries this state to the next environment step. State resets only when `done=True`.

### PPO Config Delta (R2-R4)

```python
# In train_solo_V2.py, for recurrent variants:
config = {
    "recurrent_policy": True,
    "shuffle_sequences": False,
    "max_seq_length": episode_length,  # e.g., 256 or 512
    "rollout_fragment_length": "auto",  # = max_seq_length
    # minibatch_size must be: seq_len * (batch / seq_len)
    # e.g., seq=256, batch=4096 → minibatch=256*16=4096 ✓
}
```

### Testing Checklist

- [ ] Unit test: obs-shape split produces correct `(price, task)` tensors
- [ ] Unit test: `get_initial_state` returns zero state; after forward, state is non-zero
- [ ] Unit test: state persists across 2 consecutive forward calls (simulate 2 steps)
- [ ] Unit test: state resets to zero after `done=True`
- [ ] Unit test: FiLM scale is ≈1 when task embedding is zero (identity check)
- [ ] Integration: 10-step smoke run, verify no NaN, state norm bounded
- [ ] Ablation: R0 vs R2c on small task pool (5 tasks, 500 steps), verify R2c trains without crashing
- [ ] Timescale probe: after training, correlate hidden component variance with horizon

## Relationship To Alignment Ablation

The alignment ablation (v0-v3) and the recurrence ablation (R0-R4) are **orthogonal**:

- Alignment: How are episodes batched/sequenced in PPO training?
- Recurrence: What architecture does the solver use?

Ideally, you'd run a 2D grid (4 alignment × 5 recurrent = 20 runs), but that's expensive. Practical approach:

1. Run alignment ablation on R0 (FiLM) first — cheap, establishes baseline.
2. Run recurrence ablation on the BEST alignment mode from step 1.
3. If both show gains, optionally confirm the interaction on the winning cell.

This keeps total runs to ~9 instead of 20 while capturing 90% of the information.

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Recurrent PPO is much slower (sequence-aware batching) | High | Cap `max_seq_length` at 256; use GRU (R3) as faster alternative |
| Hidden state explosion on long episodes | Medium | LayerNorm on state; clip h,c to [-10, 10]; monitor state norm |
| Task embedding doesn't condition state well | Low-Medium | R2b/R2c/R2d isolate this; fallback: concat task to input (R2a) |
| RLlib recurrent API quirks (version-specific) | Medium | Pin Ray version; test `get_initial_state` + state reset explicitly |
| OOM with recurrent models (state tensors in batch) | Medium | Same llama.cpp guard; reduce batch size for R2-R4 if needed |
| Task embedding overfitting on small task pools | Low-Medium | Weight decay on FiLM layers; freeze embeddings after warmup |
| Timescale separation doesn't emerge (LSTM acts like bag-of-recent-bars) | Medium | Check timescale separation score metric; if absent, try GRU or add explicit `t/T` feature as fallback |

## Success Criteria

R2c (true recurrent LSTM + per-step FiLM) "wins" if:
1. Held-out return on horizon≥128 tasks exceeds R0 by >15% (compounding effect visible).
2. Drawdown on long episodes is lower than R0 (memory helps risk management).
3. State norm trajectory is stable (no explosion/collapse over 500+ steps).
4. Capability report shows improvement concentrated in long-horizon + high-vol regimes (where memory matters most).
5. **Timescale separation score** confirms fast components correlate with short-horizon performance, slow components with long-horizon performance.
6. **Episode progress tracking**: solver performance degrades less in the latter half of episodes vs R0 (it "knows" it's near the end and starts reducing position).

If R2c ≈ R1 (window encoder), recurrence isn't helping beyond window encoding → the task structure doesn't need cross-step memory, and we save compute by staying on R1.

If R2b ≈ R2c, per-step FiLM is unnecessary → simpler architecture wins (init-only task conditioning).

If R2d > R2c, the value function is the bottleneck → invest in value-head task conditioning, not policy-head.

## The Overarching Goal: What This Enables For iTrader

This isn't just a better model. It's a different *kind* of trading agent:

**Current (FiLM MLP):** A reactive pattern-matcher. "Given these 30 bars, what's the best action?" No memory, no plan, no sense of progress. Every step is independent.

**Target (Recurrent FiLM LSTM):** A deliberative trader with a working model of the episode. "I'm 60% through my holding period, vol has been calm, my position is +1.2%, and I'm solving a mean-reversion task — I should hold and trail my stop." The hidden state *is* the trader's mental model of where they are in the trading process.

This is the RLVR connection: the proposer generates tasks with explicit success criteria (target Sharpe, max drawdown, horizon). The recurrent solver doesn't just optimize reward — it *tracks progress toward the task's definition of success*. The task embedding tells it what "success" looks like; the recurrence lets it track how close it is. That's verification-aware trading, and it's something no feed-forward architecture can do.
