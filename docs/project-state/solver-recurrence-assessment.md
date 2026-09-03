# Solver Recurrence Assessment

Last updated: 2026-09-03

## Short Answer

The repo currently has an LSTM model file, but the active training path defaults to `SolverMLPFiLM`.

The existing `SolverLSTMModel` is best described as a rolling-window sequence encoder:

- Input is a flattened observation window.
- The model reshapes it into `(batch, window, features)`.
- An LSTM summarizes that local window.
- The final hidden vector feeds policy and value heads.
- The recurrent hidden state is not carried across environment steps.

That is not the same as a stateful RLlib recurrent policy.

## Compatibility Warning

The existing `SolverLSTMModel` also appears stale relative to the current task-conditioned observation layout.

Current observations are shaped like:

- `price_window_flat = window_size * n_features`
- `task_embed = 32`
- full observation = `[price_window_flat, task_embed]`

The old LSTM infers `n_features = flat_dim // window_size` from the entire observation. With the common `window_size = 30`, the full observation is `30 * 5 + 32 = 182`, which does not reshape cleanly into `(30, features)`.

So before using an LSTM head again, implement a corrected model that explicitly splits price features from task embedding.

## Why This Matters

For market learning, these are different capabilities:

- `SolverMLPFiLM`: learns patterns from the flattened window and task embedding through feed-forward conditioning.
- Window-encoder LSTM: learns temporal structure inside each observed market window, but starts fresh every decision step.
- Stateful recurrent policy: carries hidden state through the episode, allowing the policy to remember information beyond the latest observation window.

The current LSTM can still learn useful within-window motifs, but it does not provide persistent episode memory. If the market signal requires remembering a path-dependent setup, hidden regime, delayed confirmation, or task-specific behavior over many steps, a true recurrent policy or attention-style sequence head is worth testing.

## Input Shape

There is likely benefit in making the model interface more native to time series:

- keep the environment observation stable for RLlib compatibility,
- let model code split `[price_window_flat, task_embed]` explicitly,
- reshape only the price slice into `(window, features)`,
- inject task embedding through FiLM, concatenation, or initial-state conditioning.

This is cleaner than inferring `n_features = flat_dim // window`, especially now that task embeddings are appended to observations.

## Candidate Head Families

### 1. FiLM MLP Baseline

Keep as the control model.

Use for:

- current solver baseline,
- proposer impact measurement,
- fast PBT and smoke tests.

### 2. Time-Series Window Encoder

Add a corrected `SolverWindowLSTMFiLM` or `SolverWindowTransformerFiLM`.

Use for:

- testing whether native temporal encoding of each market window improves solver quality,
- preserving simple rollout semantics,
- avoiding recurrent hidden-state complexity at first.

Key design:

- price window -> LSTM/GRU/TCN/Transformer encoder,
- task embedding -> FiLM or cross conditioning,
- final encoded vector -> policy and value heads.

### 3. Stateful Recurrent Policy

Add only after the window encoder baseline is measured.

Use for:

- online mode,
- partially observable market state,
- path-dependent task families,
- longer action/reward dependencies.

Key requirements:

- implement RLlib recurrent state correctly,
- define `get_initial_state`,
- return state outputs,
- validate hidden-state reset at episode boundaries,
- tune sequence length/minibatching separately from the feed-forward PPO baseline.

## Relation To Gradient Alignment

Episode-aligned optimization matters much more for a true recurrent policy than for the current baseline.

For the current PPO setup:

- `batch_mode = "complete_episodes"` already keeps rollouts episode-complete,
- PPO still mixes complete episodes inside training batches,
- that mixing is normal and often stabilizing.

For a true recurrent policy:

- sequence boundaries become a first-class hyperparameter,
- minibatches should preserve valid sequences,
- `max_seq_len` and hidden-state reset behavior matter,
- shuffling must not break temporal chunks.

## Recommended Next Experiment

Do not jump straight from the FiLM MLP to a fully stateful recurrent solver.

Recommended order:

1. Keep `SolverMLPFiLM` as the baseline.
2. Implement a corrected window encoder head that explicitly splits price features and task embedding.
3. Compare on identical task pools and held-out rolling windows.
4. Only then implement a true RLlib recurrent policy if the window encoder improves or exposes failure modes that need longer memory.

## 2026-09-03 Update: True Recurrence Is The Priority

The core value of an RNN solver is **persistent hidden state across environment steps** — not just within-window encoding. A window-encoder LSTM that resets every step forfeits the compounding benefit: reward-function bias, position awareness, and regime memory should carry forward through the episode.

### What "truly recurrent" means for iTrader

- Hidden state `(h_t, c_t)` persists from step `t` to step `t+1` within an episode.
- At each step, the model consumes the **latest observation slice** (or a short recent window), not the full flat window reshaped into pseudo-sequences.
- Episode boundary → hidden state reset (via RLlib's `get_initial_state` / `done` masking).
- Minibatching must preserve temporal chunks: PPO with `shuffle_sequences=False` or sequence-aware batching so gradients flow through valid recurrent paths.
- The task embedding conditions the **initial state** and/or is injected via FiLM at each step, rather than being baked into a flat vector that breaks reshape assumptions.

### Why this matters for trading specifically

- **Path-dependent drawdown awareness**: A recurrent solver can "remember" that it's been losing for 12 consecutive steps and modulate risk without needing an explicit drawdown feature in the observation.
- **Regime persistence**: If vol has been elevated for 50 bars, the hidden state encodes that regime context even if the latest 3-bar window looks calm.
- **Position carry-over memory**: The solver can implicitly track its own recent action sequence (buying pressure, entry timing) without a separate position-history feature.
- **Compounding reward bias**: PPO's advantage estimation + recurrent state means the value function accumulates knowledge of "what happens if I keep doing this" across the episode, not just per-window.

### Implementation path (refined)

1. **Fix obs-shape split** (prerequisite): Model receives `obs = [price_window_flat(150), task_embed(32)]`. Explicitly slice: `price = obs[:, :150]`, `task = obs[:, 150:]`. No more `flat_dim // win` inference.
2. **True recurrent step**: At each env step, feed only the **new bar** (or last K bars) into the LSTM. Hidden state carries forward. This is RLlib's native recurrent policy interface (`state_inputs`, `seq_lens`, `RNN` model type).
3. **Task conditioning via FiLM on initial state + per-step**: Task embedding modulates the LSTM's initial hidden state AND is FiLM-injected at each step so the network knows which task it's solving throughout the episode.
4. **Sequence-aware PPO config**: `shuffle_sequences=False`, appropriate `max_seq_length`, `recurrent_policy=True` in RLlib config. Minibatch size must divide sequence length cleanly.
5. **Evaluation**: Same held-out rolling windows as the alignment ablation, plus a new metric — **memory utilization** (does performance on long-horizon tasks improve more than short-horizon ones?).

### Ablation design for recurrence (to run after alignment ablation)

| Variant | Head | State | Purpose |
|---------|------|-------|---------|
| R0 | FiLM MLP | None | Baseline (control) |
| R1 | Window LSTM (encoder) | Reset every step | Within-window temporal bias only |
| R2 | True recurrent LSTM | Persists across steps, reset at episode end | Full recurrence |
| R3 | True recurrent GRU | Persists across steps, reset at episode end | Compare LSTM vs GRU (fewer params, faster) |
| R4 | Recurrent + task-FiLM on state | Persists, task-conditioned init | Isolate task conditioning benefit |

All variants: same task pool, same dataset slice, same train budget, same seed set. Evaluation: held-out rolling windows + capability report by family/regime/horizon.

## Evaluation Standard

Judge by:

- held-out episode return,
- positive-PnL fraction by evaluation slice,
- drawdown and turnover,
- entropy trajectory,
- performance by task family/regime/horizon,
- calibration of `CapabilityProfile`.

Do not judge by training reward alone.
