# Recurrent Solver + FiLM Ablation

Last updated: 2026-09-12
Status: **design contract; implementation pending; no recurrent result**

This document is the active R-series provenance and experiment proposal. It
preserves the original identities from `git show 4732a98:docs/research/recurrent-solver-ablation.md`;
it is not an accepted architecture contract. Names are design identities, not
executable flags.

## Executive position

The active branch has one implemented/default solver: `SolverMLPFiLM`. The
checked-in `SolverLSTMModel` is a legacy reset-every-forward window encoder that
reshapes the full observation and discards returned LSTM state. Because the
current observation appends a task embedding, its `flat_dim // window` inference
is incompatible with the normal `[T*F | task_dim]` layout. It is not a valid
recurrent baseline.

The research question has four separable parts:

1. Does temporal structure inside the supplied market window help?
2. Does carried `(h,c)` state help under matched input and conditioning?
3. Does task conditioning help, and at which placement?
4. Do explicit solver-state fields add information beyond OHLCV and task data?

No answer is implied before matched implementation and held-out evidence.

## Current observation contract

`ITraderEnv` owns the chronological rolling market buffer. The model-facing
layout is:

```text
obs_t = [price_window_flat(T * F) | task_embedding(D)]
price_flat = obs[..., :T*F]
task_embed = obs[..., T*F:]
price_window = reshape(price_flat, ..., T, F)
newest_bar = price_window[..., -1, :]
```

Selecting the first `F` values would select the oldest bar, not the newest. A
model must not infer `F` from the full observation width. A same-width tensor is
not evidence of the same information: record feature order, window length, task
fields, timing, warmup, and state ownership.

The current `SolverMLPFiLM` source uses:

```text
hidden = MLP(price_window_flat)
(gamma_raw, beta) = task_projection(task_embed)
gamma = tanh(gamma_raw)
conditioned_hidden = gamma * hidden + beta
```

Task embedding dropout is applied during training when configured. A
`conditioning` value other than `film` returns the unconditioned hidden state;
that path does not implement task concatenation. The identity-centred form
`(1 + scale) * hidden + shift` is a possible future factor, not current R0.

## R-series matrix and provenance

| ID | Original identity | Current status | Attribution |
|---|---|---|---|
| R0 | `SolverMLPFiLM`, no cross-step state | **Implemented/default** | Control |
| R1 | Corrected reset-every-step window LSTM + FiLM | **Unimplemented** | Within-window temporal inductive bias |
| R2a | True recurrent LSTM + task concat | **Unimplemented** | Original stateful recurrence baseline |
| R2b | True recurrent LSTM + FiLM initial state only | **Deferred** | Current initial-state hook has no task input |
| R2c | True recurrent LSTM + FiLM per step | **Active candidate; unimplemented** | Original per-step identity; placement/order remains open pending grill-me review |
| R2d | Init + per-step FiLM on value head only | **Deferred** | Requires matched policy/value placement |
| R3 | True recurrent GRU + per-step FiLM | **Planned later** | Cell type/compute comparison |
| R4-v1 | Winning head plus explicit solver-state features on matched FF/recurrent cells | **Planned, versioned** | Observability study, not historical R4 |

Migration note: `bec5c77` simplified the matrix and called per-step FiLM R2b;
that label maps to original R2c. An older init-oriented R4 maps conceptually
to original R2b; `bec5c77` instead used R4 for matched observation features.
The current matched solver-state study is retained as `R4-v1`. Do not silently
reuse an historical ID for a different factor.

## Causal controls

### R0 -> R1

This asks whether an explicit sequence encoder over the already supplied window
helps a feed-forward control. It does not test persistent memory.

### R1 -> newest-bar R2

This is an engineering progression, not an isolated memory proof. R1 sees `T`
market bars immediately at every call and resets. A newest-bar stateful model
starts with a cold hidden state (unless a causal warmup is specified) and sees
one bar per recurrent step. The comparison changes:

- accessible input history;
- cold-start/warmup information;
- task-conditioning placement;
- state lifetime and training sequence semantics.

State this compound difference in every result.

### Same-input/same-conditioning carry/reset control

Train from scratch with identical data, task pool, feature timing, optimizer,
seeds, budget, and model capacity:

- **`R2c-CARRY`** — newest-bar input, the selected per-step FiLM placement,
  carried `(h,c)` within real episodes;
- **`R2c-RESET`** — exactly the same input and conditioning, but reset `(h,c)`
  at every scored call.

The suffixes are control labels, not new R-series IDs. This pair is the minimum
carry-state attribution control. If the explicit pre-cell placement is studied,
use the same suffix pair for that placement rather than comparing two factors at
once.

A **zero-state evaluation probe** is separate: it resets a trained carry model's
state at evaluation. It is an out-of-distribution intervention because training
saw carried states. A performance change is a diagnostic of state sensitivity,
not a causal estimate on its own. Also report state permutation/history probes
with their distribution-shift caveat.

## FiLM placement vocabulary

| Placement | Task reaches | Does it change transition? | State returned |
|---|---|---:|---|
| R2a input concat | Cell input at each step | Yes, as input | Raw `(h,c)` |
| `R2c-precell` named factor | New-bar projection or cell/gate input | Yes, candidate | Raw `(h,c)` |
| R2c post-cell/readout (preferred first candidate) | Policy/value readout after cell | No | Raw `(h,c)` |
| R2b init-only | Initial `(h,c)` | Indirectly | Carried `(h,c)` |
| R2d value-only | Value readout placement by design | No policy transition claim | Matched raw state |

For any placement, state the complete task path before attributing causality. A
post-cell/readout FiLM path is only readout-only when no other task path (such as
concat) reaches the cell. If concat reaches the cell, the transition is already
task-conditioned and FiLM is an additional factor. Readout-first is a proposed
comparison order, not a settled memory-conditioning decision.

R2b remains deferred because RLlib's current `get_initial_state()` contract does
not receive the current task observation. Do not claim initial-state conditioning
just because an initial state exists.

## Time and state lifecycle

Keep three windows distinct:

1. **Input window** — bars included in one environment observation.
2. **BPTT/unroll (`max_seq_len`)** — training sequence/chunk span; detach at a
   chunk boundary does not reset environment state.
3. **Episode horizon** — task/environment episode length and termination scope.
4. **Reward-statistics window** — history used by a statistic such as Sharpe; it is
   distinct from the episode horizon and BPTT span.

A newest-bar model may use:

- **cold start**: zero initial state, then process the first available bar;
- **causal initial-window warmup**: process prior bars sequentially before the
  first scored action, with no future bars and no duplicate chronological steps.

Do not present warmup as extra prediction data, and do not count repeated
overlapping observation bars as new environment timesteps.

The current source lock is Ray/RLlib 2.58.0, Gymnasium 1.2.2, Python 3.11.
The former Ray 2.9 and Phase B Ray 2.47.1 lines are historical and not the active
API authority. R0 explicitly remains on the legacy API stack. Do not invent
`recurrent_policy=True`, `shuffle_sequences=False`, or other flags without
verifying the pinned version's native API after the RLModule/EnvRunner parity
gate.

Before any recurrent result, specify and test:

- native initial-state and recurrent-forward hooks;
- `(h,c)` ordering, shapes, dtype, and device;
- `seq_lens`/valid lengths and sequence packing;
- padding masks for policy and value losses;
- preservation of the last valid **h and c**;
- reset state when the environment actually resets at real episode termination;
- do not treat a truncated bootstrap difference as an episode reset;
- distinguish fragment detach/truncated BPTT from a real environment reset;
- vector-environment state ownership;
- checkpoint save/restore and replay state ownership;
- identical state timing and feature preprocessing in training and inference.

`batch_mode="complete_episodes"` is a collection boundary. It is not proof of
stateful rollout handling or full-episode BPTT.

## Task semantics and explicit solver state

The current default task encoder represents family, price field, reward ID,
horizon, dimensions, and a bounded seed feature. It omits reward parameters, bar
resolution, regime tags, difficulty, metadata, and start index. Reward overrides
are merged after embedding. For initial matched cells, freeze omitted fields and
report the seed shortcut, or introduce a separately versioned semantic encoder.
Do not treat the default embedding as complete task semantics.

The policy observation contains no position, previous action, realized reward,
unrealized PnL, or episode progress. OHLCV-only recurrence cannot reconstruct the
agent's sampled actions, fills, position, or reward history exactly. `R4-v1`
must define field order, normalization, timing, and terminal behavior, then add
the same explicit fields to the feed-forward control and the recurrent cell.

## Test and evaluation matrix

All matched cells use the same task pool, data eligibility, chronological splits,
feature/encoder version, reward, seeds, PPO budget, and evaluation protocol.
Record:

- held-out return, risk-adjusted metrics, drawdown, turnover, and costs;
- action distribution, entropy, and finite-loss checks;
- slices by horizon and regime only when metadata is emitted;
- steps/second, memory, parameter count, and state norms;
- valid-length/last-state and reset/truncation checks;
- checkpoint/replay state ownership;
- a causal attribution statement for every comparison.

Do not call PPO logits/value outputs calibrated confidence or expected net utility
without a separate calibration/economic-evidence procedure. A training curve is
not held-out evidence.

## AZR relationship (separate study)

The local AZR v2 source is summarized in
`docs/artifacts/research/azr-stack-review-2026-09-12.md`. Eq. 4 gives proposer
reward `0` for observed `p=0` or `p=1`, otherwise `1-p`; it is asymmetric and not
a symmetric 50% peak or measured pre/post learning progress. Table 2 values are
AZR coding/math results, not iTrader trading results. Historical OHLCV means this
project is not zero-data AZR.

`train_AZR.py` is a bootstrap task-pool generation plus solver handoff. It is not
a co-adaptive loop. Keep a separate proposer study:

1. fixed rule/random validated curriculum control;
2. frozen LLM proposer;
3. frozen LLM with solver-feedback/reference-conditioned selection;
4. adaptive proposer only after repeatable utility/uplift evidence.

Make frozen versus adaptive reference-buffer updates a named factor. Adaptation
feedback is validation. Keep an untouched final chronological test unavailable to
proposer updates, tuning, and head selection. Use deterministic, cost-aware
verification and report accepted timesteps, rejected tasks, generation calls,
tokens/latency, total compute, costs, turnover, and drawdown. Do not substitute
this study for R-series solver attribution.

## Experiment order and promotion gates (proposal)

The following order is an experiment proposal pending the user's grill-me review:
1. R0 alignment/reward controls.
2. R1 explicit split and window encoder; correct and measurable is sufficient to
   proceed, even if it does not win.
3. R2a native recurrence lifecycle.
4. R2c readout FiLM and `R2c-CARRY`/`R2c-RESET`.
5. Optional `R2c-precell`, R2b, R2d, and R3, each with named matched controls.
6. R4-v1 explicit state features matched across FF/recurrent controls.
7. Run the proposer/reference-buffer study as a separate experimental axis.
   Phase B also includes online/replay/shadow work; its infrastructure and
   validation can proceed alongside the offline track. Any deployment gate
   is separate from this proposed recurrence experiment order.

No recurrent head becomes default until lifecycle tests, matched train-from-
scratch seeds, held-out chronological evaluation, state/compute evidence, and
checkpoint/replay checks pass. R0 remains the operational control throughout.
