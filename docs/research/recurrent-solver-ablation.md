# Recurrent Solver + FiLM Ablation

Last updated: 2026-09-03
Status: **Design corrected; implementation pending behind matched baselines**

## Executive Position

A stateful recurrent solver is a compelling iTrader frontier because the
current `SolverMLPFiLM` can only infer temporal structure from the rolling
window in each observation. Persistent state could encode information beyond
that window and could support episode-aware behavior.

That value is a hypothesis, not an implemented capability. The branch currently
contains:

- `SolverMLPFiLM`, the working default;
- `SolverLSTMModel`, a legacy reset-every-step window encoder that is incompatible
  with the appended task embedding because it infers feature count from the
  entire observation width;
- no `SolverWindowLSTMFiLM` implementation;
- no RLlib stateful recurrent FiLM policy.

The research program must distinguish three effects: temporal structure inside
the supplied window, persistent state across environment steps, and FiLM task
conditioning. Mixing them in one first implementation would make any gain hard
to interpret.

## The Core Proposition

The environment emits a rolling market window and a static task embedding:

```text
observation_t = [window_t(T x F), task_embedding]
```

FiLM gives the task embedding a clear role as a condition on market features.
A stateful LSTM or GRU adds a separate memory path across episode steps. The
combination could learn task-dependent memory horizons: a breakout task may
retain volatility expansion context, while a mean-reversion task may emphasize
shorter deviations.

Possible benefits include:

- context beyond the fixed observation window;
- regime-transition memory;
- task-dependent filtering of incoming market observations;
- path-dependent value estimates;
- improved behavior on longer horizons.

These outcomes are not automatic. Recurrent state can be ignored, become
unstable, memorize episode locations, or add optimization cost without held-out
benefit. Measurements must decide.

## Critical Contract Discovery

The current observation contains market features and a task embedding, but not
explicit position, previous action, realized reward, unrealized PnL, or episode
progress. A recurrent model receiving only the current contract cannot reliably
act as a complete trade or plan tracker merely because it has hidden state.

Therefore separate these questions:

1. Does market-only recurrence help beyond the rolling window?
2. Do explicit solver-state features such as `position_t` and normalized
   `episode_progress` add value?

Do not add those features to the recurrent variant alone in the first head
ablation. That would confound architecture with observation content. A later
matched feature ablation can add them to both feed-forward and recurrent heads.

## Correct Input Semantics

The environment remains the owner of rolling-window assembly. A recurrent head
must split market and task dimensions explicitly:

```python
price_dim = window_size * n_features
price_flat = obs[..., :price_dim]
task_embed = obs[..., price_dim:]
price_window = price_flat.reshape(*price_flat.shape[:-1], window_size, n_features)
newest_bar = price_window[..., -1, :]
```

The newest bar is the last row because `ITraderEnv` builds the buffer in
chronological order. Selecting `obs[..., :n_features]` would feed the oldest bar
in the window, not the newest.

## Architecture Matrix

| ID | Head | Cross-step state | Purpose |
|---|---|---|---|
| R0 | `SolverMLPFiLM` | None | Current control |
| R1 | corrected window-LSTM + FiLM | None | Test temporal inductive bias inside the existing window |
| R2a | true recurrent LSTM + task concat | Persistent | Establish recurrence baseline |
| R2b | true recurrent LSTM + per-step FiLM | Persistent | Test repeated task conditioning |
| R3 | true recurrent GRU + per-step FiLM | Persistent | Compare lower-cost recurrent cell |
| R4 | winning head plus explicit solver-state features | Persistent | Separate market memory from trade-state observability |

R1 must explicitly split price and task dimensions. R2 and R3 should consume
the newest bar while retaining the existing environment observation contract;
this keeps rollout and evaluation infrastructure stable.

## FiLM Placement

Two task-conditioning mechanisms are worth comparing:

- **concat**: append the task embedding to the recurrent input each step;
- **per-step FiLM**: transform recurrent features with task-derived scale and
  shift before the policy and value heads.

Task-conditioned initial state is not part of the first implementation. Under
RLlib's legacy ModelV2 API, `get_initial_state()` does not receive the current
task observation. Adding observation-dependent initialization requires explicit
sequence-boundary handling and should only be attempted after per-step FiLM is
working.

Identity-centered scale is the conservative FiLM form:

```text
conditioned_h = h * (1 + scale(task)) + shift(task)
```

The implementation must test zero-task behavior, output shapes, and finite
state transitions rather than assume this parameterization is stable.

## RLlib 2.9 Integration Contract

The pinned repository stack uses Ray/RLlib 2.9 and the legacy ModelV2 path. A
true recurrent implementation should use RLlib's recurrent model contract,
including `get_initial_state`, sequence lengths, and `forward_rnn`-style batched
sequence processing. It must not detect episode starts by checking whether a
state tensor happens to sum to zero; valid learned states may be zero and batch
members may be at different boundaries.

RLlib owns state resets at episode boundaries. PPO uses truncated
backpropagation through time according to sequence chopping and
`max_seq_len`; gradients do not automatically flow through the entire episode.
`batch_mode="complete_episodes"` controls collection, not unlimited BPTT.

Configuration questions to test explicitly:

- `max_seq_len` and sequence padding;
- sequence-preserving minibatches and shuffling behavior;
- hidden-state shape under vectorized workers;
- terminal reset and truncated-sequence carry state;
- batch size and GPU memory at the pinned dependency versions.

Avoid undocumented flags such as `recurrent_policy=True`; model behavior and
RLlib's native recurrent interface should determine recurrence.

## TDD Implementation Slices

### Slice R1: Corrected Window Encoder

1. Add failing shape tests for `[T*F + task_dim]` observations.
2. Implement explicit market/task splitting.
3. Apply FiLM to the encoded window before separate policy/value heads.
4. Add a model selector to `train_solo_V2` without changing its default.
5. Run forward, backward, and 10-step PPO smoke tests.

### Slice R2: Stateful Recurrent Baseline

1. Add failing tests for RLlib initial state, two-step state carry, batch/sequence
   shapes, and terminal reset through an integration rollout.
2. Implement the smallest LSTM recurrent model using the newest market bar.
3. Keep task conditioning as concat for R2a.
4. Add per-step FiLM as R2b only after R2a is green.
5. Record parameter count, steps/second, GPU memory, and state norms.

### Slice R4: Solver-State Observability

1. Define typed observation fields for position and normalized episode progress.
2. Add the same fields to feed-forward and recurrent comparison cells.
3. Evaluate whether recurrence still adds value once state is observable.

## Experiment Order

1. Run the episode/regime alignment ablation on R0.
2. Run `pnl_linear`, `sharpe_only`, and `pnl_sharpe` on R0 using matched
   horizon cells.
3. Implement and compare R0 versus R1 under the winning data/reward setup.
4. Compare R1 versus R2a to isolate persistent memory.
5. Compare R2a versus R2b to isolate per-step FiLM.
6. Compare LSTM versus GRU only if persistent recurrence wins.
7. Add explicit solver-state features as a separate matched experiment.

This staged design costs more wall-clock time than jumping directly to R2b, but
it tells us which mechanism produced the result.

## Evaluation

Every comparison uses the same task pool, data split, seeds, PPO timestep
budget, reward, and observation contract. Report:

- held-out return, Sharpe, max drawdown, and turnover;
- action distribution and entropy;
- performance by horizon and regime;
- training steps/second and GPU memory;
- recurrent state norm and saturation;
- long-horizon uplift relative to short-horizon uplift;
- bootstrap confidence intervals across seeds.

Useful probes:

- **state ablation**: zero recurrent state during evaluation and measure loss;
- **history permutation**: preserve current window but scramble earlier episode
  order;
- **regime shift**: inject a controlled transition and measure adaptation;
- **task swap**: hold market sequence fixed and change only task embedding.

A hidden-state timescale analysis is exploratory. Channel variance alone does
not prove semantic fast/slow decomposition; probes must connect state changes to
held-out behavior.

## Promotion Gates

R1 advances only if it improves held-out behavior or sample efficiency enough
to justify its compute cost. R2 advances only if it beats R1, proving value from
persistent state rather than merely LSTM window encoding. R2b advances only if
it beats R2a, proving value from FiLM rather than recurrence alone.

No recurrent head becomes the default until:

- tests cover state lifecycle and observation splitting;
- at least three matched seeds complete;
- no NaN, action collapse, or hidden-state explosion occurs;
- gains survive held-out rolling windows and regime slices;
- throughput and memory costs are recorded;
- the default remains selectable and checkpoint compatibility is documented.

## Relationship To Task Families

Family `00` is the first experimental surface because it is implemented and
provides a stable three-action control path. Families `01` and `02` retain the
same action dimension in their current contracts, making them plausible later
users of a shared recurrent trunk. Family `03` is trade management, not a
generic trend-following family; it requires its own state and action-contract
review before sharing a head.

The long-term multi-head shape remains:

```text
shared market encoder / recurrent trunk
    -> family-routed policy head
    -> family-routed value head
```

Only one family head is routed for a given `TaskSpec`. A mixed batch may contain
multiple families, so routing must be vectorized per sample and metrics must
remain family-sliced. This remains scaffolding until families `01` and `02` are
operational.
