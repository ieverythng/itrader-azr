# Solver-state Markov completion for iTrader

**Date:** 2026-09-25  
**Scope:** Research note only; no implementation or active architecture changes  
**Question:** Which agent-owned variables must accompany the market window and
`TaskSpec` embedding when actions are target positions and the optimization reward
charges turnover plus terminal liquidation?

## Executive conclusion

For the current `net_return` environment, the minimal known internal-state
completion is:

```text
solver_state_v1 = [current_position, remaining_horizon_fraction]
```

The **current effective cost rate** must also be observable whenever it varies
between tasks or episodes, but it is task/reward context rather than mutable
portfolio state. In the present tree, `semantic_v2` includes normalized
`reward_params` in the `TaskSpec` embedding, so a fixed exact-1-bp matched test does
not need another cost field. A later cost-generalization study should expose a
smooth, explicitly normalized current-cost scalar rather than the global training
schedule.

`previous_action`, holding duration, entry price/unrealized PnL, and cumulative
turnover are not minimally required by the present exact-net-return dynamics.
They become necessary only if a future transition, reward, constraint, or fill
model depends on them. This result completes only the **known agent/episode-state
aliasing**. It does not prove that a finite OHLCV window is a Markov representation
of the market.

The clean experiment is the already reserved **R4-v1 observation factor**. Preserve
R0/R1/R2 identities. First compare R0 with and without the two-field state block
under the accepted Ray 2.58, fixed exact-1-bp contract; then cross the identical
block into the frozen recurrent control. Do not rename the recurrent series or mix
the state change with recurrence, cost curriculum, or framework migration.

## 1. Established facts

### 1.1 State sufficiency

An MDP state must retain the history information needed to determine the
conditional distribution of the next state and reward from the current state and
action. Sutton and Barto describe a Markov state as a compact summary that retains
all information relevant to the future and formalize the equality between the
history-conditioned and state-conditioned next-state/reward distributions
([*Reinforcement Learning: An Introduction*, Chapter 3](http://incompleteideas.net/book/RLbook2020.pdf)).
When this information is hidden, the agent faces a POMDP; a belief state can be a
sufficient statistic of the history, but it does not make an omitted, directly
available environment variable unnecessary
([Kaelbling, Littman, and Cassandra, 1998](https://www.cassandra.org/arc/papers/aij98.pdf)).

This criterion is about the process presented to the learner, not about whether a
neural network could sometimes infer the missing value.

### 1.2 Current iTrader transition and reward

The current working tree establishes the following:

- The observation is `price_window_flat + task_embedding`; it contains no mutable
  portfolio or episode-progress block
  ([`itrader/envs/itrader_env.py`](../../../itrader/envs/itrader_env.py), lines
  193–195 and 214–252).
- `Discrete(3)` maps to target positions `{-1, 0, +1}`. A step saves the previous
  position, overwrites position with the target, and computes
  `turnover_t = abs(target_position - previous_position)`
  ([`itrader/envs/itrader_env.py`](../../../itrader/envs/itrader_env.py), lines
  513–532).
- Reset sets position to flat and resets episode statistics, including elapsed
  step, cumulative PnL, and turnover
  ([`itrader/envs/itrader_env.py`](../../../itrader/envs/itrader_env.py), lines
  266–307).
- Episode termination depends on elapsed step versus the task horizon. The
  `net_return` reward charges ordinary turnover each step and charges
  `abs(position)` only on the terminal step
  ([`itrader/envs/itrader_env.py`](../../../itrader/envs/itrader_env.py), lines
  394–396 and 513–599;
  [`itrader/rl_lib/tasks/rewards.py`](../../../itrader/rl_lib/tasks/rewards.py),
  lines 267–293).
- `semantic_v2` embeds the task horizon, reward identifier, and normalized reward
  parameters; `legacy_v1` includes horizon but not reward parameters
  ([`itrader/rl_lib/tasks/encoder.py`](../../../itrader/rl_lib/tasks/encoder.py),
  lines 26–76). New training configuration selects `semantic_v2`
  ([`itrader/rl_lib/train/train_solo_V2.py`](../../../itrader/rl_lib/train/train_solo_V2.py),
  line 532).
- A staged optimization cost is advanced by the trainer and installed as a task
  sampler override for subsequent resets, not changed inside an active episode
  ([`itrader/rl_lib/train/train_solo_V2.py`](../../../itrader/rl_lib/train/train_solo_V2.py),
  lines 1684–1710).

The position omission therefore produces a direct state alias. The same market
window and task can be reached while short, flat, or long; the immediate cost of a
proposed target is respectively different. The elapsed-time omission is another
alias: the same market/position/task observation can occur on an ordinary step or
the final step, when liquidation is additionally charged.

### 1.3 Trading-environment precedent

This is not merely a software convention. Jiang, Xu, and Liang derive a
transaction-cost portfolio environment in which the preceding portfolio weights
influence current redistribution costs. They define the state as the pair of a
price tensor and the prior portfolio vector, and feed the prior portfolio vector
to all three of their CNN/RNN/LSTM policies
([paper, especially equations 19–20 and Sections 4.1/5.1](https://arxiv.org/abs/1706.10059)).
FinRL likewise includes cash and current holdings with prices and indicators in
its trading state; its maintained environment source carries the existing share
inventory through state updates
([FinRL environment source](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_stock_trading/env_stocktrading.py)).

These systems differ from iTrader, but they support the underlying principle:
when costs or transitions depend on inventory, inventory is environment state,
not merely a market feature.

### 1.4 Finite-horizon time awareness

iTrader's task horizon is part of the economic objective because it triggers
liquidation; it is not just a sampler-side training truncation. Pardo et al. show
that a finite-horizon environment needs a notion of remaining time in the agent's
state. Otherwise states that differ only in their proximity to termination are
aliased and the agent sees a POMDP
([Pardo et al., *Time Limits in Reinforcement Learning*, Section 2](https://proceedings.mlr.press/v80/pardo18a.html)).

Gymnasium distinguishes termination from truncation because bootstrapping depends
on why an episode ended, and requires `reset()` after either condition
([official `Env` API](https://gymnasium.farama.org/main/api/env/)). In the current
iTrader contract the horizon ends the task, terminal liquidation is part of its
reward, and the environment returns it as termination. That supports resetting
both explicit solver state and recurrent state there.

### 1.5 RLlib recurrent state does not remove the need for environment truth

Ray 2.58's old-stack recurrent interface supplies `state`, `seq_lens`, and a
model-defined `get_initial_state()`; its stock LSTM can optionally add previous
actions or rewards, but those are opt-in view requirements
([Ray 2.58 `recurrent_net.py`](https://github.com/ray-project/ray/blob/ray-2.58.0/rllib/models/torch/recurrent_net.py)).
Its sequence builder cuts on episode/unroll identities, preserves an initial
state per sequence, and pads to sequence lengths
([Ray 2.58 `rnn_sequencing.py`](https://github.com/ray-project/ray/blob/ray-2.58.0/rllib/policy/rnn_sequencing.py)).
RLlib's current first-party explanation states the same data contract: recurrent
outputs become the next state inputs, and sequence padding occurs at episode ends
([official stateful-module connector documentation](https://docs.ray.io/en/latest/rllib/connector-v2.html)).

An RNN that receives only observations cannot know a stochastically sampled action
unless that action or its resulting environment state is fed back. More
importantly, future fill models can make an intended action differ from actual
position. Therefore current position should come from `ITraderEnv` as environment
truth. It should not be reconstructed from an RNN's memory.

## 2. Variable-by-variable determination

| Candidate | Minimal for current exact `net_return`? | Determination |
|---|---:|---|
| **Current position** | **Yes** | Immediate turnover is `|target-current|`; terminal liquidation also depends on position. Encode actual current position, not an inferred prior intent. |
| **Previous action** | No, redundant today | With deterministic target-position execution, the previous action maps exactly to current position. It becomes distinct only with rejected/partial fills, execution lag, or action semantics other than target inventory. |
| **Remaining horizon / progress** | **Yes** | The terminal transition and liquidation charge depend on `_t`. Horizon alone in `TaskSpec` does not reveal elapsed time. |
| **Current effective cost rate** | Conditional: yes when variable | A constant 1 bp rate is an environment parameter and need not consume an observation dimension. If policies span rates, expose the current episode's rate through task context. The current global schedule itself is not state. |
| **Cost schedule / global sampled step** | No | Current code changes the effective task parameter only between episodes. Exposing future schedule or PBT progress would couple the policy to optimizer orchestration and lineage. |
| **Holding duration** | No | No current transition, reward, constraint, or liquidation rule reads it. Add only with minimum-hold rules, time-dependent borrow/funding, or a deliberate duration feature study. |
| **Entry price / unrealized PnL** | No | The simulator has no lot ledger; target-position mark-to-market reward depends on the next return, not entry price. Adding an ill-defined entry price would silently introduce execution semantics. |
| **Cumulative turnover** | No | Turnover costs are charged incrementally; the cumulative value is used for evidence, not future transition or optimization reward. It becomes state if a future turnover budget or tiered fee changes allowed actions/rewards. |
| **Cumulative PnL/equity/drawdown** | Not for exact `net_return` | They are not inputs to this reward's future calculation. Other registered rewards that depend on drawdown require their own state-sufficiency audit. |
| **Running Sharpe** | No | Sharpe is an evaluation statistic here. Full-episode Sharpe would leak the future; a causal running estimator is warranted only if a declared reward or constraint uses it. |

## 3. Recommended state contract

### 3.1 Minimal block

At observation time before action `a_t`, define:

```text
position_t       = actual target position currently carried, in {-1, 0, +1}
remaining_frac_t = (task.horizon - elapsed_steps_t) / task.horizon
```

Reset values are `position_t = 0` and `remaining_frac_t = 1`. After each step,
the next observation carries the newly selected position and decremented remaining
fraction. A terminal observation has remaining fraction zero.

Both values are naturally bounded. No train/validation statistic is needed to
normalize them, so there is no fitted-state leakage. A scalar position preserves
the current signed exposure semantics; a three-way one-hot is a legitimate
separate encoding control, not part of the first economic ablation.

For a variable-cost study, add:

```text
current_cost_frac = current_task.reward_params.cost_bps / declared_cost_cap_bps
```

Use a fixed, configuration-declared cap and clip to `[0, 1]`. Do not normalize by
the observed dataset, active PBT population, or future schedule. Because cost is
task semantics, the canonical value remains in `TaskSpec`; `ITraderEnv` merely
materializes the normalized solver input.

### 3.2 Ownership

- **`TaskSpec`:** owns horizon and current reward parameters, including cost.
- **`ITraderEnv`:** owns actual position, elapsed step, reset, causal computation,
  normalization, and observation assembly.
- **Model input splitter/trunk:** consumes three explicit blocks—market, task, and
  solver state. The mutable state block must reach both policy and value paths so
  the critic does not alias returns that the actor can distinguish.
- **FiLM:** remains task conditioning. Mutable position/progress must not be folded
  into the task embedding or used as FiLM parameters merely to avoid changing the
  splitter.
- **`RewardEngine`:** remains the sole reward calculation seam; adding observation
  state does not move reward logic into the model.

This preserves the architecture guardrail that `TaskSpec` is the problem contract,
`ITraderEnv` owns observation and reset semantics, and the solver learns behavior
without acquiring broker or risk authority.

## 4. Reset and recurrent lifecycle

### `stream_worker`

`stream_worker` chooses sequential episode starts, but current `reset()` still
sets position flat and clears episode PnL/turnover statistics. Therefore it is a
market-window sampling continuity mode, **not** a continuous portfolio simulation.
The explicit state block must reset on every environment reset even if return
normalization is configured to persist across the stream.

Carrying position between `stream_worker` episodes would change the environment's
economic semantics, episode evidence, liquidation accounting, and evaluation
contract. It is not an R4-v1 implementation detail.

### Recurrent PPO

- Initialize recurrent state once for each actual environment reset and each vector
  environment independently.
- Carry it across rollout fragments belonging to the same episode. `max_seq_len`
  is a training unroll boundary, not an environment reset.
- Reset on iTrader's real horizon termination. Do not carry state into the next
  `stream_worker` episode merely because its market indices are adjacent.
- Preserve valid sequence lengths and mask padding; never treat padding as flat
  position or zero remaining time.
- Feed the explicit environment state to feed-forward and recurrent models alike.
  Optional RLlib previous-action inputs are a separate factor and are not a
  substitute for actual position.

## 5. Clean matched ablation

Preserve the historical solver identities and use the existing **R4-v1** label for
the observation factor.

### First gate: isolate the known omission on R0

Use the accepted Ray 2.58 R0 contract, identical dataset/tasks/seeds/PBT cadence,
fixed exact 1 bp, and the same economic gates:

1. **R0:** current market window + `semantic_v2` task embedding.
2. **R0 + R4-v1-min:** identical R0 plus position and remaining fraction.

A small diagnostic 2×2 can attribute the two fields before the expensive matched
run:

```text
R0
R0 + position
R0 + remaining fraction
R0 + position + remaining fraction
```

This is preferable to adding every plausible portfolio statistic at once. It
tests the two proven aliases and preserves interpretability.

### Recurrent cross after identities are frozen

R1/R2 are currently unimplemented/planned identities. Do not redesign them around
this finding. Once a recurrent control is correct and frozen, compare the identical
state block across architectures, for example:

```text
R0                     vs R0 + R4-v1-min
R1                     vs R1 + R4-v1-min
R2c-CARRY (if selected) vs R2c-CARRY + R4-v1-min
```

The state-feature factor, recurrence factor, and carry/reset factor must remain
separate. Report state norms, action distribution by prior position and remaining
time, turnover, liquidation cost, held-out net return/Sharpe, throughput, and exact
fresh replay. The first mechanistic success criterion is that action probabilities
change sensibly with current position and proximity to liquidation; economic
promotion still requires the existing held-out gates.

### Later, separate cost-context study

Only after the fixed-cost state result, test a smooth current-cost input across a
predeclared set such as 0.5/1/2 bp. Compare it with `semantic_v2` alone. Do not use
a staged schedule in the same cell: that would change both observability and the
training distribution over time.

## 6. Leakage and normalization risks

- Do not expose absolute dataframe index, `start_idx`, episode seed, future price,
  terminal return, full-episode Sharpe, or regime labels computed with future bars.
- Remaining fraction is causal because it uses only declared horizon and elapsed
  actions. It should not encode absolute clock time unless session-time is a
  separately declared causal market feature.
- Position must be the state **before** the current action. Returning the target
  position in the subsequent observation is correct; returning the proposed
  current action in its own observation would be circular.
- Do not carry state-feature normalizers across train/validation/test if they were
  fitted on all splits. The proposed minimal fields need no fitted normalizer.
- A hashed task embedding distinguishes costs but does not impose smooth ordinal
  geometry. That is a generalization concern, not a Markov defect when each cost
  setting is fixed and observable.
- RNN hidden state can encode market history, but without action feedback it cannot
  exactly recover sampled actions. Even with feedback, environment truth should
  remain explicit for replay and future fill semantics.

## 7. Open questions

1. Should position use one signed scalar or a three-way one-hot? The scalar is the
   minimal semantics-preserving choice; one-hot is a cheap encoding control.
2. Should R4-v1 expose only remaining fraction, or both elapsed and remaining
   fractions? They are redundant given horizon. Use one unless evidence identifies
   an optimization issue.
3. Does any active task panel still use `legacy_v1` or task-provided `t_embed` values
   that omit cost parameters? Audit manifests before a variable-cost experiment.
4. Do proposed R1/R2 custom models receive action feedback through any non-observation
   connector? Record this explicitly, but do not let it replace position state.
5. If future rewards reintroduce drawdown, Sharpe, holding penalties, funding, or
   turnover budgets, what accumulator is the smallest sufficient statistic for
   each reward? That requires a reward-by-reward state audit rather than expanding
   R4-v1 speculatively.
6. Is the intended future online action a target position, an order intent, or a
   delta? Partial fills break the present equality between previous action and
   actual position and will require execution-state contracts outside this offline
   solver ablation.

## Sources

Primary sources only:

1. Richard S. Sutton and Andrew G. Barto,
   [*Reinforcement Learning: An Introduction*, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf)
2. Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra,
   [“Planning and Acting in Partially Observable Stochastic Domains”](https://www.cassandra.org/arc/papers/aij98.pdf)
3. Fabio Pardo et al.,
   [“Time Limits in Reinforcement Learning”](https://proceedings.mlr.press/v80/pardo18a.html)
4. Zhengyao Jiang, Dixing Xu, and Jinjun Liang,
   [“A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem”](https://arxiv.org/abs/1706.10059)
5. Farama Foundation,
   [Gymnasium `Env` API](https://gymnasium.farama.org/main/api/env/)
6. Ray project,
   [Ray 2.58 recurrent model source](https://github.com/ray-project/ray/blob/ray-2.58.0/rllib/models/torch/recurrent_net.py),
   [sequence source](https://github.com/ray-project/ray/blob/ray-2.58.0/rllib/policy/rnn_sequencing.py), and
   [stateful connector documentation](https://docs.ray.io/en/latest/rllib/connector-v2.html)
7. AI4Finance Foundation,
   [FinRL stock-trading environment source](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_stock_trading/env_stocktrading.py)
8. Current iTrader source files cited inline; these are the primary authority for
   the working-tree behavior described here.
