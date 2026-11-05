# Primary-source survey: trading benchmark and environment patterns

Date: 2026-10-06
Scope: evaluation-design evidence only; no iTrader code, panel, split, metric, or
threshold is changed by this note.

## Reading rule

The sources below verify what each platform implements or documents. They do
**not** verify that the platform, its bundled agents, or a reported return has a
replicable economic edge. The iTrader mappings are explicitly labelled as
inferences.

## Verified source features

### FinRL / FinRL-Meta

- FinRL-Meta separates market environments from its data and agent layers and
  argues that agents should be compared in the same environment and dataset.
  Its public pipeline describes training, hyperparameter adjustment on a
  separate "testing" period, and then historical backtesting or paper/live
  trading. The word *testing* in that pipeline therefore performs a role closer
  to validation than to an untouched final test
  ([official FinRL-Meta repository](https://github.com/AI4Finance-Foundation/FinRL-Meta#training-testing-trading-pipeline)).
- FinRL's current `StockTradingEnv` uses a continuous per-asset action vector,
  explicit buy/sell cost arrays, cash and share inventory, optional turbulence-
  triggered liquidation, and memories for assets, rewards, actions, states, and
  dates
  ([official environment source](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_stock_trading/env_stocktrading.py)).
- FinRL documentation says its environment layer incorporates transaction
  costs, portfolio restrictions, and risk-aversion mechanisms
  ([official environment documentation](https://finrl.readthedocs.io/en/latest/finrl_meta/Environment_layer.html)).

**iTrader inference:** retain the data/environment/agent separation and durable
action/state/date/economic ledgers. Do not import FinRL's continuous share
actions or turbulence liquidation into the `Discrete(3)` solver; forced risk
actions belong to an explicit environment contract or the deterministic online
shell.

### Microsoft Qlib

- Qlib's canonical workflow example specifies independent `train`, `valid`, and
  `test` date segments, a separate preprocessing fit interval, an explicit
  benchmark, and exchange parameters including open cost, close cost, and
  minimum cost
  ([official workflow documentation](https://github.com/microsoft/qlib/blob/main/docs/component/workflow.rst)).
- Qlib's `PortAnaRecord` retains return reports and detailed positions and
  computes benchmark-relative risk analysis both before and after costs
  ([official record implementation](https://github.com/microsoft/qlib/blob/main/qlib/workflow/record_temp.py)).
- Qlib Recorder associates metrics, parameters, tags, and artifacts with a
  concrete run
  ([official recorder documentation](https://github.com/microsoft/qlib/blob/main/docs/component/recorder.rst)).

**iTrader inference:** Qlib supports the case for keeping prediction/policy
outputs, portfolio accounting, benchmark-relative economics, and provenance as
separate artifacts. Its sample dates and costs are examples, not values to copy.

### TensorTrade

- TensorTrade exposes composable environment, action, reward, observer, data-
  feed, exchange, and portfolio components
  ([official repository](https://github.com/tensortrade-org/tensortrade)).
- Its reward interface may own state and resets that state with the environment;
  documented examples include simple profit, risk-adjusted returns, and
  position-based return `R_t = (P_t - P_{t-1}) X_t`
  ([official reward documentation](https://github.com/tensortrade-org/tensortrade/blob/master/docs/source/components/reward_scheme.md)).

**iTrader inference:** modularity is a useful seam pattern, but arbitrary
plug-compatibility is not a scientific evaluation contract. iTrader should keep
`RewardEngine`, portfolio evidence, and execution semantics separately owned
rather than infer validity from a swappable reward component.

### Gym-AnyTrading

- The base environment uses binary buy/sell actions, starts each episode short,
  changes position only when the opposite action is selected, returns windowed
  signal features, and exposes only total reward, total profit, and position in
  its base `info` record
  ([official base-environment source](https://github.com/AminHP/gym-anytrading/blob/master/gym_anytrading/envs/trading_env.py)).
- Stock/forex subclasses expose fee parameters, while its own
  `max_possible_profit` diagnostic ignores fees
  ([official repository documentation](https://github.com/AminHP/gym-anytrading)).

**iTrader inference:** this is a useful negative comparison. Implicit initial
position, a binary action space, and a fee-free oracle do not match iTrader's
target-position `short/flat/long`, terminal-liquidation accounting, or evidence
requirements.

### Gym-Trading-Env

- Gym-Trading-Env defines actions as target portfolio positions and documents
  `[-1, 0, 1]` as short, out, and long. It includes the last requested and real
  portfolio positions as dynamic features, supports windowed observations, and
  exposes trading fees and per-step borrowing interest
  ([official tutorial](https://github.com/ClementPerroud/Gym-Trading-Env/blob/main/docs/source/rl_tutorial.rst),
  [official environment description](https://github.com/ClementPerroud/Gym-Trading-Env/blob/main/docs/source/environment_desc.rst)).
- Its default reward is the log change in portfolio value, and its episode can
  end at data exhaustion or portfolio ruin
  ([official environment description](https://github.com/ClementPerroud/Gym-Trading-Env/blob/main/docs/source/environment_desc.rst)).

**iTrader inference:** the target-position action vocabulary is genuinely close
to iTrader's `Discrete(3)` semantics. The reusable pattern is the explicit
position/inventory contract, not the default reward, leverage model, or episode
termination rule.

### TradeMaster

- TradeMaster describes state, action, reward, data, algorithm, training, and
  evaluation as interacting parts of one RL-finance pipeline and notes that
  profitability depends on features and evaluation market state
  ([primary NeurIPS benchmark paper](https://papers.neurips.cc/paper_files/paper/2023/file/b8f6f7f2ba4137124ac976286eacb611-Paper-Datasets_and_Benchmarks.pdf)).
- Its reported benchmark uses rolling phases: in each phase the last year is
  test, the preceding year validation, and earlier history training. It reports
  multiple metrics and ranks across three test periods and five random seeds
  ([primary NeurIPS benchmark paper](https://papers.neurips.cc/paper_files/paper/2023/file/b8f6f7f2ba4137124ac976286eacb611-Paper-Datasets_and_Benchmarks.pdf)).

**iTrader inference:** multiple chronological test periods, multiple training
seeds, and separate profit/risk/robustness views are useful. Their particular
one-year windows, seed count, rank summaries, and algorithms are not evidence
that the same settings are adequate for one-minute overlapping iTrader tasks.

## Patterns worth mapping to current iTrader owners

| Pattern | Current owner | Bounded implication |
|---|---|---|
| Explicit chronological data roles and preprocessing-fit bounds | `DataPartition` plus sealed panel qualification | Declare acquisition universe, training, adaptive validation, terminal evaluation, and final test independently; never infer a role from the date alone. |
| Target position, current inventory, and transition cost | `ITraderEnv` / `EpisodeEvidence` | Preserve `short/flat/long`, prior position, turnover, and liquidation semantics; do not turn `WAIT` into a solver action. |
| Gross, cost, net, exposure, positions, and benchmark-relative reporting | `EpisodeEvidence` and terminal reporter | Report a metric vector and paired benchmark deltas; do not replace economics with optimized reward. |
| Run-level params, identities, artifacts, and action/state/date histories | provenance/evaluation layer | Bind every claim to dataset, source, config, checkpoint, panel, task, seed, and producer identity. |
| Multiple temporal periods and training seeds | evaluation protocol | Treat market-period and optimization-seed variation as different uncertainty dimensions; two seeds remain a screen rather than population-level optimizer inference. |

## Patterns not justified for automatic adoption

1. No surveyed platform proves that its default split, fee, reward, action
   space, or metric is correct for iTrader.
2. A framework's built-in Sharpe or benchmark report does not solve dependence,
   selection, or multiple-testing bias.
3. Older or newer dates do not certify non-exposure. Dataset role and prior
   adaptive access must be proven from metadata/provenance.
4. A convenient percentage split must not be applied to a short evaluation-only
   slice, and the existing sealed final test must not be repartitioned.
5. Risk overrides, forced liquidation, and live order semantics remain outside
   the solver unless a separately approved environment or deterministic-shell
   contract owns them.

## Conclusion

The strongest transferable lesson is not a model or headline metric. It is the
combination of explicit data roles, explicit position/cost semantics, benchmark-
relative accounting, and run-level provenance. iTrader already has owners for
those concerns; the next methodology should deepen their evidence contracts
rather than replace them with another platform's defaults.
