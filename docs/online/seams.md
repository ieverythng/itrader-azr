# Seams: where the online stack meets the rest of iTRADER

Written for alignment with the CTO. Everything under `itrader/online/` is new; nothing outside it was edited. This
page lists every point of contact, what each side assumes, and the questions that need a decision.

## What the online stack imports from the existing code

Only the typed contracts in `itrader/rl_lib/tasks/`, unchanged:

| import | used in | for |
|---|---|---|
| `TaskSpec`, `ExecutionIntent`, `CandidateTask` | `online/contracts.py` | every strategy proposal becomes a `CandidateTask`; `wait_candidate()` is the first-class WAIT |
| `CapabilityProfile` | `online/bench/profile.py`, `bench/run.py` | one profile per strategy (and one per strategy and regime) from the bench |
| `DecisionRecord` | `online/runner.py` | one record per paper cycle, appended to `data/online/runs/decisions.jsonl` |
| `validate_candidate_task`, `validate_capability_profile` | tests only | the repo's own validators accept what the online stack emits |

It does not import the environments, the solvers, the proposers, Ray, or torch. It runs without them.

## What it deliberately does not touch

* `rl_lib/tasks/families.py`: the online strategy families use ids 10 to 19 and 99 (WAIT), kept in
  `online/contracts.py::ONLINE_FAMILIES`. They are **not** in `TASK_FAMILY_REGISTRY`, so `normalize_family_id` folds them
  to "00". Decision for you: register them, or keep the online ids separate.
* The evaluation framework, the datasets, every existing test and script.

## Where your work plugs in

1. **A solver as a strategy.** `online/strategies/experimental.py` takes `entry="package.module:function"`, where the
   function receives a `Market` (price panels, funding) and returns a frame of target weights, one row per bar, using
   data up to that bar only. It is then benched, walk-forward tested, gated, risk-checked and paper-traded like every other
   sleeve, and can be added to a recipe in `online/lab/registry.py` to compete with the champion.
2. **A proposer in the framer.** `online/framer.py::MarketFramer.frame()` is where candidates are produced. Today each
   comes from a rule-based strategy with a templated rationale. A learned proposer can add candidates or rationales here;
   whatever it emits still passes `DecisionGate` and `RiskEngine`, which stay deterministic (spec 1.4).
3. **Solver scores in the gate.** `online/gate.py::DecisionGate.decide()` scores candidates with the bench's
   CapabilityProfile (confidence = probability of an edge after costs). A solver's own score per candidate can be added as
   one more logged check.
4. **DecisionRecords as data.** `decisions.jsonl` holds, per cycle: the market summary reference, every candidate with
   WAIT, the gate's checks and thresholds, the risk verdict with each adjustment, orders and fills, and the equity
   before and after. That is most of the golden-dataset row in spec 14.1; an export job is not written.
5. **"Is it statistically significant?"** `python -m itrader.online judge --returns <file> --column net_return --trials N`
   reads your `*.episodes.jsonl` (or any CSV or parquet of returns) and answers with a probabilistic and a deflated
   Sharpe ratio, a block bootstrap and a verdict. See `review_evaluation_framework.md`.

## Differences to align on

| | existing stack | online stack |
|---|---|---|
| markets | US equities, Alpaca SIP | crypto perpetuals and spot (Binance public data); 20 ETFs for one sleeve |
| bar | 1 minute | 1 day and 1 hour |
| action | 3 discrete actions per instrument | a vector of target weights across instruments |
| horizon | 96 to 200 bars per episode | positions held days to months |
| cost in evaluation | 1 bp of turnover | 5 to 10 bp fee plus spread and slippage; maker fills modelled |
| judged by | frozen replay, paired block interval | walk-forward, deflated Sharpe over all trials, a held-back final window, paper record |
| learned parts | solver, proposer | none in the loop; one slot reserved for a solver |

Questions that need an answer before the two halves share a loop:

1. Should a solver trained on 1-minute equities be asked for daily crypto weights, or should the online loop grow an
   intraday equities path (the Alpaca adapter exists but is untested against the venue)?
2. Who owns the family registry for online ids?
3. Does the proposer sit before the framer (proposing tasks to the strategies) or inside it (proposing alongside them)?
4. Which evaluation is the gate for promotion: yours, the lab's, or both in sequence?

## Honest state of the online half

A disciplined research and paper-trading platform with one unconfirmed champion: about 13% a year at a Sharpe near 1 over
three years out of sample, minus 1.9% in the most recent six months. Details and caveats are in `morning_brief.md`,
`advisor_report.md` and `handover_cto.md`. Live trading is behind four locks and has never touched a venue.
