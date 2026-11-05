# Handover to the CTO: the online stack, spec section by section

Branch `roman/itrader-final`, cut from `agent/Watson-dev` (6f5f384, 13 Oct 2026). Every file is new:
`itrader/online/**`, `itrader/tests/test_online_*.py`, `docs/online/**`. Nothing existing was edited,
nothing was pushed. Python 3.11+ (built and tested on 3.14 with pandas 3.1; the repo's pinned 3.11 /
pandas 2.1 should work, the code avoids version-specific behaviour). Deps: pandas, numpy, pyarrow,
requests (certifi), pytest. `bash itrader/online/ops/setup.sh` does the lot.

## Map

```
itrader/online/
  settings.toml, settings.py     one file, every threshold, changelog, hash in every DecisionRecord
  contracts.py                   Proposal -> CandidateTask/ExecutionIntent; wait_candidate(); ONLINE_FAMILIES 10-18, 99
  data/  feeds.py store.py universe.py      Binance mirror + OKX, parquet store, Quality (stale/gap/bad print)
  strategies/  base trend momentum reversion carry seasonality prediction_sleeve experimental, registry
  bench/  engine metrics walkforward profile report run     cost-aware backtester, PSR/DSR, evidence meter, ladder, HTML
  portfolio.py                   vol targeting, ERC budgets, Workbench allocator (cash first-class), book controls
  framer.py gate.py risk.py      MarketFramer, DecisionGate, RiskEngine
  execution/  broker paper binance_plug alpaca_plug executor
  runner.py dashboard.py alerts.py cli.py __main__.py ops/
```

## The spec (docs/iTRADER_Codex_Implementation_Spec.md), what is done and what is left

| section | status | where / note |
|---|---|---|
| 1.1 preserve working paths | done | no existing file touched; repo contracts imported, not redefined |
| 1.4 deterministic online shell | done | no learned component can place an order; `experimental.py` is the only hook and it emits weights that go through the same gate and risk |
| 1.5 WAIT first-class | done | `contracts.wait_candidate()` is in every candidate set; gate selects it when nothing clears |
| 2.2 online architecture | done | data -> framer -> gate -> portfolio -> risk -> executor -> DecisionRecord -> profile |
| 3.3 MarketFramer | done | `framer.py`: 1 to 3 candidates + WAIT, market summary with ref hash, Quality carried to risk |
| 3.4 DecisionGate | done | `gate.py`: validity, confidence floor, utility floor per day, utility over holding period vs 1.5x cost, risk lock; every check logged with value and threshold |
| 3.5 RiskEngine | done | `risk.py`: the 12.1 list plus kill file, gap sizing, fast cancel, keyword calendar; 12.2 modes paper and replay run the same code; live sits behind four locks |
| 3.6 Executor | done | `execution/executor.py` + `PaperBroker`; live adapters exist, guarded, untested against a venue |
| 3.7 WatsonOW | left | nothing built; the hooks are the DecisionRecord stream and `alerts.py` |
| 4 contracts | done | TaskSpec, ExecutionIntent, CandidateTask, CapabilityProfile, DecisionRecord used as is; `validate_*` accepts them. ONLINE_FAMILIES are not in `TASK_FAMILY_REGISTRY` (your file); register ids 10-18 and 99 there if you want `normalize_family_id` to keep them instead of folding to "00" |
| 5 task families | partial | online strategy families 10-18 defined in `contracts.py`; not wired to `families.py` (see above) |
| 6 solver | left | `strategies/experimental.py` has the adapter slot: give it `entry="module:function"` returning a weight frame from a `Market`, and it is benched and gated like the rest |
| 10.2 online loop steps 1-11 | done except 6 | step 6 "solver scores each candidate" is served by the bench's CapabilityProfile (PSR as confidence, mean net per bar as utility). A real solver plugs in via the experimental slot |
| 10.3 candidate count | done | `[gate] max_candidates = 3` |
| 10.4 WAIT semantics | done | all four conditions; test `test_gate_selects_wait_when_nothing_clears` |
| 10.5 outputs | done | `data/online/runs/decisions.jsonl`, one repo `DecisionRecord` per cycle |
| 11 gate standard | done | thresholds in settings, logged, no hidden heuristics |
| 12 risk standard | done | see `operator_guide.md` for every lock and how it clears |
| 13 executor standard | done | fills, acks, errors, audit in the record; never changes a weight except lot rounding |
| 14.1 golden dataset | left | DecisionRecords hold the fields; an export job to the 14.1 row shape is not written |
| 14.2 capability dataset | partial | `profiles.json` from the bench; no regime/horizon breakdown yet |
| 17 tests | done | 40 tests: contracts, no-lookahead for every strategy and for the maker engine, cost accounting, funding signs, metrics, ladder, gate/WAIT, every risk block, paper fills, resting orders (fill, chase, stale range), live guard, walk-forward simulation uses only the past, regimes, bootstrap, the audit fixes, full replay through the loop |
| 14.2 by regime | done | `profiles_by_regime.json`: one repo-format CapabilityProfile per strategy and regime (up/down x calm/wild from BTC, known a day ahead) |
| 18.2 online metrics | mostly | dashboard shows WAIT rate, selected candidate, confidence, utility, risk adjustments, fees, slippage, equity; calibration by family and regime over time is left |
| 19/20 phases | n/a | this is the Phase B "online shadow stack" (workstream 7) in paper form |

## What the bench found (4 years, 10 USDT pairs, holdout = last 182 days)

See `docs/online/morning_brief.md` for the tables (second round, which supersedes the first night's numbers).
In one line: Turtle and funding carry have a positive record after costs, modern trend a weak one; momentum,
pairs and seasonality lose; mean reversion looked excellent until a 23-hour lookahead in its daily trend
filter was fixed, and after the audit fixes it is positive on development and loses badly on the holdout.
No sleeve passes the paper gate. Walk-forward (weekly re-allocation from past data only, three years, live
caps applied) the Workbench portfolio made about 1.6% a year at 10% vol, Sharpe 0.21, alpha against BTC
3.5% a year with a t-statistic of 0.6. BTC sized to the same risk made 8.5%. No edge has been demonstrated.

## The independent review (risk.py and execution/)

A separate reviewer went through `risk.py` and `execution/` and reported 21 findings. Fixed tonight:
unlock resetting the peak (it relocked at once before), NaN equity and NaN marks bypassing locks,
non-atomic state files and an unreadable state silently clearing a lock (now fails closed), spot legs
reaching the futures API, dust positions that could not be flattened (rounding and minimum notional),
stale-data and spread rules letting sign flips through, spot legs not clamped long-only, spot short
guard only at submit, inconsistent mark lookups, fast-cancel reference prices lost for a missing symbol,
live adapters not recording fills and one failed cancel aborting a cancel-all, keys read at construction,
order ids not deterministic, realised pnl gross of fees, missing marks valued at cost, funding not
idempotent, closed trades never reaching the cooldown rules. Left, by choice or for you:

* paper broker does not restore open orders after a restart (they are cancelled: the safe reading);
* `cooldown_bars_*` are hours (the cycle bar is 1h); rename if you change the cycle;
* `realised_today` is mark-to-market since the day start, which is what the daily lock should look at, but the name is loose;
* `BinanceBroker.book()` is the one unguarded call (public endpoint, no key);
* the Alpaca adapter maps spot legs to `BASE/USD` and refuses perp legs; equities pass through; untested;
* no margin or liquidation model in paper; gross is capped well below where it would matter.

## Known limits, said plainly

* Costs are modelled, not measured. The paper broker uses the live top-of-book spread, a fixed fee and 2 bps
  slippage. The ladder's `slippage_ratio` gate exists to catch the model being wrong.
* Carry ignores basis. Spot and perp are treated as price-neutral; the holdout had no funding above the hurdle.
* The bench's strategies have published, fixed parameters (`n_variants_tried` 1 or 2). If anyone starts tuning,
  bump that counter: the deflated Sharpe depends on it.
* The prediction sleeve (Polymarket) reports through the polyedge kit and never trades here. Polymarket's
  terms restrict the country the operator lives in; this stack does nothing to work around that.

## Second round (5 Nov, daytime): what was added and what it showed

New modules: `bench/portfolio_sim.py` (walk-forward portfolio simulation, regimes, alpha/beta against a benchmark),
`bench/stress.py` (block bootstrap, wealth projection), `backtest_maker` in `bench/engine.py`, limit orders in the paper
broker and executor, `ops/install_launchd.sh` and its plist, `python -m itrader.online plan`.

**Independent audit of the strategy code.** A second reviewer read every strategy and the taker engine and ran two
lookahead tests on the real data: truncation at many cut points, and replacing every future price, funding rate and
daily row with noise. No lookahead; the mean-reversion fix is confirmed. It then found logic and accounting faults, all fixed:

| where | fault | fix |
|---|---|---|
| `Pairs` | stop and time-out were inert (re-entry on the next bar); legs 1:1 although the spread used the hedge ratio | blocked until the spread is back inside the exit band; no entry beyond the stop level; legs sized 1 : beta, beta fixed at entry |
| `Seasonality` | calendar read from the decision row, so the window traded an hour late; "last day" meant the last two | calendar read from the bar that is held; last day means one day |
| `TrendModern` | inverse-vol normalisation removed forecast strength: always fully invested | position = forecast x inverse-vol share of the universe |
| `CostModel` | spot+perp charged the perp fee on both legs (16 bps instead of 21) | per-leg average of spot and perp fees |
| `FundingCarry` | hurdle used a third of the true round-trip cost; weights summed to 1.0 (no room for margin); edge-proportional sizing re-sized daily; missing funding treated as zero | engine cost in the hurdle, 30-day amortisation; 80% of equity; fixed slots with hysteresis; no data means no position |
| `funding_per_bar` | a settlement at 00:00 was charged to the bar opening then | attributed to the bar that ends at the settlement |
| `Turtle` | breakouts on closes not highs/lows; unit size and stop re-computed daily from current N; skipped breakouts not tracked | highs and lows; N, unit size and stop fixed at the fill; shadow trades for the System 1 filter |
| `CrossSectionalMomentum` | weights re-scaled daily between weekly rebalances; rebalance day counted from the first row of the data | whole book frozen between Monday rebalances |
| `MeanReversion` | hourly re-sizing and re-normalisation; daily trend shifted by rows | size fixed at entry; daily series re-labelled to the time it is known |
| `walk_forward` | one-bar overlaps between folds | half-open slices |
| `Sortino` | spread of losing bars around their own mean | downside deviation from zero |
| data | funding archive up to a month stale | tail filled from OKX public history |

`n_variants_tried` was raised on every strategy that changed. Not changed: the engine still treats weights as constant
within a bar (no cost for drift); the 1h store has one two-hour gap (24 Mar 2023).

**Maker execution.** `[execution] style = "maker"`. Bench: a limit at the touch, filled only on a trade-through of
`maker_through_bps`, the fill marked to the bar's close (so adverse selection is in the numbers), taker fallback after
`maker_max_wait_bars`. Paper: the order rests for one cycle; the broker fills it only if the last completed bar, which must
have begun after the order was placed, traded through it; otherwise the executor cancels it and crosses. Risk-forced
trades and hedged spot/perp pairs are always takers. The gate still prices trades at the taker cost (the conservative
bound). Fill rates in the bench are 93 to 99%; paper has to confirm that before anyone believes it.

**Walk-forward portfolio.** `simulate()` re-allocates weekly from past data only: profile each sleeve (development
plus a trailing pseudo-holdout), apply the gate's confidence floor, vol-scale, ERC budgets, allocator, portfolio
vol scale, cap each sleeve's exposure where the live per-asset cap would clip it, charge re-allocation at the taker
cost. Variants run and reported, so the selection is visible: the allocator as designed, without the gate floor, regime-aware,
equal weight, and the other execution style. An earlier run, before the audit fixes and without gate or caps, gave -1.9%;
with the audit fixes and the gate, 3.5%; with the live caps, 1.6%. The last is the one that matches the running system.

**What to take from it.** The allocator beats equal weight by about two points a year and loses to vol-matched BTC by
about seven. Its timing of the trend sleeves is performance-chasing (trailing six-month vote) and cost it in 2024.
If you redesign one thing, redesign that, and judge the redesign with `simulate()` and a count of what you tried.

## Third round (5 Nov, later): the lab

New package `itrader/online/lab/`:

| file | what it is |
|---|---|
| `registry.py` | Pre-registered recipes (hypothesis, sleeves, policy, universe) and the selection criteria. Committed before any evaluation (`3b54e6f`). Add a recipe here to test an idea; it is counted as one more trial. |
| `policy.py` | `exposures(policy, past, metas, s, state)`: exposure multiplier per sleeve from past returns only. `steady` (slow gate with memory, equal risk, monthly, 20% band) and `stack` (market-neutral core at its own notional, directional overlay sized by risk, gap budget shared). The same function drives the walk-forward simulation and the live loop. |
| `evaluate.py` | `Lab.cycle()`: sleeve returns per universe and cost level, walk-forward per recipe, research and vault windows, deflated Sharpe over all trials, eligibility constraints, champion and challenger rule, `ledger.jsonl`, `champion.json`, `league.json`, `returns.parquet`, sleeve returns and metadata for the live loop. |
| `advisor.py` | `write_report()`: deterministic review from the league, the champion, the ledger and the paper records. |

New sleeves: `carry_core` (hurdle at the cost of cash plus amortised costs, two-week funding, more and smaller slots),
`trend_long` (the modern trend forecast, long side only), `macro_trend` (the same rule on 20 ETFs, bar `1b`, instrument
`etf`, excess returns over a T-bill ETF, 2 bps a side, a borrow fee on shorts). New feed `Yahoo` (daily, adjusted). The
crypto universe is 25 coins for daily sleeves; hourly sleeves stay on the ten majors (`universe_1h`).

Live wiring: `PaperRunner` adopts `champion.json` (its sleeves, universe and policy), computes exposures with
`lab.policy.exposures` every `rebalance_days`, builds the book with `portfolio.build_from_exposures`, syncs the listed
markets once a day, and once a week re-runs the bench and the lab and applies the champion rule. Without a champion the
second-round Workbench path runs unchanged.

Three live settings were changed so that what runs is what was tested, each in the changelog: the gate's absolute
utility floor (1 bp a day) became "positive after costs", because it shut out carry; `max_candidates` went from 3 to
10, because a stacked book approves each sleeve on its own (the spec's 1 to 3 is a default, see 10.3); and the gate's
holding-period estimate is now position-weighted (twice average gross over turnover), because counting bars between
any trade made a carry position look like a ten-day trade. The risk engine gained `hedged_gap_factor` (0.2 of one-leg
notional for a hedged pair) and `gap_move_listed` (5% for listed markets); gross caps are 2.5 so a carry core fits.

Results and caveats are in `morning_brief.md` and `advisor_report.md` (a snapshot is committed in `docs/online/`). The
short version: champion `R7_stack_wide_all`, 11.0% a year at Sharpe 1.21 on the research window, -1.0% in the vault,
58% deflated probability after 13 trials. Unconfirmed. The allocator fix (R0 to R1) is the gain I would bet on.

Things I would look at first if this were mine: (1) the survivor universe: rebuild it point-in-time from exchange
listings; (2) the macro sleeve's instrument: it needs a real margined venue and its own cost model; (3) carry's churn
around the hurdle in low-funding regimes, visible in the vault; (4) the lab's research window is two and a half years
of one crypto regime: every conclusion is conditional on that.

**Trial 14 (same day).** `R8_stack_core_long` (ten majors, carry core, classic sleeves plus `trend_long`) was registered at
Roman's request after R0 to R7 had been seen, committed before evaluation (`9fd0845`), and replaced R7 under the challenger
rule (t 1.5): research 17.0% a year at Sharpe 1.30, whole three years 13.3% at 1.04, vault -1.9%. Unconfirmed, and the
vault has now been opened twice, so treat it as spent. The paper loop runs R8 on the ten-coin universe.
