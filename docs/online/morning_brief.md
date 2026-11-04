# Morning brief, Wednesday 5 November 2026

Roman, here is where things stand. Short version: the online half of iTRADER exists, runs, and is
honest about what it can and cannot earn. It is not a money machine. It is the instrument that will
tell you, in paper, over the next few months, whether any of it deserves real money, and it will say
no clearly when the answer is no.

## What was built (branch `roman/itrader-final`, local only, nothing pushed)

Five local commits (the WIP patch plus four tonight), all new files under `itrader/online/`, `itrader/tests/test_online_*.py` and
`docs/online/`. The CTO's files are untouched; the Desktop steps doc shows the command that proves it.

* **Data**: 4 years of 1d and 1h candles and funding for 10 USDT pairs from Binance's public mirror with OKX
  fallback (no keys), parquet store, data quality checks that the risk engine obeys. 22 minutes to download.
* **Strategies**: Turtle 1983, modern multi-speed trend, cross-sectional momentum, hourly mean reversion, funding
  carry, pairs, seasonality, a prediction sleeve that reports your polyedge paper farm, and an experimental slot
  for the CTO's solver. One interface, no-lookahead test on every one.
* **Bench**: one backtester with fees, spread, slippage and funding; walk-forward folds; the final six months as an
  untouched holdout; probabilistic and deflated Sharpe; an evidence meter; capacity; CapabilityProfiles in the
  repo's format; a promotion ladder with numeric gates; an HTML report.
* **The loop**: MarketFramer (candidates plus WAIT), DecisionGate (every threshold logged), Portfolio (vol targeting,
  risk parity, the Workbench allocator with cash as an asset), RiskEngine (kill file, drawdown and daily-loss
  locks, cooldowns, stale data, keyword calendar, spread cap, caps, gap sizing, fast cancel), PaperBroker (live
  spread, fees, partial fills, latency, funding), Executor, one DecisionRecord per cycle, dashboard, alerts,
  settings, CLI, setup script, systemd unit.
* **Tests**: 26, all passing, including a full replay through the loop on the real data.
* **Review**: an independent pass over the risk engine and execution found 21 issues; 15 were real and are fixed,
  the rest are listed for the CTO in `handover_cto.md`.

## What the bench says

Costs per side: 5 bps fee (10 on spot), 1 bps half spread, 2 bps slippage. Each sleeve is shown at its own
raw size (gross up to 1), before the portfolio scales it down. Development is Nov 2022 to May 2026; the
holdout is 5 May to 3 Nov 2026 and no decision used it.

| sleeve | dev CAGR | dev Sharpe | DSR | dev max DD | holdout CAGR | holdout Sharpe | folds positive | costs / gross | ladder |
|---|---|---|---|---|---|---|---|---|---|
| turtle | 3.3% | 0.33 | 72% | -62% | 52.5% | 1.28 | 4 of 7 | 20% | bench |
| trend_modern | 10.8% | 0.46 | 79% | -52% | 3.7% | 0.29 | 4 of 7 | 7% | bench |
| funding_carry | 1.6% | 2.94 | 100% | -0.8% | 0.0% | 0.00 | 2 of 7 | 54% | bench |
| xs_momentum | -3.9% | -0.11 | 42% | -30% | -18.3% | -1.40 | 3 of 7 | 485% | fails |
| mean_reversion | -16.6% | -0.31 | 14% | -64% | -20.3% | -0.78 | 3 of 7 | 238% | fails |
| pairs | -7.1% | -0.82 | 6% | -27% | -8.9% | -1.33 | 1 of 7 | large | fails |
| seasonality | -23.2% | -1.26 | 1% | -65% | 2.0% | 0.21 | 0 of 7 | large | fails |

Read it like this:

* **Nothing passes the paper gate yet.** Trend's drawdowns at raw size are far too deep (that is what vol
  targeting is for), carry had nothing to earn in 2026 (funding sat below the hurdle), and the other four lose to
  costs. This is the expected shape: single to low double digits a year after costs from the classics, and
  several published effects that do not survive taker fees.
* **Mean reversion is the lesson of the night.** It showed 29% a year on development and 25% on the holdout, the
  best sleeve by far, until a test caught that its daily trend filter saw each day's close 23 hours early. One
  line fixed; the edge was the peek. If you ever see a number like that again, assume a leak until proven otherwise.
* **Turtle's 52% holdout is one good six-month run**, not a result. Its development Sharpe is 0.33.
* **Carry is real but small**: 1.6% a year at raw size with a 0.8% worst drawdown. It is the one thing here that
  behaves like income rather than a bet, and the allocator treats it as such.

### The Workbench portfolio (the surprise, and what it actually does)

Strategies are skills with measured capability. The allocator scores each one on its development Sharpe,
discounted by the chance the edge is real, by how much of the needed evidence exists, by the holdout's vote,
and by how much of the target vol it can reach at sane leverage; cash sits in the same softmax at a score of
zero and never falls below 10%. Then risk parity on correlations, then a 12% vol target for the book.
No weight used any holdout data.

| | CAGR | vol | Sharpe | max DD | worst day |
|---|---|---|---|---|---|
| Workbench, development | 5.4% | 9.7% | 0.59 | -15.1% | -4.2% |
| Workbench, holdout | 5.8% | 7.4% | 0.80 | -3.1% | -1.1% |
| equal weight, development | -4.3% | 7.2% | -0.56 | -19.1% | -2.5% |
| equal weight, holdout | -3.7% | 5.4% | -0.66 | -3.7% | -0.9% |

Allocation today: trend_modern 26%, turtle 25%, funding_carry 33%, cash 16%, the four losers 0%. That is the
surprise: not a bigger number, but a system that puts the losers to zero on its own and holds cash as a
position. The evidence meter says a 0.8 Sharpe needs about four years of record for 95% confidence that it is
above zero; a 0.6 Sharpe needs about eight. Six months of holdout is a hint, not proof. On 10,000 USDT this
portfolio is a few hundred a year. Say that out loud before anyone says "bank".

### The 16/16 lesson, as a number

A no-edge bettor at 98.7c wins 16 in a row 81% of the time. To reject "no edge" at 5% from a streak alone
takes 229 straight wins. The polyedge sleeve stays paper for that reason as much as for the terms of service.

## The paper loop tonight

Two cycles ran on live prices. The gate approved trend_modern (short across the ten pairs, in line with the
trend), held turtle because entering was not worth 1.5x its cost, and rejected pairs on utility. The risk
engine passed the book at gross 0.26, ten orders filled on the live book at 2.7 bps slippage, fees 1.30 on
10,000. The second cycle traded nothing (no-trade band). Cash 50% (one sleeve approved; the allocator caps
any single sleeve at half). Dashboard: `data/online/runs/dashboard.html`.

## What is not done

* Live adapters exist behind four locks and have never touched a venue. Alpaca's symbol map is partial.
* WatsonOW, the golden-dataset export and calibration-by-regime are not built (listed in `handover_cto.md`).
* The experimental slot is empty until the CTO gives it an adapter entry.

## Three next steps

1. **Run paper for 60 days without touching a threshold.** `nohup .venv/bin/python -m itrader.online paper &`,
   look at the dashboard on Sundays, read `decisions.jsonl` when something surprises you. The ladder gate is
   60 days, PSR 0.8, tracking error under 0.5.
2. **Decide carry's leverage with a basis model, or drop it.** At 2x it earns 3% a year; it only matters at 5x,
   and 5x needs a view on basis and exchange risk that this bench does not have.
3. **Give the experimental slot a solver.** `entry="module:function"` returning a weight frame from a `Market`
   puts the CTO's R1/LSTM work on the same bench, same costs, same gate as everything else. That is the fair test.

Files: the branch zip and patch are in your Downloads (`itrader_online_branch.zip`); `docs/online/github_desktop_steps.md`
shows how to look at it without pushing anything.
