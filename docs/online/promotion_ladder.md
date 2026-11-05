# Promotion ladder

Four rungs. Every gate is a number in `itrader/online/settings.toml` under `[gates.*]`, every check is
written into the strategy's CapabilityProfile with the value it saw. The ladder is advisory: it
never flips live on. Live needs four separate locks (see `operator_guide.md`).

| rung | what the record is | gates (defaults) |
|---|---|---|
| **bench** | 4 years of data, costs modelled, final 6 months untouched | entry rung |
| **paper** | bench passed | dev DSR >= 0.90; holdout Sharpe >= 0.30 and holdout return > 0; dev max drawdown <= 35%; at least 60% of walk-forward folds positive; costs eat <= 50% of gross |
| **small live** | paper record exists | >= 60 paper days; paper PSR >= 0.80; paper Sharpe within 0.5 of the bench holdout Sharpe; paper record covers >= 50% of what the evidence meter asks for |
| **full** | small live record exists | >= 90 live days; realised cost <= 1.5x modelled; DSR including live >= 0.95; capital capped by `[live] max_equity_fraction` until then |

## Why these numbers

* **DSR 0.90**: the deflated Sharpe already discounts for the number of variants tried. 0.90 means the
  development record would be a 1-in-10 fluke under no edge, after that discount.
* **Holdout Sharpe 0.30**: lower than the development bar on purpose. Six months is short; the holdout
  exists to catch the strategy that only worked in the period it was designed on, not to demand
  a second full proof.
* **Folds 60%**: a strategy that made all its money in one six-month window is a different bet from
  one that made a little in most windows. Trend following fails this gate in long sideways periods
  and that is the correct answer.
* **Cost share 50%**: beyond this the strategy is mostly trading for the venue.
* **Evidence ratio 50%**: the evidence meter (bench report) converts a Sharpe into the years of record
  needed before 95% confidence that the edge is above zero. A paper record half that long is enough
  to size small money; the full length is for full size.
* **Slippage ratio 1.5x**: if real fills cost more than one and a half times the model, the bench
  numbers no longer describe the strategy and it goes back a rung.

## The 16/16 lesson, as a number

A no-edge bettor at an implied 98.7% wins 16 in a row 81% of the time. To reject "no edge" at 5%
from a streak alone needs 229 straight wins. `itrader.online.bench.binary_evidence(0.987, 16)`
prints this. Streaks are not evidence; records of known length with known variance are.

## The benchmark question

Passing the ladder says a strategy has a record that is unlikely to be luck. It does not say the strategy is worth
running. For that, look at two more numbers the bench prints for every sleeve and for the portfolio: alpha against
BTC with its t-statistic (below 2 means it could be chance), and the vol-matched BTC line in the portfolio table.
Over 2023 to 2026 holding BTC at the portfolio's own risk made 8.5% a year; a strategy that cannot beat that on a
risk-adjusted basis is complexity without reward. As of the second round nothing here does.

## Demotion

Automatic, each cycle, through the Workbench allocator: a sleeve's capital share is its capability
score (development Sharpe, discounted by DSR and evidence, with the holdout as a vote) blended with
its paper Sharpe as the paper record grows. A sleeve whose paper record diverges loses capital
before anyone reads a report. A drawdown lock or a daily loss lock takes the whole book to cash.
