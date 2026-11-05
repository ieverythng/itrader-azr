# Status brief, 5 November 2026 (second round)

Roman, this replaces this morning's brief. The system is considerably better than it was twelve hours
ago. The headline number is worse, and that is the improvement: the earlier figure was too kind, and the
new tests are the ones that would have cost you money to learn any other way.

## The one paragraph to read

Tested the hard way (re-allocating every week using only what was known at the time, for three years, with
realistic costs and the live system's own limits), the portfolio made **about 1.6% a year at 10% volatility**.
That is not distinguishable from zero: there is roughly a 38% chance the true record is a losing one. Over the
same three years, simply holding BTC at the same level of risk made about 8.5% a year. **No strategy here has
earned promotion past paper, and the honest reading is that no edge has been demonstrated.** What you own is
an instrument that can tell real edge from luck, and it is currently saying "not yet".

## What I got wrong this morning, and why

This morning's brief said the Workbench portfolio made 5.4% a year on development and 5.8% on the holdout.
Both numbers were computed correctly and both were too generous, for three reasons that the second round found:

1. The allocator's weights for the development period were set with knowledge of the whole development period.
   The holdout was honest, but it was a single six-month window. The new walk-forward simulation never lets the
   allocator see past the day it is deciding on, and covers three years.
2. An independent audit of the strategy code found no lookahead (it scrambled all future data and nothing
   changed), but it found real faults in six of the seven strategies. Fixing them moved results in both directions.
3. The first simulation levered the carry sleeve to 1.7x, which the live book's per-asset cap would never allow.
   Capped where the live system caps it, the result dropped from 3.5% to 1.6%.

## What changed in the system

* **Independent audit of every strategy, all findings fixed.** Pairs: the stop never stopped (it re-entered on the
  next bar), and the legs ignored the hedge ratio. Seasonality: traded its window an hour late. Modern trend: ignored
  forecast strength and was always fully invested. Carry: costs understated by a third, sized as if the account had
  more capital than it does, re-sized daily for no reason. Turtle: used closes where the 1983 rules use highs and
  lows, and re-sized units daily instead of fixing them at entry. Momentum: churned daily and rebalanced on different
  days in backtest and live. Each strategy's honesty counter (`n_variants_tried`) went up accordingly.
* **Maker execution with an honest fill model.** Orders rest at the touch and fill only if the market trades through
  them; whatever does not fill within one bar is crossed as a taker at full cost. In the bench this cut trend's costs
  from about 20% of gross profit to 2%. It did not rescue any losing strategy. The paper loop now trades this way;
  anything the risk engine forces is still sent at once as a taker.
* **Walk-forward portfolio simulation**, with BTC as the benchmark, a block bootstrap and a cost stress.
* **Regime-aware capability profiles** (four regimes from BTC trend and volatility, per-regime CapabilityProfiles as the
  spec's section 14.2 asks). Tested out of sample, regime-aware allocation was worse (0.6% against 1.6%), so it is built,
  reported, and switched off.
* **Alpha against BTC** for every sleeve and for the portfolio, so "is this just being long crypto?" has a number.
* **Fresh funding data** (the exchange archive runs a month behind; the gap is now filled from OKX), a daily summary
  alert, a wall-clock sleep so a sleeping laptop cannot skip cycles, Mac service files.
* **40 tests**, all passing.

## The numbers

Per strategy, maker execution, each at its own raw size. Development is Nov 2022 to May 2026; the holdout is the last 182 days.

| sleeve | dev CAGR | dev Sharpe | DSR | dev max DD | holdout CAGR | holdout Sharpe | alpha vs BTC (t) | verdict |
|---|---|---|---|---|---|---|---|---|
| turtle | 14.1% | 0.54 | 67% | -44% | 9.9% | 0.45 | 15.6%/yr (0.8) | best of the set, not proven |
| trend_modern | 4.5% | 0.30 | 50% | -30% | 1.6% | 0.18 | 12.6%/yr (0.9) | weak |
| funding_carry | 2.7% | 5.24 | 100% | -0.6% | 0.0% | 0.00 | 2.2%/yr (9.4) | real but small; nothing to earn since May |
| mean_reversion | 4.0% | 0.28 | 37% | -39% | -26.7% | -1.38 | 5.1%/yr (0.4) | fails the holdout |
| pairs | -1.2% | -0.10 | 24% | -14% | -3.0% | -0.48 | 0.4%/yr (0.1) | no edge |
| xs_momentum | -5.2% | -0.16 | 21% | -38% | -28.3% | -2.20 | -5.3%/yr (-0.6) | loses |
| seasonality | -24.5% | -1.36 | 0% | -69% | -8.0% | -0.45 | -38.9%/yr (-5.6) | loses, badly |

A t-statistic below 2 means the figure could easily be chance. Only carry's is above 2, and carry earns about
2% a year. Its Sharpe looks enormous because the bench does not model the gap between spot and perp prices,
which is the real risk in that trade; do not read 5.24 as safety.

The whole portfolio, walk-forward, every day out of sample (5 Nov 2023 to 5 Nov 2026):

| | CAGR | vol | Sharpe | max DD | alpha vs BTC (t) | last 6 months |
|---|---|---|---|---|---|---|
| Workbench (as it runs live) | 1.6% | 10.3% | 0.21 | -17.4% | 3.5%/yr (0.6) | -5.2% |
| Workbench, regime-aware | 0.6% | 8.3% | 0.12 | -18.0% | 0.7%/yr (0.1) | 0.0% |
| equal weight across sleeves | -0.5% | 8.1% | -0.02 | -16.7% | -0.7%/yr (-0.2) | -7.4% |
| BTC, buy and hold | 31.9% | 46.6% | 0.83 | -53.0% | | |
| BTC, sized to the same risk | 8.5% | 12.7% | 0.71 | -22.8% | | |

* Resampled 2,000 times, the Workbench record ranges from -6.7% a year (bad case) to +10.8% (good case), median 1.6%.
* At double the assumed costs it makes 1.3%; at triple, it loses 5.7%.
* The allocator does beat equal weight, by about two points a year, by keeping the proven losers at zero. It does
  not beat the benchmark, and its own timing of the trend sleeves cost it: it added to them after good runs.
* BTC's own Sharpe over three years of a bull market is not proof of anything either. It is the bar a crypto strategy
  has to clear to justify its complexity, and nothing here clears it.

## Your second surprise: the plan, in your own numbers

`python -m itrader.online plan --capital 1000 --monthly 200` projects an account forward from the measured record:

| years | paid in | bad case | middle | good case | chance of less than you paid in |
|---|---|---|---|---|---|
| 1 | 3,400 | 3,122 | 3,418 | 3,830 | 47% |
| 3 | 8,200 | 7,220 | 8,437 | 9,964 | 39% |
| 5 | 13,000 | 11,235 | 13,594 | 16,862 | 36% |

Read the "paid in" column first. After five years, 13,000 of the 13,594 in the middle case is money you put in.
The trading adds about 600, and a third of the time it takes some away. At this size, **what you save decides the
outcome and what the system earns barely registers**. That is the real wealth code, and it is unglamorous: keep
adding, do not blow up, and do not pay real money for an edge that has not been shown. The tool takes any measured
return stream (`--source`), so when a strategy does earn its place, you will see it in this table before you risk a cent.

## Running now

The paper loop is running again on the upgraded code, in maker style, with the trend sleeves approved and the rest at
zero. Equity stands near 9,991 on a 10,000 paper start. To keep it alive across restarts on the Mac:
`bash itrader/online/ops/install_launchd.sh` (it needs the Mac awake; a closed laptop on battery will pause it).

## What is not done

* Live adapters remain behind four locks and untested against a venue. Nothing here justifies opening them.
* WatsonOW and the golden-dataset export are still for the CTO (see `handover_cto.md`).

## Three next steps

1. **Leave the paper loop alone for 60 days.** The useful question now is whether paper matches the bench, especially
   the maker fill rate. If real fills are worse than modelled, the ladder's cost gate will say so.
2. **Put the CTO's solver in the experimental slot.** It gets the same bench, the same walk-forward test and the same
   benchmark. If it has an edge, this is now a fair and strict place to show it.
3. **Bring one new idea at a time**, state the rule before looking at the result, and let the bench judge it. Every
   variant tried is counted against it. That discipline is worth more than any strategy in this repository.

This is a measurement of software against historical data, not financial advice.
