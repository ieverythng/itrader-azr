# Status brief, 5 November 2026 (third round: the lab)

Roman, the numbers went up, by a lot, and I can show you exactly why and exactly how far to trust them.

## Update, same day: trial 14 is the new champion

You approved one more experiment, registered before it was run: the ten original coins, carry as the core, the
classic sleeves and long-only trend on top (`R8_stack_core_long`). Under the lab's own rule it replaced the champion
described below (its daily returns beat R7's with a t-statistic of 1.5).

| | per year | vol | Sharpe | worst DD | at 2x costs | last six months (vault) |
|---|---|---|---|---|---|---|
| R7, the previous champion | 11.0% | 8.9% | 1.21 | -8.7% | 0.68 | -1.0% |
| **R8, the champion now** | **17.0%** | **12.8%** | **1.30** | **-8.2%** | **1.00** | **-1.9%** |

Over the whole three years out of sample R8 comes to **13.3% a year at a Sharpe of 1.04**, worst drawdown -8.2%.
Resampled, the research record runs from 3.0% a year in the bad case to 35.5% in the good case, median 16.6%.
The five-year middle of your plan (1,000 plus 200 a month) rises to 18,400 against 13,000 paid in.

Read it with three cautions, each heavier than for R7:

* **The vault said no again:** -1.9% over the last six months. Two champions, two losing vaults. The recent market has
  not paid this style, and carry in particular has cost money since May.
* **This idea came from looking at the results.** That makes its research figure weaker evidence than the others'. After
  14 trials the probability it beats luck is 62%; established needs 90%.
* **The vault has now been opened twice.** It is no longer an untouched test. From here the only clean evidence is the
  paper record going forward.

About 7 of the 17 points come from long-only trend, which is owning crypto while it rises. The paper loop now runs R8.
Everything below describes the round that led here and still holds, except that R7 is no longer the champion.

## The one paragraph to read

Over the same three years, every day out of sample, the portfolio went **from 1.6% a year at a Sharpe of 0.21
to 8.7% a year at a Sharpe of 0.99**, with a smaller worst drawdown (-8.7% against -17.4%). On the two and a half
years used for choosing it reads 11.0% at 1.21. The catch, stated as plainly as the gain: on the most recent
six months, which were held back and looked at once, it lost 1.0%. So this is a much better system that is
**not yet confirmed**. It is running in paper now, and the next 60 days of that record are the test nobody has
seen the answer to.

## How it got better: a research loop that cannot fool itself

You asked for something that notices its mistakes and improves on itself. That is now a module, `itrader/online/lab/`:

1. **Rules first.** Eight experiments, their hypotheses and the way a winner is chosen were written into
   `lab/registry.py` and committed before any of them was run (commit `dbbe1a4`).
2. **Walk-forward judging.** Each recipe is run month by month using only what was known at the time, with costs,
   double costs, 2,000 resampled histories and benchmarks.
3. **Every attempt is counted.** The ledger keeps every trial. Thirteen variants have been tried in all, and every
   Sharpe ratio is marked down for that.
4. **A vault.** The most recent six months choose nothing. They are opened once, for the winner, and reported.
5. **Champion and challenger.** A new recipe replaces the champion only if it is better after the mark-down and its
   daily returns beat the champion's with a t-statistic of at least 1.
6. **An advisor.** Each run writes `advisor_report.md`: where things stand, what each change was worth, what to be
   careful about, whether paper trading agrees with the model, and what to try next. Fixed rules over measured
   numbers; it does not flatter.
7. **It repeats.** Once a week the paper loop re-runs the bench and the lab on fresh data and applies the same rule.

## What the experiments found

Research window, 6 Nov 2023 to 7 May 2026, every day out of sample:

| recipe | what changed | per year | vol | Sharpe | worst DD | Sharpe at 2x costs |
|---|---|---|---|---|---|---|
| R0 baseline | the system as it was | 3.1% | 10.6% | 0.34 | -16.2% | -0.06 |
| R1 steady | a slower allocator, no trailing vote | 8.0% | 10.6% | 0.78 | -11.8% | 0.50 |
| R2 stack | carry as the core, directional sleeves on top | 8.8% | 11.5% | 0.79 | -12.0% | 0.46 |
| R3 + carry core | carry hurdle set by the cost of cash | 10.0% | 11.1% | 0.91 | -9.8% | 0.53 |
| R4 + 25 coins | wider universe | 5.9% | 10.3% | 0.61 | -12.2% | 0.00 |
| R5 + long-only trend | owning crypto while its trend is up | 13.7% | 11.4% | 1.18 | -10.4% | 0.65 |
| R6 + multi-asset trend | the trend rule on 20 ETFs | 4.8% | 8.2% | 0.62 | -9.2% | -0.02 |
| **R7 all three** | **champion** | **11.0%** | **8.9%** | **1.21** | **-8.7%** | **0.68** |

What each change was worth, measured against the recipe before it:

* **The allocator was the biggest leak.** Slowing it down and removing its six-month performance vote more than doubled
  the Sharpe ratio (+0.45). This is the most trustworthy gain: it is a logic fix, not a new bet.
* **Long-only trend added the most return** (+7.8 points a year, t 2.4). Be clear about what it is: owning crypto while it
  rises and standing aside when it falls. That is market exposure with a brake, not an edge. In a long decline it earns
  nothing.
* **Carry with a sensible hurdle helped a little** (+0.12 Sharpe) and then lost about 2% a year in the last six months,
  when funding hovered around its hurdle and it paid to get in and out. That is most of the vault's loss.
* **Twenty-five coins instead of ten made it worse** (-0.31). More names, more cost, no more independence.
* **Multi-asset trend did not raise the Sharpe ratio here.** On 20 years of ETF history, after costs and financing, the rule
  earns a Sharpe of only 0.22. It lowered volatility and drawdown, which is why the combined recipe edged out R5.

## How far to trust it

* **Chosen with hindsight of the research window.** After marking down for 13 trials, the probability the champion's
  Sharpe is better than luck is 58%. The bar for "established" is 90%.
* **The vault did not confirm it:** -1.0% over the last six months (Sharpe -0.20, worst drawdown -5.3%).
* **Alpha is borderline:** 10.5% a year against BTC (t 1.9) and 11.6% against the S&P 500 (t 2.1). Around 2 is where
  "probably real" begins; this sits on the line.
* **Resampled:** from 0.7% a year in the bad case to 23.4% in the good case, median 10.7%.
* **Benchmarks in the same window:** the S&P 500 made 24.8% a year at 15.7% volatility, an exceptional run against a
  long-run average near 10%. BTC made 40% with a 50% drawdown.

## On 18 to 22% a year

The champion runs at about 9% volatility because the risk limits hold it there. At the stock market's volatility (about
16%) the same record would read roughly 19% a year in the research window and 15% over the whole three years, with
drawdowns nearly doubled. That dial exists (`portfolio_vol_target` and the gap rule) and I have not turned it. Turning up
risk on a result the vault has not confirmed is how accounts are lost. If 60 days of paper agree with the model, it is
a two-line settings change, and the advisor's report will say when that case can be made.

## Your plan, on the new record

`python -m itrader.online plan --capital 1000 --monthly 200` (it now uses the champion):

| years | paid in | bad case | middle | good case | chance of less than you paid in |
|---|---|---|---|---|---|
| 1 | 3,400 | 3,256 | 3,568 | 4,035 | 21% |
| 3 | 8,200 | 7,998 | 9,371 | 11,418 | 9% |
| 5 | 13,000 | 13,244 | 16,281 | 20,973 | 4% |

This morning the five-year middle was 13,594. It is now 16,281, and even the bad case ends above what you paid in.
The saving still does most of the work; the trading now adds something visible.

## Running now

The paper loop runs the champion: carry core on eight coins, long-only trend, Turtle, modern trend and the multi-asset
sleeve, with resting limit orders and the weekly self-review. Entering the carry core cost about 0.14% of equity, as
modelled. Dashboard and `python -m itrader.online lab --action show` show the current state.

## What is not done, and what I would not claim

* This is not a finished product a buyer could value. It is a disciplined research platform with one unconfirmed
  champion. Its worth depends on a live record it does not have yet.
* The multi-asset sleeve assumes margined instruments; ETF history stands in for them and the paper loop marks them at
  daily closes.
* The 25-coin list contains only coins that survived, which flatters long-biased results a little.
* Carry's real risks (an exchange failing, spot and perpetual prices pulling apart) are not in these numbers.
* Live trading is still behind four locks, untested against a venue, and nothing here justifies opening them.

## Three next steps

1. **Leave it alone for 60 days.** Read the advisor's report each Monday. Change nothing.
2. **One observation worth a new experiment:** the wider universe hurt and long-only trend helped, so "ten coins plus
   long-only trend" may beat the champion. I did not test it, because adding recipes after seeing results is exactly
   how backtests lie. If you want it, it goes into the registry as trial 14 and the lab judges it.
3. **Give the experimental slot the CTO's solver.** It is judged by the same lab against the same champion.

This is a measurement of software against historical data, not financial advice.
