# Advisor report, 05 November 2026 09:56 UTC

## Where the system stands

**Champion: `R7_stack_wide_all`** (stack policy, wide25, adds market exposure; survivor universe). first champion: highest deflated Sharpe among eligible recipes.

* Research window (2023-11-06 to 2026-05-07, every day out of sample): **11.0% a year at 8.9% volatility, Sharpe 1.21**, worst drawdown -8.7%, worst day -2.6%.
* Deflated for the 13 variants tried so far, the probability this is better than nothing is 58%.
* Vault (2026-05-07 onward, never used to choose): -1.0% in total, Sharpe -0.20, worst drawdown -5.3%. **NOT confirmed**.
* At double the assumed costs: Sharpe 0.68. Chance the research record is a losing one: 4%.
* Against BTC: beta 0.01, alpha 10.5% a year (t 1.9). Against the S&P 500: beta -0.04, alpha 11.6% a year (t 2.1).

Benchmarks over the same research window: btc 40.1% a year at 48.4% vol (Sharpe 0.94, drawdown -49.5%); btc vol matched 9.9% a year at 12.8% vol (Sharpe 0.80, drawdown -19.6%); spy 24.8% a year at 15.7% vol (Sharpe 1.49, drawdown -18.8%).

## Every recipe tried

| recipe | CAGR | vol | Sharpe | max DD | Sharpe at 2x costs | P(losing) | deflated | BTC beta | alpha t | eligible |
|---|---|---|---|---|---|---|---|---|---|---|
| **R7_stack_wide_all** | 11.0% | 8.9% | 1.21 | -8.7% | 0.68 | 4% | 58% | 0.01 | 1.9 | yes |
| R5_stack_wide_long | 13.7% | 11.4% | 1.18 | -10.4% | 0.65 | 4% | 56% | -0.00 | 1.9 | yes |
| R3_stack_carry_core | 10.0% | 11.1% | 0.91 | -9.8% | 0.53 | 6% | 39% | -0.02 | 1.6 | yes |
| R2_stack | 8.8% | 11.5% | 0.79 | -12.0% | 0.46 | 9% | 32% | -0.02 | 1.4 | yes |
| R1_steady | 8.0% | 10.6% | 0.78 | -11.8% | 0.50 | 10% | 32% | -0.01 | 1.3 | yes |
| R6_stack_wide_macro | 4.8% | 8.2% | 0.62 | -9.2% | -0.02 | 14% | 23% | -0.04 | 1.3 | no |
| R4_stack_wide | 5.9% | 10.3% | 0.61 | -12.2% | 0.00 | 15% | 22% | -0.05 | 1.4 | yes |
| R0_baseline | 3.1% | 10.6% | 0.34 | -16.2% | -0.06 | 32% | 12% | -0.03 | 0.8 | no |

Research window only. "Deflated" is the probability the Sharpe ratio is above what the best of 13 no-skill variants would show by luck.

## What each change was worth

* **a slower allocator with no trailing vote** (R1_steady against R0_baseline): Sharpe +0.45, return +5.0 points a year, t 1.2. It helped, but the difference could be chance.
* **carry as the core with a directional overlay (capital stacking)** (R2_stack against R1_steady): Sharpe +0.01, return +0.8 points a year, t 0.8. It made no real difference.
* **a carry hurdle set by the cost of cash instead of 13%** (R3_stack_carry_core against R2_stack): Sharpe +0.12, return +1.2 points a year, t 1.7. It helped, but the difference could be chance.
* **25 coins instead of 10** (R4_stack_wide against R3_stack_carry_core): Sharpe -0.31, return -4.2 points a year, t -1.3. It made it worse.
* **adding long-only trend (market exposure)** (R5_stack_wide_long against R4_stack_wide): Sharpe +0.58, return +7.8 points a year, t 2.4. It helped, and the difference is unlikely to be chance.
* **adding trend on 20 ETFs across asset classes** (R6_stack_wide_macro against R4_stack_wide): Sharpe +0.01, return -1.0 points a year, t -0.7. It made no real difference.
* **adding both long-only trend and multi-asset trend** (R7_stack_wide_all against R4_stack_wide): Sharpe +0.61, return +5.1 points a year, t 1.5. It helped, but the difference could be chance.

## What to be careful about

* The champion lost money in the vault. Its research record did not travel to the most recent six months. Treat it as unproven.
* After counting 13 trials, the champion's Sharpe is not statistically established (deflated probability 58%; 90% is the bar).
* Alpha against BTC has a t-statistic of 1.9. Below 2, it cannot be told apart from chance.
* The long-only trend sleeve earns by owning crypto while it rises and standing aside when it falls. That is market exposure with a brake, not an edge: in a long decline it earns nothing, and in a choppy one it loses.
* The 25-coin universe is a list of coins that survived. Results on it are a little better than a trader could have had at the time.
* The core is spot-against-perpetual carry. Its real risks, an exchange failing or spot and perpetual prices pulling apart, do not appear in its volatility or its drawdown here.
* The multi-asset sleeve assumes margined instruments (futures or perpetuals on the same markets); ETF history is the proxy and the paper loop marks it at daily closes.
* Sleeves that were switched on in under a fifth of months and add little: pairs, seasonality.
* Cost-fragile recipes (Sharpe falls by more than 0.6 at double costs): R4_stack_wide, R6_stack_wide_macro.

## Is paper trading agreeing with the model?

* 11 cycles over 0.4 days. Equity 10,000.00 to 9,972.58 (-0.27%). WAIT in 0% of cycles. Fees paid 11.73.
* Resting orders: 67 posted, 0 filled, 0 chased. Too few to judge the fill model; it needs at least a few dozen.
* The paper record is 0 days old. Nothing can be read from its return yet; the first honest comparison with the model is at about 60 days.

## What to do next

1. The scarce thing now is evidence, not ideas. Leave the champion running in paper for 60 days without touching a threshold; that record is the only test left that has not been looked at.
2. Carry is carrying weight. Before real money: a second venue to split exchange risk, and a model of the spot-perpetual gap in a violent rally.
3. Give the experimental slot the CTO's solver (`entry="module:function"`). It is judged by this same lab, against this champion.
4. To try an idea: write the rule and the hypothesis into `itrader/online/lab/registry.py` first, commit it, then run `python -m itrader.online lab`. Every recipe added is one more trial counted against all of them.

## Champion history

* 2026-11-05T09:34: R7_stack_wide_all (first champion: highest deflated Sharpe among eligible recipes)

This report is generated from measured results by fixed rules. It describes software tested on historical data; it is not financial advice.
