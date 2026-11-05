# Advisor report, 05 November 2026 10:14 UTC

## Where the system stands

**Champion: `R8_stack_core_long`** (stack policy, core10, adds market exposure; registered after seeing earlier results). challenger R8_stack_core_long replaces R7_stack_wide_all: t-statistic 1.50.

* Research window (2023-11-06 to 2026-05-07, every day out of sample): **17.0% a year at 12.8% volatility, Sharpe 1.30**, worst drawdown -8.2%, worst day -3.7%.
* Deflated for the 14 variants tried so far, the probability this is better than nothing is 62%.
* Vault (2026-05-07 onward, never used to choose): -1.9% in total, Sharpe -0.22, worst drawdown -5.8%. **NOT confirmed**.
* At double the assumed costs: Sharpe 1.00. Chance the research record is a losing one: 2%.
* Against BTC: beta 0.02, alpha 15.5% a year (t 1.9). Against the S&P 500: beta -0.04, alpha 17.5% a year (t 2.2).

Benchmarks over the same research window: btc 40.1% a year at 48.4% vol (Sharpe 0.94, drawdown -49.5%); btc vol matched 9.9% a year at 12.8% vol (Sharpe 0.80, drawdown -19.6%); spy 24.8% a year at 15.7% vol (Sharpe 1.49, drawdown -18.8%).

## Every recipe tried

| recipe | CAGR | vol | Sharpe | max DD | Sharpe at 2x costs | P(losing) | deflated | BTC beta | alpha t | eligible |
|---|---|---|---|---|---|---|---|---|---|---|
| **R8_stack_core_long** | 17.0% | 12.8% | 1.30 | -8.2% | 1.00 | 2% | 62% | 0.02 | 1.9 | yes |
| R7_stack_wide_all | 11.0% | 8.9% | 1.21 | -8.7% | 0.68 | 4% | 57% | 0.01 | 1.9 | yes |
| R5_stack_wide_long | 13.7% | 11.4% | 1.18 | -10.4% | 0.65 | 4% | 55% | -0.00 | 1.9 | yes |
| R3_stack_carry_core | 10.0% | 11.1% | 0.91 | -9.8% | 0.53 | 6% | 38% | -0.02 | 1.6 | yes |
| R2_stack | 8.8% | 11.5% | 0.79 | -12.0% | 0.46 | 9% | 31% | -0.02 | 1.4 | yes |
| R1_steady | 8.0% | 10.6% | 0.78 | -11.8% | 0.50 | 10% | 31% | -0.01 | 1.3 | yes |
| R6_stack_wide_macro | 4.8% | 8.2% | 0.62 | -9.2% | -0.02 | 14% | 22% | -0.04 | 1.3 | no |
| R4_stack_wide | 5.9% | 10.3% | 0.61 | -12.2% | 0.00 | 15% | 21% | -0.05 | 1.4 | yes |
| R0_baseline | 3.1% | 10.6% | 0.34 | -16.2% | -0.06 | 32% | 11% | -0.03 | 0.8 | no |

Research window only. "Deflated" is the probability the Sharpe ratio is above what the best of 14 no-skill variants would show by luck.

## What each change was worth

* **a slower allocator with no trailing vote** (R1_steady against R0_baseline): Sharpe +0.45, return +5.0 points a year, t 1.2. It helped, but the difference could be chance.
* **carry as the core with a directional overlay (capital stacking)** (R2_stack against R1_steady): Sharpe +0.01, return +0.8 points a year, t 0.8. It made no real difference.
* **a carry hurdle set by the cost of cash instead of 13%** (R3_stack_carry_core against R2_stack): Sharpe +0.12, return +1.2 points a year, t 1.7. It helped, but the difference could be chance.
* **25 coins instead of 10** (R4_stack_wide against R3_stack_carry_core): Sharpe -0.31, return -4.2 points a year, t -1.3. It made it worse.
* **adding long-only trend (market exposure)** (R5_stack_wide_long against R4_stack_wide): Sharpe +0.58, return +7.8 points a year, t 2.4. It helped, and the difference is unlikely to be chance.
* **adding trend on 20 ETFs across asset classes** (R6_stack_wide_macro against R4_stack_wide): Sharpe +0.01, return -1.0 points a year, t -0.7. It made no real difference.
* **adding both long-only trend and multi-asset trend** (R7_stack_wide_all against R4_stack_wide): Sharpe +0.61, return +5.1 points a year, t 1.5. It helped, but the difference could be chance.
* **adding long-only trend on the ten majors (registered after seeing R0 to R7)** (R8_stack_core_long against R3_stack_carry_core): Sharpe +0.38, return +7.0 points a year, t 2.3. It helped, and the difference is unlikely to be chance.
* **ten coins instead of 25, with long-only trend in both** (R8_stack_core_long against R5_stack_wide_long): Sharpe +0.11, return +3.3 points a year, t 1.0. It helped, but the difference could be chance.

## What to be careful about

* The champion lost money in the vault. Its research record did not travel to the most recent six months. Treat it as unproven.
* After counting 14 trials, the champion's Sharpe is not statistically established (deflated probability 62%; 90% is the bar).
* Alpha against BTC has a t-statistic of 1.9. Below 2, it cannot be told apart from chance.
* The long-only trend sleeve earns by owning crypto while it rises and standing aside when it falls. That is market exposure with a brake, not an edge: in a long decline it earns nothing, and in a choppy one it loses.
* The core is spot-against-perpetual carry. Its real risks, an exchange failing or spot and perpetual prices pulling apart, do not appear in its volatility or its drawdown here.
* Sleeves that were switched on in under a fifth of months and add little: pairs, seasonality.
* Cost-fragile recipes (Sharpe falls by more than 0.6 at double costs): R4_stack_wide, R6_stack_wide_macro.

## Is paper trading agreeing with the model?

* 13 cycles over 0.4 days. Equity 10,000.00 to 9,971.29 (-0.29%). WAIT in 0% of cycles. Fees paid 13.55.
* Resting orders: 86 posted, 0 filled as maker, 19 chased as taker: a fill rate of 0% against the 93 to 99% the bench assumes. That is BELOW the model: the bench is flattering the maker style and its numbers should be read down.
* The paper record is 0 days old. Nothing can be read from its return yet; the first honest comparison with the model is at about 60 days.

## What to do next

1. The scarce thing now is evidence, not ideas. Leave the champion running in paper for 60 days without touching a threshold; that record is the only test left that has not been looked at.
2. Carry is carrying weight. Before real money: a second venue to split exchange risk, and a model of the spot-perpetual gap in a violent rally.
3. Give the experimental slot the CTO's solver (`entry="module:function"`). It is judged by this same lab, against this champion.
4. To try an idea: write the rule and the hypothesis into `itrader/online/lab/registry.py` first, commit it, then run `python -m itrader.online lab`. Every recipe added is one more trial counted against all of them.

## Champion history

* 2026-11-05T09:34: R7_stack_wide_all (first champion: highest deflated Sharpe among eligible recipes)
* 2026-11-05T10:14: R8_stack_core_long (challenger R8_stack_core_long replaces R7_stack_wide_all: t-statistic 1.50)

This report is generated from measured results by fixed rules. It describes software tested on historical data; it is not financial advice.
