"""Pre-registered experiments.

This file was written and committed BEFORE any recipe in it was evaluated, so that the rules could
not be bent to fit the results. Each recipe states a hypothesis, the sleeves it uses, how capital is
allocated, and nothing else may be tuned after the fact. Adding a recipe later is allowed; it is one
more trial and is counted as such against every result (see `CRITERIA`).

Policies (implemented in `policy.py`):

* workbench  the allocator as built in the second round: weekly, capability scores with a trailing
             six-month vote, capital shared out between sleeves. The control.
* steady     the same sleeves, a slower hand: a sleeve is on when the probability its whole record is
             better than nothing reaches `gate_on` and off below `gate_off`; equal risk between the
             sleeves that are on; monthly; an exposure is only changed when it has moved by more than
             `exposure_band`. No trailing vote, no softmax.
* stack      capital stacking. Market-neutral carry is the core and holds the capital (its own spot
             notional, at most 80% of equity). Directional sleeves are an overlay on perpetuals, which
             need margin, not capital; they are sized by risk to the portfolio's volatility target with
             the "steady" rules, and scaled down if the gap rule's budget would be exceeded.

The numbers in `[lab]` of settings.toml (monthly, 0.55 / 0.50, 90 and 180 days, 20% band) were fixed
with this file and are not searched over.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

CLASSIC = ("turtle", "trend_modern", "xs_momentum", "mean_reversion", "pairs", "seasonality")


@dataclass(frozen=True)
class Recipe:
    name: str
    hypothesis: str
    policy: str                      # workbench | steady | stack
    universe: str                    # core10 | wide25
    overlay: Tuple[str, ...]         # directional sleeves (all sleeves, for workbench and steady)
    core: Tuple[str, ...] = ()       # market-neutral core sleeves (stack only; for the others they join the common pool)
    label: str = ""                  # an honest tag shown wherever the result is shown

    @property
    def sleeves(self) -> Tuple[str, ...]:
        return tuple(self.core) + tuple(self.overlay)


RECIPES: List[Recipe] = [
    Recipe("R0_baseline",
           "Control: the system exactly as it ran after the second round.",
           "workbench", "core10", CLASSIC + ("funding_carry",)),
    Recipe("R1_steady",
           "The allocator loses what the sleeves earn by re-weighting weekly on a trailing six-month vote (performance chasing). "
           "A slower rule with no vote should keep more of it.",
           "steady", "core10", CLASSIC + ("funding_carry",)),
    Recipe("R2_stack",
           "Carry is market neutral and needs the capital; directional sleeves need only margin. Making them compete for the same "
           "capital wastes it. Carry as the core with a directional overlay should add the two returns together.",
           "stack", "core10", CLASSIC, core=("funding_carry",)),
    Recipe("R3_stack_carry_core",
           "The carry hurdle (13%) is above the perpetual's baseline funding (about 11%), so carry sits out most of the time. A hurdle "
           "at the opportunity cost of cash plus amortised costs (about 6%), averaged over two weeks, should earn the baseline.",
           "stack", "core10", CLASSIC, core=("carry_core",)),
    Recipe("R4_stack_wide",
           "Ten coins give carry few slots and trend little breadth. Twenty-five give carry more, richer and more diversified slots "
           "and trend a few more partly independent bets.",
           "stack", "wide25", CLASSIC, core=("carry_core",), label="survivor universe"),
    Recipe("R5_stack_wide_long",
           "In crypto the short side of trend following has lost money. Adding a long-only trend sleeve should raise return. "
           "This is market exposure with a brake, not edge.",
           "stack", "wide25", CLASSIC + ("trend_long",), core=("carry_core",), label="adds market exposure; survivor universe"),
    Recipe("R6_stack_wide_macro",
           "Trend following is paid for breadth. Adding the same rule on twenty ETFs across asset classes, which barely co-move with "
           "crypto, should raise the Sharpe ratio more than any change to the crypto sleeves can.",
           "stack", "wide25", CLASSIC + ("macro_trend",), core=("carry_core",), label="survivor universe"),
    Recipe("R7_stack_wide_all",
           "All three additions together: wider universe, long-only trend, multi-asset trend.",
           "stack", "wide25", CLASSIC + ("trend_long", "macro_trend"), core=("carry_core",), label="adds market exposure; survivor universe"),
]

# How a winner is chosen. Fixed with the recipes.
CRITERIA: Dict[str, str] = {
    "window": "Walk-forward, every day out of sample. The research window runs from one year after the data starts to the start of the vault. "
              "The vault is the most recent `vault_days`; it is not used to choose anything.",
    "primary": "Sharpe ratio on the research window, deflated for the number of trials: every recipe ever evaluated plus `prior_variants`.",
    "eligible": "Maximum drawdown no worse than `max_drawdown`; block-bootstrap chance of a losing record at most `max_p_losing`; "
                "Sharpe at double costs above `min_cost2x_sharpe`.",
    "champion": "The eligible recipe with the highest deflated Sharpe. If none is eligible there is no champion and the book stays as it was.",
    "challenger": "Later, a recipe replaces the champion only if it is eligible, has a higher deflated Sharpe, and its daily returns on the "
                  "research window exceed the champion's with a t-statistic of at least `switch_t`.",
    "vault": "Opened once, for the champion, and reported whatever it shows. A champion that loses money in the vault is marked unconfirmed.",
    "truth": "None of this is proof. The only clean test left is the paper record from here forward.",
}


def by_name(name: str) -> Recipe:
    for r in RECIPES:
        if r.name == name:
            return r
    raise KeyError(name)
