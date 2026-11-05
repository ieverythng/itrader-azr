"""Funding-rate carry: own the coin in spot, short the same amount in the perpetual, and collect
the funding that longs pay shorts. Market-neutral; the risks are that funding turns negative,
that spot and perp drift apart (basis), and the exchange itself.

A coin is switched on when its trailing funding, annualised, clears a hurdle after costs, and
switched off only when it falls below half the hurdle, so the book does not flap around the
line. Each coin that is on gets a fixed slot, not a size that changes every day with the
funding print. The spot notional is capped at 80% of equity because the perp short needs
margin too: a carry book of 1.0 would need more than the account holds.

The cost used in the hurdle is the engine's own round trip for the two legs (about 42 bps: spot
and perp fees, spread and slippage on both, in and out). Where funding data is missing (the
exchange archive runs behind) the coin is treated as off: no data, no position.

Expected utility here is unusually honest: it is the funding you were paid last week, less the
cost of the round trip, with no price forecast at all. (`n_variants_tried` is 2: sizing and
costs were corrected once after an independent audit.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Market, Strategy, clean


class FundingCarry(Strategy):
    name = "funding_carry"
    family_id = "14"
    bar = "1d"
    instrument = "spot+perp"
    description = "Spot long / perp short where trailing funding beats a hurdle; fixed slots, hysteresis, at most 80% of equity."
    params = {"lookback_days": 7, "hurdle_ann": 0.08, "round_trip_cost": 0.0042, "hold_days_assumed": 30, "exit_fraction": 0.5,
              "gross": 0.80, "min_slots": 3, "max_single": 0.34}
    n_variants_tried = 2

    def targets(self, m: Market) -> pd.DataFrame:
        fd = m.funding_daily()
        c = m.close
        w = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        if fd is None:
            return w
        p = self.params
        fd = fd.reindex(index=c.index, columns=c.columns)                 # missing days stay NaN: no data, no position
        ann = (fd.rolling(p["lookback_days"], min_periods=p["lookback_days"]).mean() * 365.0).to_numpy()
        hurdle = p["hurdle_ann"] + p["round_trip_cost"] * 365.0 / p["hold_days_assumed"]
        on = np.zeros(ann.shape, dtype=bool)
        for j in range(ann.shape[1]):
            state = False
            for i in range(ann.shape[0]):
                a = ann[i, j]
                if not np.isfinite(a):
                    state = False
                elif state:
                    state = a > hurdle * p["exit_fraction"]
                else:
                    state = a > hurdle
                on[i, j] = state
        n_on = on.sum(axis=1)
        slot = np.minimum(p["max_single"], p["gross"] / np.maximum(n_on, p["min_slots"]))
        # the weight is the spot-long notional; the perp short is the same size (the backtester treats
        # "spot+perp" as price-neutral, credits funding on the short leg and charges both legs' costs)
        return clean(pd.DataFrame(on * slot[:, None], index=c.index, columns=c.columns))
