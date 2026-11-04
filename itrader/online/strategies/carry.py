"""Funding-rate carry: own the coin in spot, short the same amount in the perpetual, and collect
the funding that longs pay shorts. Market-neutral; the risk is that funding turns negative or
the exchange has a problem. We hold the trade only where the trailing funding, annualised, is
above a hurdle after costs, and size by the expected carry so the best-paid coins get more.

Expected utility here is unusually honest: it is the funding you were paid last week, less the
round-trip cost, with no price forecast at all.
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
    description = "Spot long / perp short where trailing funding beats a hurdle; sized by expected carry."
    params = {"lookback_days": 7, "hurdle_ann": 0.08, "max_single": 0.34, "round_trip_cost": 0.0012}
    n_variants_tried = 1

    def targets(self, m: Market) -> pd.DataFrame:
        fd = m.funding_daily()
        c = m.close
        w = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        if fd is None:
            return w
        fd = fd.reindex(c.index).fillna(0.0)
        p = self.params
        ann = fd.rolling(p["lookback_days"], min_periods=p["lookback_days"]).mean() * 365.0
        # weekly turnover means the round-trip cost is amortised over ~7 days
        hurdle = p["hurdle_ann"] + p["round_trip_cost"] * 365.0 / 7.0
        edge = (ann - hurdle).clip(lower=0.0)
        g = edge.sum(axis=1).replace(0, np.nan)
        w = edge.div(g, axis=0).fillna(0.0).clip(upper=p["max_single"])
        # the weight is the spot-long notional; the perp short is the same size (the backtester
        # treats instrument "spot+perp" as price-neutral and credits funding on the short leg)
        return clean(w)
