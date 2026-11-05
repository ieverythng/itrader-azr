"""Cross-sectional momentum: own what has been strongest, avoid (or short) what has been weakest.

Jegadeesh and Titman (1993) for equities; Liu and Tsyvinski (2021) and others find the same
in crypto. We rank the universe on 1 to 3 month returns skipping the most recent week (the
short-term reversal), hold the top third and short the bottom third, size by inverse
volatility, and rebalance once a week. Long-only in spot mode.

The whole book is frozen between rebalances: weights are set on Monday's close and held, so
no turnover is generated on the other six days by volatility estimates moving. The rebalance
day is a calendar weekday, not a count from the first row of the data, so the backtest and the
live loop trade on the same days whatever the length of their history. (Both points come from
the independent audit; `n_variants_tried` is 2.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Market, Strategy, clean, inverse_vol_weights


class CrossSectionalMomentum(Strategy):
    name = "xs_momentum"
    family_id = "12"
    bar = "1d"
    instrument = "perp"
    description = "Rank on 30/60/90-day returns (skip last 7), long top third, short bottom third, set on Mondays and held."
    params = {"lookbacks": (30, 60, 90), "skip": 7, "top_frac": 0.34, "rebalance_weekday": 0, "vol_span": 30, "long_only": False}
    n_variants_tried = 2

    def targets(self, m: Market) -> pd.DataFrame:
        c = m.close
        p = self.params
        score = sum(c.shift(p["skip"]).pct_change(lb - p["skip"]) for lb in p["lookbacks"]) / len(p["lookbacks"])
        n = score.notna().sum(axis=1)
        k = (n * p["top_frac"]).round().clip(lower=1)
        sig = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        order = score.rank(axis=1, ascending=False)
        sig[order.le(k, axis=0)] = 1.0
        if not p["long_only"]:
            order_lo = score.rank(axis=1, ascending=True)
            sig[order_lo.le(k, axis=0) & (sig == 0.0)] = -1.0          # never overwrite a long (matters when few names are valid)
        sig = sig.where(score.notna(), 0.0)
        w = inverse_vol_weights(sig, m.vol(p["vol_span"]), gross=1.0, max_single=0.35)
        keep = pd.Series(c.index.dayofweek == int(p["rebalance_weekday"]), index=c.index)
        w = w.where(keep, np.nan).ffill().fillna(0.0)                  # the book set on the rebalance day is held as it is
        return clean(w)
