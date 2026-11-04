"""Cross-sectional momentum: own what has been strongest, avoid (or short) what has been weakest.

Jegadeesh and Titman (1993) for equities; Liu and Tsyvinski (2021) and others find the same
in crypto. We rank the universe on 1 to 3 month returns skipping the most recent week (the
short-term reversal), hold the top third and short the bottom third, rebalance weekly, and
size by inverse volatility. Long-only in spot mode.
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
    description = "Rank on 30/60/90-day returns (skip last 7), long top third, short bottom third, weekly rebalance."
    params = {"lookbacks": (30, 60, 90), "skip": 7, "top_frac": 0.34, "rebalance_days": 7, "vol_span": 30, "long_only": False}
    n_variants_tried = 1

    def targets(self, m: Market) -> pd.DataFrame:
        c = m.close
        p = self.params
        score = sum(c.shift(p["skip"]).pct_change(lb - p["skip"]) for lb in p["lookbacks"]) / len(p["lookbacks"])
        # normalise cross-sectionally each day (z-score of ranks) so the signal is comparable through time
        rank = score.rank(axis=1, pct=True)
        n = rank.notna().sum(axis=1)
        k = (n * p["top_frac"]).round().clip(lower=1)
        sig = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        order = score.rank(axis=1, ascending=False)
        sig[order.le(k, axis=0)] = 1.0
        if not p["long_only"]:
            order_lo = score.rank(axis=1, ascending=True)
            sig[order_lo.le(k, axis=0)] = -1.0
        sig = sig.where(score.notna(), 0.0)
        # rebalance weekly: hold the signal between rebalance days
        idx = np.arange(len(sig))
        keep = (idx % p["rebalance_days"]) == 0
        sig = sig.where(pd.Series(keep, index=sig.index), np.nan).ffill().fillna(0.0)
        w = inverse_vol_weights(sig, m.vol(p["vol_span"]), gross=1.0, max_single=0.35)
        return clean(w)
