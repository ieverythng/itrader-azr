"""Seasonality: calendar and time-of-day effects that recur because of how markets are plumbed,
not because of any forecast.

* Turn of the month: equity indices (and, more weakly, BTC) tend to rise from the last trading
  day of the month through the first three, when pension and payroll flows arrive.
* Time of day: crypto's US-session hours (13:00-21:00 UTC) have carried most of the return
  since 2020; the Asian overnight has carried little or negative return.

These effects are small, so this sleeve is sized small and judged strictly: the bench will drop
it if it does not survive the holdout. It earns its place by not moving with the others.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Market, Strategy, clean


class Seasonality(Strategy):
    name = "seasonality"
    family_id = "16"
    bar = "1h"
    instrument = "perp"
    description = "Long BTC/ETH in US hours (13-21 UTC) and across the turn of the month; flat otherwise."
    params = {"us_hours": (13, 21), "tom_days": (-1, 3), "symbols": ("BTCUSDT", "ETHUSDT"), "gross": 0.5}
    n_variants_tried = 1

    def targets(self, m: Market) -> pd.DataFrame:
        c = m.close
        idx = c.index
        p = self.params
        hour = idx.hour
        day = idx.day
        days_in_month = idx.days_in_month
        us = (hour >= p["us_hours"][0]) & (hour < p["us_hours"][1])
        tom = (day <= p["tom_days"][1]) | (day >= days_in_month + p["tom_days"][0])
        on = pd.Series((us | tom).astype(float), index=idx)
        w = pd.DataFrame(0.0, index=idx, columns=c.columns)
        syms = [s for s in p["symbols"] if s in c.columns]
        for s in syms:
            w[s] = on * p["gross"] / max(1, len(syms))
        return clean(w)
