"""Seasonality: calendar and time-of-day effects that recur because of how markets are plumbed,
not because of any forecast.

* Turn of the month: equity indices (and, more weakly, BTC) tend to rise from the last day of
  the month through the first three, when pension and payroll flows arrive.
* Time of day: crypto's US-session hours (13:00-21:00 UTC) have carried most of the return
  since 2020; the Asian overnight has carried little or negative return.

The calendar is read off the bar the position is HELD through, not the bar the row is decided
on: row t is held during bar t+1, so row t asks "is bar t+1 inside the window?". (An earlier
version read the calendar off row t and so traded 14:00-22:00, an hour late.)

These effects are small, so this sleeve is sized small and judged strictly: the bench drops it
if it does not survive the holdout. Two honesty notes: the hours were chosen with knowledge of
the 2020 onward record, and the alignment was corrected once, so `n_variants_tried` is 2. And
at taker fees an eight-hour window cannot be harvested by switching in and out every day; the
bench shows that plainly.
"""
from __future__ import annotations

import pandas as pd

from .base import Market, Strategy, clean

_STEP = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "15m": pd.Timedelta(minutes=15), "1d": pd.Timedelta(days=1)}


class Seasonality(Strategy):
    name = "seasonality"
    family_id = "16"
    bar = "1h"
    instrument = "perp"
    description = "Long BTC/ETH in US hours (13-21 UTC) and from the last day of the month through the third; flat otherwise."
    params = {"us_hours": (13, 21), "tom_days": (-1, 3), "symbols": ("BTCUSDT", "ETHUSDT"), "gross": 0.5}
    n_variants_tried = 2

    def targets(self, m: Market) -> pd.DataFrame:
        c = m.close
        idx = c.index
        p = self.params
        held = idx + _STEP.get(m.bar, pd.Timedelta(hours=1))            # open time of the bar this row's weight is held through
        us = (held.hour >= p["us_hours"][0]) & (held.hour < p["us_hours"][1])
        last_n = -int(p["tom_days"][0])                                   # tom_days = (-1, 3): the last 1 day and the first 3
        tom = (held.day <= p["tom_days"][1]) | (held.day > held.days_in_month - last_n)
        on = pd.Series((us | tom).astype(float), index=idx)
        w = pd.DataFrame(0.0, index=idx, columns=c.columns)
        syms = [s for s in p["symbols"] if s in c.columns]
        for s in syms:
            w[s] = on * p["gross"] / max(1, len(syms))
        return clean(w)
