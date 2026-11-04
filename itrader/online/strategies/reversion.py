"""Short-term mean reversion inside the trend, and pairs trading.

MeanReversion (hourly): buy a sharp dip in a coin whose daily trend is up, sell a sharp spike in
one whose daily trend is down. The dip is measured as a z-score of the 24-hour return against
its own recent distribution. Hold until the z-score crosses back through zero or 48 bars pass.
The trend filter is what keeps this from catching falling knives; without it the strategy
loses, which the bench will show if you switch it off.

Pairs (hourly): two coins that move together (BTC/ETH, ETH/SOL...). Fit the hedge ratio on a
rolling window of log prices, trade the spread when its z-score stretches past 2, exit at 0.5,
stop at 4. Market-neutral by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Market, Strategy, clean, inverse_vol_weights
from ..data.universe import DEFAULT_PAIRS


class MeanReversion(Strategy):
    name = "mean_reversion"
    family_id = "13"
    bar = "1h"
    instrument = "perp"
    description = "Buy 24h dips (z < -2) in daily uptrends, sell spikes in downtrends; exit at z=0 or 48 bars."
    params = {"ret_bars": 24, "z_window": 24 * 30, "entry_z": 2.0, "exit_z": 0.0, "max_hold": 48, "trend_days": 50,
              "vol_span": 24 * 5, "trend_filter": True}
    n_variants_tried = 2  # with and without the trend filter were both looked at

    def targets(self, m: Market) -> pd.DataFrame:
        c = m.close
        p = self.params
        r = c.pct_change(p["ret_bars"])
        z = (r - r.rolling(p["z_window"], min_periods=p["z_window"] // 2).mean()) / r.rolling(p["z_window"], min_periods=p["z_window"] // 2).std()
        if p["trend_filter"] and m.daily_close is not None:
            d = m.daily_close
            trend_d = np.sign(d - d.rolling(p["trend_days"]).mean())
            # Daily bars are indexed by OPEN time and hold that day's close, so the row for day D is only known
            # at the end of D. Shift by one DAY before spreading onto the hourly index: every hour of day D sees
            # the trend as of the close of D-1. (A one-hour shift here was a 23-hour lookahead.)
            trend = trend_d.shift(1).reindex(c.index, method="ffill")
        else:
            trend = pd.DataFrame(1.0, index=c.index, columns=c.columns)
        sig = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        zv, tv = z.to_numpy(), trend.reindex(columns=c.columns).to_numpy()
        out = np.zeros_like(zv)
        for j in range(zv.shape[1]):
            state, held = 0.0, 0
            for i in range(zv.shape[0]):
                zz, tt = zv[i, j], tv[i, j]
                if not np.isfinite(zz):
                    out[i, j] = state
                    continue
                if state == 0.0:
                    if zz < -p["entry_z"] and (tt > 0 or not p["trend_filter"]):
                        state, held = 1.0, 0
                    elif zz > p["entry_z"] and (tt < 0 or not p["trend_filter"]):
                        state, held = -1.0, 0
                else:
                    held += 1
                    if (state > 0 and zz >= p["exit_z"]) or (state < 0 and zz <= -p["exit_z"]) or held >= p["max_hold"]:
                        state = 0.0
                out[i, j] = state
        sig[:] = out
        return clean(inverse_vol_weights(sig, m.vol(p["vol_span"]), gross=1.0, max_single=0.5))


class Pairs(Strategy):
    name = "pairs"
    family_id = "15"
    bar = "1h"
    instrument = "perp"
    description = "Rolling-hedge spread on co-moving pairs; enter |z|>2, exit |z|<0.5, stop |z|>4."
    params = {"window": 24 * 20, "entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0, "max_hold": 24 * 10, "pairs": DEFAULT_PAIRS}
    n_variants_tried = 1

    def targets(self, m: Market) -> pd.DataFrame:
        c = m.close
        p = self.params
        lp = np.log(c)
        w = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        n_pairs = sum(1 for a, b in p["pairs"] if a in c.columns and b in c.columns)
        if n_pairs == 0:
            return w
        for a, b in p["pairs"]:
            if a not in c.columns or b not in c.columns:
                continue
            x, y = lp[b], lp[a]
            cov = x.rolling(p["window"]).cov(y)
            var = x.rolling(p["window"]).var()
            beta = (cov / var.replace(0, np.nan))
            spread = y - beta * x
            z = (spread - spread.rolling(p["window"]).mean()) / spread.rolling(p["window"]).std()
            zv = z.to_numpy()
            state, held = 0.0, 0
            pos = np.zeros(len(zv))
            for i in range(len(zv)):
                zz = zv[i]
                if not np.isfinite(zz):
                    pos[i] = state
                    continue
                if state == 0.0:
                    if zz > p["entry_z"]:
                        state, held = -1.0, 0   # spread rich: short a, long b
                    elif zz < -p["entry_z"]:
                        state, held = 1.0, 0
                else:
                    held += 1
                    if abs(zz) < p["exit_z"] or abs(zz) > p["stop_z"] or held >= p["max_hold"]:
                        state = 0.0
                pos[i] = state
            pos_s = pd.Series(pos, index=c.index)
            # dollar-neutral legs, each pair gets an equal slice of gross
            leg = 0.5 / n_pairs
            w[a] = w[a] + pos_s * leg
            w[b] = w[b] - pos_s * leg * np.sign(beta.fillna(1.0)).clip(lower=0.0).replace(0, 1.0)
        return clean(w)
