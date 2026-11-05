"""Short-term mean reversion inside the trend, and pairs trading.

MeanReversion (hourly): buy a sharp dip in a coin whose daily trend is up, sell a sharp spike in
one whose daily trend is down. The dip is measured as a z-score of the 24-hour return against
its own recent distribution. Hold until the z-score crosses back through zero or 48 bars pass.
A position's size is set once, at entry (its inverse-volatility share of the universe times a
constant), and held: re-sizing every hour as volatility moves, and every time another coin
enters or leaves, only pays the venue.

Pairs (hourly): two coins that move together (BTC/ETH, ETH/SOL...). Fit the hedge ratio on a
rolling window of log prices, trade the spread when its z-score stretches past 2, exit at 0.5,
stop at 4. The legs are sized by the hedge ratio at entry, so the position is the spread whose
z-score triggered it. After a stop or a time-out the pair is blocked until the spread has come
back inside the exit band; without that the stop is no stop, because the next bar re-enters.

Daily data inside an hourly strategy. Daily bars are indexed by open time and hold that day's
close, so a daily value is only known a day after its label. The trend series is therefore
re-labelled to the time it becomes available (label + 1 day) before it is spread onto the
hourly index. An earlier version shifted by one hour, a 23-hour lookahead that made this
strategy look like the best in the book.

Honesty counter: MeanReversion has been looked at with and without the trend filter and with
two sizing rules (3 variants); Pairs with two leg-sizing rules (2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Market, Strategy, clean
from ..data.universe import DEFAULT_PAIRS


class MeanReversion(Strategy):
    name = "mean_reversion"
    family_id = "13"
    bar = "1h"
    instrument = "perp"
    description = "Buy 24h dips (z < -2) in daily uptrends, sell spikes in downtrends; size fixed at entry; exit at z=0 or 48 bars."
    params = {"ret_bars": 24, "z_window": 24 * 30, "entry_z": 2.0, "exit_z": 0.0, "max_hold": 48, "trend_days": 50,
              "vol_span": 24 * 5, "trend_filter": True, "unit_scale": 3.0, "max_single": 0.35}
    n_variants_tried = 3

    def targets(self, m: Market) -> pd.DataFrame:
        c = m.close
        p = self.params
        r = c.pct_change(p["ret_bars"])
        z = (r - r.rolling(p["z_window"], min_periods=p["z_window"] // 2).mean()) / r.rolling(p["z_window"], min_periods=p["z_window"] // 2).std()
        if p["trend_filter"] and m.daily_close is not None:
            d = m.daily_close
            trend_d = np.sign(d - d.rolling(p["trend_days"]).mean())
            trend_d.index = trend_d.index + pd.Timedelta(days=1)        # the time each daily value becomes known
            trend = trend_d.reindex(c.index, method="ffill")
        else:
            trend = pd.DataFrame(1.0, index=c.index, columns=c.columns)
        vol = m.vol(p["vol_span"])
        inv = 1.0 / vol.replace(0, np.nan)
        share = (inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0) * p["unit_scale"]).clip(upper=p["max_single"]).to_numpy()
        zv, tv = z.to_numpy(), trend.reindex(columns=c.columns).to_numpy()
        out = np.zeros_like(zv)
        for j in range(zv.shape[1]):
            state, held, size = 0.0, 0, 0.0
            for i in range(zv.shape[0]):
                zz, tt = zv[i, j], tv[i, j]
                if np.isfinite(zz):
                    if state == 0.0:
                        sz = share[i, j]
                        if np.isfinite(sz) and sz > 0:
                            if zz < -p["entry_z"] and (tt > 0 or not p["trend_filter"]):
                                state, held, size = 1.0, 0, sz
                            elif zz > p["entry_z"] and (tt < 0 or not p["trend_filter"]):
                                state, held, size = -1.0, 0, sz
                    else:
                        held += 1
                        if (state > 0 and zz >= p["exit_z"]) or (state < 0 and zz <= -p["exit_z"]) or held >= p["max_hold"]:
                            state, size = 0.0, 0.0
                out[i, j] = state * size
        w = pd.DataFrame(out, index=c.index, columns=c.columns)
        gross = w.abs().sum(axis=1)
        over = gross > 1.0
        if over.any():
            w.loc[over] = w.loc[over].div(gross[over], axis=0)
        return clean(w)


class Pairs(Strategy):
    name = "pairs"
    family_id = "15"
    bar = "1h"
    instrument = "perp"
    description = "Rolling-hedge spread on co-moving pairs; enter |z|>2, exit |z|<0.5, stop |z|>4 then wait for the spread to return."
    params = {"window": 24 * 20, "entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0, "max_hold": 24 * 10, "pairs": DEFAULT_PAIRS, "beta_cap": 3.0}
    n_variants_tried = 2

    def targets(self, m: Market) -> pd.DataFrame:
        c = m.close
        p = self.params
        lp = np.log(c)
        w = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        live = [(a, b) for a, b in p["pairs"] if a in c.columns and b in c.columns]
        if not live:
            return w
        leg = 0.5 / len(live)                       # each pair's two legs together use at most 1/n of gross
        for a, b in live:
            x, y = lp[b], lp[a]
            beta = x.rolling(p["window"]).cov(y) / x.rolling(p["window"]).var().replace(0, np.nan)
            spread = y - beta * x
            z = (spread - spread.rolling(p["window"]).mean()) / spread.rolling(p["window"]).std()
            zv, bv = z.to_numpy(), beta.to_numpy()
            state, held, blocked, be = 0.0, 0, False, 0.0
            wa, wb = np.zeros(len(zv)), np.zeros(len(zv))
            for i in range(len(zv)):
                zz = zv[i]
                if np.isfinite(zz):
                    if state == 0.0:
                        if blocked:
                            blocked = abs(zz) >= p["exit_z"]           # wait until the spread is back inside the band
                        elif p["entry_z"] < abs(zz) <= p["stop_z"] and np.isfinite(bv[i]):   # never open beyond the stop level
                            state, held = (-1.0 if zz > 0 else 1.0), 0   # spread rich: short a, long beta of b
                            be = float(np.clip(bv[i], -p["beta_cap"], p["beta_cap"]))
                    else:
                        held += 1
                        if abs(zz) < p["exit_z"]:
                            state = 0.0
                        elif abs(zz) > p["stop_z"] or held >= p["max_hold"]:
                            state, blocked = 0.0, True
                if state != 0.0:
                    scale = 2.0 * leg / (1.0 + abs(be))                # |w_a| + |w_b| = 2 x leg, in the ratio 1 : beta
                    wa[i], wb[i] = state * scale, -state * be * scale
            w[a] = w[a] + wa
            w[b] = w[b] + wb
        return clean(w)
