"""Trend following, in its original and its modern form.

Turtle (1983): Dennis and Eckhardt's rules as taught to the Turtles. Enter on a 20-day breakout
(System 1) or a 55-day breakout (System 2), size each unit by N (the 20-day ATR) so one unit
risks about 1% of equity, exit on a 10-day (S1) or 20-day (S2) reverse breakout, and stop at
2N. System 1 skips a breakout if the previous one would have won. We run both systems with
half the capital each, as the Turtles did.

TrendModern: what systematic trend funds run today (Carver's "Systematic Trading", AQR's
"A Century of Evidence on Trend-Following"). Several EWMA crossover speeds averaged into one
continuous forecast, scaled by volatility so the position changes smoothly instead of all-in
or all-out, capped, and with a no-trade buffer so small forecast changes do not pay costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Market, Strategy, clean, inverse_vol_weights


class Turtle(Strategy):
    name = "turtle"
    family_id = "10"
    bar = "1d"
    instrument = "perp"
    description = "Original Turtle rules: 20/55-day breakouts, N-sized units, 2N stop, 10/20-day exits."
    params = {"s1_entry": 20, "s1_exit": 10, "s2_entry": 55, "s2_exit": 20, "atr": 20, "risk_per_unit": 0.01,
              "max_units": 4, "stop_n": 2.0}
    n_variants_tried = 1  # the published rules, not optimised

    def _system(self, m: Market, entry: int, exit_: int, skip_after_winner: bool) -> pd.DataFrame:
        c, h, l = m.close, m.high, m.low
        tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()]).groupby(level=0).max()
        n = tr.ewm(alpha=1 / self.params["atr"], min_periods=self.params["atr"]).mean()
        hi_e, lo_e = c.rolling(entry).max().shift(), c.rolling(entry).min().shift()
        hi_x, lo_x = c.rolling(exit_).max().shift(), c.rolling(exit_).min().shift()
        pos = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        units = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        for s in c.columns:
            state, entry_px, stop, last_trade_won, u = 0, np.nan, np.nan, False, 0
            cs, hs, ls, ns = c[s].to_numpy(), hi_e[s].to_numpy(), lo_e[s].to_numpy(), n[s].to_numpy()
            hx, lx = hi_x[s].to_numpy(), lo_x[s].to_numpy()
            col = np.zeros(len(cs))
            ucol = np.zeros(len(cs))
            for i in range(len(cs)):
                px, nn = cs[i], ns[i]
                if not np.isfinite(nn) or nn <= 0:
                    continue
                if state == 0:
                    brk_up, brk_dn = px > hs[i], px < ls[i]
                    if (brk_up or brk_dn) and (not skip_after_winner or not last_trade_won):
                        state = 1 if brk_up else -1
                        entry_px, stop, u = px, px - state * self.params["stop_n"] * nn, 1
                    elif (brk_up or brk_dn) and skip_after_winner and last_trade_won:
                        last_trade_won = False  # the skipped breakout resets the filter
                else:
                    # pyramid every 0.5N in our favour, up to max_units
                    if u < self.params["max_units"] and state * (px - entry_px) > 0.5 * nn * u:
                        u += 1
                        stop = px - state * self.params["stop_n"] * nn
                    exited = (state == 1 and (px < lx[i] or px < stop)) or (state == -1 and (px > hx[i] or px > stop))
                    if exited:
                        last_trade_won = state * (px - entry_px) > 0
                        state, u = 0, 0
                col[i], ucol[i] = state, u
            pos[s], units[s] = col, ucol
        # unit size: risk_per_unit of equity per N move -> weight = risk / (N / price) per unit
        w = pos * units * self.params["risk_per_unit"] / (n / c).replace(0, np.nan)
        return clean(w)

    def targets(self, m: Market) -> pd.DataFrame:
        p = self.params
        w = 0.5 * self._system(m, p["s1_entry"], p["s1_exit"], True) + 0.5 * self._system(m, p["s2_entry"], p["s2_exit"], False)
        gross = w.abs().sum(axis=1)
        over = gross > 1.0
        w.loc[over] = w.loc[over].div(gross[over], axis=0)
        return clean(w)


class TrendModern(Strategy):
    name = "trend_modern"
    family_id = "11"
    bar = "1d"
    instrument = "perp"
    description = "Three EWMA crossover speeds averaged, vol-scaled continuous forecast, 10% no-trade buffer."
    params = {"speeds": ((8, 32), (16, 64), (32, 128)), "vol_span": 36, "forecast_cap": 2.0, "buffer": 0.10, "max_single": 0.30}
    n_variants_tried = 1  # the standard speeds from the literature

    def targets(self, m: Market) -> pd.DataFrame:
        c = m.close
        vol_px = c.pct_change().ewm(span=self.params["vol_span"], min_periods=12).std() * c  # price vol in price units
        fcs = []
        for fast, slow in self.params["speeds"]:
            raw = (c.ewm(span=fast, min_periods=fast).mean() - c.ewm(span=slow, min_periods=slow).mean()) / vol_px.replace(0, np.nan)
            # scale each speed so its average absolute forecast is 1 (Carver), estimated on a trailing window
            scalar = 1.0 / raw.abs().rolling(250, min_periods=60).mean().replace(0, np.nan)
            fcs.append((raw * scalar).clip(-self.params["forecast_cap"], self.params["forecast_cap"]))
        fc = sum(fcs) / len(fcs)
        fc = fc.where(np.isfinite(fc), 0.0)
        w = inverse_vol_weights(fc / self.params["forecast_cap"], m.vol(self.params["vol_span"]), gross=1.0, max_single=self.params["max_single"])
        return clean(apply_buffer(w, self.params["buffer"]))


def apply_buffer(w: pd.DataFrame, buffer: float) -> pd.DataFrame:
    """Only move the position when the target has drifted more than `buffer` x its size.
    Cuts turnover (and costs) sharply with almost no loss of signal."""
    if buffer <= 0:
        return w
    out = w.copy()
    vals = w.to_numpy()
    held = np.zeros(vals.shape[1])
    res = np.zeros_like(vals)
    for i in range(vals.shape[0]):
        tgt = vals[i]
        band = np.maximum(np.abs(tgt) * buffer, 0.005)
        move = np.abs(tgt - held) > band
        held = np.where(move, tgt, held)
        res[i] = held
    out[:] = res
    return out
