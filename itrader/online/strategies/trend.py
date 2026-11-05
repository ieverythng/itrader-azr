"""Trend following, in its original and its modern form.

Turtle (1983): Dennis and Eckhardt's rules as taught to the Turtles. Enter when the close
breaks the highest high (or lowest low) of the previous 20 days (System 1) or 55 days (System 2);
size a unit by N, the 20-day ATR at the moment of entry, so one unit risks a fixed share of
equity per N; add a unit every half N in favour, up to four; exit on a 10-day (S1) or 20-day
(S2) reverse breakout, or at 2N from the last fill. System 1 ignores a breakout if the previous
one would have been a winner, whether or not it was taken (the skipped trade is tracked as a
shadow so the filter knows). Both systems run with half the capital each. Unit sizes are frozen
at the fill, as the rules say: a position is not re-sized every day because volatility moved.

TrendModern: what systematic trend funds run today (Carver's "Systematic Trading", AQR's
"A Century of Evidence on Trend-Following"). Several EWMA crossover speeds averaged into one
continuous forecast, each scaled so its average absolute value is one, capped; the position is
the forecast times an inverse-volatility share, so a weak forecast means a small position
and a strong one a full position; and a no-trade buffer so small changes do not pay costs.

Honesty counter: both classes were revised once after an independent audit (breakouts on highs
and lows and frozen unit sizes for Turtle; forecast strength no longer normalised away for
TrendModern), so `n_variants_tried` is 2 for each.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Market, Strategy, clean


class Turtle(Strategy):
    name = "turtle"
    family_id = "10"
    bar = "1d"
    instrument = "perp"
    description = "Original Turtle rules: 20/55-day breakouts of highs and lows, units sized by N at entry, 2N stop, 10/20-day exits."
    # risk_per_unit is a scale only (the portfolio vol-targets the sleeve): 0.25% per N keeps a full book near gross 1
    params = {"s1_entry": 20, "s1_exit": 10, "s2_entry": 55, "s2_exit": 20, "atr": 20, "risk_per_unit": 0.0025,
              "max_units": 4, "stop_n": 2.0, "add_n": 0.5}
    n_variants_tried = 2

    def _system(self, m: Market, entry: int, exit_: int, skip_after_winner: bool) -> pd.DataFrame:
        p = self.params
        c = m.close
        h = m.high if m.high is not None else c
        l = m.low if m.low is not None else c
        tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()]).groupby(level=0).max()
        n = tr.ewm(alpha=1 / p["atr"], min_periods=p["atr"]).mean()
        hi_e, lo_e = h.rolling(entry).max().shift(), l.rolling(entry).min().shift()
        hi_x, lo_x = h.rolling(exit_).max().shift(), l.rolling(exit_).min().shift()
        out = np.zeros(c.shape)
        for j, s in enumerate(c.columns):
            cs, ns = c[s].to_numpy(), n[s].to_numpy()
            he, le, hx, lx = hi_e[s].to_numpy(), lo_e[s].to_numpy(), hi_x[s].to_numpy(), lo_x[s].to_numpy()
            state, shadow, last_won, u = 0, False, False, 0
            last_fill = n_fill = sum_px = unit_w = stop = np.nan
            for i in range(len(cs)):
                px = cs[i]
                if np.isfinite(px):
                    if state == 0:
                        nn = ns[i]
                        if np.isfinite(nn) and nn > 0 and np.isfinite(he[i]) and np.isfinite(le[i]):
                            up, dn = px > he[i], px < le[i]
                            if up or dn:
                                state = 1 if up else -1
                                shadow = bool(skip_after_winner and last_won)   # skipped, but followed to know how it ended
                                n_fill, last_fill, sum_px, u = nn, px, px, 1
                                unit_w = p["risk_per_unit"] * px / nn             # frozen: one unit risks risk_per_unit per N
                                stop = px - state * p["stop_n"] * n_fill
                    else:
                        if u < p["max_units"] and state * (px - last_fill) >= p["add_n"] * n_fill:
                            u += 1
                            last_fill, sum_px = px, sum_px + px
                            stop = px - state * p["stop_n"] * n_fill
                        out_long = state == 1 and (px < lx[i] or px < stop)
                        out_short = state == -1 and (px > hx[i] or px > stop)
                        if out_long or out_short:
                            last_won = state * (px - sum_px / u) > 0
                            state, u, shadow = 0, 0, False
                out[i, j] = 0.0 if (state == 0 or shadow) else state * u * unit_w
        return clean(pd.DataFrame(out, index=c.index, columns=c.columns))

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
    description = "Three EWMA crossover speeds averaged; position = forecast strength x inverse-vol share; 10% no-trade buffer."
    params = {"speeds": ((8, 32), (16, 64), (32, 128)), "vol_span": 36, "forecast_cap": 2.0, "buffer": 0.10, "max_single": 0.30}
    n_variants_tried = 2

    def targets(self, m: Market) -> pd.DataFrame:
        c = m.close
        p = self.params
        vol_px = c.pct_change().ewm(span=p["vol_span"], min_periods=12).std() * c  # price vol in price units
        fcs = []
        for fast, slow in p["speeds"]:
            raw = (c.ewm(span=fast, min_periods=fast).mean() - c.ewm(span=slow, min_periods=slow).mean()) / vol_px.replace(0, np.nan)
            # scale each speed so its average absolute forecast is 1 (Carver), estimated on a trailing window
            scalar = 1.0 / raw.abs().rolling(250, min_periods=60).mean().replace(0, np.nan)
            fcs.append((raw * scalar).clip(-p["forecast_cap"], p["forecast_cap"]))
        fc = sum(fcs) / len(fcs)
        sig = (fc / p["forecast_cap"]).where(np.isfinite(fc), 0.0)          # in [-1, 1]
        vol = m.vol(p["vol_span"])
        inv = 1.0 / vol.replace(0, np.nan)
        share = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0)          # inverse-vol share of the whole universe, sums to 1
        w = (sig * share).fillna(0.0).clip(-p["max_single"], p["max_single"])  # gross = average forecast strength, at most 1
        return clean(apply_buffer(w, p["buffer"]))


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
