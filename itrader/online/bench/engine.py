"""Backtesters over target weights: taker (vectorised) and maker with taker fallback.

Conventions (the same ones the live executor follows):

* The weight in row t is decided at the close of bar t and held through bar t+1. The taker
  engine shifts weights by one bar before multiplying by returns, so there is no lookahead by
  construction, whatever a strategy does inside.
* Costs: every change in weight is turnover; each unit of turnover pays fee + half spread +
  slippage per side. "spot+perp" trades two legs, so it pays twice.
* Instruments: "perp" is signed exposure and pays funding when long and the rate is positive
  (receives when short). "spot" is long only and never pays funding. "spot+perp" is price
  neutral: long spot, short the same perp notional, receives funding on the short leg.
* Funding settlements are attributed to the bar that contains them.

Maker with taker fallback (`backtest_maker`, perps only). At the close of bar t the change in
position is posted as a resting limit order at the touch: a buy at close x (1 - half spread), a
sell at close x (1 + half spread). During bar t+1 it fills only if the market trades THROUGH
the level by `maker_through_bps` (a touch is not a fill: we assume we are last in the queue).
A fill pays the maker fee and no spread or slippage. The position held during the bar is the
old one, and the filled change earns from the fill price to the bar's close, so adverse
selection is in the numbers: orders fill exactly when the price comes to us, and the ones that
run away are missed. An order still unfilled after `maker_max_wait_bars` is crossed at that
bar's close as a taker, at full taker cost, toward the newest target.

What neither engine models: basis between spot and perp, liquidation, exchange outages, and
our own market impact beyond the slippage constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..strategies.base import Market

PPY = {"1d": 365.0, "1b": 252.0, "1h": 365.0 * 24, "4h": 365.0 * 6, "15m": 365.0 * 96, "5m": 365.0 * 288}
FLOOR = {"1d": "D", "1b": "D", "1h": "h", "4h": "4h", "15m": "15min", "5m": "5min"}


@dataclass(frozen=True)
class CostModel:
    fee_bps: float = 5.0
    spot_fee_bps: float = 10.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0
    maker_fee_bps: float = 2.0
    maker_through_bps: float = 1.0
    maker_max_wait_bars: int = 1
    etf_cost_bps: float = 2.0            # listed ETFs: no commission, about 1 bp half spread and 1 bp slippage
    etf_short_borrow_ann: float = 0.01   # cost of borrowing an ETF to short it, per year

    def per_side(self, instrument: str) -> float:
        """Cost of trading one unit of weight on ONE leg. For "spot+perp" the two legs pay different fees, so this is
        their average: multiplied by legs() it gives spot fee + perp fee + spread and slippage on both."""
        if instrument == "etf":
            return self.etf_cost_bps / 1e4
        if instrument == "spot":
            fee = self.spot_fee_bps
        elif instrument == "spot+perp":
            fee = (self.spot_fee_bps + self.fee_bps) / 2.0
        else:
            fee = self.fee_bps
        return (fee + self.half_spread_bps + self.slippage_bps) / 1e4

    @staticmethod
    def legs(instrument: str) -> int:
        return 2 if instrument == "spot+perp" else 1

    def round_trip(self, instrument: str) -> float:
        return 2.0 * self.per_side(instrument) * self.legs(instrument)

    def scaled(self, k: float) -> "CostModel":
        """Every cost multiplied by k (stress tests). The spread also widens, so maker limits sit further away."""
        return replace(self, fee_bps=self.fee_bps * k, spot_fee_bps=self.spot_fee_bps * k, half_spread_bps=self.half_spread_bps * k,
                       slippage_bps=self.slippage_bps * k, maker_fee_bps=self.maker_fee_bps * k, maker_through_bps=self.maker_through_bps * k,
                       etf_cost_bps=self.etf_cost_bps * k)

    @classmethod
    def from_settings(cls, s) -> "CostModel":
        c = s.costs
        return cls(fee_bps=float(c.fee_bps), spot_fee_bps=float(c.spot_fee_bps), half_spread_bps=float(c.half_spread_bps),
                   slippage_bps=float(c.slippage_bps), maker_fee_bps=float(c.get("maker_fee_bps", 2.0)),
                   maker_through_bps=float(c.get("maker_through_bps", 1.0)), maker_max_wait_bars=int(c.get("maker_max_wait_bars", 1)), etf_cost_bps=float(c.get("etf_cost_bps", 2.0)),
                   etf_short_borrow_ann=float(c.get("etf_short_borrow_ann", 0.01)))


@dataclass
class BacktestResult:
    name: str
    bar: str
    instrument: str
    weights: pd.DataFrame      # what was held during each bar
    gross: pd.Series           # price return per bar
    funding: pd.Series         # funding paid (negative) or received (positive) per bar
    cost: pd.Series            # transaction cost per bar (positive number, subtracted)
    turnover: pd.Series        # sum |dw| per bar
    net: pd.Series             # gross + funding - cost
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.net).cumprod()

    @property
    def ppy(self) -> float:
        return PPY[self.bar]

    def daily(self) -> pd.Series:
        """Net return per calendar day (compounded within the day)."""
        return (1.0 + self.net).resample("D").prod() - 1.0

    def slice(self, start=None, end=None) -> "BacktestResult":
        sl = slice(start, end)
        return BacktestResult(self.name, self.bar, self.instrument, self.weights.loc[sl], self.gross.loc[sl],
                              self.funding.loc[sl], self.cost.loc[sl], self.turnover.loc[sl], self.net.loc[sl], dict(self.meta))

    def components(self) -> Dict[str, float]:
        return {"gross": float(self.gross.sum()), "funding": float(self.funding.sum()),
                "cost": float(self.cost.sum()), "net": float(self.net.sum())}


def funding_per_bar(funding: Optional[pd.DataFrame], index: pd.DatetimeIndex, bar: str, columns=None) -> pd.DataFrame:
    """Settlement rates summed into the bar that ENDS at or contains each settlement; zero where none.
    A settlement at exactly 00:00 belongs to the bar that closes at 00:00: funding is paid on the position
    held across the settlement, not on the one opened at that instant."""
    cols = list(columns) if columns is not None else (list(funding.columns) if funding is not None else [])
    if funding is None or not len(funding):
        return pd.DataFrame(0.0, index=index, columns=cols)
    f = funding.copy()
    f.index = (f.index - pd.Timedelta(milliseconds=1)).floor(FLOOR[bar])
    f = f.groupby(level=0).sum()
    return f.reindex(index=index, columns=cols).fillna(0.0)


def _clean_weights(weights: pd.DataFrame, close: pd.DataFrame, instrument: str) -> pd.DataFrame:
    w = weights.reindex(index=close.index, columns=close.columns).astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if instrument in ("spot", "spot+perp"):
        w = w.clip(lower=0.0)
    return w


def backtest(weights: pd.DataFrame, market: Market, costs: CostModel, instrument: str = "perp", name: str = "") -> BacktestResult:
    """Taker execution: every change in weight is traded at the close and pays the full taker cost."""
    close = market.close
    w = _clean_weights(weights, close, instrument)
    w_app = w.shift(1).fillna(0.0)                       # held during the bar
    r = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fpb = funding_per_bar(market.funding, close.index, market.bar, close.columns)

    if instrument == "perp":
        gross = (w_app * r).sum(axis=1)
        fund = -(w_app * fpb).sum(axis=1)
    elif instrument == "spot":
        gross = (w_app * r).sum(axis=1)
        fund = pd.Series(0.0, index=close.index)
    elif instrument == "spot+perp":
        gross = pd.Series(0.0, index=close.index)
        fund = (w_app * fpb).sum(axis=1)
    elif instrument == "etf":                              # signed, no funding; shorts pay a borrow fee
        gross = (w_app * r).sum(axis=1)
        fund = -(w_app.clip(upper=0.0).abs().sum(axis=1)) * costs.etf_short_borrow_ann / PPY[market.bar]
    else:
        raise ValueError(f"unknown instrument {instrument}")

    turnover = (w_app - w_app.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * costs.per_side(instrument) * costs.legs(instrument)
    net = gross + fund - cost
    total = float(turnover.sum())
    return BacktestResult(name or "strategy", market.bar, instrument, w_app, gross, fund, cost, turnover, net,
                          {"style": "taker", "maker_share": 0.0, "cost_per_turnover_bps": 1e4 * float(cost.sum()) / total if total > 0 else 0.0})


def backtest_maker(weights: pd.DataFrame, market: Market, costs: CostModel, instrument: str = "perp", name: str = "") -> BacktestResult:
    """Post-only limit orders at the touch, filled only on a trade-through, taker fallback after the wait.
    Falls back to the taker engine where a resting order makes no sense here (two-legged carry, spot, no high/low)."""
    if instrument != "perp" or market.high is None or market.low is None:
        return backtest(weights, market, costs, instrument, name)
    close = market.close
    w = _clean_weights(weights, close, instrument)
    C = close.to_numpy(dtype=float)
    H = market.high.reindex(index=close.index, columns=close.columns).to_numpy(dtype=float)
    L = market.low.reindex(index=close.index, columns=close.columns).to_numpy(dtype=float)
    W = w.to_numpy(dtype=float)
    F = funding_per_bar(market.funding, close.index, market.bar, close.columns).to_numpy(dtype=float)
    n, k = C.shape
    hs, thr = costs.half_spread_bps / 1e4, costs.maker_through_bps / 1e4
    mk_fee, tk_cost, max_wait = costs.maker_fee_bps / 1e4, costs.per_side("perp"), max(1, int(costs.maker_max_wait_bars))

    held = np.zeros(k)
    wait = np.zeros(k, dtype=int)
    gross, fund, cost, turn = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    held_hist = np.zeros((n, k))
    maker_notional = taker_notional = 0.0
    posted = filled_n = 0
    for t in range(1, n):
        c0, c1 = C[t - 1], C[t]
        ok = np.isfinite(c0) & np.isfinite(c1) & (c0 > 0)
        r = np.where(ok, c1 / np.where(ok, c0, 1.0) - 1.0, 0.0)
        tgt = W[t - 1]                                    # decided at the close of bar t-1
        d = tgt - held
        working = np.abs(d) > 1e-9
        buy = d > 0
        px = np.where(buy, c0 * (1.0 - hs), c0 * (1.0 + hs))
        through = np.where(buy, L[t] <= px * (1.0 - thr), H[t] >= px * (1.0 + thr))
        fill = working & ok & np.isfinite(L[t]) & np.isfinite(H[t]) & through
        d_fill = np.where(fill, d, 0.0)
        # the old position earns the whole bar; the filled change earns from its fill price to the close
        g = float((held * r).sum()) + float((d_fill * np.where(fill, c1 / np.where(fill, px, 1.0) - 1.0, 0.0)).sum())
        f = -float((held * F[t]).sum())
        mk = float(np.abs(d_fill).sum())
        c = mk * mk_fee
        posted += int(working.sum())
        filled_n += int(fill.sum())
        held_hist[t] = held                                # exposure carried through the bar (pre-fill, conservative for stats)
        held = np.where(fill, tgt, held)
        wait = np.where(working & ~fill, wait + 1, 0)
        chase = wait >= max_wait
        tk = 0.0
        if chase.any():                                    # cross at this bar's close toward the newest target
            dd = np.where(chase, W[t] - held, 0.0)
            tk = float(np.abs(dd).sum())
            c += tk * tk_cost
            held = np.where(chase, W[t], held)
            wait = np.where(chase, 0, wait)
        gross[t], fund[t], cost[t], turn[t] = g, f, c, mk + tk
        maker_notional += mk
        taker_notional += tk
    idx = close.index
    total = maker_notional + taker_notional
    meta = {"style": "maker", "maker_share": maker_notional / total if total > 0 else 0.0, "orders_posted": posted, "orders_filled": filled_n,
            "fill_rate": filled_n / posted if posted else 0.0, "cost_per_turnover_bps": 1e4 * float(cost.sum()) / total if total > 0 else 0.0,
            "maker_notional": maker_notional, "taker_notional": taker_notional}
    gs, fs, cs, ts = (pd.Series(x, index=idx) for x in (gross, fund, cost, turn))
    return BacktestResult(name or "strategy", market.bar, instrument, pd.DataFrame(held_hist, index=idx, columns=close.columns),
                          gs, fs, cs, ts, gs + fs - cs, meta)


def run_style(style: str, weights: pd.DataFrame, market: Market, costs: CostModel, instrument: str, name: str = "") -> BacktestResult:
    return (backtest_maker if style == "maker" else backtest)(weights, market, costs, instrument, name)
