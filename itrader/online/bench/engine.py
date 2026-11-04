"""Vectorised backtester over target weights.

Conventions (the same ones the live executor follows):

* The weight in row t is decided at the close of bar t and held through bar t+1. The engine
  shifts weights by one bar before multiplying by returns, so there is no lookahead by
  construction, whatever a strategy does inside.
* Costs: every change in weight is turnover; each unit of turnover pays fee + half spread +
  slippage per side. "spot+perp" trades two legs, so it pays twice.
* Instruments: "perp" is signed exposure and pays funding when long and the rate is positive
  (receives when short). "spot" is long only and never pays funding. "spot+perp" is price
  neutral: long spot, short the same perp notional, receives funding on the short leg.
* Funding settlements are attributed to the bar that contains them.

What the bench does not model: basis between spot and perp (small on the majors, but it is
a real risk for carry), liquidation, borrow for spot shorts (not allowed), exchange outages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..strategies.base import Market

PPY = {"1d": 365.0, "1h": 365.0 * 24, "4h": 365.0 * 6, "15m": 365.0 * 96, "5m": 365.0 * 288}
FLOOR = {"1d": "D", "1h": "h", "4h": "4h", "15m": "15min", "5m": "5min"}


@dataclass(frozen=True)
class CostModel:
    fee_bps: float = 5.0
    spot_fee_bps: float = 10.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0

    def per_side(self, instrument: str) -> float:
        fee = self.spot_fee_bps if instrument == "spot" else self.fee_bps
        return (fee + self.half_spread_bps + self.slippage_bps) / 1e4

    @staticmethod
    def legs(instrument: str) -> int:
        return 2 if instrument == "spot+perp" else 1

    def round_trip(self, instrument: str) -> float:
        return 2.0 * self.per_side(instrument) * self.legs(instrument)

    @classmethod
    def from_settings(cls, s) -> "CostModel":
        c = s.costs
        return cls(fee_bps=float(c.fee_bps), spot_fee_bps=float(c.spot_fee_bps),
                   half_spread_bps=float(c.half_spread_bps), slippage_bps=float(c.slippage_bps))


@dataclass
class BacktestResult:
    name: str
    bar: str
    instrument: str
    weights: pd.DataFrame      # applied weights (already shifted): what was held during each bar
    gross: pd.Series           # price return per bar
    funding: pd.Series         # funding paid (negative) or received (positive) per bar
    cost: pd.Series            # transaction cost per bar (positive number, subtracted)
    turnover: pd.Series        # sum |dw| per bar
    net: pd.Series             # gross + funding - cost

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
                              self.funding.loc[sl], self.cost.loc[sl], self.turnover.loc[sl], self.net.loc[sl])

    def components(self) -> Dict[str, float]:
        return {"gross": float(self.gross.sum()), "funding": float(self.funding.sum()),
                "cost": float(self.cost.sum()), "net": float(self.net.sum())}


def funding_per_bar(funding: Optional[pd.DataFrame], index: pd.DatetimeIndex, bar: str, columns=None) -> pd.DataFrame:
    """Settlement rates summed into the bar that contains each settlement; zero where none."""
    cols = list(columns) if columns is not None else (list(funding.columns) if funding is not None else [])
    if funding is None or not len(funding):
        return pd.DataFrame(0.0, index=index, columns=cols)
    f = funding.copy()
    f.index = f.index.floor(FLOOR[bar])
    f = f.groupby(level=0).sum()
    return f.reindex(index=index, columns=cols).fillna(0.0)


def backtest(weights: pd.DataFrame, market: Market, costs: CostModel, instrument: str = "perp", name: str = "") -> BacktestResult:
    close = market.close
    w = weights.reindex(index=close.index, columns=close.columns).astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if instrument in ("spot", "spot+perp"):
        w = w.clip(lower=0.0)
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
    else:
        raise ValueError(f"unknown instrument {instrument}")

    turnover = (w_app - w_app.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * costs.per_side(instrument) * costs.legs(instrument)
    net = gross + fund - cost
    return BacktestResult(name or "strategy", market.bar, instrument, w_app, gross, fund, cost, turnover, net)
