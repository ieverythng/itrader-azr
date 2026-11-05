"""The strategy interface. One code path for the bench and for live.

A strategy sees a `Market` (price panels and funding) and returns target weights: a frame
indexed by bar time, one column per symbol, values = fraction of equity (signed). The weight
in row t may only use data up to and including bar t; the bench applies it from bar t+1.
Each strategy sizes by inverse volatility so every instrument contributes similar risk, and
normalises so gross exposure is at most 1.0; the portfolio layer scales from there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ANN = {"1d": 365.0, "1b": 252.0, "1h": 365.0 * 24, "4h": 365.0 * 6, "15m": 365.0 * 96, "5m": 365.0 * 288}


@dataclass
class Market:
    bar: str
    close: pd.DataFrame
    high: Optional[pd.DataFrame] = None
    low: Optional[pd.DataFrame] = None
    volume: Optional[pd.DataFrame] = None
    funding: Optional[pd.DataFrame] = None     # 8-hourly rates, index ts, columns symbols
    daily_close: Optional[pd.DataFrame] = None  # for hourly strategies that need the daily trend

    @property
    def symbols(self) -> List[str]:
        return list(self.close.columns)

    def returns(self) -> pd.DataFrame:
        return self.close.pct_change()

    def vol(self, span: int = 30) -> pd.DataFrame:
        """Exponentially weighted volatility per bar (not annualised)."""
        return self.returns().ewm(span=span, min_periods=max(5, span // 3)).std()

    def ann_factor(self) -> float:
        return float(np.sqrt(ANN[self.bar]))

    def funding_daily(self) -> Optional[pd.DataFrame]:
        """Funding per day per symbol (sum of the day's 8-hourly settlements), aligned to daily bars."""
        if self.funding is None or not len(self.funding):
            return None
        return self.funding.resample("1D").sum(min_count=1)


def inverse_vol_weights(signal: pd.DataFrame, vol: pd.DataFrame, gross: float = 1.0, max_single: float = 0.35) -> pd.DataFrame:
    """signal in [-1, 1] per symbol -> weights, each scaled by 1/vol, gross exposure <= `gross`."""
    v = vol.replace(0, np.nan)
    raw = signal / v
    raw = raw.where(np.isfinite(raw), 0.0)
    g = raw.abs().sum(axis=1).replace(0, np.nan)
    w = raw.div(g, axis=0).fillna(0.0) * gross
    return w.clip(-max_single, max_single)


class Strategy:
    name: str = "base"
    family_id: str = "18"
    bar: str = "1d"
    instrument: str = "perp"       # "perp" allows shorts and pays funding; "spot" is long-only, no funding
    description: str = ""
    params: Dict[str, float] = {}
    n_variants_tried: int = 1       # honesty counter for the deflated Sharpe: how many parameter sets were looked at

    def __init__(self, **params) -> None:
        self.params = {**self.__class__.params, **params}

    def targets(self, m: Market) -> pd.DataFrame:
        raise NotImplementedError

    def rationale(self, m: Market, w: pd.Series) -> str:
        longs = [s for s, x in w.items() if x > 1e-9]
        shorts = [s for s, x in w.items() if x < -1e-9]
        return f"{self.name}: long {longs or 'none'}, short {shorts or 'none'}"

    def describe(self) -> Dict[str, object]:
        return {"name": self.name, "family_id": self.family_id, "bar": self.bar, "instrument": self.instrument,
                "params": dict(self.params), "description": self.description, "n_variants_tried": self.n_variants_tried}


def clean(w: pd.DataFrame) -> pd.DataFrame:
    return w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
