"""Metrics that are hard to fool.

* Sharpe is reported with its probabilistic version (PSR: the probability the true Sharpe is
  above zero given the sample's length, skew and fat tails) and its deflated version (DSR:
  the same after correcting for how many variants were tried; Bailey and Lopez de Prado,
  2014). A strategy whose author tried 50 parameter sets needs a far higher Sharpe to mean
  anything, and `n_variants_tried` on each strategy is where that honesty is recorded.
* The evidence meter answers "how long until this record means something": the number of
  bars (and days, and trades) a strategy needs at its observed Sharpe before PSR reaches 95%.
  It is also what says that 16 wins in a row at 98.7c is not evidence (p = 0.81 under no edge).
* Capacity is the equity at which our daily trading would be `participation` of the pair's
  daily volume on the venue, for the most constrained pair.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .engine import PPY, BacktestResult

EULER_GAMMA = 0.5772156649015329


def periods_per_year(bar: str) -> float:
    return PPY[bar]


# ----------------------------------------------------------------------------- normal helpers
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Acklam's rational approximation (|error| < 4.5e-4 relative), enough for these uses."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ----------------------------------------------------------------------------- basic stats
def _moments(r: pd.Series):
    r = r.dropna()
    n = len(r)
    if n < 3:
        return n, 0.0, 0.0, 0.0, 3.0
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    sk = float(r.skew()) if n > 2 else 0.0
    ku = float(r.kurt()) + 3.0 if n > 3 else 3.0       # pandas gives excess kurtosis
    if not np.isfinite(sk):
        sk = 0.0
    if not np.isfinite(ku):
        ku = 3.0
    return n, mu, sd, sk, ku


def sharpe(r: pd.Series, ppy: float) -> float:
    n, mu, sd, _, _ = _moments(r)
    return float(mu / sd * math.sqrt(ppy)) if sd > 0 and n > 2 else 0.0


def cagr(net: pd.Series, ppy: float) -> float:
    n = int(net.notna().sum())
    if n == 0:
        return 0.0
    eq = float((1.0 + net.fillna(0.0)).prod())
    if eq <= 0:
        return -1.0
    return eq ** (ppy / n) - 1.0


def ann_vol(r: pd.Series, ppy: float) -> float:
    _, _, sd, _, _ = _moments(r)
    return float(sd * math.sqrt(ppy))


def max_drawdown(net: pd.Series) -> float:
    eq = (1.0 + net.fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min()) if len(dd) else 0.0


def probabilistic_sharpe(sr_period: float, n: int, skew: float, kurt: float, sr_bench_period: float = 0.0) -> float:
    """P(true SR > sr_bench) given the estimate, in per-period units (Bailey and Lopez de Prado)."""
    if n <= 1:
        return 0.5
    denom = 1.0 - skew * sr_period + (kurt - 1.0) / 4.0 * sr_period ** 2
    if denom <= 0:
        denom = 1e-9
    z = (sr_period - sr_bench_period) * math.sqrt(n - 1) / math.sqrt(denom)
    return norm_cdf(z)


def deflated_sharpe(sr_period: float, n: int, skew: float, kurt: float, n_trials: int = 1) -> Dict[str, float]:
    """DSR = PSR against the Sharpe one would expect from the best of `n_trials` no-skill variants.

    The variance of the trial Sharpes is not observed here, so it is proxied by the sampling
    variance of this estimate, (1 - g3 SR + (g4 - 1)/4 SR^2) / (n - 1). With one trial the
    benchmark is zero and DSR equals PSR."""
    n_trials = max(1, int(n_trials))
    if n_trials == 1 or n <= 1:
        sr0 = 0.0
    else:
        var_sr = max(1e-12, (1.0 - skew * sr_period + (kurt - 1.0) / 4.0 * sr_period ** 2) / max(1, n - 1))
        sr0 = math.sqrt(var_sr) * ((1 - EULER_GAMMA) * norm_ppf(1 - 1.0 / n_trials) + EULER_GAMMA * norm_ppf(1 - 1.0 / (n_trials * math.e)))
    return {"dsr": probabilistic_sharpe(sr_period, n, skew, kurt, sr0), "sr0_period": sr0, "n_trials": n_trials}


def evidence_meter(sr_period: float, skew: float, kurt: float, ppy: float, have_bars: int, trades_per_bar: float = 0.0,
                   confidence: float = 0.95) -> Dict[str, float]:
    """Bars needed before PSR against zero reaches `confidence` at the observed per-period Sharpe."""
    z = norm_ppf(confidence)
    if sr_period <= 0:
        need = math.inf
    else:
        denom = max(1e-9, 1.0 - skew * sr_period + (kurt - 1.0) / 4.0 * sr_period ** 2)
        need = 1.0 + (z / sr_period) ** 2 * denom
    days = need / ppy * 365.0 if math.isfinite(need) else math.inf
    return {
        "bars_needed": need,
        "days_needed": days,
        "years_needed": days / 365.0 if math.isfinite(days) else math.inf,
        "trades_needed": need * trades_per_bar if (math.isfinite(need) and trades_per_bar > 0) else math.nan,
        "bars_have": int(have_bars),
        "ratio": (have_bars / need) if (math.isfinite(need) and need > 0) else 0.0,
        "confidence": confidence,
    }


def binary_evidence(p_null: float, streak: int = 16, alpha: float = 0.05) -> Dict[str, float]:
    """The 16/16 lesson: how likely a no-edge bettor at implied probability `p_null` wins
    `streak` in a row, and how long a streak would be needed to reject no-edge at `alpha`."""
    p_streak = p_null ** streak
    needed = math.ceil(math.log(alpha) / math.log(p_null)) if 0 < p_null < 1 else math.inf
    return {"p_null": p_null, "streak": streak, "p_streak_under_no_edge": p_streak, "streak_needed_for_alpha": needed, "alpha": alpha}


def capacity_usd(weights: pd.DataFrame, close: pd.DataFrame, volume: Optional[pd.DataFrame], participation: float,
                 ppy: float, lookback_days: int = 90) -> Dict[str, object]:
    """Equity at which the strategy's own daily trading in its most constrained pair reaches
    `participation` of that pair's daily USD volume on this venue."""
    if volume is None or not len(volume):
        return {"capacity_usd": math.nan, "binding_symbol": None}
    bars_per_day = ppy / 365.0
    dw = weights.diff().abs().fillna(0.0)
    n = int(lookback_days * bars_per_day)
    dw_daily = dw.tail(n).sum() / max(1.0, lookback_days)                  # weight units traded per day per symbol
    usd_vol = (volume * close).reindex(columns=close.columns)
    adv = usd_vol.tail(n).resample("D").sum().replace(0, np.nan).median()  # USD per day per symbol
    cap = (participation * adv) / dw_daily.replace(0, np.nan)
    cap = cap.dropna()
    if not len(cap):
        return {"capacity_usd": math.inf, "binding_symbol": None}
    s = cap.idxmin()
    return {"capacity_usd": float(cap.min()), "binding_symbol": str(s), "adv_usd": {k: float(v) for k, v in adv.dropna().items()}}


def summarize(res: BacktestResult, n_trials: int = 1, participation: float = 0.01, close: Optional[pd.DataFrame] = None,
              volume: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    ppy = res.ppy
    net = res.net
    n, mu, sd, sk, ku = _moments(net)
    sr_p = (mu / sd) if sd > 0 else 0.0
    daily = res.daily()
    exposed = int((res.weights.abs().sum(axis=1) > 1e-9).sum())
    trades = int((res.turnover > 1e-9).sum())
    comp = res.components()
    years = max(1e-9, n / ppy)
    gross_total = comp["gross"] + comp["funding"]
    out: Dict[str, object] = {
        "name": res.name, "bar": res.bar, "instrument": res.instrument,
        "start": str(net.index[0]) if n else None, "end": str(net.index[-1]) if n else None,
        "bars": n, "years": years, "bars_exposed": exposed, "exposure_share": exposed / n if n else 0.0,
        "cagr": cagr(net, ppy), "total_return": float((1.0 + net.fillna(0.0)).prod() - 1.0),
        "ann_vol": ann_vol(net, ppy), "sharpe": sharpe(net, ppy), "sharpe_period": sr_p,
        "sortino": float(mu / net[net < 0].std(ddof=1) * math.sqrt(ppy)) if (net < 0).sum() > 2 and net[net < 0].std(ddof=1) > 0 else 0.0,
        "max_dd": max_drawdown(net), "worst_day": float(daily.min()) if len(daily) else 0.0,
        "best_day": float(daily.max()) if len(daily) else 0.0,
        "hit_rate_days": float((daily > 0).sum() / max(1, (daily != 0).sum())),
        "skew": sk, "kurtosis": ku,
        "turnover_per_year": float(res.turnover.sum() / years), "trade_bars": trades,
        "gross_return_total": gross_total, "cost_total": comp["cost"], "funding_total": comp["funding"],
        "cost_share": (comp["cost"] / abs(gross_total)) if abs(gross_total) > 1e-12 else math.inf,
        "psr": probabilistic_sharpe(sr_p, n, sk, ku, 0.0),
        "mean_net_per_bar": mu,
    }
    out.update(deflated_sharpe(sr_p, n, sk, ku, n_trials))
    out["evidence"] = evidence_meter(sr_p, sk, ku, ppy, n, trades_per_bar=trades / n if n else 0.0)
    if close is not None:
        out["capacity"] = capacity_usd(res.weights, close, volume, participation, ppy)
    return out
