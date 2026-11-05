"""Walk-forward simulation of the whole portfolio, and market regimes.

The single holdout in the per-strategy bench answers "did this sleeve's numbers travel?". This
module answers the harder question: "what would the Workbench allocator itself have earned,
deciding week by week with only what it could have known?".

`simulate` walks through the sleeves' daily net returns. Every `sim_rebalance_days` it builds a
capability profile for each sleeve from the PAST ONLY (development and a trailing pseudo-holdout,
exactly like the bench), vol-scales the sleeves, sets risk budgets, asks the allocator for weights
(cash included), sets the portfolio vol scale, charges a cost for changing exposures, and then
lives with those weights until the next rebalance. Every day in the output is out of sample.

Regimes. Four labels from BTC alone, each known at the previous day's close: trend up or down
(close against its 100-day mean) and calm or wild (30-day realised vol against its one-year
median). A sleeve's profile carries its record per regime; the allocator can use the record for
today's regime, shrunk toward the overall record because per-regime samples are small. Whether
that helps is an empirical question the bench answers out of sample, not an assumption.

Limits: the simulation works on sleeve returns, so asset-level caps and the no-trade band are not
applied (they only ever reduce exposure), and sleeve returns already include each sleeve's own
trading costs at its own size.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..portfolio import CASH, WorkbenchAllocator, realised_vol, risk_budgets
from ..settings import Settings
from .metrics import _moments, deflated_sharpe, evidence_meter, max_drawdown, probabilistic_sharpe

REGIMES = ("up_calm", "up_wild", "down_calm", "down_wild")


def _regime_raw(btc_close: pd.Series, ma_days: int = 100, vol_days: int = 30, vol_ref_days: int = 365) -> pd.Series:
    """Label computed from closes up to and including each row's own close."""
    c = btc_close.astype(float)
    ma = c.rolling(ma_days, min_periods=ma_days // 2).mean()
    vol = c.pct_change().rolling(vol_days, min_periods=10).std()
    ref = vol.rolling(vol_ref_days, min_periods=60).median()
    lab = pd.Series(np.where(c > ma, "up", "down"), index=c.index) + "_" + pd.Series(np.where(vol > ref, "wild", "calm"), index=c.index)
    lab[ma.isna() | vol.isna() | ref.isna()] = "unknown"
    return lab


def regime_labels(btc_close: pd.Series, **kw) -> pd.Series:
    """Regime for each day's RETURN: built from closes through the day before (one-day shift, no lookahead)."""
    return _regime_raw(btc_close, **kw).shift(1).fillna("unknown")


def regime_now(btc_close: pd.Series, **kw) -> str:
    """Regime for the decision being taken now, from every completed daily bar."""
    lab = _regime_raw(btc_close, **kw)
    return str(lab.iloc[-1]) if len(lab) else "unknown"


def regime_table(r: pd.Series, regimes: Optional[pd.Series]) -> Dict[str, Dict[str, float]]:
    if regimes is None or not len(r):
        return {}
    lab = regimes.reindex(r.index).fillna("unknown")
    out: Dict[str, Dict[str, float]] = {}
    for name, x in r.groupby(lab):
        n, mu, sd, sk, ku = _moments(x)
        srp = mu / sd if sd > 0 else 0.0
        out[str(name)] = {"n": int(n), "mean_daily": float(mu), "sharpe": float(srp * math.sqrt(365.0)), "psr": probabilistic_sharpe(srp, n, sk, ku)}
    return out


def profile_lite(r: pd.Series, n_trials: int = 1, holdout_days: int = 182, regimes: Optional[pd.Series] = None) -> Dict[str, Any]:
    """The fields the allocator scores on, from a sleeve's daily returns known so far."""
    r = r.dropna()
    if len(r) < 30:
        return {"estimated_p_solve": 0.5, "expected_net_utility": 0.0, "metadata": {"dev": {"sharpe": 0.0, "dsr": 0.5}, "holdout": {"sharpe": 0.0}, "evidence": {"ratio": 0.0}}}
    cut = r.index[-1] - pd.Timedelta(days=holdout_days)
    dev, hold = r[r.index < cut], r[r.index >= cut]
    if len(dev) < 90:
        dev, hold = r, r
    n, mu, sd, sk, ku = _moments(dev)
    srp = mu / sd if sd > 0 else 0.0
    d = deflated_sharpe(srp, n, sk, ku, n_trials)
    ev = evidence_meter(srp, sk, ku, 365.0, n)
    hn, hmu, hsd, _, _ = _moments(hold)
    meta = {"dev": {"sharpe": float(srp * math.sqrt(365.0)), "dsr": float(d["dsr"])},
            "holdout": {"sharpe": float(hmu / hsd * math.sqrt(365.0)) if hsd > 0 else 0.0},
            "evidence": {"ratio": float(min(1.0, ev["ratio"]))}}
    if regimes is not None:
        meta["regimes"] = regime_table(dev, regimes)
    return {"estimated_p_solve": probabilistic_sharpe(srp, n, sk, ku), "expected_net_utility": float(mu), "metadata": meta}


def stats(x: pd.Series) -> Dict[str, float]:
    x = x.dropna()
    n, mu, sd, sk, ku = _moments(x)
    srp = mu / sd if sd > 0 else 0.0
    return {"days": int(n), "cagr": float((1.0 + x).prod() ** (365.0 / max(n, 1)) - 1.0) if n else 0.0, "vol": float(sd * math.sqrt(365.0)),
            "sharpe": float(srp * math.sqrt(365.0)), "max_dd": max_drawdown(x), "worst_day": float(x.min()) if n else 0.0,
            "psr": probabilistic_sharpe(srp, n, sk, ku), "total_return": float((1.0 + x).prod() - 1.0) if n else 0.0}


def alpha_beta(r: pd.Series, bench: pd.Series) -> Dict[str, float]:
    """How much of a return stream is just the benchmark: regression of daily returns on the benchmark's.
    alpha_ann is what is left per year after the benchmark exposure is taken out; alpha_t is its t-statistic."""
    df = pd.concat([r.rename("y"), bench.rename("x")], axis=1).dropna()
    n = len(df)
    if n < 60 or float(df.x.var()) <= 0 or float(df.y.var()) <= 0:
        return {}
    beta = float(df.y.cov(df.x) / df.x.var())
    alpha = float(df.y.mean() - beta * df.x.mean())
    resid = df.y - alpha - beta * df.x
    se = float(resid.std(ddof=2) / math.sqrt(n))
    return {"beta": beta, "alpha_ann": alpha * 365.0, "alpha_t": (alpha / se) if se > 0 else 0.0, "corr": float(df.y.corr(df.x)), "days": n}


def simulate(R: pd.DataFrame, n_trials: Dict[str, int], s: Settings, regimes: Optional[pd.Series] = None, regime_aware: bool = False,
             mode: str = "workbench", rebalance_days: Optional[int] = None, min_history_days: Optional[int] = None,
             realloc_cost_per_side: float = 0.0008, apply_gate: bool = True, expo_caps: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """mode: "workbench" (capability-weighted, cash first-class) or "equal" (same vol rules, equal capital).
    apply_gate: a sleeve gets capital only if it would clear the live DecisionGate's standing checks (bench
    confidence at or above `[gate] min_confidence`, positive expected utility), as in the real loop.
    expo_caps: the most a sleeve's own weights can be multiplied by before the live per-asset cap would clip
    them (max_single_asset / the sleeve's usual largest weight). Without it a low-vol sleeve such as carry is
    levered here far beyond what the live book allows."""
    p, a = s.portfolio, s.allocator
    R = R.sort_index().fillna(0.0)
    names = list(R.columns)
    reb = int(rebalance_days or s.bench.get("sim_rebalance_days", 7))
    warm = int(min_history_days or s.bench.get("sim_min_history_days", 365))
    hold_days = int(s.bench.holdout_days)
    tgt_vol, port_vol, floor = float(p.strategy_vol_target), float(p.portfolio_vol_target), float(p.min_vol_floor)
    vwin, cwin = int(p.vol_window_days), int(p.corr_window_days)
    alloc = WorkbenchAllocator(s)
    if len(R) <= warm + 5:
        return {"returns": pd.Series(dtype=float), "history": pd.DataFrame()}
    out = np.zeros(len(R) - warm)
    hist = []
    prev = pd.Series(0.0, index=names)
    i = warm
    while i < len(R):
        past = R.iloc[:i]
        today = R.index[i]
        vol = {c: realised_vol(past[c], vwin, floor) for c in names}
        scales = pd.Series({c: (min(2.0, tgt_vol / vol[c]) if np.isfinite(vol[c]) and vol[c] > 0 else 1.0) for c in names})
        vr = {c: (min(1.0, vol[c] * scales[c] / tgt_vol) if np.isfinite(vol[c]) else 1.0) for c in names}
        scaled_past = past * scales
        regime = None
        if mode == "equal":
            cash = float(a.cash_floor)
            capital = pd.Series((1.0 - cash) / len(names), index=names)
        else:
            regime = str(regimes.loc[today]) if (regime_aware and regimes is not None and today in regimes.index) else None
            profs = {c: profile_lite(past[c], int(n_trials.get(c, 1)), hold_days, regimes if regime_aware else None) for c in names}
            if apply_gate:
                floor_conf = float(s.gate.min_confidence)
                profs = {c: pr for c, pr in profs.items() if pr["estimated_p_solve"] >= floor_conf and pr["expected_net_utility"] > 0}
            dec = alloc.allocate(profs, None, vr, regime=regime)
            w = dec.weights.drop(CASH)
            cash = float(dec.weights[CASH])
            rb = risk_budgets(scaled_past, float(p.corr_penalty), cwin, tgt_vol)
            tilt = rb.reindex(w.index).fillna(0.0) * w
            capital = (tilt / tilt.sum() * (1.0 - cash)) if tilt.sum() > 0 else tilt * 0.0
            capital = capital.reindex(names).fillna(0.0)
        pv = realised_vol((scaled_past * capital).sum(axis=1), vwin, floor)
        ps = float(min(2.0, port_vol / pv)) if capital.abs().sum() > 0 and np.isfinite(pv) and pv > 0 else 1.0
        expo = capital * scales * ps                       # multiplier on each sleeve's own return stream
        if expo_caps:
            expo = expo.clip(upper=pd.Series({c: float(expo_caps.get(c, np.inf)) for c in names}))
        j = min(len(R), i + reb)
        block = (R.iloc[i:j] * expo).sum(axis=1).to_numpy(dtype=float).copy()
        block[0] -= float((expo - prev).abs().sum()) * realloc_cost_per_side
        out[i - warm:j - warm] = block
        hist.append({"date": str(today.date()), "regime": regime, "cash": cash, "port_scale": ps, **{c: float(expo[c]) for c in names}})
        prev = expo
        i = j
    return {"returns": pd.Series(out, index=R.index[warm:]), "history": pd.DataFrame(hist)}
