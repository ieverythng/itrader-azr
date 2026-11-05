"""Stress tests: how fragile is a result?

* `block_bootstrap`: resample a daily return series in blocks (so runs of good and bad days stay
  together) and look at the spread of outcomes. One backtest is one path through history; this
  shows the others that history could as easily have taken. The honest headline is the 5th
  percentile and the probability of a losing record, not the median.
* `wealth_paths`: the same resampling carried forward for years, with a monthly contribution. It
  answers "where might this account be", as a range, and shows how much of the answer is the
  saving and how much is the return.

Both assume the future is drawn from the measured past. It will not be, exactly. Treat the
ranges as optimistic at both ends: a strategy can stop working altogether, and that is not in
any resample of the years in which it worked.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


def _resample(r: np.ndarray, length: int, block: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """draws x length matrix of block-resampled returns (circular blocks of fixed size)."""
    T = len(r)
    nb = int(math.ceil(length / block))
    starts = rng.integers(0, T, size=(draws, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % T
    return r[idx.reshape(draws, nb * block)[:, :length]]


def block_bootstrap(r: pd.Series, block: int = 20, draws: int = 2000, seed: int = 7) -> Dict[str, Any]:
    x = r.dropna().to_numpy(dtype=float)
    T = len(x)
    if T < 3 * block:
        return {"days": T, "note": "too short to bootstrap"}
    sims = _resample(x, T, block, draws, np.random.default_rng(seed))
    eq = np.cumprod(1.0 + sims, axis=1)
    cagr = eq[:, -1] ** (365.0 / T) - 1.0
    sd = sims.std(axis=1, ddof=1)
    sharpe = np.where(sd > 0, sims.mean(axis=1) / sd * math.sqrt(365.0), 0.0)
    dd = (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)
    q = lambda a, p: float(np.percentile(a, p))  # noqa: E731
    return {"days": T, "block_days": block, "draws": draws,
            "cagr": {"p05": q(cagr, 5), "p25": q(cagr, 25), "p50": q(cagr, 50), "p75": q(cagr, 75), "p95": q(cagr, 95)},
            "sharpe": {"p05": q(sharpe, 5), "p50": q(sharpe, 50), "p95": q(sharpe, 95)},
            "max_dd": {"p05": q(dd, 5), "p50": q(dd, 50), "p95": q(dd, 95)},
            "p_losing_record": float((cagr < 0).mean()), "p_drawdown_worse_than_20pct": float((dd < -0.20).mean())}


def wealth_paths(r: pd.Series, start: float = 1000.0, monthly: float = 0.0, years: Iterable[int] = (1, 3, 5, 10), block: int = 20,
                 draws: int = 2000, seed: int = 11) -> Dict[str, Any]:
    """Percentiles of account value at each horizon, contributions added every 30 days."""
    x = r.dropna().to_numpy(dtype=float)
    years = sorted(int(y) for y in years)
    horizon = max(years) * 365
    if len(x) < 3 * block:
        return {"note": "too short to project"}
    sims = _resample(x, horizon, block, draws, np.random.default_rng(seed))
    value = np.full(draws, float(start))
    rows: List[Dict[str, Any]] = []
    marks = {y * 365: y for y in years}
    for d in range(horizon):
        if d > 0 and d % 30 == 0:
            value = value + monthly
        value = value * (1.0 + sims[:, d])
        if (d + 1) in marks:
            y = marks[d + 1]
            paid = start + monthly * ((d + 1) // 30)
            rows.append({"years": y, "paid_in": float(paid), "p05": float(np.percentile(value, 5)), "p25": float(np.percentile(value, 25)),
                         "p50": float(np.percentile(value, 50)), "p75": float(np.percentile(value, 75)), "p95": float(np.percentile(value, 95)),
                         "p_below_paid_in": float((value < paid).mean())})
    return {"start": start, "monthly": monthly, "block_days": block, "draws": draws, "source_days": len(x), "rows": rows}


def wealth_table(plan: Dict[str, Any], currency: str = "USDT") -> str:
    if "rows" not in plan:
        return plan.get("note", "no projection")
    lines = [f"Start {plan['start']:,.0f} {currency}, adding {plan['monthly']:,.0f} a month. {plan['draws']} resampled futures from {plan['source_days']} measured days.",
             "", "| years | paid in | bad case (5%) | middle | good case (95%) | chance of having less than you paid in |", "|---|---|---|---|---|---|"]
    for row in plan["rows"]:
        lines.append(f"| {row['years']} | {row['paid_in']:,.0f} | {row['p05']:,.0f} | {row['p50']:,.0f} | {row['p95']:,.0f} | {100 * row['p_below_paid_in']:.0f}% |")
    return "\n".join(lines)
