"""Is it statistically significant? One answer for any series of returns.

    python -m itrader.online judge --returns runs/episodes.jsonl --column net_return --trials 8

Give it per-period returns from anything: a solver's evaluation episodes, a backtest, a paper record.
It reports the mean and its t-statistic, the Sharpe ratio per period, the probability that the true
Sharpe is above zero given the sample's length, skew and fat tails (PSR), the same after allowing for
how many variants were tried (deflated, DSR), a block bootstrap of the mean, and how long a record
would be needed at this Sharpe to be 95% sure. The verdict is "significant" only if the deflated
probability is at least 95% AND the bootstrap leaves at most a 5% chance that the mean is not positive.

`--trials` is the honest part. It is the number of solvers, seeds, checkpoints and hyperparameter
settings that were looked at before this one was picked. With eight solvers and a few seeds each,
it is not 1.

Caveats the numbers cannot fix: periods must be comparable and roughly non-overlapping (overlapping
episodes overstate the sample; use `--block` at least as long as the overlap), and a series selected
on the same window it is judged on is flattered however it is tested.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..bench.metrics import _moments, deflated_sharpe, evidence_meter, max_drawdown, norm_cdf, probabilistic_sharpe
from ..bench.stress import _resample


def load_series(path: str | Path, column: Optional[str] = None, equity: bool = False) -> pd.Series:
    """CSV, parquet or JSON-lines. `column` names the return field (default: the last numeric column)."""
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".jsonl", ".ndjson"):
        df = pd.DataFrame([json.loads(line) for line in p.read_text().splitlines() if line.strip()])
    else:
        df = pd.read_csv(p)
    if column is None:
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num:
            raise ValueError("no numeric column found; name one with --column")
        column = num[-1]
    x = pd.to_numeric(df[column], errors="coerce").dropna().reset_index(drop=True)
    return x.pct_change().dropna().reset_index(drop=True) if equity else x


def judge(r: pd.Series, trials: int = 1, ppy: Optional[float] = None, block: int = 20, draws: int = 2000, seed: int = 7) -> Dict[str, Any]:
    r = pd.Series(r).dropna().astype(float)
    n, mu, sd, sk, ku = _moments(r)
    if n < 10 or sd <= 0:
        return {"n": int(n), "verdict": "too few observations to say anything", "significant": False}
    srp = mu / sd
    t = srp * math.sqrt(n)
    dsr = deflated_sharpe(srp, n, sk, ku, trials)
    block = int(max(1, min(block, n // 5)))
    means = _resample(r.to_numpy(), n, block, draws, np.random.default_rng(seed)).mean(axis=1)
    ev = evidence_meter(srp, sk, ku, float(ppy or 1.0), n)
    out: Dict[str, Any] = {
        "n": int(n), "mean": float(mu), "std": float(sd), "t_stat": float(t), "p_value_one_sided": float(1.0 - norm_cdf(t)),
        "sharpe_per_period": float(srp), "sharpe_annualised": float(srp * math.sqrt(ppy)) if ppy else None,
        "skew": float(sk), "kurtosis": float(ku), "psr": float(probabilistic_sharpe(srp, n, sk, ku)),
        "dsr": float(dsr["dsr"]), "trials": int(dsr["n_trials"]), "luck_sharpe_per_period": float(dsr["sr0_period"]),
        "bootstrap_mean": {"p05": float(np.percentile(means, 5)), "p50": float(np.percentile(means, 50)), "p95": float(np.percentile(means, 95)),
                           "block": block, "draws": draws},
        "p_mean_not_positive": float((means <= 0).mean()),
        "periods_needed_for_95pct": float(ev["bars_needed"]), "share_of_needed_record": float(min(ev["ratio"], 99.0)),
        "max_drawdown": float(max_drawdown(r)), "hit_rate": float((r > 0).mean()),
    }
    out["significant"] = bool(out["dsr"] >= 0.95 and out["p_mean_not_positive"] <= 0.05)
    if out["significant"]:
        out["verdict"] = f"significant: unlikely to be luck even after {out['trials']} trial(s)"
    elif out["psr"] >= 0.95 and trials > 1:
        out["verdict"] = f"significant on its own, NOT after allowing for {out['trials']} trials: it could be the best of several lucky draws"
    elif mu > 0:
        out["verdict"] = "positive but not significant: this sample cannot tell it apart from luck"
    else:
        out["verdict"] = "not positive: nothing to test"
    return out


def format_judgement(j: Dict[str, Any], name: str = "series") -> str:
    if "mean" not in j:
        return f"{name}: {j.get('verdict')} (n = {j.get('n')})"
    need = j["periods_needed_for_95pct"]
    b = j["bootstrap_mean"]
    lines = [
        f"{name}: {j['n']} periods",
        f"  mean {j['mean']:+.6f} per period (t = {j['t_stat']:.2f}, one-sided p = {j['p_value_one_sided']:.3f}); std {j['std']:.6f}; hit rate {100 * j['hit_rate']:.0f}%",
        f"  Sharpe per period {j['sharpe_per_period']:.3f}" + (f" (annualised {j['sharpe_annualised']:.2f})" if j.get("sharpe_annualised") is not None else "")
        + f"; skew {j['skew']:.2f}, kurtosis {j['kurtosis']:.1f}; worst drawdown {100 * j['max_drawdown']:.1f}%",
        f"  probability the true Sharpe is above zero: {100 * j['psr']:.1f}%",
        f"  the same after {j['trials']} trial(s): {100 * j['dsr']:.1f}%  (the best of that many no-skill variants would show about {j['luck_sharpe_per_period']:.3f} by luck)",
        f"  block bootstrap of the mean ({b['draws']} draws, blocks of {b['block']}): 5% {b['p05']:+.6f}, median {b['p50']:+.6f}, 95% {b['p95']:+.6f}; chance the mean is not positive {100 * j['p_mean_not_positive']:.1f}%",
        f"  record needed for 95% confidence at this Sharpe: " + (f"{need:,.0f} periods; this sample is {100 * min(j['share_of_needed_record'], 9.99):.0f}% of that" if math.isfinite(need) else "no length would do, the mean is not positive"),
        f"  VERDICT: {j['verdict']}",
    ]
    return "\n".join(lines)
