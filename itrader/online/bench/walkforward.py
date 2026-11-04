"""Walk-forward and holdout.

The last `holdout_days` are the holdout: no design decision in this stack was made by looking
at them, and the report shows them separately so the reader can see whether the development
numbers survive. The development period is cut into fixed folds and each fold is reported
on its own: a strategy that made all its money in one fold is a different animal from one that
made a little in every fold, even if the totals match.

The strategies here have fixed, published parameters (n_variants_tried is 1 or 2), so there is
no re-fitting per fold. If someone adds a fitted strategy, fit on folds before t and report on t;
the fold boundaries below are where that goes.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .engine import BacktestResult
from .metrics import summarize


def split_points(index: pd.DatetimeIndex, holdout_days: int = 182, fold_days: int = 182) -> Dict[str, object]:
    if not len(index):
        return {"dev_start": None, "dev_end": None, "holdout_start": None, "end": None, "folds": []}
    end = index[-1]
    holdout_start = end - pd.Timedelta(days=holdout_days)
    dev_start = index[0]
    folds: List[tuple] = []
    cur = holdout_start
    while cur - pd.Timedelta(days=fold_days) > dev_start:
        folds.append((cur - pd.Timedelta(days=fold_days), cur))
        cur = cur - pd.Timedelta(days=fold_days)
    if cur > dev_start + pd.Timedelta(days=30):
        folds.append((dev_start, cur))
    folds.reverse()
    return {"dev_start": dev_start, "dev_end": holdout_start, "holdout_start": holdout_start, "end": end, "folds": folds}


def walk_forward(res: BacktestResult, n_trials: int = 1, holdout_days: int = 182, fold_days: int = 182, warmup_bars: int = 0,
                 participation: float = 0.01, close: Optional[pd.DataFrame] = None, volume: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    """Full-sample, development, holdout and per-fold summaries for one strategy."""
    idx = res.net.index
    if warmup_bars and len(idx) > warmup_bars:
        res = res.slice(idx[warmup_bars], None)
        idx = res.net.index
    sp = split_points(idx, holdout_days, fold_days)
    full = summarize(res, n_trials, participation, close, volume)
    dev = summarize(res.slice(None, sp["dev_end"]), n_trials, participation, close, volume) if sp["dev_end"] is not None else full
    hold = summarize(res.slice(sp["holdout_start"], None), n_trials, participation, close, volume) if sp["holdout_start"] is not None else full
    folds = []
    for a, b in sp["folds"]:
        s = summarize(res.slice(a, b), n_trials, participation)
        folds.append({"start": str(a.date()), "end": str(b.date()), "net": s["total_return"], "sharpe": s["sharpe"],
                      "max_dd": s["max_dd"], "bars": s["bars"], "turnover_per_year": s["turnover_per_year"]})
    pos = [f for f in folds if f["net"] > 0]
    return {
        "name": res.name, "bar": res.bar, "instrument": res.instrument,
        "split": {k: (str(v) if not isinstance(v, list) else len(v)) for k, v in sp.items()},
        "full": full, "dev": dev, "holdout": hold, "folds": folds,
        "folds_positive_share": (len(pos) / len(folds)) if folds else 0.0,
        "worst_fold_sharpe": min((f["sharpe"] for f in folds), default=0.0),
        "worst_fold_net": min((f["net"] for f in folds), default=0.0),
    }
