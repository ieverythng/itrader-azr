"""Run the bench end to end on the local data store.

    python -m itrader.online bench            (or)   from itrader.online.bench.run import run_bench

Outputs in <out_root>/bench/:
    results.json          every strategy: walk-forward summary, promotion stage, description
    results.csv           the one-line-per-strategy table
    profiles.json         CapabilityProfiles in the repo's format (list of dicts)
    returns_daily.parquet daily net returns per strategy, for the portfolio layer and the allocator
    report.html           the human report
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd

from ..data.store import DataStore
from ..settings import Settings, settings as default_settings
from ..strategies import BENCHABLE, Market, Strategy, all_strategies
from .engine import BacktestResult, CostModel, backtest
from .profile import capability_profile, promotion_stage
from .report import render
from .walkforward import walk_forward

WARMUP_BARS = {"1d": 130, "1h": 24 * 31}   # longest lookback a strategy needs before its weights mean anything


def load_market(store: DataStore, universe: Iterable[str], bar: str, daily_close: Optional[pd.DataFrame] = None) -> Market:
    universe = list(universe)
    close = store.panel(universe, bar, "close")
    if close.empty:
        raise RuntimeError(f"no {bar} data in {store.root}; run the data sync first")
    high = store.panel(universe, bar, "high").reindex(columns=close.columns)
    low = store.panel(universe, bar, "low").reindex(columns=close.columns)
    vol = store.panel(universe, bar, "volume").reindex(columns=close.columns)
    funding = store.funding_panel(universe)
    funding = funding.reindex(columns=close.columns) if len(funding) else None
    return Market(bar=bar, close=close, high=high, low=low, volume=vol, funding=funding, daily_close=daily_close)


def bench_one(strategy: Strategy, market: Market, costs: CostModel, s: Settings) -> Dict[str, Any]:
    t0 = time.time()
    w = strategy.targets(market)
    res: BacktestResult = backtest(w, market, costs, strategy.instrument, name=strategy.name)
    wf = walk_forward(res, n_trials=strategy.n_variants_tried, holdout_days=int(s.bench.holdout_days), fold_days=int(s.bench.fold_days),
                      warmup_bars=WARMUP_BARS.get(market.bar, 0), participation=float(s.bench.participation),
                      close=market.close, volume=market.volume)
    stage = promotion_stage(wf, s.gates)
    prof = capability_profile(strategy, wf, stage)
    return {"name": strategy.name, "bar": strategy.bar, "instrument": strategy.instrument, "description": strategy.description,
            "family_id": strategy.family_id, "params": prof.metadata["params"], "n_variants_tried": strategy.n_variants_tried,
            "walk_forward": wf, "promotion": stage, "profile": prof.to_dict(), "seconds": round(time.time() - t0, 1),
            "_result": res}


def run_bench(s: Optional[Settings] = None, store_root: Optional[str] = None, out_dir: Optional[str] = None,
              names: Optional[Iterable[str]] = None, log: Callable[[str], None] = print) -> Dict[str, Any]:
    s = s or default_settings()
    root = s.resolve(store_root or s.general.data_root)
    out = Path(out_dir) if out_dir else s.resolve(s.general.out_root) / "bench"
    out.mkdir(parents=True, exist_ok=True)
    store = DataStore(root)
    costs = CostModel.from_settings(s)
    universe = list(s.data.universe)
    names = list(names) if names else list(BENCHABLE)
    strategies = [x for x in all_strategies(names=names) if x.name in BENCHABLE]

    markets: Dict[str, Market] = {}
    daily = load_market(store, universe, "1d")
    markets["1d"] = daily
    if any(x.bar == "1h" for x in strategies):
        markets["1h"] = load_market(store, universe, "1h", daily_close=daily.close)
    for bar, m in markets.items():
        log(f"  {bar}: {len(m.close)} bars x {len(m.close.columns)} symbols, {m.close.index[0].date()} to {m.close.index[-1].date()}, "
            f"funding {'yes' if m.funding is not None else 'no'}")

    results: List[Dict[str, Any]] = []
    equities: Dict[str, pd.Series] = {}
    daily_returns: Dict[str, pd.Series] = {}
    for st in strategies:
        if st.bar not in markets:
            log(f"  skip {st.name}: no {st.bar} data")
            continue
        try:
            r = bench_one(st, markets[st.bar], costs, s)
        except Exception as exc:  # noqa: BLE001
            log(f"  [!] {st.name} failed: {type(exc).__name__}: {exc}")
            continue
        res: BacktestResult = r.pop("_result")
        equities[st.name] = res.equity
        daily_returns[st.name] = res.daily()
        results.append(r)
        d, h = r["walk_forward"]["dev"], r["walk_forward"]["holdout"]
        log(f"  {st.name:16s} dev CAGR {100 * d['cagr']:6.1f}%  Sharpe {d['sharpe']:5.2f}  DSR {100 * d['dsr']:3.0f}%  maxDD {100 * d['max_dd']:6.1f}%  | "
            f"holdout CAGR {100 * h['cagr']:6.1f}%  Sharpe {h['sharpe']:5.2f}  | folds+ {100 * r['walk_forward']['folds_positive_share']:3.0f}%  "
            f"cost share {100 * min(d['cost_share'], 9.99):4.0f}%  -> {r['promotion']['stage']}  ({r['seconds']}s)")

    # artefacts
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    (out / "profiles.json").write_text(json.dumps([r["profile"] for r in results], indent=1, default=str))
    table = pd.DataFrame([{
        "strategy": r["name"], "bar": r["bar"], "instrument": r["instrument"], "stage": r["promotion"]["stage"],
        "dev_cagr": r["walk_forward"]["dev"]["cagr"], "dev_sharpe": r["walk_forward"]["dev"]["sharpe"], "dev_dsr": r["walk_forward"]["dev"]["dsr"],
        "dev_max_dd": r["walk_forward"]["dev"]["max_dd"], "dev_worst_day": r["walk_forward"]["dev"]["worst_day"],
        "holdout_cagr": r["walk_forward"]["holdout"]["cagr"], "holdout_sharpe": r["walk_forward"]["holdout"]["sharpe"],
        "holdout_max_dd": r["walk_forward"]["holdout"]["max_dd"], "folds_positive": r["walk_forward"]["folds_positive_share"],
        "turnover_per_year": r["walk_forward"]["dev"]["turnover_per_year"], "cost_share": r["walk_forward"]["dev"]["cost_share"],
        "years_needed": r["walk_forward"]["dev"]["evidence"]["years_needed"], "capacity_usd": r["walk_forward"]["dev"].get("capacity", {}).get("capacity_usd"),
    } for r in results])
    table.to_csv(out / "results.csv", index=False)
    if daily_returns:
        pd.DataFrame(daily_returns).sort_index().to_parquet(out / "returns_daily.parquet")
    cost_info = {**{k: float(getattr(costs, k)) for k in ("fee_bps", "spot_fee_bps", "half_spread_bps", "slippage_bps")},
                 "participation": float(s.bench.participation), "holdout_days": int(s.bench.holdout_days)}
    notes = ["Expect single to low double digit annual returns from classic strategies after costs. A high development Sharpe with a weak holdout "
             "is the normal shape of an overfit; a strategy that fails here is reported as failing, not tuned until it passes.",
             "The prediction sleeve (Polymarket, paper only) is not benched here; it reports through its own kit."]
    (out / "report.html").write_text(render(results, equities, f"iTRADER online bench ({root.name})", cost_info, notes))
    log(f"  wrote {out / 'report.html'}")
    return {"results": results, "out_dir": str(out), "table": table, "equities": equities, "daily_returns": daily_returns}
