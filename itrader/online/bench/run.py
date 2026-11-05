"""Run the bench end to end on the local data store.

    python -m itrader.online bench            (or)   from itrader.online.bench.run import run_bench

Per strategy: both execution styles (taker, and maker with taker fallback), walk-forward folds,
the untouched holdout, the promotion stage and a CapabilityProfile, judged on the style set in
`[execution] style`. Then the portfolio: a walk-forward simulation of the Workbench allocator
(weekly, past data only), the same with regime-aware scoring, an equal-weight control, BTC as
the benchmark, a block bootstrap of the result and a cost stress.

Outputs in <out_root>/bench/:
    results.json / results.csv     every strategy, both styles
    profiles.json                  CapabilityProfiles in the repo's format (with a per-regime record in metadata)
    profiles_by_regime.json        one CapabilityProfile per strategy and regime (spec 14.2)
    returns_daily.parquet          daily net returns per strategy at the configured style
    portfolio.json                 the portfolio simulations, bootstrap, stress, latest allocation
    portfolio_returns.parquet      daily out-of-sample returns of each portfolio variant
    report.html                    the human report
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from itrader.rl_lib.tasks.capability_profile import CapabilityProfile

from ..data.feeds import Yahoo
from ..data.store import DataStore
from ..settings import Settings, settings as default_settings
from ..strategies import BENCHABLE, Market, Strategy, all_strategies
from .engine import BacktestResult, CostModel, run_style
from .metrics import sharpe as _sharpe
from .portfolio_sim import REGIMES, alpha_beta, regime_labels, regime_now, regime_table, simulate, stats
from .profile import capability_profile, promotion_stage
from .report import render
from .stress import block_bootstrap
from .walkforward import walk_forward

WARMUP_BARS = {"1d": 130, "1h": 24 * 31, "1b": 260}   # longest lookback a strategy needs before its weights mean anything
BARS_PER_DAY = {"1d": 1.0, "1h": 24.0, "1b": 1.0}


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


def load_macro_market(s: Settings, cutoff: Optional[pd.Timestamp] = None) -> Optional[Market]:
    """Daily ETF panel for the macro trend sleeve (bar "1b"), with the risk-free return from a T-bill ETF.
    `cutoff` hides everything not yet known at that time (replay)."""
    uni = list(s.data.get("macro_universe", []))
    if not uni:
        return None
    store = DataStore(s.resolve(s.data.get("macro_root", "data/online/macro")), feed=Yahoo())
    close = store.panel(uni, "1d", "close")
    if close.empty:
        return None
    high = store.panel(uni, "1d", "high").reindex(columns=close.columns)
    low = store.panel(uni, "1d", "low").reindex(columns=close.columns)
    vol = store.panel(uni, "1d", "volume").reindex(columns=close.columns)
    bil = store.load("BIL", "1d")
    rf = bil.set_index("ts")["close"].pct_change().reindex(close.index).fillna(0.0) if len(bil) else None
    if cutoff is not None:
        keep = close.index + pd.Timedelta(days=1) <= pd.Timestamp(cutoff)
        close, high, low, vol = close[keep], high[keep], low[keep], vol[keep]
        rf = rf[keep] if rf is not None else None
    return Market(bar="1b", close=close, high=high, low=low, volume=vol, funding=None, rf=rf)


def evaluate(strategy: Strategy, weights: pd.DataFrame, market: Market, costs: CostModel, s: Settings, style: str) -> Tuple[Dict[str, Any], BacktestResult]:
    res = run_style(style, weights, market, costs, strategy.instrument, strategy.name)
    wf = walk_forward(res, n_trials=strategy.n_variants_tried, holdout_days=int(s.bench.holdout_days), fold_days=int(s.bench.fold_days),
                      warmup_bars=WARMUP_BARS.get(market.bar, 0), participation=float(s.bench.participation),
                      close=market.close, volume=market.volume)
    return wf, res


def _slim(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if not isinstance(v, (dict, list))}


def bench_one(strategy: Strategy, market: Market, costs: CostModel, s: Settings) -> Dict[str, Any]:
    """One strategy at the configured style (kept for callers that want a single result)."""
    style = str(s.execution.get("style", "taker"))
    wf, res = evaluate(strategy, strategy.targets(market), market, costs, s, style)
    stage = promotion_stage(wf, s.gates)
    prof = capability_profile(strategy, wf, stage)
    return {"name": strategy.name, "bar": strategy.bar, "instrument": strategy.instrument, "description": strategy.description,
            "family_id": strategy.family_id, "params": prof.metadata["params"], "n_variants_tried": strategy.n_variants_tried,
            "walk_forward": wf, "promotion": stage, "profile": prof.to_dict(), "_result": res}


def _views(x: pd.Series, hold_start: pd.Timestamp) -> Dict[str, Any]:
    return {"oos": stats(x), "pre_holdout": stats(x[x.index < hold_start]), "holdout": stats(x[x.index >= hold_start])}


def _fmt(st: Dict[str, float]) -> str:
    return (f"CAGR {100 * st['cagr']:6.1f}%  vol {100 * st['vol']:5.1f}%  Sharpe {st['sharpe']:5.2f}  maxDD {100 * st['max_dd']:6.1f}%  "
            f"worst day {100 * st['worst_day']:5.1f}%  PSR {100 * st['psr']:3.0f}%")


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
    style = str(s.execution.get("style", "taker"))
    other = "taker" if style == "maker" else "maker"
    hold_days = int(s.bench.holdout_days)

    markets: Dict[str, Market] = {}
    daily = load_market(store, universe, "1d")
    markets["1d"] = daily
    if any(x.bar == "1h" for x in strategies):
        markets["1h"] = load_market(store, list(s.data.get("universe_1h", universe)), "1h", daily_close=daily.close)
    if any(x.bar == "1b" for x in strategies):
        mm = load_macro_market(s)
        if mm is not None:
            markets["1b"] = mm
    for bar, m in markets.items():
        log(f"  {bar}: {len(m.close)} bars x {len(m.close.columns)} symbols, {m.close.index[0].date()} to {m.close.index[-1].date()}, "
            f"funding {'yes' if m.funding is not None else 'no'}")
    log(f"  execution style judged: {style} (the other, {other}, is shown alongside)")

    results: List[Dict[str, Any]] = []
    equities: Dict[str, pd.Series] = {}
    daily_returns: Dict[str, pd.Series] = {}
    daily_alt: Dict[str, pd.Series] = {}
    cache: Dict[str, Tuple[Strategy, Market, pd.DataFrame]] = {}
    caps: Dict[str, float] = {}                                    # sleeve exposure the live per-asset cap allows
    for st in strategies:
        if st.bar not in markets:
            log(f"  skip {st.name}: no {st.bar} data")
            continue
        t0 = time.time()
        try:
            m = markets[st.bar]
            w = st.targets(m)
            wf, res = evaluate(st, w, m, costs, s, style)
            wf_alt, res_alt = evaluate(st, w, m, costs, s, other)
        except Exception as exc:  # noqa: BLE001
            log(f"  [!] {st.name} failed: {type(exc).__name__}: {exc}")
            continue
        stage = promotion_stage(wf, s.gates)
        prof = capability_profile(st, wf, stage)
        cache[st.name] = (st, m, w)
        wmax = res.weights.abs().max(axis=1)
        wmax = wmax[wmax > 1e-9]
        caps[st.name] = float(s.portfolio.max_single_asset) / float(wmax.quantile(0.95)) if len(wmax) else float("inf")
        equities[st.name] = res.equity
        daily_returns[st.name] = res.daily()
        daily_alt[st.name] = res_alt.daily()
        r = {"name": st.name, "bar": st.bar, "instrument": st.instrument, "description": st.description, "family_id": st.family_id,
             "params": prof.metadata["params"], "n_variants_tried": st.n_variants_tried, "execution_style": res.meta.get("style", style),
             "execution": dict(res.meta), "walk_forward": wf, "promotion": stage, "profile": prof.to_dict(),
             "alt": {"style": res_alt.meta.get("style", other), "dev": _slim(wf_alt["dev"]), "holdout": _slim(wf_alt["holdout"]),
                     "folds_positive_share": wf_alt["folds_positive_share"]},
             "seconds": round(time.time() - t0, 1)}
        results.append(r)
        d, h, ad, ah = wf["dev"], wf["holdout"], wf_alt["dev"], wf_alt["holdout"]
        log(f"  {st.name:15s} [{r['execution_style']:5s}] dev CAGR {100 * d['cagr']:6.1f}% SR {d['sharpe']:5.2f} DSR {100 * d['dsr']:3.0f}% maxDD {100 * d['max_dd']:6.1f}% | "
            f"holdout CAGR {100 * h['cagr']:6.1f}% SR {h['sharpe']:5.2f} | folds+ {100 * wf['folds_positive_share']:3.0f}% cost {100 * min(d['cost_share'], 9.99):4.0f}% "
            f"fill {100 * res.meta.get('fill_rate', 0):3.0f}% | {r['alt']['style']}: dev SR {ad['sharpe']:5.2f} holdout SR {ah['sharpe']:5.2f} -> {stage['stage']}")

    # ------------------------------------------------------------------ regimes
    btc = daily.close["BTCUSDT"] if "BTCUSDT" in daily.close.columns else daily.close.iloc[:, 0]
    reg = regime_labels(btc)
    hold_start = daily.close.index[-1] - pd.Timedelta(days=hold_days)
    by_regime: List[Dict[str, Any]] = []
    for r in results:
        dr = daily_returns[r["name"]]
        table = regime_table(dr[dr.index < hold_start], reg)          # development days only: the holdout stays untouched
        r["profile"]["metadata"]["regimes"] = table
        r["regimes"] = table
        overall = float(r["walk_forward"]["dev"]["sharpe"])
        for name in REGIMES:
            if name in table:
                t = table[name]
                by_regime.append(CapabilityProfile(
                    task_family=str(r["family_id"]), bar_resolution=str(r["bar"]), regime_tags=[name], horizon_bars=1,
                    estimated_p_solve=float(t["psr"]), expected_net_utility=float(t["mean_daily"]) / BARS_PER_DAY.get(r["bar"], 1.0),
                    calibration_error=float(abs(t["sharpe"] - overall)), sample_count=int(t["n"]),
                    last_updated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    metadata={"strategy": r["name"], "regime_sharpe": t["sharpe"], "overall_dev_sharpe": overall, "unit": "sample_count is days",
                              "source": "itrader.online.bench"}).to_dict())

    # ------------------------------------------------------------------ portfolio: walk-forward, benchmark, bootstrap, stress
    portfolio: Dict[str, Any] = {}
    port_view = None
    if len(daily_returns) >= 2:
        t0 = time.time()
        ntr = {st.name: int(st.n_variants_tried) for st in strategies}
        R = pd.DataFrame(daily_returns).sort_index().fillna(0.0)
        R_alt = pd.DataFrame(daily_alt).sort_index().fillna(0.0)
        tk = costs.per_side("perp")
        sims = {"workbench": simulate(R, ntr, s, reg, False, realloc_cost_per_side=tk, expo_caps=caps),
                "workbench_regime": simulate(R, ntr, s, reg, True, realloc_cost_per_side=tk, expo_caps=caps),
                "workbench_no_gate": simulate(R, ntr, s, reg, False, realloc_cost_per_side=tk, apply_gate=False, expo_caps=caps),
                "equal_weight": simulate(R, ntr, s, mode="equal", realloc_cost_per_side=tk, expo_caps=caps),
                f"workbench_{other}": simulate(R_alt, ntr, s, reg, False, realloc_cost_per_side=tk, expo_caps=caps)}
        chosen = "workbench_regime" if bool(s.allocator.get("regime_aware", False)) else "workbench"
        x = sims[chosen]["returns"]
        if len(x):
            btc_all = btc.pct_change()
            btc_r = btc_all.reindex(x.index).fillna(0.0)
            btc_scale = (float(s.portfolio.portfolio_vol_target) / (btc_all.rolling(60, min_periods=20).std() * np.sqrt(365.0))).shift(1).clip(upper=2.0)
            btc_vm = btc_r * btc_scale.reindex(x.index).fillna(0.0)
            variants = {k: _views(v["returns"], hold_start) for k, v in sims.items()}
            for k, v in sims.items():
                variants[k]["vs_btc"] = alpha_beta(v["returns"], btc_r)
            for r in results:                                           # how much of each sleeve is simply BTC exposure
                r["vs_btc"] = alpha_beta(daily_returns[r["name"]], btc_all)
            stress: Dict[str, Any] = {}
            for mult in list(s.bench.get("stress_cost_multiples", [1.0, 2.0, 3.0])):
                cm = costs.scaled(float(mult))
                Rm = pd.DataFrame({n: run_style(style, w, m, cm, st.instrument, n).daily() for n, (st, m, w) in cache.items()}).sort_index().fillna(0.0)
                xm = simulate(Rm, ntr, s, reg, chosen == "workbench_regime", realloc_cost_per_side=cm.per_side("perp"), expo_caps=caps)["returns"]
                stress[f"{float(mult):g}x"] = {"portfolio": stats(xm), "sleeve_sharpe": {n: float(_sharpe(Rm[n], 365.0)) for n in Rm.columns}}
            hist = sims[chosen]["history"]
            sleeve_cols = [c for c in hist.columns if c in R.columns]
            portfolio = {
                "execution_style": style, "chosen": chosen, "oos_start": str(x.index[0].date()), "oos_end": str(x.index[-1].date()),
                "holdout_start": str(hold_start.date()), "rebalance_days": int(s.bench.get("sim_rebalance_days", 7)),
                "variants": variants,
                "benchmark": {"btc_buy_and_hold": _views(btc_r, hold_start), "btc_vol_matched": _views(btc_vm, hold_start)},
                "bootstrap": block_bootstrap(x, int(s.bench.get("bootstrap_block_days", 20)), int(s.bench.get("bootstrap_draws", 2000))),
                "cost_stress": stress,
                "latest_allocation": hist.iloc[-1].to_dict() if len(hist) else {},
                "average_exposure": {c: float(hist[c].mean()) for c in sleeve_cols}, "average_cash": float(hist["cash"].mean()) if len(hist) else None,
                "share_of_weeks_all_cash": float((hist[sleeve_cols].abs().sum(axis=1) < 1e-9).mean()) if len(hist) else None,
                "sleeve_exposure_caps": {k: (None if v == float("inf") else round(v, 3)) for k, v in caps.items()},
                "regime_now": regime_now(btc), "regime_days": {k: int(v) for k, v in reg.value_counts().items()},
                "method": "weekly re-allocation from past data only; sleeve vol 10% (cap 2x); ERC budgets x capability weights with cash; portfolio vol 12% (cap 2x); "
                          "sleeve exposure capped where the live per-asset cap would clip it; re-allocation charged at the taker cost; every day is out of sample",
            }
            pd.DataFrame({k: v["returns"] for k, v in sims.items()}).assign(btc=btc_r, btc_vol_matched=btc_vm).to_parquet(out / "portfolio_returns.parquet")
            lines = [f"Walk-forward portfolio, out of sample {portfolio['oos_start']} to {portfolio['oos_end']} (holdout from {portfolio['holdout_start']}), execution {style}."]
            for k, v in variants.items():
                ab = v.get("vs_btc") or {}
                lines.append(f"  {k:22s} all OOS  {_fmt(v['oos'])}  | beta to BTC {ab.get('beta', 0):5.2f}, alpha {100 * ab.get('alpha_ann', 0):5.1f}%/yr (t {ab.get('alpha_t', 0):4.1f})")
                lines.append(f"  {'':22s} holdout  {_fmt(v['holdout'])}")
            lines.append("  sleeves against BTC (full sample): " + "; ".join(
                f"{r['name']} beta {(r.get('vs_btc') or {}).get('beta', 0):.2f} alpha {100 * (r.get('vs_btc') or {}).get('alpha_ann', 0):.1f}%/yr t {(r.get('vs_btc') or {}).get('alpha_t', 0):.1f}"
                for r in results if r.get("vs_btc")))
            for k, v in portfolio["benchmark"].items():
                lines.append(f"  {k:22s} all OOS  {_fmt(v['oos'])}")
            b = portfolio["bootstrap"]
            if "cagr" in b:
                lines.append(f"Bootstrap of {chosen} ({b['draws']} draws, {b['block_days']}-day blocks): CAGR 5% {100 * b['cagr']['p05']:.1f}%, median {100 * b['cagr']['p50']:.1f}%, "
                             f"95% {100 * b['cagr']['p95']:.1f}%; max drawdown 5% {100 * b['max_dd']['p05']:.1f}%; chance of a losing record {100 * b['p_losing_record']:.0f}%.")
            for k, v in stress.items():
                lines.append(f"  costs {k:4s} portfolio {_fmt(v['portfolio'])}")
            la = portfolio["latest_allocation"]
            lines.append("Latest allocation (exposure multiplier per sleeve): " + ", ".join(f"{c} {la.get(c, 0):.2f}" for c in sleeve_cols if abs(la.get(c, 0)) > 1e-9)
                         + f"; cash {100 * float(la.get('cash', 0)):.0f}%; regime now {portfolio['regime_now']}.")
            portfolio["text"] = "\n".join(lines)
            for ln in lines:
                log("  " + ln)
            log(f"  (portfolio simulations {time.time() - t0:.0f}s)")
            port_view = {"equity": (1.0 + x).cumprod(), "text": portfolio["text"]}

    # ------------------------------------------------------------------ artefacts
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    (out / "profiles.json").write_text(json.dumps([r["profile"] for r in results], indent=1, default=str))
    (out / "profiles_by_regime.json").write_text(json.dumps(by_regime, indent=1, default=str))
    (out / "portfolio.json").write_text(json.dumps(portfolio, indent=1, default=str))
    table = pd.DataFrame([{
        "strategy": r["name"], "bar": r["bar"], "instrument": r["instrument"], "style": r["execution_style"], "stage": r["promotion"]["stage"],
        "dev_cagr": r["walk_forward"]["dev"]["cagr"], "dev_sharpe": r["walk_forward"]["dev"]["sharpe"], "dev_dsr": r["walk_forward"]["dev"]["dsr"],
        "dev_max_dd": r["walk_forward"]["dev"]["max_dd"], "dev_worst_day": r["walk_forward"]["dev"]["worst_day"],
        "holdout_cagr": r["walk_forward"]["holdout"]["cagr"], "holdout_sharpe": r["walk_forward"]["holdout"]["sharpe"],
        "holdout_max_dd": r["walk_forward"]["holdout"]["max_dd"], "folds_positive": r["walk_forward"]["folds_positive_share"],
        "turnover_per_year": r["walk_forward"]["dev"]["turnover_per_year"], "cost_share": r["walk_forward"]["dev"]["cost_share"],
        "fill_rate": r["execution"].get("fill_rate"), "maker_share": r["execution"].get("maker_share"),
        "alt_style": r["alt"]["style"], "alt_dev_sharpe": r["alt"]["dev"]["sharpe"], "alt_holdout_sharpe": r["alt"]["holdout"]["sharpe"],
        "years_needed": r["walk_forward"]["dev"]["evidence"]["years_needed"], "capacity_usd": r["walk_forward"]["dev"].get("capacity", {}).get("capacity_usd"),
    } for r in results])
    table.to_csv(out / "results.csv", index=False)
    if daily_returns:
        pd.DataFrame(daily_returns).sort_index().to_parquet(out / "returns_daily.parquet")
    cost_info = {**{k: float(getattr(costs, k)) for k in ("fee_bps", "spot_fee_bps", "half_spread_bps", "slippage_bps", "maker_fee_bps", "maker_through_bps")},
                 "participation": float(s.bench.participation), "holdout_days": hold_days, "style": style}
    notes = ["Expect single to low double digit annual returns from classic strategies after costs. A high development Sharpe with a weak holdout "
             "is the normal shape of an overfit; a strategy that fails here is reported as failing, not tuned until it passes.",
             f"Execution judged: {style}. Maker means resting limit orders at the touch that fill only on a trade-through, with a taker fallback after "
             f"{costs.maker_max_wait_bars} bar; the other style is shown beside it. Paper trading must confirm the fill model before it is trusted.",
             "The prediction sleeve (Polymarket, paper only) is not benched here; it reports through its own kit."]
    (out / "report.html").write_text(render(results, equities, f"iTRADER online bench ({root.name})", cost_info, notes, portfolio=port_view))
    log(f"  wrote {out / 'report.html'}")
    return {"results": results, "out_dir": str(out), "table": table, "equities": equities, "daily_returns": daily_returns, "portfolio": portfolio}
