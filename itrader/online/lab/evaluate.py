"""The lab's engine: evaluate recipes, keep the ledger, choose (or keep) a champion.

    Lab(s).cycle()   evaluate every registered recipe on the current data, apply the champion rule,
                     write league.json / champion.json / returns.parquet and the advisor's report.

Honesty rules built in:
* Every evaluation is appended to `ledger.jsonl`. The number of distinct recipes ever evaluated, plus
  the variants tried before the lab existed, is the trial count that deflates every Sharpe ratio.
* Selection uses the research window only. The vault (the most recent `vault_days`) is opened for
  the champion alone, after it is chosen, and reported whatever it shows.
* A challenger replaces a champion only with a higher deflated Sharpe AND daily returns that beat the
  champion's with a t-statistic of at least `switch_t`. Otherwise the incumbent stays.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..bench.engine import CostModel, run_style
from ..bench.metrics import _moments, deflated_sharpe
from ..bench.portfolio_sim import alpha_beta, simulate, stats
from ..bench.run import load_macro_market, load_market
from ..bench.stress import block_bootstrap
from ..data.store import DataStore
from ..settings import Settings, settings as default_settings
from ..strategies import BENCHABLE, make
from .policy import SleeveMeta, simulate_policy
from .registry import CRITERIA, RECIPES, Recipe


def paired_t(a: pd.Series, b: pd.Series) -> float:
    """t-statistic of the mean daily difference a - b."""
    d = (a - b).dropna()
    if len(d) < 30 or float(d.std(ddof=1)) <= 0:
        return 0.0
    return float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))


class Lab:
    def __init__(self, s: Optional[Settings] = None, log: Callable[[str], None] = print) -> None:
        self.s = s or default_settings()
        self.log = log
        self.root = self.s.resolve(self.s.lab.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = DataStore(self.s.resolve(self.s.general.data_root))
        self.costs = CostModel.from_settings(self.s)
        self.style = str(self.s.execution.get("style", "taker"))
        self._markets: Dict[str, Dict[str, Any]] = {}
        self._targets: Dict[Any, pd.DataFrame] = {}
        self._sets: Dict[Any, Dict[str, Dict[str, Any]]] = {}
        self.returns: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------ sleeves
    def markets(self, universe_key: str) -> Dict[str, Any]:
        if universe_key not in self._markets:
            uni = list(self.s.data.get("universe_core", self.s.data.universe)) if universe_key == "core10" else list(self.s.data.universe)
            d = load_market(self.store, uni, "1d")
            mk: Dict[str, Any] = {"1d": d, "1h": load_market(self.store, list(self.s.data.get("universe_1h", uni)), "1h", daily_close=d.close)}
            mm = load_macro_market(self.s)
            if mm is not None:
                mk["1b"] = mm
            self._markets[universe_key] = mk
        return self._markets[universe_key]

    def sleeve_set(self, universe_key: str, cost_mult: float = 1.0) -> Dict[str, Dict[str, Any]]:
        key = (universe_key, cost_mult)
        if key in self._sets:
            return self._sets[key]
        mk = self.markets(universe_key)
        costs = self.costs.scaled(cost_mult) if cost_mult != 1.0 else self.costs
        rk, pf = self.s.risk, self.s.portfolio
        out: Dict[str, Dict[str, Any]] = {}
        for name in BENCHABLE:
            if name == "experimental":
                continue
            st = make(name)
            m = mk.get(st.bar)
            if m is None:
                continue
            tk = (universe_key, name)
            if tk not in self._targets:
                self._targets[tk] = st.targets(m)
            res = run_style(self.style, self._targets[tk], m, costs, st.instrument, name)
            gross = res.weights.abs().sum(axis=1)
            gross_mean = float(gross[gross > 1e-9].mean()) if (gross > 1e-9).any() else 0.0
            wmax = res.weights.abs().max(axis=1)
            wmax = wmax[wmax > 1e-9]
            cap = float(pf.max_single_asset) / float(wmax.quantile(0.95)) if len(wmax) else float("inf")
            if st.instrument == "spot+perp":
                gap, cost = float(rk.get("hedged_gap_factor", 0.2)) * gross_mean, gross_mean * costs.per_side(st.instrument) * costs.legs(st.instrument)
            elif st.instrument == "etf":
                gap, cost = gross_mean * float(rk.get("gap_move_listed", rk.gap_move)) / float(rk.gap_move), gross_mean * costs.per_side("etf")
            else:
                gap, cost = gross_mean, gross_mean * costs.per_side("perp")
            out[name] = {"daily": res.daily(), "bar": st.bar, "instrument": st.instrument, "gross": gross_mean,
                         "meta": SleeveMeta(name, "overlay", int(st.n_variants_tried), cap, gap, cost)}
        self._sets[key] = out
        return out

    # ------------------------------------------------------------------ one recipe
    def run_recipe(self, recipe: Recipe, cost_mult: float = 1.0) -> Dict[str, Any]:
        sl = self.sleeve_set(recipe.universe, cost_mult)
        names = [n for n in recipe.sleeves if n in sl]
        start = sl["turtle"]["daily"].index[0]                           # the crypto data's first day: the common window
        R = pd.DataFrame({n: sl[n]["daily"] for n in names})
        R = R[R.index >= start].fillna(0.0)
        priors = {n: sl[n]["daily"][sl[n]["daily"].index < start] for n in names if sl[n]["daily"].index[0] < start}
        metas = {n: replace(sl[n]["meta"], role=("core" if n in recipe.core else "overlay")) for n in names}
        if recipe.policy == "workbench":
            return simulate(R, {n: metas[n].n_trials for n in names}, self.s, None, False,
                            realloc_cost_per_side=self.costs.per_side("perp") * cost_mult, expo_caps={n: metas[n].expo_cap for n in names})
        return simulate_policy(R, metas, self.s, recipe.policy, priors)

    def vault_start(self, x: pd.Series) -> pd.Timestamp:
        return x.index[-1] - pd.Timedelta(days=int(self.s.lab.vault_days))

    def evaluate(self, recipe: Recipe) -> Dict[str, Any]:
        sim = self.run_recipe(recipe)
        x = sim["returns"]
        self.returns[recipe.name] = x
        vs = self.vault_start(x)
        research = x[x.index < vs]
        x2 = self.run_recipe(recipe, 2.0)["returns"]
        n, mu, sd, sk, ku = _moments(research)
        btc, spy = self.benchmarks(research.index)
        hist = sim["history"]
        cols = [c for c in hist.columns if c in recipe.sleeves] if len(hist) else []
        return {"name": recipe.name, "hypothesis": recipe.hypothesis, "policy": recipe.policy, "universe": recipe.universe, "label": recipe.label,
                "core": list(recipe.core), "overlay": list(recipe.overlay), "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "data_end": str(x.index[-1].date()), "research_start": str(research.index[0].date()), "research_end": str(research.index[-1].date()),
                "settings_hash": self.s.hash, "execution_style": self.style,
                "research": stats(research), "cost2x": stats(x2[x2.index < vs]),
                "bootstrap": block_bootstrap(research, int(self.s.bench.get("bootstrap_block_days", 20)), int(self.s.bench.get("bootstrap_draws", 2000))),
                "vs_btc": alpha_beta(research, btc), "vs_spy": alpha_beta(research, spy),
                "moments": {"n": int(n), "sr_period": (mu / sd) if sd > 0 else 0.0, "skew": sk, "kurt": ku},
                "avg_exposure": {c: float(hist[c].mean()) for c in cols},
                "share_on": {c: float((hist[c] > 1e-9).mean()) for c in cols},
                "latest_exposure": {c: float(hist[c].iloc[-1]) for c in cols} if len(hist) else {}}

    def benchmarks(self, index: pd.DatetimeIndex) -> tuple:
        d = self.markets("core10")["1d"].close
        btc = d["BTCUSDT"].pct_change().reindex(index).fillna(0.0)
        mm = self.markets("core10").get("1b")
        spy = (1.0 + mm.close["SPY"].pct_change()).resample("D").prod().sub(1.0).reindex(index).fillna(0.0) if mm is not None and "SPY" in mm.close else pd.Series(0.0, index=index)
        return btc, spy

    # ------------------------------------------------------------------ ledger and champion
    def ledger(self) -> List[Dict[str, Any]]:
        p = self.root / "ledger.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def champion(self) -> Optional[Dict[str, Any]]:
        p = self.root / "champion.json"
        return json.loads(p.read_text()) if p.exists() else None

    def decide(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        lab = self.s.lab
        seen = {e["name"] for e in self.ledger()} | {r["name"] for r in results}
        n_trials = int(lab.prior_variants) + len(seen)
        for r in results:
            mo = r["moments"]
            d = deflated_sharpe(mo["sr_period"], mo["n"], mo["skew"], mo["kurt"], n_trials)
            r["dsr"], r["n_trials"] = float(d["dsr"]), n_trials
            checks = {"max_drawdown": abs(r["research"]["max_dd"]) <= float(lab.max_drawdown),
                      "p_losing": float(r["bootstrap"].get("p_losing_record", 1.0)) <= float(lab.max_p_losing),
                      "cost2x_sharpe": r["cost2x"]["sharpe"] > float(lab.min_cost2x_sharpe)}
            r["checks"], r["eligible"] = checks, all(checks.values())
        ranked = sorted([r for r in results if r["eligible"]], key=lambda r: r["dsr"], reverse=True)
        inc = self.champion()
        note, chosen = "", None
        if not ranked:
            note = "no recipe met the eligibility constraints; no champion change"
            chosen = next((r for r in results if inc and r["name"] == inc.get("name")), None)
        else:
            best = ranked[0]
            cur = next((r for r in results if inc and r["name"] == inc.get("name")), None)
            if cur is None or best["name"] == cur["name"]:
                chosen, note = best, ("champion re-confirmed on new data" if cur is not None else "first champion: highest deflated Sharpe among eligible recipes")
            else:
                t = paired_t(self.returns[best["name"]][: self.vault_start(self.returns[best["name"]])],
                             self.returns[cur["name"]][: self.vault_start(self.returns[cur["name"]])])
                if cur["eligible"] and (best["dsr"] <= cur["dsr"] or t < float(lab.switch_t)):
                    chosen, note = cur, f"challenger {best['name']} not adopted: t-statistic {t:.2f} against the champion is below {float(lab.switch_t):.1f}"
                else:
                    chosen, note = best, f"challenger {best['name']} replaces {cur['name']}: t-statistic {t:.2f}" + ("" if cur["eligible"] else " (the old champion is no longer eligible)")
        champ: Dict[str, Any] = {"decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": note, "n_trials": n_trials, "name": None}
        if chosen is not None:
            x = self.returns[chosen["name"]]
            v = x[x.index >= self.vault_start(x)]
            vault = stats(v)                                              # the vault is opened here, for the champion only
            prior = inc.get("history", []) if inc else []
            changed = (inc or {}).get("name") != chosen["name"]
            champ.update({"name": chosen["name"], "policy": chosen["policy"], "universe": chosen["universe"], "core": chosen["core"], "overlay": chosen["overlay"],
                          "label": chosen["label"], "since": champ["decided_at"] if changed else (inc or {}).get("since", champ["decided_at"]),
                          "research": chosen["research"], "dsr": chosen["dsr"], "vault": vault, "vault_start": str(v.index[0].date()) if len(v) else None,
                          "confirmed": bool(vault["total_return"] > 0), "latest_exposure": chosen["latest_exposure"],
                          "history": prior + ([{"at": champ["decided_at"], "name": chosen["name"], "note": note}] if changed or not prior else [])})
            for r in results:
                r["champion"] = r["name"] == chosen["name"]
        (self.root / "champion.json").write_text(json.dumps(champ, indent=1, default=str))
        with (self.root / "ledger.jsonl").open("a") as fh:
            for r in results:
                fh.write(json.dumps({k: v for k, v in r.items() if k != "hypothesis"}, default=str) + "\n")
        return champ

    # ------------------------------------------------------------------ the whole loop
    def cycle(self, recipes: Optional[List[Recipe]] = None) -> Dict[str, Any]:
        recipes = recipes or RECIPES
        results = []
        for rc in recipes:
            try:
                r = self.evaluate(rc)
            except Exception as exc:  # noqa: BLE001
                self.log(f"  [!] {rc.name} failed: {type(exc).__name__}: {exc}")
                continue
            results.append(r)
            rs, c2 = r["research"], r["cost2x"]
            self.log(f"  {rc.name:22s} research CAGR {100 * rs['cagr']:6.1f}%  vol {100 * rs['vol']:5.1f}%  Sharpe {rs['sharpe']:5.2f}  maxDD {100 * rs['max_dd']:6.1f}%  "
                     f"| 2x costs Sharpe {c2['sharpe']:5.2f} | P(losing) {100 * r['bootstrap'].get('p_losing_record', 1):3.0f}% | beta BTC {r['vs_btc'].get('beta', 0):5.2f} "
                     f"alpha t {r['vs_btc'].get('alpha_t', 0):4.1f}")
        champ = self.decide(results)
        idx = self.returns[results[0]["name"]].index if results else pd.DatetimeIndex([])
        if len(idx):
            btc, spy = self.benchmarks(idx)
            scale = (float(self.s.portfolio.portfolio_vol_target) / (self.markets("core10")["1d"].close["BTCUSDT"].pct_change().rolling(60, min_periods=20).std() * np.sqrt(365.0))).shift(1).clip(upper=2.0)
            bench = {"btc": btc, "btc_vol_matched": btc * scale.reindex(idx).fillna(0.0), "spy": spy}
            pd.DataFrame({**self.returns, **bench}).to_parquet(self.root / "returns.parquet")
            vs = self.vault_start(btc)
            league = {"decided_at": champ["decided_at"], "n_trials": champ["n_trials"], "criteria": CRITERIA, "results": results,
                      "benchmarks": {k: stats(v[v.index < vs]) for k, v in bench.items()}, "research_window": [str(idx[0].date()), str(vs.date())]}
            (self.root / "league.json").write_text(json.dumps(league, indent=1, default=str))
        for key in {rc.universe for rc in recipes}:                       # what the live loop needs to run the champion's policy
            sl = self.sleeve_set(key)
            pd.DataFrame({n: v["daily"] for n, v in sl.items()}).to_parquet(self.root / f"sleeve_returns_{key}.parquet")
            (self.root / f"sleeve_meta_{key}.json").write_text(json.dumps({n: {**v["meta"].__dict__, "gross": v["gross"], "instrument": v["instrument"]} for n, v in sl.items()}, indent=1, default=str))
        self.log(f"  champion: {champ.get('name')} ({champ.get('note')}); trials counted: {champ['n_trials']}")
        return {"champion": champ, "results": results}
