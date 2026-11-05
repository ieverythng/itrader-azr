"""The lab: allocation policies, the walk-forward evaluator's honesty rules, the champion rule, the advisor."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from itrader.online.bench import CostModel, backtest
from itrader.online.bench.portfolio_sim import stats
from itrader.online.lab.advisor import write_report
from itrader.online.lab.evaluate import Lab, paired_t
from itrader.online.lab.policy import PolicyState, SleeveMeta, exposures, record, simulate_policy
from itrader.online.lab.registry import CRITERIA, RECIPES, by_name
from itrader.online.portfolio import build_from_exposures
from itrader.online.settings import load_settings
from itrader.online.strategies import REGISTRY, Market, make


def _R(n=700, seed=0):
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"good": rng.normal(0.0015, 0.01, n), "weak": rng.normal(0.0, 0.01, n), "bad": rng.normal(-0.0015, 0.01, n),
                         "carry": rng.normal(0.0002, 0.0003, n)}, index=idx)


def _metas(core=("carry",)):
    return {c: SleeveMeta(c, "core" if c in core else "overlay", 1, 1.0 if c in core else 5.0, 0.16 if c in core else 0.4, 0.0008) for c in ("good", "weak", "bad", "carry")}


def test_registry_is_consistent_and_criteria_are_stated():
    assert len({r.name for r in RECIPES}) == len(RECIPES) >= 8
    for r in RECIPES:
        assert r.policy in ("workbench", "steady", "stack") and r.universe in ("core10", "wide25") and r.hypothesis
        assert all(n in REGISTRY for n in r.sleeves) and (r.policy == "stack") == bool(r.core)
    assert by_name("R0_baseline").policy == "workbench" and {"primary", "eligible", "champion", "challenger", "vault"} <= set(CRITERIA)


def test_slow_gate_and_stack_exposures():
    s = load_settings()
    R = _R()
    st = PolicyState()
    expo, info = exposures("stack", R, _metas(), s, st)
    assert info["on"]["good"] and not info["on"]["bad"] and info["on"]["carry"]
    assert expo["carry"] == 1.0 and expo["bad"] == 0.0 and expo["good"] > 0          # the core holds its own notional; losers get nothing
    budget = s.risk.max_gap_loss_pct / s.risk.gap_move
    assert sum(expo[c] * _metas()[c].gap_per_unit for c in expo.index) <= budget + 0.25 * budget   # within the band's tolerance of the gap budget
    again, _ = exposures("stack", R, _metas(), s, st)
    assert (again == expo).all()                                                     # nothing moved: nothing re-traded
    steady, _ = exposures("steady", R, _metas(core=()), s, PolicyState())
    assert steady["bad"] == 0.0 and 0 < steady["carry"] <= 5.0                       # in the common pool carry is one sleeve among others, under its cap
    short = R.iloc[:200]
    young, info2 = exposures("stack", short, _metas(), s, PolicyState())
    assert young.sum() == 0.0 and not any(info2["on"].values())                      # under a year of record: nothing is switched on
    psr, mu, n = record(R["good"])
    assert psr > 0.95 and mu > 0 and n == len(R)


def test_gate_has_memory():
    s = load_settings()
    idx = pd.date_range("2023-01-01", periods=600, freq="D", tz="UTC")
    r = pd.Series(np.random.default_rng(1).normal(0.0, 0.01, 600), index=idx)
    r = r - r.mean() + 0.01 * 0.004                                                 # a record whose PSR sits between gate_off and gate_on
    psr, _, _ = record(r)
    assert s.lab.gate_off < psr < s.lab.gate_on
    R = pd.DataFrame({"x": r})
    m = {"x": SleeveMeta("x", "overlay", 1, 5.0, 0.4, 0.0008)}
    assert not exposures("steady", R, m, s, PolicyState())[1]["on"]["x"]             # not enough to switch on
    assert exposures("steady", R, m, s, PolicyState(on={"x": True}))[1]["on"]["x"]   # enough to stay on


def test_policy_simulation_uses_only_the_past_and_charges_for_changes():
    s = load_settings()
    R = _R()
    base = simulate_policy(R, _metas(), s, "stack")
    R2 = R.copy()
    R2.iloc[600:] = np.random.default_rng(9).normal(0.0, 0.05, (100, 4))
    alt = simulate_policy(R2, _metas(), s, "stack")["returns"]
    k = 600 - int(s.lab.min_history_days)
    assert np.allclose(base["returns"].iloc[:k], alt.iloc[:k]) and not np.allclose(base["returns"].iloc[k:], alt.iloc[k:])
    free = {c: SleeveMeta(m.name, m.role, 1, m.expo_cap, m.gap_per_unit, 0.0) for c, m in _metas().items()}
    assert simulate_policy(R, free, s, "stack")["returns"].sum() > base["returns"].sum()   # re-allocation is not free
    assert len(base["history"]) == int(np.ceil((len(R) - 365) / int(s.lab.rebalance_days)))


def _result(name, sharpe, n=900, dd=-0.1, p_lose=0.05, c2=0.5):
    return {"name": name, "hypothesis": "", "policy": "stack", "universe": "wide25", "label": "", "core": ["carry_core"], "overlay": ["turtle"],
            "research": {"sharpe": sharpe, "max_dd": dd, "cagr": 0.1, "vol": 0.1, "worst_day": -0.02, "psr": 0.9, "total_return": 0.2, "days": n},
            "cost2x": {"sharpe": c2}, "bootstrap": {"p_losing_record": p_lose}, "vs_btc": {"beta": 0.0, "alpha_ann": 0.1, "alpha_t": 1.5},
            "vs_spy": {"beta": 0.0, "alpha_ann": 0.1, "alpha_t": 1.5}, "moments": {"n": n, "sr_period": sharpe / np.sqrt(365), "skew": 0.0, "kurt": 3.0},
            "avg_exposure": {}, "share_on": {}, "latest_exposure": {}}


def _lab(tmp_path):
    lab = Lab.__new__(Lab)                          # no data store needed for the decision rule
    lab.s = load_settings(overrides={"lab": {"root": str(tmp_path)}})
    lab.root, lab.log, lab.returns = tmp_path, (lambda *_: None), {}
    return lab


def test_champion_rule_eligibility_challenger_and_vault(tmp_path):
    lab = _lab(tmp_path)
    idx = pd.date_range("2024-01-01", periods=900, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0.0006, 0.006, 900), index=idx)
    wiggle = 0.0003 * np.where(np.arange(900) % 2 == 0, 1.0, -1.0)               # differs from A every day, by nothing on average
    lab.returns = {"A": a, "B": a + wiggle, "C": a + 0.0015, "X": a}
    # first champion: the best ELIGIBLE recipe, not the best Sharpe (X breaks the drawdown limit)
    c1 = lab.decide([_result("A", 1.0), _result("X", 2.0, dd=-0.4)])
    assert c1["name"] == "A" and c1["n_trials"] == lab.s.lab.prior_variants + 2 and "first champion" in c1["note"]
    assert c1["confirmed"] == (stats(a[a.index >= a.index[-1] - pd.Timedelta(days=182)])["total_return"] > 0) and c1["vault_start"]
    # a challenger with a slightly higher Sharpe whose returns do not clearly beat the champion's is NOT adopted
    c2 = lab.decide([_result("A", 1.0), _result("B", 1.05)])
    assert c2["name"] == "A" and "not adopted" in c2["note"] and paired_t(lab.returns["B"], a) < lab.s.lab.switch_t
    # one that beats it day after day is
    c3 = lab.decide([_result("A", 1.0), _result("C", 1.6)])
    assert c3["name"] == "C" and "replaces A" in c3["note"] and [h["name"] for h in c3["history"]] == ["A", "C"]
    # every trial ever made is counted, and nothing eligible means no change
    c4 = lab.decide([_result("C", 1.6, p_lose=0.9)])
    assert c4["name"] == "C" and "no recipe met" in c4["note"] and c4["n_trials"] == lab.s.lab.prior_variants + 4
    assert len(lab.ledger()) == 7


def test_advisor_writes_a_report_from_the_evidence(tmp_path):
    lab = _lab(tmp_path)
    idx = pd.date_range("2024-01-01", periods=900, freq="D", tz="UTC")
    a = pd.Series(np.random.default_rng(3).normal(0.0006, 0.006, 900), index=idx)
    lab.returns = {"R0_baseline": a - 0.0003, "R1_steady": a}
    res = [_result("R0_baseline", 0.3), _result("R1_steady", 1.0)]
    champ = lab.decide(res)
    pd.DataFrame(lab.returns).to_parquet(tmp_path / "returns.parquet")
    (tmp_path / "league.json").write_text(json.dumps({"results": res, "n_trials": champ["n_trials"], "benchmarks": {"btc": stats(a)},
                                                      "research_window": [str(idx[0].date()), str((idx[-1] - pd.Timedelta(days=182)).date())]}, default=str))
    text = write_report(lab.s, tmp_path, tmp_path / "runs").read_text()
    assert "Champion: `R1_steady`" in text and "a slower allocator with no trailing vote" in text and "What to be careful about" in text
    assert "not statistically established" in text and "No paper cycles recorded yet" in text and "not financial advice" in text


def test_book_from_exposures_and_listed_market_costs():
    s = load_settings()
    legs = {"carry_core": pd.Series({"BTCUSDT.SPOT": 0.2, "BTCUSDT": -0.2, "ETHUSDT.SPOT": 0.2, "ETHUSDT": -0.2}),
            "trend_long": pd.Series({"BTCUSDT": 0.5, "SOLUSDT": 0.5}), "macro_trend": pd.Series({"SPY": 0.3, "TLT": -0.2})}
    d = build_from_exposures(s, legs, {"carry_core": 1.0, "trend_long": 0.2, "macro_trend": 0.5}, policy="test")
    w = d.asset_weights
    assert w["BTCUSDT.SPOT"] == pytest.approx(0.2) and w["BTCUSDT"] == pytest.approx(-0.2 + 0.1) and w["SPY"] == pytest.approx(0.15)
    assert d.cash == pytest.approx(0.6) and d.gross <= s.portfolio.gross_cap and d.sleeve_capital["trend_long"] == 0.2
    # listed markets: margined positions earn the return above the risk-free rate, and shorts pay to borrow
    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    close = pd.DataFrame({"SPY": [100, 101, 102.01, 103.0301, 104.060401, 105.10100501]}, index=idx)
    rf = pd.Series(0.0002, index=idx)
    m = Market("1b", close, close, close, close * 0 + 1, None, None, rf)
    long = backtest(pd.DataFrame({"SPY": 1.0}, index=idx), m, CostModel(etf_cost_bps=0.0), "etf")
    assert long.gross.iloc[2] == pytest.approx(0.01 - 0.0002) and long.ppy == 252.0
    short = backtest(pd.DataFrame({"SPY": -1.0}, index=idx), m, CostModel(etf_cost_bps=0.0, etf_short_borrow_ann=0.0252), "etf")
    assert short.funding.iloc[2] == pytest.approx(-0.0001) and short.gross.iloc[2] == pytest.approx(-(0.01 - 0.0002))
    assert make("macro_trend").bar == "1b" and make("trend_long").params["long_only"] and make("carry_core").params["hurdle_ann"] == 0.04


def test_judge_separates_skill_from_luck_and_counts_trials(tmp_path):
    from itrader.online.lab.judge import format_judgement, judge, load_series

    rng = np.random.default_rng(11)
    strong, noise = pd.Series(rng.normal(0.002, 0.01, 1500)), pd.Series(rng.normal(0.0, 0.01, 1500))
    a, b = judge(strong), judge(noise)
    assert a["significant"] and a["psr"] > 0.99 and a["p_mean_not_positive"] < 0.01
    assert not b["significant"] and "not" in b["verdict"]
    weak = pd.Series(rng.normal(0.0008, 0.01, 600))                       # looks good alone, not once the search is counted
    alone, searched = judge(weak, trials=1), judge(weak, trials=200)
    assert searched["dsr"] < alone["dsr"] and searched["luck_sharpe_per_period"] > 0 and not searched["significant"]
    assert judge(pd.Series([0.01, 0.02]))["significant"] is False
    path = tmp_path / "episodes.jsonl"
    path.write_text("\n".join(json.dumps({"symbol": "X", "net_return": float(v), "horizon": 96}) for v in strong.head(200)))
    loaded = load_series(path, "net_return")
    assert len(loaded) == 200 and "VERDICT" in format_judgement(judge(loaded, trials=8), "episodes.jsonl")
    eq = tmp_path / "eq.csv"
    pd.DataFrame({"equity": (1 + strong.head(100)).cumprod()}).to_csv(eq, index=False)
    assert len(load_series(eq, "equity", equity=True)) == 99
