"""Cost accounting, funding conventions, metrics and the ladder."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from itrader.online.bench import CostModel, backtest, binary_evidence, deflated_sharpe, evidence_meter, probabilistic_sharpe, summarize, walk_forward
from itrader.online.bench.profile import capability_profile, promotion_stage
from itrader.online.settings import load_settings
from itrader.online.strategies import Market, make
from itrader.tests.test_online_contracts import synthetic_market


def _flat_market(n=10, price=100.0, rate=0.001):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = pd.DataFrame({"BTCUSDT": [price] * n, "ETHUSDT": [price] * n}, index=idx)
    funding = pd.DataFrame({"BTCUSDT": [rate] * n, "ETHUSDT": [rate] * n}, index=idx + pd.Timedelta(hours=8))
    return Market("1d", close, close, close, close * 0 + 1, funding)


def test_turnover_and_costs_exact():
    m = _flat_market(rate=0.0)
    w = pd.DataFrame(0.0, index=m.close.index, columns=m.close.columns)
    w.iloc[2:5, 0] = 0.5                                # decided at bars 2,3,4 -> held at 3,4,5
    cm = CostModel(fee_bps=5, half_spread_bps=1, slippage_bps=2)
    r = backtest(w, m, cm, "perp")
    per_side = 8 / 1e4
    assert r.turnover.iloc[3] == 0.5 and r.turnover.iloc[6] == 0.5 and r.turnover.sum() == 1.0
    assert math.isclose(r.cost.sum(), 1.0 * per_side)
    assert r.gross.abs().sum() == 0.0                   # flat prices, no price pnl
    assert (r.weights.iloc[3:6, 0] == 0.5).all() and r.weights.iloc[2, 0] == 0.0   # one-bar shift


def test_funding_sign_conventions():
    m = _flat_market(rate=0.001)
    w = pd.DataFrame(0.0, index=m.close.index, columns=m.close.columns)
    w.iloc[:, 0] = 1.0
    cm = CostModel(0, 0, 0, 0)
    perp = backtest(w, m, cm, "perp")
    assert perp.funding.iloc[1:].lt(0).all(), "a perp long pays positive funding"
    assert math.isclose(perp.funding.iloc[2], -0.001)
    carry = backtest(w, m, cm, "spot+perp")
    assert carry.funding.iloc[1:].gt(0).all() and carry.gross.abs().sum() == 0.0, "carry is price neutral and receives funding"
    assert math.isclose(backtest(w, m, CostModel(5, 10, 0, 0), "spot+perp").cost.iloc[1], 1.0 * (5 + 10) / 1e4), "two legs: the spot fee plus the perp fee"
    spot = backtest(-w, m, cm, "spot")
    assert spot.weights.abs().sum().sum() == 0.0, "spot cannot be short"


def test_metrics_sanity_and_evidence():
    assert 0.49 < probabilistic_sharpe(0.0, 500, 0, 3) < 0.51
    assert probabilistic_sharpe(0.1, 500, 0, 3) > 0.95
    d1, d20 = deflated_sharpe(0.1, 500, 0, 3, 1), deflated_sharpe(0.1, 500, 0, 3, 20)
    assert d1["dsr"] > d20["dsr"] and d20["sr0_period"] > 0
    ev = evidence_meter(0.05, 0, 3, 365, 100)
    assert ev["bars_needed"] > 1000 and ev["ratio"] < 0.1
    assert evidence_meter(-0.1, 0, 3, 365, 100)["bars_needed"] == math.inf
    b = binary_evidence(0.987, 16)
    assert 0.80 < b["p_streak_under_no_edge"] < 0.82 and b["streak_needed_for_alpha"] == 229


def test_walk_forward_holdout_and_profile_format():
    s = load_settings()
    st = make("trend_modern")
    m = synthetic_market("1d", n=1200, seed=3)
    r = backtest(st.targets(m), m, CostModel.from_settings(s), st.instrument, name=st.name)
    wf = walk_forward(r, 1, 182, 182, 130, 0.01, m.close, m.volume)
    assert wf["holdout"]["bars"] in (182, 183) and wf["dev"]["bars"] + wf["holdout"]["bars"] >= len(r.net) - 131
    assert len(wf["folds"]) >= 3 and all(f["bars"] > 100 for f in wf["folds"])
    stage = promotion_stage(wf, s.gates)
    assert stage["stage"] in ("bench", "paper") and stage["rungs"][0]["checks"]
    prof = capability_profile(st, wf, stage)
    d = prof.to_dict()
    assert d["task_family"] == "11" and 0 <= d["estimated_p_solve"] <= 1 and d["metadata"]["n_variants_tried"] == st.n_variants_tried >= 1
    from itrader.rl_lib.tasks.validation import validate_capability_profile

    assert validate_capability_profile(d)["sample_count"] == d["sample_count"]


def test_summary_components_add_up():
    m = synthetic_market("1d", n=500, seed=5)
    st = make("xs_momentum")
    r = backtest(st.targets(m), m, CostModel(5, 10, 1, 2), st.instrument)
    s = summarize(r, 1, 0.01, m.close, m.volume)
    comp = r.components()
    assert math.isclose(comp["gross"] + comp["funding"] - comp["cost"], comp["net"], rel_tol=1e-9)
    assert s["bars"] == len(r.net) and 0 <= s["psr"] <= 1 and np.isfinite(s["max_dd"])
