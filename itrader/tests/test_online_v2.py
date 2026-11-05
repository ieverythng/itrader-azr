"""Second round: maker execution (bench and paper), walk-forward portfolio simulation, regimes,
stress tests, and the strategy fixes that came out of the independent audit."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from itrader.online.bench import CostModel, backtest, backtest_maker
from itrader.online.bench.portfolio_sim import REGIMES, alpha_beta, profile_lite, regime_labels, regime_now, simulate
from itrader.online.bench.stress import block_bootstrap, wealth_paths, wealth_table
from itrader.online.execution import Executor, PaperBroker
from itrader.online.portfolio import CASH, WorkbenchAllocator
from itrader.online.settings import load_settings
from itrader.online.strategies import Market, make
from itrader.tests.test_online_contracts import synthetic_market

NOW = datetime(2026, 1, 10, 12, tzinfo=timezone.utc)
MARKS = {"BTCUSDT": 50000.0, "ETHUSDT": 2000.0}
CM = CostModel(fee_bps=5, spot_fee_bps=10, half_spread_bps=1, slippage_bps=2, maker_fee_bps=2, maker_through_bps=1, maker_max_wait_bars=1)


def _flat(n=12, price=100.0, rng=0.5):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = pd.DataFrame({"BTCUSDT": [price] * n}, index=idx)
    return Market("1d", close, close + rng, close - rng, close * 0 + 1, None)


def _in_and_out(m):
    w = pd.DataFrame(0.0, index=m.close.index, columns=m.close.columns)
    w.iloc[2:6, 0] = 0.5
    return w


# ----------------------------------------------------------------------------- maker engine
def test_maker_fills_on_trade_through_and_pays_only_the_maker_fee():
    m = _flat(rng=0.5)
    r = backtest_maker(_in_and_out(m), m, CM, "perp")
    assert r.meta["fill_rate"] == 1.0 and r.meta["maker_share"] == 1.0
    assert math.isclose(r.cost.sum(), 1.0 * 2e-4)                               # 0.5 in and 0.5 out at 2 bps
    assert math.isclose(r.gross.iloc[3], 0.5 * (100 / 99.99 - 1), rel_tol=1e-9)  # bought at the bid, marked at the close
    assert r.turnover.iloc[3] == 0.5 and r.turnover.iloc[7] == 0.5


def test_maker_falls_back_to_taker_one_bar_late_when_nothing_trades_through():
    m = _flat(rng=0.0)                                                           # high == low == close: a touch is not a fill
    r = backtest_maker(_in_and_out(m), m, CM, "perp")
    assert r.meta["fill_rate"] == 0.0 and r.meta["maker_share"] == 0.0
    assert math.isclose(r.cost.sum(), 1.0 * 8e-4)                               # full taker cost both ways
    assert r.weights.iloc[3, 0] == 0.0 and r.weights.iloc[4, 0] == 0.5          # a bar later than the taker engine
    assert backtest(_in_and_out(m), m, CM, "perp").weights.iloc[3, 0] == 0.5
    carry = backtest_maker(_in_and_out(m), m, CM, "spot+perp")                  # two-legged carry never rests
    assert carry.meta["style"] == "taker"


def test_maker_engine_has_no_lookahead_and_costs_scale():
    m = synthetic_market("1d", n=400, seed=2)
    w = make("turtle").targets(m)
    full = backtest_maker(w, m, CM, "perp")
    cut = 300
    close_t = m.close.index[cut - 1] + pd.Timedelta(days=1)
    m2 = Market("1d", m.close.iloc[:cut], m.high.iloc[:cut], m.low.iloc[:cut], m.volume.iloc[:cut], m.funding[m.funding.index <= close_t])
    part = backtest_maker(w.iloc[:cut], m2, CM, "perp")
    assert np.allclose(full.net.iloc[:cut].to_numpy(), part.net.to_numpy(), atol=1e-12)
    assert backtest(w, m, CM.scaled(2.0), "perp").cost.sum() == pytest.approx(2 * backtest(w, m, CM, "perp").cost.sum())
    assert CM.round_trip("spot+perp") == pytest.approx(2 * (10 + 5 + 2 * 3) / 1e4)   # both fees and both spreads, in and out


# ----------------------------------------------------------------------------- maker orders in the paper broker
def _maker_settings():
    return load_settings(overrides={"execution": {"use_live_book": False, "partial_fill_prob": 0.0, "style": "maker"}})


def test_paper_limit_order_rests_then_fills_when_the_market_trades_through(tmp_path):
    s = _maker_settings()
    b = PaperBroker(s, state_path=tmp_path / "b.json", equity_start=10000)
    ex = Executor(b, s)
    rep = ex.rebalance(pd.Series({"BTCUSDT": 0.2}), MARKS, NOW)
    o = rep.orders[0]
    assert rep.maker["posted"] == 1 and o.type == "limit" and o.open and not b.positions()
    assert o.limit_price == pytest.approx(50000 * (1 - 1e-4))                    # the bid
    later = NOW + timedelta(hours=1)
    rep2 = ex.rebalance(pd.Series({"BTCUSDT": 0.2}), MARKS, later, ranges={"BTCUSDT": (o.limit_price * 0.999, 50500.0, NOW)})
    assert rep2.maker["filled_since_last_cycle"] == 1 and rep2.maker["chased_as_taker"] == 0
    f = o.fills[0]
    assert f.price == o.limit_price and f.fee == pytest.approx(f.qty * f.price * 2e-4) and o.meta["filled_as"] == "maker"
    assert ex.current_weights(MARKS)["BTCUSDT"] == pytest.approx(0.2, abs=2e-3) and not b.open_orders()


def test_paper_limit_order_is_chased_as_a_taker_when_it_does_not_fill(tmp_path):
    s = _maker_settings()
    b = PaperBroker(s, state_path=tmp_path / "b.json", equity_start=10000)
    ex = Executor(b, s)
    o = ex.rebalance(pd.Series({"BTCUSDT": 0.2}), MARKS, NOW).orders[0]
    later = NOW + timedelta(hours=1)
    touch_only = {"BTCUSDT": (o.limit_price, 50500.0, NOW)}                      # touched the bid, did not trade through it
    rep2 = ex.rebalance(pd.Series({"BTCUSDT": 0.2}), MARKS, later, ranges=touch_only)
    assert o.state.value == "cancelled" and rep2.maker["unfilled_cancelled"] == 1 and rep2.maker["chased_as_taker"] == 1
    taker = [x for x in rep2.orders if x.type == "market"][0]
    assert taker.state.value == "filled" and taker.meta["filled_as"] == "taker" and taker.fills[0].price > 50000
    # a range from a bar that began before the order was placed proves nothing and must not fill it
    b2 = PaperBroker(s, state_path=tmp_path / "b2.json", equity_start=10000)
    ex2 = Executor(b2, s)
    o2 = ex2.rebalance(pd.Series({"BTCUSDT": 0.2}), MARKS, NOW).orders[0]
    b2.poll(later, MARKS, {"BTCUSDT": (1.0, 99999.0, NOW - timedelta(hours=1))})
    assert o2.open and not o2.fills


def test_urgent_orders_never_rest(tmp_path):
    s = _maker_settings()
    b = PaperBroker(s, state_path=tmp_path / "b.json", equity_start=10000)
    ex = Executor(b, s)
    rep = ex.rebalance(pd.Series({"BTCUSDT": 0.2}), MARKS, NOW, urgent=True)
    assert rep.orders[0].type == "market" and rep.orders[0].state.value == "filled" and rep.maker["style"] == "taker"
    hedged = ex.rebalance(pd.Series({"BTCUSDT": 0.2, "ETHUSDT.SPOT": 0.1, "ETHUSDT": -0.1}), MARKS, NOW + timedelta(minutes=5))
    assert {o.symbol: o.type for o in hedged.orders} == {"ETHUSDT.SPOT": "market", "ETHUSDT": "market"}   # a hedged pair goes together


# ----------------------------------------------------------------------------- portfolio simulation, regimes, stress
def _returns(n=520, seed=0, cols=("a", "b", "c")):
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(np.random.default_rng(seed).normal(0.0004, 0.01, (n, len(cols))), index=idx, columns=list(cols))


def test_walk_forward_simulation_uses_only_the_past():
    s = load_settings(overrides={"bench": {"sim_min_history_days": 200, "sim_rebalance_days": 7}})
    R = _returns()
    base = simulate(R, {}, s)
    R2 = R.copy()
    R2.iloc[400:] = np.random.default_rng(9).normal(0.0, 0.05, (120, 3))        # scramble everything from day 400 on
    alt = simulate(R2, {}, s)["returns"]
    assert len(base["returns"]) == 320 and np.allclose(base["returns"].iloc[:200], alt.iloc[:200])
    assert not np.allclose(base["returns"].iloc[200:], alt.iloc[200:])
    eq = simulate(R, {}, s, mode="equal")
    assert len(eq["history"]) == len(base["history"]) and (base["history"]["cash"] >= s.allocator.cash_floor - 1e-9).all()


def test_simulation_gate_keeps_unproven_sleeves_out():
    s = load_settings(overrides={"bench": {"sim_min_history_days": 200}})
    R = _returns(cols=("good", "bad"))
    R["good"] = R["good"] + 0.002
    R["bad"] = R["bad"] - 0.002
    h = simulate(R, {}, s)["history"]
    assert (h["bad"] == 0).all() and (h["good"] > 0).any()
    assert profile_lite(R["bad"])["expected_net_utility"] < 0 < profile_lite(R["good"])["estimated_p_solve"]


def test_regime_labels_never_use_the_day_they_label():
    idx = pd.date_range("2023-01-01", periods=500, freq="D", tz="UTC")
    btc = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.02, 500))), index=idx)
    lab = regime_labels(btc)
    bumped = btc.copy()
    bumped.iloc[-1] *= 3.0
    assert (regime_labels(bumped) == lab).all()
    assert set(lab.unique()) <= set(REGIMES) | {"unknown"} and regime_now(btc) in set(REGIMES)
    alloc = WorkbenchAllocator(load_settings())
    prof = {"x": {"estimated_p_solve": 0.9, "metadata": {"dev": {"sharpe": 1.0, "dsr": 0.9}, "holdout": {"sharpe": 1.0}, "evidence": {"ratio": 1.0},
                                                         "regimes": {"down_wild": {"n": 180, "sharpe": -1.0}}}}}
    assert alloc.allocate(prof).weights["x"] > 0
    assert alloc.allocate(prof, regime="down_wild").weights[CASH] == 1.0        # half weight on a -1 regime Sharpe: score falls to zero


def test_bootstrap_and_wealth_paths():
    r = pd.Series(np.random.default_rng(5).normal(0.0005, 0.01, 800))
    b1, b2 = block_bootstrap(r, draws=300), block_bootstrap(r, draws=300)
    assert b1 == b2 and b1["cagr"]["p05"] < b1["cagr"]["p50"] < b1["cagr"]["p95"] and 0 <= b1["p_losing_record"] <= 1
    w0 = wealth_paths(r, 1000, 0, (1, 3), draws=300)
    w1 = wealth_paths(r, 1000, 200, (1, 3), draws=300)
    assert w1["rows"][-1]["paid_in"] == 1000 + 200 * 36 and w1["rows"][-1]["p50"] > w0["rows"][-1]["p50"]
    assert "paid in" in wealth_table(w1) and block_bootstrap(r.iloc[:30]).get("note")
    ab = alpha_beta(0.5 * r + 0.001, r)
    assert ab["beta"] == pytest.approx(0.5) and ab["alpha_ann"] == pytest.approx(0.365)


# ----------------------------------------------------------------------------- audit fixes to the strategies
def test_seasonality_reads_the_calendar_of_the_bar_it_holds():
    idx = pd.date_range("2024-03-10", periods=72, freq="h", tz="UTC")
    close = pd.DataFrame({"BTCUSDT": 100.0, "ETHUSDT": 50.0}, index=idx)
    on = make("seasonality").targets(Market("1h", close))["BTCUSDT"] > 0
    t = lambda x: pd.Timestamp(x, tz="UTC")  # noqa: E731
    assert on[t("2024-03-11 12:00")] and not on[t("2024-03-11 11:00")]           # row 12:00 holds the 13:00 bar
    assert on[t("2024-03-11 19:00")] and not on[t("2024-03-11 20:00")]           # row 20:00 would hold the 21:00 bar


def test_carry_needs_funding_data_and_leaves_room_for_margin():
    m = synthetic_market("1d", n=200, seed=4)
    fund = pd.DataFrame(0.0006, index=m.funding.index, columns=m.funding.columns)   # 0.06% per 8 hours: well over the hurdle
    w = make("funding_carry").targets(Market("1d", m.close, m.high, m.low, m.volume, fund))
    last = w.iloc[-1]
    assert (last > 0).all() and last.sum() <= 0.80 + 1e-9 and last.max() <= 0.34
    assert (w.diff().abs().sum(axis=1).iloc[30:] < 1e-12).all()                  # fixed slots: no daily re-sizing
    stale = make("funding_carry").targets(Market("1d", m.close, m.high, m.low, m.volume, fund.iloc[:-30]))
    assert stale.iloc[-1].sum() == 0.0                                           # no data, no position


def test_pairs_legs_follow_the_hedge_ratio():
    idx = pd.date_range("2024-01-01", periods=24 * 90, freq="h", tz="UTC")
    rng = np.random.default_rng(8)
    x = np.cumsum(rng.normal(0, 0.004, len(idx)))
    noise = np.zeros(len(idx))
    for i in range(1, len(idx)):
        noise[i] = 0.97 * noise[i - 1] + rng.normal(0, 0.004)
    close = pd.DataFrame({"AAAUSDT": 100 * np.exp(0.5 * x + noise), "BBBUSDT": 100 * np.exp(x)}, index=idx)
    w = make("pairs", pairs=(("AAAUSDT", "BBBUSDT"),)).targets(Market("1h", close))
    live = w[(w["AAAUSDT"] != 0)]
    assert len(live) > 50 and np.isfinite(w.to_numpy()).all()
    assert ((live["AAAUSDT"] * live["BBBUSDT"]) < 0).mean() > 0.95               # opposite legs
    assert ((live.abs().sum(axis=1) - 1.0).abs() < 1e-9).all()                   # the pair uses its whole slice, split 1 : beta
    ratio = (live["BBBUSDT"].abs() / live["AAAUSDT"].abs()).median()
    assert 0.2 < ratio < 1.0                                                     # near the true 0.5, not the old fixed 1.0


def test_momentum_book_is_frozen_between_calendar_rebalances():
    m = synthetic_market("1d", n=400, seed=6)
    w = make("xs_momentum").targets(m)
    changed = w.diff().abs().sum(axis=1) > 1e-12
    assert set(w.index[changed].dayofweek) <= {0}                                # only Mondays
    later_start = make("xs_momentum").targets(Market("1d", m.close.iloc[3:], m.high.iloc[3:], m.low.iloc[3:], m.volume.iloc[3:], m.funding))
    assert np.allclose(w.iloc[200:].to_numpy(), later_start.iloc[197:].to_numpy(), atol=1e-6)   # same days whatever the history length
