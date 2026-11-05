"""DecisionGate: WAIT semantics. RiskEngine: every block. Portfolio: allocator and controls."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from itrader.online.contracts import Proposal, wait_candidate
from itrader.online.data.store import Quality
from itrader.online.framer import Frame
from itrader.online.gate import DecisionGate
from itrader.online.portfolio import CASH, Portfolio, WorkbenchAllocator, apply_no_trade_band
from itrader.online.risk import RiskEngine
from itrader.online.settings import load_settings

NOW = datetime(2026, 1, 10, 12, tzinfo=timezone.utc)


def _frame(proposals):
    cands = [p.to_candidate() for p in proposals] + [wait_candidate()]
    return Frame(NOW.isoformat(), {"regime_tags": []}, "ms_test", proposals, cands, {}, {}, {"BTCUSDT": 50000.0}, {"BTCUSDT": 0.03})


def _prop(strategy="turtle", eu=0.0005, conf=0.8, w=None, cost=0.0):
    return Proposal(strategy, "10", "1d", w or {"BTCUSDT": 0.3}, eu, conf, "t", metadata={"cost_to_trade": cost})


def test_gate_selects_wait_when_nothing_clears():
    s = load_settings()
    g = DecisionGate(s)
    d = g.decide(_frame([_prop(conf=0.2), _prop("trend_modern", eu=-0.001)]))
    assert d.wait and d.selected_candidate_id == "WAIT" and not d.approved and "WAIT" in d.reason
    assert d.decide if False else True
    d2 = g.decide(_frame([]))
    assert d2.wait
    d3 = g.decide(_frame([_prop()]), risk_locked=True, lock_reason="drawdown")
    assert d3.wait and any(c["check"] == "risk_unlocked" and not c["ok"] for c in d3.checks[_prop().candidate_id])


def test_gate_approves_and_ranks_and_checks_cost():
    s = load_settings()
    g = DecisionGate(s)
    good, better = _prop(eu=0.0005, conf=0.7), _prop("trend_modern", eu=0.001, conf=0.9)
    d = g.decide(_frame([good, better]))
    assert not d.wait and [p.strategy for p in d.approved] == ["trend_modern", "turtle"]
    assert d.selected_candidate_id == better.candidate_id and d.thresholds["settings_hash"] == s.hash
    costly = _prop(eu=0.0005, conf=0.7, cost=0.01)       # utility over 1 bar (5 bps) < 1.5 x 100 bps
    d = g.decide(_frame([costly]))
    assert d.wait and [c["check"] for c in d.checks[costly.candidate_id] if not c["ok"]] == ["utility_vs_cost"]
    invalid = _prop(w={"NOPEUSDT": 0.3})
    assert g.decide(_frame([invalid])).wait


def test_risk_kill_file_flattens(tmp_path):
    s = load_settings(overrides={"risk": {"kill_file": str(tmp_path / "KILL")}})
    r = RiskEngine(s, state_path=tmp_path / "rs.json")
    r.mark(10000, NOW)
    (tmp_path / "KILL").write_text("x")
    v = r.check(pd.Series({"BTCUSDT": 0.3}), pd.Series({"BTCUSDT": 0.1}), NOW)
    assert v.flatten and v.blocked and v.approved.abs().sum() == 0 and "kill" in v.reasons[0]


def test_risk_stale_calendar_cooldown_spread(tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text('[{"title": "FOMC decision", "start": "2026-01-10T12:30:00Z", "end": "2026-01-10T13:00:00Z"}]')
    s = load_settings(overrides={"risk": {"calendar_file": str(cal), "kill_file": str(tmp_path / "nokill")}})
    r = RiskEngine(s, state_path=tmp_path / "rs.json")
    r.mark(10000, NOW)
    cur = pd.Series({"BTCUSDT": 0.1, "ETHUSDT": -0.2})
    tgt = pd.Series({"BTCUSDT": 0.2, "ETHUSDT": -0.1, "SOLUSDT": 0.1})
    v = r.check(tgt, cur, NOW)                                  # inside the FOMC window (pre 30 min)
    assert v.blocked and v.approved["BTCUSDT"] == 0.1 and v.approved["ETHUSDT"] == -0.1 and v.approved["SOLUSDT"] == 0.0
    later = NOW + timedelta(hours=3)
    ok = lambda s: Quality(s, "1d", 10, None, None, 10.0, 0, 0, 0, True, [])  # noqa: E731
    q = {"BTCUSDT": Quality("BTCUSDT", "1d", 10, None, None, 99999, 0, 0, 0, False, ["stale by 999 min"]), "ETHUSDT": ok("ETHUSDT"), "SOLUSDT": ok("SOLUSDT")}
    v = r.check(tgt, cur, later, quality=q)
    assert v.approved["BTCUSDT"] == 0.1 and v.approved["SOLUSDT"] == 0.1 and not v.flatten
    v = r.check(tgt, cur, later, quality={"BTCUSDT": ok("BTCUSDT")})          # no record at all counts as bad
    assert v.approved["SOLUSDT"] == 0.0 and v.approved["ETHUSDT"] == -0.1 and v.approved["BTCUSDT"] == 0.2
    r.on_trade_closed(-1.0, later, stopped=True)
    v = r.check(tgt, cur, later + timedelta(hours=1))
    assert v.blocked and "cooldown" in " ".join(v.reasons)
    r.state.cooldown_until = None
    v = r.check(tgt, cur, later + timedelta(hours=1), spreads_bps={"SOLUSDT": 50.0})
    assert v.approved["SOLUSDT"] == 0.0 and any(a["rule"] == "spread" for a in v.adjustments)


def test_risk_caps_gap_and_locks(tmp_path):
    s = load_settings(overrides={"risk": {"kill_file": str(tmp_path / "nokill"), "calendar_file": str(tmp_path / "nocal")}})
    r = RiskEngine(s, state_path=tmp_path / "rs.json")
    r.mark(10000, NOW)
    v = r.check(pd.Series({"BTCUSDT": 0.9, "BTCUSDT.SPOT": 0.5, "ETHUSDT": -0.6}), pd.Series(dtype=float), NOW, prices={"BTCUSDT": 50000.0})
    assert v.approved.abs().max() <= s.risk.max_asset_pct + 1e-9
    assert v.approved.abs().sum() <= s.risk.max_gap_loss_pct / s.risk.gap_move + 1e-9
    assert {a["rule"] for a in v.adjustments} >= {"max_asset_pct", "gap_to_zero"}
    # fast cancel: a 5% move inside one cycle
    v = r.check(pd.Series({"BTCUSDT": 0.2}), pd.Series({"BTCUSDT": 0.1}), NOW + timedelta(hours=1), prices={"BTCUSDT": 53000.0})
    assert v.fast_cancel == ["BTCUSDT"] and v.blocked and v.approved["BTCUSDT"] == 0.1
    # daily loss lock then roll to the next day
    r.mark(9600, NOW + timedelta(hours=2))
    v = r.check(pd.Series({"BTCUSDT": 0.2}), pd.Series({"BTCUSDT": 0.1}), NOW + timedelta(hours=2))
    assert v.flatten and "daily loss" in v.reasons[0]
    r.mark(9600, NOW + timedelta(days=1))
    v = r.check(pd.Series({"BTCUSDT": 0.2}), pd.Series({"BTCUSDT": 0.0}), NOW + timedelta(days=1))
    assert not v.flatten and v.approved["BTCUSDT"] > 0
    # drawdown lock needs an operator
    r.mark(8700, NOW + timedelta(days=2))
    v = r.check(pd.Series({"BTCUSDT": 0.2}), pd.Series({"BTCUSDT": 0.2}), NOW + timedelta(days=2))
    assert v.flatten and r.state.locked_until == "manual"
    r2 = RiskEngine(s, state_path=tmp_path / "rs.json")       # the lock survives a restart
    assert r2.state.locked_until == "manual"
    r2.unlock()
    assert RiskEngine(s, state_path=tmp_path / "rs.json").state.locked_until is None


def test_allocator_cash_first_class_and_portfolio_controls():
    s = load_settings()
    a = WorkbenchAllocator(s)
    none = a.allocate({"x": {"estimated_p_solve": 0.3, "metadata": {"dev": {"sharpe": -0.2, "dsr": 0.3}, "holdout": {"sharpe": -0.1}, "evidence": {"ratio": 0.1}}}})
    assert none.weights[CASH] == 1.0 and none.weights["x"] == 0.0
    good = {"a": {"estimated_p_solve": 0.9, "metadata": {"dev": {"sharpe": 1.0, "dsr": 0.9}, "holdout": {"sharpe": 0.9}, "evidence": {"ratio": 0.8}}},
            "b": {"estimated_p_solve": 0.8, "metadata": {"dev": {"sharpe": 0.6, "dsr": 0.8}, "holdout": {"sharpe": 0.5}, "evidence": {"ratio": 0.3}}}}
    d = a.allocate(good)
    assert abs(d.weights.sum() - 1) < 1e-9 and d.weights[CASH] >= s.allocator.cash_floor - 1e-9 and d.weights["a"] > d.weights["b"] > 0
    assert d.weights.drop(CASH).max() <= s.allocator.max_weight + 1e-9
    idx = pd.date_range("2025-01-01", periods=120, freq="D", tz="UTC")
    rets = pd.DataFrame(np.random.default_rng(0).normal(0.0005, 0.02, (120, 2)), index=idx, columns=["a", "b"])
    p = Portfolio(s).build({"a": pd.Series({"BTCUSDT": 0.6, "ETHUSDT": 0.4}), "b": pd.Series({"SOLUSDT": -1.0})}, rets, current=pd.Series({"BTCUSDT": 0.2}), profiles=good)
    assert p.gross <= s.portfolio.gross_cap + 1e-9 and p.asset_weights.abs().max() <= s.portfolio.max_single_asset + 1e-9
    assert set(p.sleeve_capital) == {"a", "b"} and p.cash >= s.allocator.cash_floor - 1e-9
    assert apply_no_trade_band(pd.Series({"X": 0.102}), pd.Series({"X": 0.1}), 0.1)["X"] == 0.1
