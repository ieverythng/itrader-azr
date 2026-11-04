"""Paper fills, the live guard, and a full replay through the loop on the real store when present."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from itrader.online.execution import Executor, LiveDisabled, Order, OrderState, PaperBroker
from itrader.online.execution.alpaca_plug import AlpacaBroker
from itrader.online.execution.binance_plug import BinanceBroker
from itrader.online.settings import load_settings, repo_root

NOW = datetime(2026, 1, 10, 12, tzinfo=timezone.utc)
MARKS = {"BTCUSDT": 50000.0, "ETHUSDT": 2000.0}


def _s(**x):
    return load_settings(overrides={"execution": {"use_live_book": False, **x}})


def test_paper_fill_price_fee_and_determinism(tmp_path):
    s = _s(partial_fill_prob=0.0)
    b = PaperBroker(s, state_path=tmp_path / "b.json", equity_start=10000)
    o = b.submit(Order("BTCUSDT", "buy", 0.01, 500), 50000.0, NOW)
    assert o.state == OrderState.ACKED and not o.fills
    assert b.poll(NOW, MARKS) == []                       # latency not elapsed
    b.poll(NOW + timedelta(seconds=1), MARKS)
    assert o.state == OrderState.FILLED
    f = o.fills[0]
    assert f.price == pytest.approx(50000 * (1 + (1 + 2) / 1e4)) and f.fee == pytest.approx(0.01 * f.price * 5 / 1e4)
    assert b.equity(MARKS) == pytest.approx(10000 - 0.01 * 50000 * 3 / 1e4 - f.fee)
    b2 = PaperBroker(s, state_path=tmp_path / "b2.json", equity_start=10000)
    o2 = Order("BTCUSDT", "buy", 0.01, 500, id=o.id)
    b2.submit(o2, 50000.0, NOW)
    b2.poll(NOW + timedelta(seconds=1), MARKS)
    assert o2.fills[0].price == f.price


def test_paper_partials_rejections_funding_and_realised(tmp_path):
    s = _s(partial_fill_prob=1.0, partial_fill_min=0.3)
    b = PaperBroker(s, state_path=tmp_path / "b.json", equity_start=10000)
    o = b.submit(Order("ETHUSDT", "sell", 1.0, 2000), 2000.0, NOW)
    b.poll(NOW + timedelta(seconds=1), MARKS)
    assert o.state == OrderState.PARTIAL and 0 < o.filled_qty < 1
    b.poll(NOW + timedelta(seconds=2), MARKS)
    assert o.state in (OrderState.PARTIAL, OrderState.FILLED) and len(o.fills) >= 2
    assert b.submit(Order("BTCUSDT.SPOT", "sell", 0.1, 5000), 50000.0, NOW).state == OrderState.REJECTED
    assert b.submit(Order("BTCUSDT", "buy", 0.0001, 5), 50000.0, NOW).state == OrderState.REJECTED
    paid = b.apply_funding({"ETHUSDT": 0.001}, MARKS, NOW)
    assert paid < 0, "a short receives positive funding"
    # realised pnl when closing the short lower
    while o.open:
        b.poll(NOW + timedelta(seconds=5), MARKS)
    c = b.submit(Order("ETHUSDT", "buy", 1.0, 1900), 1900.0, NOW)
    b.poll(NOW + timedelta(seconds=10), {"ETHUSDT": 1900.0})
    assert b.realised["ETHUSDT"] > 0 and b.closed_trades[-1]["pnl"] > 0
    saved = json.loads((tmp_path / "b.json").read_text())
    assert "cash" in saved and saved["fees_paid"] > 0


def test_executor_rebalance_and_min_notional(tmp_path):
    s = _s(partial_fill_prob=0.0, min_order_notional=50.0)
    b = PaperBroker(s, state_path=tmp_path / "b.json", equity_start=10000)
    ex = Executor(b, s)
    rep = ex.rebalance(pd.Series({"BTCUSDT": 0.2, "ETHUSDT": -0.1, "ETHUSDT.SPOT": 0.002}), MARKS, NOW)
    assert [o.state.value for o in rep.orders] == ["filled", "filled"] and rep.skipped[0]["symbol"] == "ETHUSDT.SPOT"
    w = ex.current_weights(MARKS)
    assert w["BTCUSDT"] == pytest.approx(0.2, abs=1e-3) and w["ETHUSDT"] == pytest.approx(-0.1, abs=1e-3)
    rep2 = ex.rebalance(pd.Series({"BTCUSDT": 0.2, "ETHUSDT": -0.1}), MARKS, NOW + timedelta(minutes=1))
    assert all(o.notional < 50 for o in rep2.orders) or not rep2.orders   # nothing meaningful to trade


def test_live_adapters_refuse_without_all_locks(tmp_path, monkeypatch):
    s = load_settings()
    for cls in (BinanceBroker, AlpacaBroker):
        br = cls(s, root=tmp_path)
        with pytest.raises(LiveDisabled):
            br.submit(Order("BTCUSDT", "buy", 0.01, 500), 50000.0, NOW)
        assert not br.guard["open"] and br.dry_run(Order("BTCUSDT", "buy", 0.01, 500))["guard"]["open"] is False
    monkeypatch.setenv("ITRADER_LIVE", "1")
    monkeypatch.setenv("ITRADER_BINANCE_KEY", "k")
    monkeypatch.setenv("ITRADER_BINANCE_SECRET", "s")
    br = BinanceBroker(s, root=tmp_path, live=True)
    with pytest.raises(LiveDisabled) as e:                 # settings.live.enabled and the confirmation file still closed
        br.submit(Order("BTCUSDT", "buy", 0.01, 500), 50000.0, NOW)
    assert "settings.live.enabled" in str(e.value) and "confirmation" in str(e.value)


@pytest.mark.skipif(not (repo_root() / "data" / "online" / "1d" / "BTCUSDT.parquet").exists(), reason="no local data store")
def test_full_replay_through_the_loop(tmp_path):
    from itrader.online.runner import PaperRunner

    s = load_settings(overrides={"execution": {"use_live_book": False}, "risk": {"kill_file": str(tmp_path / "nokill"), "calendar_file": str(tmp_path / "nocal")}})
    r = PaperRunner(s, out_dir=tmp_path / "run", sync=False, alerts_enabled=False,
                    broker=PaperBroker(s, state_path=tmp_path / "pb.json", equity_start=10000))
    end = datetime.now(timezone.utc) - timedelta(days=200)
    recs = r.replay(end - timedelta(days=2), end, step=timedelta(days=1), max_cycles=3)
    assert len(recs) == 3
    for d in recs:
        assert d["market_summary_ref"].startswith("ms_") and d["candidate_tasks"][-1]["metadata"]["candidate_id"] == "WAIT"
        assert "gate" in d["solver_scores"] and "approved" in d["risk_decisions"] and "orders" in d["executor_action"]
        assert d["metadata"]["settings_hash"] == s.hash
    lines = (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()
    assert len(lines) == 3
    from itrader.rl_lib.tasks.decision_record import DecisionRecord

    back = DecisionRecord.from_dict(json.loads(lines[-1]))
    assert back.candidate_tasks and isinstance(back.wait_selected, bool)
    from itrader.online.dashboard import render_dashboard

    assert "<html" in render_dashboard(tmp_path / "run", s)
