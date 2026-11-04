"""Contracts bridge and the no-lookahead rule for every benchable strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from itrader.online.contracts import ONLINE_FAMILIES, Proposal, candidate_id, wait_candidate
from itrader.online.strategies import BENCHABLE, Market, all_strategies, make
from itrader.rl_lib.tasks.candidate_task import CandidateTask
from itrader.rl_lib.tasks.validation import validate_candidate_task


def synthetic_market(bar: str = "1d", n: int = 900, seed: int = 0, with_daily: bool = True) -> Market:
    rng = np.random.default_rng(seed)
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LTCUSDT", "ADAUSDT", "XRPUSDT"]
    freq = "D" if bar == "1d" else "h"
    idx = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC")
    drift = rng.normal(0.0004, 0.0002, len(syms)) if bar == "1d" else rng.normal(0.00002, 0.00001, len(syms))
    vol = 0.03 if bar == "1d" else 0.006
    rets = rng.normal(drift, vol, (n, len(syms)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=syms)
    high = close * (1 + np.abs(rng.normal(0, 0.005, close.shape)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, close.shape)))
    volume = pd.DataFrame(rng.uniform(1e3, 1e4, close.shape), index=idx, columns=syms)
    fidx = pd.date_range(idx[0], idx[-1], freq="8h", tz="UTC")
    funding = pd.DataFrame(rng.normal(0.0001, 0.0002, (len(fidx), len(syms))), index=fidx, columns=syms)
    daily = close.resample("D").last() if (bar != "1d" and with_daily) else None
    return Market(bar, close, high, low, volume, funding, daily)


def test_online_families_do_not_collide_with_repo_families():
    from itrader.rl_lib.tasks.families import TASK_FAMILY_REGISTRY

    assert not set(ONLINE_FAMILIES) & set(TASK_FAMILY_REGISTRY)
    assert ONLINE_FAMILIES["99"]["name"] == "wait"


def test_proposal_to_candidate_roundtrip_and_validation():
    p = Proposal("turtle", "10", "1d", {"BTCUSDT": 0.3, "ETHUSDT": -0.2}, 0.0004, 0.7, "test", metadata={"regime_tags": ["normal_vol"]})
    c = p.to_candidate()
    assert isinstance(c, CandidateTask)
    assert candidate_id(c) == p.candidate_id and p.candidate_id.startswith("turtle:")
    d = c.to_dict()
    back = CandidateTask.from_dict(d)
    assert back.execution_intent.entry_logic["params"]["weights"] == p.weights
    assert abs(back.execution_intent.risk_envelope["gross"] - 0.5) < 1e-12
    validated = validate_candidate_task(d)      # the repo's own validator accepts it (family falls back, intent clamps)
    assert validated["execution_intent"]["abstain_allowed"] is True
    w = wait_candidate("nothing cleared")
    assert w.abstain_option and candidate_id(w) == "WAIT" and w.expected_utility == 0.0


@pytest.mark.parametrize("name", [n for n in BENCHABLE if n != "experimental"])
def test_no_lookahead(name):
    st = make(name)
    m = synthetic_market(st.bar, n=900 if st.bar == "1d" else 24 * 60)
    full = st.targets(m)
    cut = len(m.close) - 40
    # Bars are indexed by OPEN time; the decision at row t is taken at that bar's CLOSE. So the truncated market
    # may contain funding settlements inside bar t (before its close) and only daily rows whose day has ended.
    step = pd.Timedelta(days=1) if st.bar == "1d" else pd.Timedelta(hours=1)
    close_t = m.close.index[cut - 1] + step
    daily = None
    if m.daily_close is not None:
        daily = m.daily_close[m.daily_close.index + pd.Timedelta(days=1) <= close_t]
    m2 = Market(m.bar, m.close.iloc[:cut], m.high.iloc[:cut], m.low.iloc[:cut], m.volume.iloc[:cut],
                m.funding[m.funding.index < close_t], daily)
    part = st.targets(m2)
    a, b = full.iloc[:cut].to_numpy(), part.to_numpy()
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=1e-9), f"{name}: weights before the cut changed when later data was removed"
    assert np.isfinite(full.to_numpy()).all()
    # a no-trade buffer holds old weights while others move, so gross can sit up to ~10% over 1 between rebalances;
    # the framer normalises to 1 before anything is traded. Anything beyond that would be a sizing bug.
    assert (full.abs().sum(axis=1) <= 1.15).all(), f"{name}: gross well above 1"


def test_every_strategy_runs_and_registry_is_consistent():
    for st in all_strategies():
        m = synthetic_market(st.bar if st.bar in ("1d", "1h") else "1d", n=400 if st.bar == "1d" else 24 * 40)
        w = st.targets(m)
        assert list(w.columns) == list(m.close.columns) and len(w) == len(m.close)
        assert st.family_id in ONLINE_FAMILIES
        d = st.describe()
        assert d["name"] == st.name and d["n_variants_tried"] >= 1
