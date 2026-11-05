"""MarketFramer: strategies -> candidate tasks (+ WAIT).

Each cycle the framer loads the panels, asks every strategy for its latest target weights, and
turns each non-flat answer into a `Proposal` and then a repo `CandidateTask`. The expected
utility and confidence on a proposal come from the strategy's CapabilityProfile (the bench),
not from the strategy's own opinion of itself. WAIT is always in the set. The framer also builds
the market summary the DecisionRecord references and carries the data Quality so the RiskEngine
can refuse stale symbols.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from itrader.rl_lib.tasks.candidate_task import CandidateTask

from .bench.engine import CostModel
from .bench.portfolio_sim import regime_now
from .bench.run import load_macro_market, load_market
from .contracts import Proposal, wait_candidate
from .data.feeds import Yahoo
from .data.store import DataStore, Quality
from .settings import Settings, settings as default_settings
from .strategies import BENCHABLE, Market, Strategy


@dataclass
class Frame:
    now: str
    market_summary: Dict[str, Any]
    market_summary_ref: str
    proposals: List[Proposal]
    candidates: List[CandidateTask]            # proposals as CandidateTasks, plus WAIT last
    sleeve_targets: Dict[str, pd.Series]        # strategy -> asset weights (latest row)
    quality: Dict[str, Quality]
    prices: Dict[str, float]
    vols: Dict[str, float]                      # daily vol per symbol (fraction)
    notes: List[str] = field(default_factory=list)
    ranges: Dict[str, Any] = field(default_factory=dict)   # symbol -> (low, high, bar open time) of the last completed bar
    regime: Optional[str] = None                # market regime from completed daily bars (see bench.portfolio_sim)

    @property
    def wait(self) -> CandidateTask:
        return self.candidates[-1]


class MarketFramer:
    def __init__(self, strategies: List[Strategy], store: DataStore, s: Optional[Settings] = None,
                 profiles: Optional[Dict[str, Dict[str, Any]]] = None, universe: Optional[List[str]] = None) -> None:
        self.s = s or default_settings()
        self.strategies = [x for x in strategies if x.name in BENCHABLE]
        self.store = store
        self.profiles = profiles or {}
        self.costs = CostModel.from_settings(self.s)
        self.universe = list(universe or self.s.data.universe)
        self.universe_1h = [x for x in self.s.data.get("universe_1h", self.universe) if x in self.universe]
        self.macro_store = DataStore(self.s.resolve(self.s.data.get("macro_root", "data/online/macro")), feed=Yahoo())

    # ------------------------------------------------------------------ data
    def markets(self) -> Dict[str, Market]:
        daily = load_market(self.store, self.universe, "1d")
        out = {"1d": daily}
        if any(x.bar == "1h" for x in self.strategies):
            out["1h"] = load_market(self.store, self.universe_1h, "1h", daily_close=daily.close)
        if any(x.bar == "1b" for x in self.strategies):
            mm = load_macro_market(self.s, cutoff=getattr(self.store, "cutoff", None))   # a replay store hides the future here too
            if mm is not None and len(mm.close):
                out["1b"] = mm
        return out

    def summary(self, m: Market, now: datetime) -> Dict[str, Any]:
        c = m.close
        r1 = c.pct_change().iloc[-1]
        r7 = c.pct_change(7).iloc[-1]
        vol = c.pct_change().ewm(span=30, min_periods=10).std().iloc[-1]
        fund = None
        if m.funding is not None and len(m.funding):
            fund = m.funding.tail(3).sum().reindex(c.columns)
        per = {}
        for sym in c.columns:
            per[sym] = {"close": float(c[sym].iloc[-1]), "ret_1d": float(r1[sym]) if np.isfinite(r1[sym]) else None,
                        "ret_7d": float(r7[sym]) if np.isfinite(r7[sym]) else None,
                        "vol_1d": float(vol[sym]) if np.isfinite(vol[sym]) else None,
                        "funding_1d": float(fund[sym]) if fund is not None and np.isfinite(fund[sym]) else None}
        rets = c.pct_change().tail(30)
        corr = rets.corr().values
        avg_corr = float(np.nanmean(corr[np.triu_indices_from(corr, 1)])) if corr.size > 1 else 0.0
        btc = per.get("BTCUSDT", {})
        regime = "high_vol" if (btc.get("vol_1d") or 0) > 0.04 else ("low_vol" if (btc.get("vol_1d") or 0) < 0.02 else "normal_vol")
        return {"as_of": str(c.index[-1]), "now": now.isoformat(timespec="seconds"), "bar": m.bar, "symbols": per,
                "avg_pairwise_corr_30d": avg_corr, "regime_tags": [regime, "crypto_majors"]}

    # ------------------------------------------------------------------ frame
    def frame(self, now: Optional[datetime] = None, current: Optional[pd.Series] = None, max_candidates: Optional[int] = None) -> Frame:
        now = now or datetime.now(timezone.utc)
        mk = self.markets()
        daily = mk["1d"]
        summ = self.summary(daily, now)
        ref = "ms_" + hashlib.sha1(json.dumps(summ, sort_keys=True, default=str).encode()).hexdigest()[:12]
        quality = {sym: self.store.quality(sym, "1d", max_stale_bars=float(self.s.data.max_stale_bars), now=now) for sym in self.universe}
        if "1h" in mk:
            qh = {sym: self.store.quality(sym, "1h", max_stale_bars=float(self.s.data.max_stale_bars), now=now) for sym in self.universe_1h}
            for sym, q in qh.items():
                if not q.ok:
                    quality[sym] = q
        # marks: the freshest close we have (the 1h panel when loaded), never the still-open daily bar's stale close
        prices = {sym: float(daily.close[sym].iloc[-1]) for sym in daily.close.columns if np.isfinite(daily.close[sym].iloc[-1])}
        if "1h" in mk:
            hc = mk["1h"].close
            prices.update({sym: float(hc[sym].iloc[-1]) for sym in hc.columns if np.isfinite(hc[sym].iloc[-1])})
        if "1b" in mk:                                   # listed markets: last daily close; weekends and holidays are normal, so staleness is judged loosely
            mc = mk["1b"].close
            prices.update({sym: float(mc[sym].iloc[-1]) for sym in mc.columns if np.isfinite(mc[sym].iloc[-1])})
            for sym in mc.columns:
                quality[sym] = self.macro_store.quality(sym, "1d", max_stale_bars=5.0, now=now)
        vols = {sym: float(v) for sym, v in daily.close.pct_change().ewm(span=30, min_periods=10).std().iloc[-1].items() if np.isfinite(v)}
        cur = current if current is not None else pd.Series(dtype=float)
        # the last completed bar's traded range: what a resting order placed at the previous cycle has lived through
        rm = mk.get("1h", daily)
        ranges: Dict[str, Any] = {}
        if rm.high is not None and rm.low is not None and len(rm.close):
            bar_open = rm.close.index[-1].to_pydatetime()
            for sym in rm.close.columns:
                lo, hi = rm.low[sym].iloc[-1], rm.high[sym].iloc[-1]
                if np.isfinite(lo) and np.isfinite(hi):
                    ranges[sym] = (float(lo), float(hi), bar_open)
        btc = daily.close["BTCUSDT"] if "BTCUSDT" in daily.close.columns else daily.close.iloc[:, 0]
        regime = regime_now(btc)
        summ["regime"] = regime

        proposals: List[Proposal] = []
        sleeve_targets: Dict[str, pd.Series] = {}
        notes: List[str] = []
        for st in self.strategies:
            m = mk.get(st.bar)
            if m is None:
                notes.append(f"{st.name}: no {st.bar} data")
                continue
            try:
                w = st.targets(m).iloc[-1]
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{st.name}: targets failed: {type(exc).__name__}: {exc}")
                continue
            w = w[np.isfinite(w)]
            w = w[w.abs() > 1e-6]
            sleeve_targets[st.name] = w
            if w.empty:
                continue
            g = float(w.abs().sum())
            if g > 1.0:                                   # buffers can leave gross a hair above 1; the contract is <= 1
                w = w / g
            prof = self.profiles.get(st.name, {})
            meta = prof.get("metadata", {}) if prof else {}
            dev = meta.get("dev") or {}
            # expected utility of THIS trade: the bench's mean net return per bar is averaged over flat bars too,
            # so condition on being in a position (divide by the share of bars with exposure)
            eu_all = float(prof.get("expected_net_utility", 0.0)) if prof else 0.0
            eu = eu_all / max(0.05, float(dev.get("exposure_share", 1.0) or 1.0))
            conf = float(prof.get("estimated_p_solve", 0.0)) if prof else 0.0
            cost = float((w.reindex(w.index).fillna(0.0) - cur.reindex(w.index).fillna(0.0)).abs().sum() * self.costs.per_side(st.instrument) * self.costs.legs(st.instrument))
            # expected holding period of a unit of position, from the bench (twice the average gross over the turnover per
            # bar), capped at 180 days. An earlier version counted bars between ANY trade in the book, which made a
            # carry position held for months look like a ten-day trade and kept it from ever being opened.
            cap = 180 * (24 if st.bar == "1h" else 1)
            hold = dev.get("holding_bars")
            if hold is None:
                hold = 2.0 * float(dev.get("bars_exposed", 0) or 0) / max(1.0, float(dev.get("trade_bars", 0) or 1))
            ttl = int(max(1, min(cap, round(float(hold)))))
            proposals.append(Proposal(
                strategy=st.name, family_id=st.family_id, bar=st.bar, weights={k: float(v) for k, v in w.items()},
                expected_utility=eu, confidence=conf, rationale=st.rationale(m, w), instrument=st.instrument, ttl_bars=ttl,
                metadata={"regime_tags": summ["regime_tags"], "cost_to_trade": cost, "holding_bars": ttl, "expected_utility_unconditional": eu_all,
                          "profile_stage": (meta.get("promotion") or {}).get("stage", "none"),
                          "dev_sharpe": (meta.get("dev") or {}).get("sharpe"), "holdout_sharpe": (meta.get("holdout") or {}).get("sharpe"),
                          "evidence_ratio": (meta.get("evidence") or {}).get("ratio"), "has_profile": bool(prof)}))
        # rank by utility x confidence and keep the small set the spec asks for
        proposals.sort(key=lambda p: p.expected_utility * max(p.confidence, 1e-6), reverse=True)
        k = int(max_candidates or self.s.gate.max_candidates)
        kept, dropped = proposals[:k], proposals[k:]
        for p in dropped:
            notes.append(f"{p.strategy}: not among the top {k} candidates this cycle")
        cands = [p.to_candidate(price_id="online_panel") for p in kept]
        cands.append(wait_candidate("WAIT is always available"))
        return Frame(now.isoformat(timespec="seconds"), summ, ref, kept, cands, sleeve_targets, quality, prices, vols, notes, ranges, regime)
