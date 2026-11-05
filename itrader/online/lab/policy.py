"""Allocation policies: from sleeves' return records to an exposure multiplier per sleeve.

One function, `exposures`, is used both by the walk-forward simulation (`simulate_policy`) and by the
live portfolio, so what is tested is what runs. A policy sees only the past.

    steady   every sleeve in one pool: on or off by a slow gate on its whole record, equal risk between
             the ones that are on, portfolio scaled to its volatility target, a floor of cash kept back.
    stack    "core" sleeves (market-neutral carry) hold the capital at their own notional when their gate
             is on; the rest are an overlay sized by risk exactly as in `steady`, without the cash floor,
             and shrunk if the gap rule's budget would be exceeded.

The gate has memory (on at `gate_on`, off below `gate_off`) and an exposure is only changed when it has
moved by more than `exposure_band`, so the book does not churn on noise. All numbers come from `[lab]`,
`[portfolio]` and `[risk]` in settings.toml and were fixed before any recipe was evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..bench.metrics import _moments, probabilistic_sharpe
from ..portfolio import realised_vol, risk_budgets
from ..settings import Settings


@dataclass
class SleeveMeta:
    name: str
    role: str = "overlay"            # "core" | "overlay"
    n_trials: int = 1
    expo_cap: float = float("inf")   # multiplier beyond which the live per-asset cap would clip the sleeve's weights
    gap_per_unit: float = 1.0        # gap-rule exposure used per unit of multiplier (hedged and listed sleeves use less)
    cost_per_unit: float = 0.0008    # cost of changing the multiplier by one (the sleeve's gross times its cost per side)


@dataclass
class PolicyState:
    on: Dict[str, bool] = field(default_factory=dict)
    expo: Dict[str, float] = field(default_factory=dict)
    psr: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"on": dict(self.on), "expo": dict(self.expo), "psr": dict(self.psr)}

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "PolicyState":
        d = d or {}
        return PolicyState(dict(d.get("on", {})), {k: float(v) for k, v in d.get("expo", {}).items()}, dict(d.get("psr", {})))


def record(r: pd.Series, prior: Optional[pd.Series] = None) -> Tuple[float, float, int]:
    """(probability the record is better than nothing, mean daily return, days) for a sleeve's whole history."""
    x = pd.concat([prior, r]) if prior is not None and len(prior) else r
    x = x.dropna()
    n, mu, sd, sk, ku = _moments(x)
    if n < 30 or sd <= 0:
        return 0.5, float(mu), int(n)
    return float(probabilistic_sharpe(mu / sd, n, sk, ku)), float(mu), int(n)


def exposures(policy: str, past: pd.DataFrame, metas: Dict[str, SleeveMeta], s: Settings, state: PolicyState,
              priors: Optional[Dict[str, pd.Series]] = None) -> Tuple[pd.Series, Dict[str, Any]]:
    if policy not in ("steady", "stack"):
        raise ValueError(f"unknown policy {policy}")
    lab, p, rk = s.lab, s.portfolio, s.risk
    names = list(past.columns)
    priors = priors or {}
    tgt, port, floor = float(p.strategy_vol_target), float(p.portfolio_vol_target), float(p.min_vol_floor)
    vwin, cwin = int(lab.vol_window_days), int(lab.corr_window_days)

    on: Dict[str, bool] = {}
    for c in names:
        psr, mu, n = record(past[c], priors.get(c))
        bar = float(lab.gate_off) if state.on.get(c, False) else float(lab.gate_on)
        on[c] = bool(n >= int(lab.min_history_days) and mu > 0 and psr >= bar)
        state.psr[c] = psr

    expo = pd.Series(0.0, index=names)
    core = [c for c in names if metas[c].role == "core"] if policy == "stack" else []
    pool = [c for c in names if c not in core]
    for c in core:
        if on[c]:
            expo[c] = min(1.0, metas[c].expo_cap)                     # the sleeve's own notional (at most 80% of equity)
    active = [c for c in pool if on[c]]
    info: Dict[str, Any] = {"on": on, "portfolio_scale": 1.0}
    if active:
        vol = {c: realised_vol(past[c], vwin, floor) for c in active}
        scale = pd.Series({c: (min(2.0, tgt / vol[c]) if np.isfinite(vol[c]) and vol[c] > 0 else 1.0) for c in active})
        scaled = past[active].fillna(0.0) * scale
        rb = risk_budgets(scaled, float(p.corr_penalty), cwin, tgt)
        rb = rb / rb.sum() if rb.sum() > 0 else rb
        capital = rb * ((1.0 - float(s.allocator.cash_floor)) if policy == "steady" else 1.0)
        pv = realised_vol((scaled * capital).sum(axis=1), vwin, floor)
        ps = float(min(2.0, port / pv)) if np.isfinite(pv) and pv > 0 else 1.0
        info["portfolio_scale"] = ps
        for c in active:
            expo[c] = min(float(capital[c] * scale[c] * ps), metas[c].expo_cap)

    # the gap rule: what a gap against everything at once may cost; the core is served first
    budget = float(rk.max_gap_loss_pct) / float(rk.gap_move)
    used_core = float(sum(expo[c] * metas[c].gap_per_unit for c in core))
    used_pool = float(sum(expo[c] * metas[c].gap_per_unit for c in pool))
    room = max(0.0, budget - used_core)
    if used_pool > room and used_pool > 0:
        for c in pool:
            expo[c] *= room / used_pool
        info["gap_scaled"] = room / used_pool

    band = float(lab.exposure_band)
    for c in names:
        old, new = float(state.expo.get(c, 0.0)), float(expo[c])
        if old > 0 and new > 0 and abs(new - old) <= band * old:
            expo[c] = old
    state.on = on
    state.expo = {c: float(expo[c]) for c in names}
    return expo, info


def simulate_policy(R: pd.DataFrame, metas: Dict[str, SleeveMeta], s: Settings, policy: str,
                    priors: Optional[Dict[str, pd.Series]] = None) -> Dict[str, Any]:
    """Walk forward: decide exposures from the past every `rebalance_days`, live with them, pay to change them."""
    lab = s.lab
    R = R.sort_index().fillna(0.0)
    names = list(R.columns)
    warm, reb = int(lab.min_history_days), int(lab.rebalance_days)
    if len(R) <= warm + 5:
        return {"returns": pd.Series(dtype=float), "history": pd.DataFrame()}
    state = PolicyState()
    out = np.zeros(len(R) - warm)
    hist = []
    prev = pd.Series(0.0, index=names)
    i = warm
    while i < len(R):
        expo, info = exposures(policy, R.iloc[:i], metas, s, state, priors)
        j = min(len(R), i + reb)
        block = (R.iloc[i:j] * expo).sum(axis=1).to_numpy(dtype=float).copy()
        block[0] -= float(sum(abs(expo[c] - prev[c]) * metas[c].cost_per_unit for c in names))
        out[i - warm:j - warm] = block
        hist.append({"date": str(R.index[i].date()), "sleeves_on": int(sum(info["on"].values())), "portfolio_scale": info["portfolio_scale"],
                     **{c: float(expo[c]) for c in names}})
        prev = expo
        i = j
    return {"returns": pd.Series(out, index=R.index[warm:]), "history": pd.DataFrame(hist)}
