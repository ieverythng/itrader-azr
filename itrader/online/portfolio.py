"""Portfolio construction: from each strategy's target weights to one set of asset weights.

Three layers, each a plain function with its numbers logged:

1. Sleeve vol targeting. Each strategy's recent realised vol is scaled to `strategy_vol_target`
   so a quiet strategy and a wild one contribute comparable risk. Scale is capped.
2. Risk budgets across sleeves. Equal risk contribution on a blended covariance
   (corr_penalty = 0 is plain inverse vol, 1 is full risk parity on the sample correlation), then
   tilted by the Workbench allocator's capability weights. Cash (WAIT) is a sleeve too.
3. Portfolio controls. Scale the whole book to `portfolio_vol_target`, cap gross and per-asset
   exposure, and apply a no-trade band so small drifts do not pay costs.

The Workbench allocator (the "surprise" in the handover). In the Neural Workbench framing a
strategy is a skill with a measured capability, and the system composes skills rather than
betting on one. The allocator turns each CapabilityProfile into a score (development Sharpe,
discounted by the probability the edge is real and by how much of the required evidence
exists), puts cash in the same softmax with a score of zero, shrinks toward equal weight
because samples are small, and never lets cash fall below `cash_floor`. A strategy whose
evidence is thin gets little capital until the record earns it; a strategy whose paper record
diverges from its bench loses capital automatically. Nothing is learned online; every weight
can be recomputed by hand from the profiles and the settings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .settings import Settings, settings as default_settings

CASH = "CASH"


# --------------------------------------------------------------------------- layer 1: sleeve vol
def realised_vol(returns: pd.Series, window_days: int, floor: float) -> float:
    r = returns.dropna().tail(int(window_days))
    if len(r) < 10:
        return float("nan")
    return max(float(r.std(ddof=1) * math.sqrt(365.0)), floor)


def vol_scale(returns: pd.Series, target: float, window_days: int, floor: float, max_scale: float = 2.0) -> float:
    v = realised_vol(returns, window_days, floor)
    if not np.isfinite(v) or v <= 0:
        return 1.0
    return float(min(max_scale, target / v))


# --------------------------------------------------------------------------- layer 2: risk budgets
def erc_weights(cov: pd.DataFrame, iters: int = 200, tol: float = 1e-10) -> pd.Series:
    """Equal risk contribution by multiplicative updates. Falls back to inverse vol if degenerate."""
    names = list(cov.columns)
    S = cov.to_numpy(dtype=float)
    n = len(names)
    if n == 0:
        return pd.Series(dtype=float)
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    w = (1.0 / d)
    w = w / w.sum()
    for _ in range(iters):
        rc = w * (S @ w)
        total = float(w @ S @ w)
        if total <= 0:
            break
        target = total / n
        upd = np.sqrt(np.clip(target / np.clip(rc, 1e-14, None), 0.0, None))
        w_new = w * upd
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    if not np.all(np.isfinite(w)):
        w = (1.0 / d) / (1.0 / d).sum()
    return pd.Series(w, index=names)


def risk_budgets(returns: pd.DataFrame, corr_penalty: float, window_days: int, floor: float) -> pd.Series:
    """Risk-parity budgets across sleeves on the blended covariance."""
    r = returns.dropna(how="all").tail(int(window_days))
    cols = [c for c in r.columns if r[c].notna().sum() >= 10]
    if not cols:
        return pd.Series(1.0 / max(1, len(returns.columns)), index=returns.columns)
    r = r[cols].fillna(0.0)
    cov = r.cov().to_numpy()
    vol = np.sqrt(np.clip(np.diag(cov), (floor / math.sqrt(365.0)) ** 2, None))
    corr = cov / np.outer(vol, vol)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    blended = (1.0 - corr_penalty) * np.eye(len(cols)) + corr_penalty * corr
    cov_b = pd.DataFrame(blended * np.outer(vol, vol), index=cols, columns=cols)
    w = erc_weights(cov_b)
    out = pd.Series(0.0, index=returns.columns)
    out.loc[cols] = w
    missing = [c for c in returns.columns if c not in cols]
    if missing:   # no history: give the newcomers the smallest existing budget, then renormalise
        out.loc[missing] = float(w.min()) if len(w) else 1.0
        out = out / out.sum()
    return out


@dataclass
class AllocatorDecision:
    weights: pd.Series                      # strategies + CASH, sums to 1
    scores: Dict[str, float]
    raw_softmax: Dict[str, float]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": {k: float(v) for k, v in self.weights.items()}, "scores": self.scores, "raw_softmax": self.raw_softmax, "notes": self.notes}


class WorkbenchAllocator:
    """Capability-weighted meta-allocation with cash as a first-class asset."""

    def __init__(self, s: Optional[Settings] = None) -> None:
        self.s = s or default_settings()
        self.cfg = self.s.allocator

    @staticmethod
    def score(profile: Dict[str, Any], paper: Optional[Dict[str, Any]] = None, evidence_discount: bool = True,
              regime: Optional[str] = None, regime_shrink_days: float = 180.0) -> Dict[str, float]:
        """One number per strategy from its CapabilityProfile (dict form) and, if present, its paper record.
        With `regime`, the development Sharpe is replaced by the sleeve's Sharpe in that regime, shrunk toward
        the overall figure in proportion to how few days the regime has (n / (n + regime_shrink_days))."""
        meta = profile.get("metadata", {}) or {}
        dev = meta.get("dev", {}) or {}
        sharpe = float(dev.get("sharpe", 0.0) or 0.0)
        if regime:
            rg = (meta.get("regimes") or {}).get(regime)
            if rg and float(rg.get("n", 0) or 0) > 0:
                lam = float(rg["n"]) / (float(rg["n"]) + float(regime_shrink_days))
                sharpe = lam * float(rg.get("sharpe", 0.0) or 0.0) + (1.0 - lam) * sharpe
        p_edge = float(profile.get("estimated_p_solve", 0.0) or 0.0)      # PSR on the development period
        dsr = float(dev.get("dsr", p_edge) or 0.0)
        ev = meta.get("evidence", {}) or {}
        ratio = float(ev.get("ratio", 0.0) or 0.0)
        ratio = 0.0 if not np.isfinite(ratio) else min(1.0, ratio)
        hold_sharpe = float((meta.get("holdout", {}) or {}).get("sharpe", 0.0) or 0.0)
        # the holdout gets a vote: if it disagrees with development, the score drops toward it
        base = 0.5 * sharpe + 0.5 * min(sharpe, hold_sharpe)
        s = base * dsr
        if evidence_discount:
            s *= (0.25 + 0.75 * ratio)        # thin evidence keeps a quarter of its score, never zero by construction
        if paper and paper.get("days", 0) >= 20:
            # live/paper tracking: shrink toward the paper Sharpe as the paper record grows
            lam = min(1.0, paper["days"] / 180.0)
            s = (1 - lam) * s + lam * float(paper.get("sharpe", 0.0)) * float(paper.get("psr", 0.5))
        return {"score": float(max(0.0, s)), "sharpe_dev": sharpe, "sharpe_holdout": hold_sharpe, "dsr": dsr, "evidence_ratio": ratio}

    def allocate(self, profiles: Dict[str, Dict[str, Any]], paper_stats: Optional[Dict[str, Dict[str, Any]]] = None,
                 vol_ratio: Optional[Dict[str, float]] = None, regime: Optional[str] = None) -> AllocatorDecision:
        """`vol_ratio[n]` = the vol a sleeve actually reaches at its allowed leverage / the target vol (<= 1).
        A sleeve that cannot reach the target (carry at 2x) is scored on the return it can deliver, not on
        its Sharpe alone; otherwise risk parity hands most of the book to the lowest-vol sleeve.
        `regime`: score each sleeve on its record in this market regime (see `score`)."""
        cfg = self.cfg
        names = list(profiles)
        shrink = float(cfg.get("regime_shrink_days", 180.0))
        details = {n: self.score(profiles[n], (paper_stats or {}).get(n), bool(cfg.evidence_discount), regime, shrink) for n in names}
        for n in names:
            vr = float(min(1.0, max(0.0, (vol_ratio or {}).get(n, 1.0))))
            details[n]["vol_ratio"] = vr
            details[n]["score"] *= vr
        scores = {n: d["score"] for n, d in details.items()}
        scores[CASH] = float(cfg.cash_score)
        keys = list(scores)
        x = np.array([scores[k] for k in keys], dtype=float) / max(1e-9, float(cfg.temperature))
        x = x - x.max()
        sm = np.exp(x)
        sm = sm / sm.sum()
        raw = dict(zip(keys, map(float, sm)))
        # zero-score strategies get nothing; their share goes to cash
        w = np.array([raw[k] if (k == CASH or scores[k] > 0) else 0.0 for k in keys])
        w = w / w.sum()
        # shrink toward equal weight among the deployed strategies (small samples)
        active = [i for i, k in enumerate(keys) if k != CASH and w[i] > 0]
        if active:
            eq = np.zeros_like(w)
            eq[active] = (1.0 - float(cfg.cash_floor)) / len(active)
            eq[keys.index(CASH)] = float(cfg.cash_floor)
            w = (1.0 - float(cfg.shrink_to_equal)) * w + float(cfg.shrink_to_equal) * eq
        # caps and the cash floor
        ci = keys.index(CASH)
        for i, k in enumerate(keys):
            if k != CASH:
                w[i] = min(float(cfg.max_weight), max(float(cfg.min_weight), w[i]))
        strat_total = w.sum() - w[ci]
        max_strat = 1.0 - float(cfg.cash_floor)
        if strat_total > max_strat:
            for i, k in enumerate(keys):
                if k != CASH:
                    w[i] *= max_strat / strat_total
            strat_total = max_strat
        w[ci] = 1.0 - strat_total
        notes = [f"{n}: score {details[n]['score']:.3f} (dev SR {details[n]['sharpe_dev']:.2f}, holdout SR {details[n]['sharpe_holdout']:.2f}, "
                 f"DSR {details[n]['dsr']:.2f}, evidence {details[n]['evidence_ratio']:.0%}, reachable vol {details[n]['vol_ratio']:.0%} of target)" for n in names]
        if regime:
            notes.insert(0, f"regime: {regime} (scores use each sleeve's record in this regime, shrunk toward its overall record)")
        if not active:
            notes.append("no strategy has a positive capability score: all capital in cash (WAIT)")
        return AllocatorDecision(pd.Series(w, index=keys), scores, raw, notes)


# --------------------------------------------------------------------------- layer 3: book controls
def apply_no_trade_band(target: pd.Series, current: pd.Series, band: float, floor: float = 0.005) -> pd.Series:
    cur = current.reindex(target.index).fillna(0.0)
    tol = np.maximum(target.abs() * band, floor)
    move = (target - cur).abs() > tol
    return target.where(move, cur)


@dataclass
class PortfolioDecision:
    asset_weights: pd.Series            # final targets per asset after all controls
    pre_band_weights: pd.Series
    sleeve_scales: Dict[str, float]
    risk_budgets: Dict[str, float]
    allocator: Optional[Dict[str, Any]]
    sleeve_capital: Dict[str, float]    # budget x allocator, per sleeve (fraction of equity)
    cash: float
    portfolio_scale: float
    gross: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"asset_weights": {k: float(v) for k, v in self.asset_weights.items() if abs(v) > 1e-12},
                "pre_band_weights": {k: float(v) for k, v in self.pre_band_weights.items() if abs(v) > 1e-12},
                "sleeve_scales": self.sleeve_scales, "risk_budgets": self.risk_budgets, "allocator": self.allocator,
                "sleeve_capital": self.sleeve_capital, "cash": self.cash, "portfolio_scale": self.portfolio_scale,
                "gross": self.gross, "notes": self.notes}


class Portfolio:
    def __init__(self, s: Optional[Settings] = None) -> None:
        self.s = s or default_settings()
        self.allocator = WorkbenchAllocator(self.s) if bool(self.s.allocator.enabled) else None

    def build(self, sleeve_targets: Dict[str, pd.Series], sleeve_returns: pd.DataFrame, current: Optional[pd.Series] = None,
              profiles: Optional[Dict[str, Dict[str, Any]]] = None, paper_stats: Optional[Dict[str, Dict[str, Any]]] = None,
              regime: Optional[str] = None) -> PortfolioDecision:
        """sleeve_targets: strategy -> Series(asset -> weight) as the strategy emitted them (gross <= 1).
        sleeve_returns: daily net returns per strategy (bench history, extended by the paper record)."""
        p = self.s.portfolio
        names = list(sleeve_targets)
        notes: List[str] = []
        rets = sleeve_returns.reindex(columns=names) if len(sleeve_returns) else pd.DataFrame(columns=names)

        # 1. vol scale each sleeve
        scales = {n: vol_scale(rets[n] if n in rets else pd.Series(dtype=float), float(p.strategy_vol_target), int(p.vol_window_days), float(p.min_vol_floor)) for n in names}
        scaled_rets = pd.DataFrame({n: rets[n] * scales[n] for n in names if n in rets}) if len(rets) else pd.DataFrame(columns=names)

        # 2. budgets: risk parity x capability. After scaling every sleeve sits at or below the target vol, so the
        #    vol floor is the target itself: budgets then come from correlations, not from who could not be levered.
        budgets = risk_budgets(scaled_rets if len(scaled_rets) else pd.DataFrame(np.nan, index=[0], columns=names), float(p.corr_penalty), int(p.corr_window_days), float(p.strategy_vol_target))
        vol_ratio = {}
        for n in names:
            v = realised_vol(rets[n], int(p.vol_window_days), float(p.min_vol_floor)) if n in rets else float("nan")
            vol_ratio[n] = min(1.0, v * scales[n] / float(p.strategy_vol_target)) if np.isfinite(v) and v > 0 else 1.0
        alloc = None
        if self.allocator is not None and profiles:
            use_regime = regime if bool(self.s.allocator.get("regime_aware", False)) else None
            dec = self.allocator.allocate({n: profiles[n] for n in names if n in profiles}, paper_stats, vol_ratio, regime=use_regime)
            alloc = dec.to_dict()
            cap_w = dec.weights.drop(CASH)
            cash = float(dec.weights[CASH])
            tilt = budgets.reindex(cap_w.index).fillna(0.0) * cap_w
            tilt = tilt / tilt.sum() * (1.0 - cash) if tilt.sum() > 0 else tilt
            sleeve_capital = {n: float(tilt.get(n, 0.0)) for n in names}
            notes.extend(dec.notes)
        else:
            cash = float(self.s.allocator.cash_floor)
            sleeve_capital = {n: float(budgets[n] * (1.0 - cash)) for n in names}

        # 3. combine, then portfolio controls
        assets = sorted({a for t in sleeve_targets.values() for a in t.index})
        book = pd.Series(0.0, index=assets)
        for n in names:
            book = book.add(sleeve_targets[n].reindex(assets).fillna(0.0) * scales[n] * sleeve_capital[n], fill_value=0.0)
        # portfolio vol: from the combined sleeve return history under the same scales and capital
        if len(scaled_rets):
            port_hist = sum(scaled_rets[n].fillna(0.0) * sleeve_capital[n] for n in scaled_rets.columns)
            pv = realised_vol(port_hist, int(p.vol_window_days), float(p.min_vol_floor))
            port_scale = float(min(2.0, float(p.portfolio_vol_target) / pv)) if np.isfinite(pv) and pv > 0 else 1.0
        else:
            port_scale = 1.0
            notes.append("no sleeve return history: portfolio vol scale left at 1.0")
        book = book * port_scale
        gross = float(book.abs().sum())
        if gross > float(p.gross_cap):
            book = book * (float(p.gross_cap) / gross)
            notes.append(f"gross {gross:.2f} capped to {float(p.gross_cap):.2f}")
        book = book.clip(-float(p.max_single_asset), float(p.max_single_asset))
        pre = book.copy()
        if current is not None:
            book = apply_no_trade_band(book, current, float(p.no_trade_band))
        return PortfolioDecision(book, pre, scales, {k: float(v) for k, v in budgets.items()}, alloc, sleeve_capital, cash, port_scale,
                                 float(book.abs().sum()), notes)


def load_bench_returns(bench_dir: str | Path) -> pd.DataFrame:
    p = Path(bench_dir) / "returns_daily.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_profiles(bench_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    import json

    p = Path(bench_dir) / "profiles.json"
    if not p.exists():
        return {}
    out = {}
    for prof in json.loads(p.read_text()):
        out[prof.get("metadata", {}).get("strategy", prof.get("task_family"))] = prof
    return out
