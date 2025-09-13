from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

# Registry of reward callables
_REGISTRY: Dict[str, Any] = {}

def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco

class RewardEngine:
    @staticmethod
    def compute(reward_id: str, params: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Compute per-step reward.
        ctx keys (conventions):
          price_t, price_t1, position_t, action_t, pnl_t, pnl_t1, dd_t1, turnover_t
        Returns (reward, components)
        """
        fn = _REGISTRY.get(reward_id)
        if fn is None:
            raise KeyError(f"Unknown reward_id: {reward_id}")
        r, comps = fn(params or {}, ctx or {})
        # Ensure finite floats
        def _f(x):
            try:
                import math
                return 0.0 if (x is None or not math.isfinite(float(x))) else float(x)
            except Exception:
                return 0.0
        r = _f(r)
        comps = {k: _f(v) for k, v in (comps or {}).items()}
        return r, comps

@register("pnl_linear")
def reward_pnl_linear(params: dict, ctx: dict) -> tuple[float, dict]:
    """Linear mix of pnl_delta, turnover, and drawdown with modes.

    params:
    - pnl_w: float (default 1.0)
    - turnover_w: float (default 0.0) # abstract cost per turnover unit
    - dd_mode: "none"|"delta"|"terminal"
    - dd_w: float (if delta)
    - dd_T_weight: float (if terminal)
    ctx:
    - pnl_delta, turnover_t, dd_inc, max_dd
    - terminal: optional bool (env may add into comps or call compute again on terminal)
    """
    pnl_w = float(params.get("pnl_w", 1.0))
    to_w = float(params.get("turnover_w", 0.0))
    dd_mode = str(params.get("dd_mode", "none"))
    dd_w = float(params.get("dd_w", 0.0))
    dd_T_weight = float(params.get("dd_T_weight", 0.0))

    to_cap = float(params.get("turnover_cap", float("inf")))
    dd_cap = float(params.get("dd_cap", float("inf")))

    pnl_delta = float(ctx.get("pnl_delta", 0.0))
    turnover_t = float(ctx.get("turnover_t", 0.0))

    r = 0.0
    comps = {}

    # pnl contribution
    r += pnl_w * pnl_delta
    comps["pnl_delta"] = pnl_w * pnl_delta

    # trading cost: choose either concrete fee or abstract turnover weight (both allowed but keep small)
    # turnover penalty (capped)
    trade_cost_raw = to_w * turnover_t
    trade_cost = min(abs(trade_cost_raw), to_cap) * (1.0 if trade_cost_raw >= 0 else -1.0)
    r -= trade_cost
    comps["turnover"] = -trade_cost

    # drawdown contribution
    # drawdown
    if dd_mode == "delta":
        dd_inc = float(ctx.get("dd_inc", 0.0))
        dd_term_raw = dd_w * dd_inc
        dd_term = min(abs(dd_term_raw), dd_cap) * (1.0 if dd_term_raw >= 0 else -1.0)
        r -= dd_term
        comps["dd"] = -dd_term
    elif dd_mode == "terminal" and bool(ctx.get("terminal", False)):
        max_dd = float(ctx.get("max_dd", 0.0))
        dd_term_raw = dd_T_weight * max_dd
        dd_term = min(abs(dd_term_raw), dd_cap) * (1.0 if dd_term_raw >= 0 else -1.0)
        r -= dd_term
        comps["dd"] = -dd_term
    else:
        comps["dd"] = 0.0

    return r, comps

@register("pnl_sharpe")
def reward_pnl_sharpe(params: Dict[str, Any], ctx: Dict[str, Any]):
    pnl_w = float(params.get("pnl_w", 1.0))
    dd_w = float(params.get("dd_w", 0.0))
    turnover_w = float(params.get("turnover_w", 0.0))

    # primitives
    pnl_delta = float(ctx.get("pnl_t1", 0.0) - ctx.get("pnl_t", 0.0))
    dd = float(ctx.get("dd_t1", 0.0))  # positive number
    turnover = float(ctx.get("turnover_t", 0.0))

    reward = pnl_w * pnl_delta - dd_w * dd - turnover_w * turnover
    return reward, {
        "pnl_delta": pnl_delta,
        "dd": dd,
        "turnover": turnover,
    }

@register("composed_linear")
def reward_composed_linear(params: Dict[str, Any], ctx: Dict[str, Any]):
    # Weighted sum over provided components (extensible without env changes)
    total = 0.0
    comps = {}
    alias = {
        "dd": "dd_t1",
        "turnover": "turnover_t",
        "pnl_delta": "pnl_delta",   # already friendly
        "ret": "ret",               # optional if you add it
    }
    for k, w in params.items():
        if not k.endswith("_w"):
            continue
        base = k[:-2]  # strip _w
        key  = alias.get(base, base)
        v    = float(ctx.get(key, 0.0))
        total += float(w) * v
        comps[base] = v
    return total, comps