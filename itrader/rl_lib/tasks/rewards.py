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
        if reward_id not in _REGISTRY:
            raise KeyError(f"Unknown reward_id: {reward_id}")
        r, comps = _REGISTRY[reward_id](params, ctx)
        return float(r), comps

@register("pnl_sharpe")
def reward_pnl_sharpe(params: Dict[str, Any], ctx: Dict[str, Any]):
    pnl_w = float(params.get("pnl_w", 1.0))
    dd_w = float(params.get("dd_w", 0.1))
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
    for k, w in params.items():
        if not k.endswith("_w"):
            continue
        base = k[:-2]  # strip _w
        v = float(ctx.get(base, 0.0))
        total += float(w) * v
        comps[base] = v
    return total, comps