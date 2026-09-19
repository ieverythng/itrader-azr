from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Dict, Optional, Tuple


RewardFn = Callable[[dict[str, Any], dict[str, Any]], tuple[float, dict[str, float]]]
RewardParamNormalizer = Callable[[dict[str, Any]], dict[str, Any]]
RewardParamSampler = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True)
class RewardSpec:
    fn: RewardFn
    normalize_params: Optional[RewardParamNormalizer] = None
    sample_params: Optional[RewardParamSampler] = None


_REGISTRY: dict[str, RewardSpec] = {}


def _finite_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    return value if math.isfinite(value) else 0.0


def _clamp_signed(value: float, cap: float) -> float:
    value = _finite_float(value)
    cap = float(cap)
    if not math.isfinite(cap):
        return value
    return min(abs(value), cap) * (1.0 if value >= 0.0 else -1.0)


def _bounded_float(value: Any, default: float, *, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    value = _finite_float(value if value is not None else default)
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalized_mode(value: Any, *, default: str, allowed: set[str]) -> str:
    value = str(value if value is not None else default).lower()
    return value if value in allowed else default


def _rng_choice(rng: Any, choices: list[Any]) -> Any:
    if hasattr(rng, "choice"):
        try:
            value = rng.choice(choices)
            return value.item() if hasattr(value, "item") else value
        except Exception:
            pass
    raise TypeError("Reward parameter sampler requires an RNG with choice() support")


def _rng_uniform(rng: Any, lo: float, hi: float) -> float:
    if hasattr(rng, "uniform"):
        try:
            return float(rng.uniform(lo, hi))
        except Exception:
            pass
    raise TypeError("Reward parameter sampler requires an RNG with uniform() support")


def _normalize_pnl_linear_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    normalized["pnl_w"] = _bounded_float(normalized.get("pnl_w"), 1.0, lo=0.0, hi=10.0)
    normalized["turnover_w"] = _bounded_float(normalized.get("turnover_w"), 0.0, lo=0.0, hi=0.01)
    normalized["dd_mode"] = _normalized_mode(
        normalized.get("dd_mode"),
        default="delta",
        allowed={"none", "delta", "terminal"},
    )
    normalized["dd_w"] = _bounded_float(normalized.get("dd_w"), 0.04, lo=0.0, hi=0.2)
    normalized["dd_T_weight"] = _bounded_float(normalized.get("dd_T_weight"), 0.0, lo=0.0, hi=1.0)
    normalized["turnover_cap"] = _bounded_float(normalized.get("turnover_cap"), 0.02, lo=0.0, hi=0.1)
    normalized["dd_cap"] = _bounded_float(normalized.get("dd_cap"), 0.05, lo=0.0, hi=0.2)
    normalized["reward_scale"] = _bounded_float(normalized.get("reward_scale"), 1.0, lo=0.1, hi=10.0)
    normalized["normalize"] = _coerce_bool(normalized.get("normalize"), True)
    normalized["norm_mode"] = _normalized_mode(
        normalized.get("norm_mode"),
        default="all",
        allowed={"all", "penalties", "off"},
    )
    normalized["unit_clip"] = _bounded_float(normalized.get("unit_clip"), 100000000.0, lo=1.0)
    return normalized


def _normalize_sharpe_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    normalized["w_sharpe"] = _bounded_float(params.get("w_sharpe"), 0.5, lo=0.0, hi=2.0)
    normalized["sharpe_window"] = int(
        _bounded_float(params.get("sharpe_window"), 32.0, lo=2.0, hi=512.0)
    )
    normalized["sharpe_min_samples"] = int(
        _bounded_float(
            params.get("sharpe_min_samples"),
            8.0,
            lo=2.0,
            hi=float(normalized["sharpe_window"]),
        )
    )
    normalized["sharpe_target"] = _bounded_float(
        params.get("sharpe_target"), 1.0, lo=1e-3, hi=10.0
    )
    return normalized


def _normalize_net_return_params(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize an economically explicit return objective.

    ``reward_scale`` changes PPO's numerical scale without changing which
    policy maximizes the episode objective. Unlike the legacy adaptive return
    normalization, it preserves the exact ratio between return and costs.
    """
    return {
        "cost_bps": _bounded_float(params.get("cost_bps"), 1.0, lo=0.0, hi=100.0),
        "reward_scale": _bounded_float(
            params.get("reward_scale"),
            1000.0,
            lo=1.0,
            hi=1_000_000.0,
        ),
    }


def _normalize_pnl_sharpe_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_pnl_linear_params(params)
    normalized.update(_normalize_sharpe_params(params))
    return normalized


def _normalize_composed_linear_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    for key, value in list(normalized.items()):
        if not key.endswith("_w"):
            continue
        if key == "pnl_w":
            normalized[key] = _bounded_float(value, 1.0, lo=0.0, hi=10.0)
        elif key == "turnover_w":
            normalized[key] = _bounded_float(value, 0.0, lo=0.0, hi=0.01)
        elif key == "dd_w":
            normalized[key] = _bounded_float(value, 0.04, lo=0.0, hi=0.2)
        elif key == "dd_T_weight":
            normalized[key] = _bounded_float(value, 0.0, lo=0.0, hi=1.0)
        else:
            normalized[key] = _finite_float(value)
    return normalized


def _sample_pnl_linear_params(rng: Any) -> dict[str, Any]:
    return {
        "pnl_w": 1.0,
        "turnover_w": _rng_choice(rng, [0.0, 0.0005, 0.001, 0.002]),
        "dd_mode": "delta",
        "dd_w": round(_rng_uniform(rng, 0.0, 0.08), 6),
        "dd_T_weight": 0.0,
        "turnover_cap": 0.02,
        "dd_cap": 0.1,
        "reward_scale": 1.0,
        "normalize": True,
        "norm_mode": "all",
        "unit_clip": 100000000.0,
    }


def _sample_sharpe_params(rng: Any) -> dict[str, Any]:
    return {
        "w_sharpe": _rng_choice(rng, [0.25, 0.5, 0.75]),
        "sharpe_window": _rng_choice(rng, [16, 32, 64]),
        "sharpe_min_samples": 8,
        "sharpe_target": 1.0,
    }


def _sample_net_return_params(rng: Any) -> dict[str, Any]:
    del rng
    return {"cost_bps": 1.0, "reward_scale": 1000.0}


def _sample_pnl_sharpe_params(rng: Any) -> dict[str, Any]:
    params = _sample_pnl_linear_params(rng)
    params.update(_sample_sharpe_params(rng))
    return params


def _sample_composed_linear_params(rng: Any) -> dict[str, Any]:
    return {
        "pnl_w": 1.0,
        "dd_w": round(_rng_uniform(rng, 0.0, 0.08), 6),
        "turnover_w": _rng_choice(rng, [0.0, 0.0005, 0.001, 0.002]),
    }


def register(
    name: str,
    *,
    normalize_params: Optional[RewardParamNormalizer] = None,
    sample_params: Optional[RewardParamSampler] = None,
) -> Callable[[RewardFn], RewardFn]:
    def decorator(fn: RewardFn) -> RewardFn:
        _REGISTRY[name] = RewardSpec(
            fn=fn,
            normalize_params=normalize_params,
            sample_params=sample_params,
        )
        return fn

    return decorator


class RewardEngine:
    @staticmethod
    def compute(
        reward_id: str,
        params: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """Compute one reward step and sanitize the outputs."""
        spec = _REGISTRY.get(reward_id)
        if spec is None:
            raise KeyError(f"Unknown reward_id: {reward_id}")

        reward, components = spec.fn(params or {}, ctx or {})
        clean_components = {key: _finite_float(value) for key, value in (components or {}).items()}
        return _finite_float(reward), clean_components

    @staticmethod
    def normalize_params(reward_id: str, params: Dict[str, Any] | None) -> Dict[str, Any]:
        spec = _REGISTRY.get(reward_id)
        payload = dict(params or {})
        if spec is None or spec.normalize_params is None:
            return payload
        return spec.normalize_params(payload)

    @staticmethod
    def sample_params(reward_id: str, rng: Any) -> Dict[str, Any]:
        spec = _REGISTRY.get(reward_id)
        if spec is None or spec.sample_params is None:
            return {}
        return RewardEngine.normalize_params(reward_id, spec.sample_params(rng))

    @staticmethod
    def registered_ids() -> tuple[str, ...]:
        return tuple(sorted(_REGISTRY.keys()))


@register(
    "net_return",
    normalize_params=_normalize_net_return_params,
    sample_params=_sample_net_return_params,
)
def reward_net_return(params: dict, ctx: dict) -> tuple[float, dict]:
    """Cost-adjusted return with exact episode-level economic arithmetic.

    The reward charges each position change and the terminal liquidation used
    by :class:`EpisodeEvidence`. A constant scale keeps PPO gradients practical
    while preserving the objective's economic ordering.
    """
    normalized = _normalize_net_return_params(params)
    scale = normalized["reward_scale"]
    cost_per_turnover = normalized["cost_bps"] * 1e-4
    pnl_term = _finite_float(ctx.get("pnl_delta", 0.0)) * scale
    turnover = max(0.0, _finite_float(ctx.get("turnover_t", 0.0)))
    liquidation = (
        abs(_finite_float(ctx.get("position_t", 0.0)))
        if bool(ctx.get("terminal", False))
        else 0.0
    )
    components = {
        "pnl_delta": pnl_term,
        "turnover_cost": -(turnover * cost_per_turnover * scale),
        "liquidation_cost": -(liquidation * cost_per_turnover * scale),
    }
    return sum(components.values()), components


@register(
    "pnl_linear",
    normalize_params=_normalize_pnl_linear_params,
    sample_params=_sample_pnl_linear_params,
)
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
    r_scale    = float(params.get("reward_scale", 1.0))
    normalize  = bool(params.get("normalize", True))
    norm_mode  = str(params.get("norm_mode", "all"))   # "all" | "penalties" | "off"
    unit_clip  = float(params.get("unit_clip", float("inf")))

    pnl_delta = float(ctx.get("pnl_delta", 0.0))
    turnover_t = float(ctx.get("turnover_t", 0.0))
    is_terminal = bool(ctx.get("terminal", False))

    # dynamic scale so a "typical" return ~ 1.0
    raw_unit = float(ctx.get("ret_unit", 1.0)) if normalize else 1.0
    unit = _clamp_signed(raw_unit, unit_clip)
    if unit < 0.0:
        unit = 0.0

    r = 0.0
    comps = {}

    # pnl contribution (optionally normalized)
    pnl_term = pnl_w * pnl_delta * (unit if norm_mode == "all" else 1.0)
    r += pnl_term
    comps["pnl_delta"] = pnl_term

    # trading cost: choose either concrete fee or abstract turnover weight.
    trade_cost_raw = to_w * turnover_t
    trade_cost = _clamp_signed(trade_cost_raw, to_cap)
    r -= trade_cost
    comps["turnover"] = -trade_cost

    # drawdown contribution
    if dd_mode == "delta":
        dd_inc = float(ctx.get("dd_inc", 0.0))
        dd_term_raw = (dd_w * dd_inc) * (unit if norm_mode in ("all", "penalties") else 1.0)
        dd_term = _clamp_signed(dd_term_raw, dd_cap)
        r -= dd_term
        comps["dd"] = -dd_term
    elif dd_mode == "terminal" and is_terminal:
        max_dd = float(ctx.get("max_dd", 0.0))
        dd_term_raw = (dd_T_weight * max_dd) * (unit if norm_mode in ("all", "penalties") else 1.0)
        dd_term = _clamp_signed(dd_term_raw, dd_cap)
        r -= dd_term
        comps["dd"] = -dd_term
    else:
        comps["dd"] = 0.0

    if r_scale != 1.0:
        r *= r_scale
        for k in ("pnl_delta", "turnover", "dd"):
            comps[k] = comps.get(k, 0.0) * r_scale

    return r, comps

def _bounded_sharpe_score(
    strategy_returns: Any,
    *,
    window: int,
    min_samples: int,
    target: float,
) -> float:
    try:
        values = [_finite_float(value) for value in strategy_returns]
    except TypeError:
        return 0.0
    values = values[-window:]
    if len(values) < min_samples:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(max(0.0, variance))
    if std <= 1e-12:
        return 0.0
    return math.tanh((mean / std) / target)


def _incremental_sharpe(params: dict[str, Any], ctx: dict[str, Any]) -> float:
    try:
        history = tuple(ctx.get("strategy_returns", ()))
    except TypeError:
        history = ()
    window = int(params.get("sharpe_window", 32))
    min_samples = int(params.get("sharpe_min_samples", 8))
    target = max(1e-3, float(params.get("sharpe_target", 1.0)))
    weight = float(params.get("w_sharpe", 0.5))

    current_score = _bounded_sharpe_score(
        history,
        window=window,
        min_samples=min_samples,
        target=target,
    )
    previous_score = _bounded_sharpe_score(
        history[:-1],
        window=window,
        min_samples=min_samples,
        target=target,
    )
    return weight * (current_score - previous_score)


@register(
    "sharpe_only",
    normalize_params=_normalize_sharpe_params,
    sample_params=_sample_sharpe_params,
)
def reward_sharpe_only(params: Dict[str, Any], ctx: Dict[str, Any]):
    sharpe_delta = _incremental_sharpe(params, ctx)
    return sharpe_delta, {"sharpe_delta": sharpe_delta}


@register(
    "pnl_sharpe",
    normalize_params=_normalize_pnl_sharpe_params,
    sample_params=_sample_pnl_sharpe_params,
)
def reward_pnl_sharpe(params: Dict[str, Any], ctx: Dict[str, Any]):
    base_reward, components = reward_pnl_linear(params, ctx)
    sharpe_delta = _incremental_sharpe(params, ctx)
    components["sharpe_delta"] = sharpe_delta
    return base_reward + sharpe_delta, components

@register(
    "composed_linear",
    normalize_params=_normalize_composed_linear_params,
    sample_params=_sample_composed_linear_params,
)
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
