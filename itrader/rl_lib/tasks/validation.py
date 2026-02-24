from __future__ import annotations

import random
from typing import Any, Dict


_DD_MODES = {"none", "delta", "terminal"}
_NORM_MODES = {"all", "penalties", "off"}


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_bool(x: Any, default: bool) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return bool(default)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(x)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def validate_and_clamp_task(
    task_dict: Dict[str, Any] | None,
    env_config: Dict[str, Any] | None = None,
    *,
    fallback_seed: int | None = None,
) -> Dict[str, Any]:
    """Normalize raw proposer task dict into a safe TaskSpec-compatible payload."""
    cfg = dict(env_config or {})
    out: Dict[str, Any] = dict(task_dict or {})

    window_size = _as_int(cfg.get("window_size", 30), 30)
    n_features = _as_int(cfg.get("n_features", 5), 5)
    t_embed_dim = _as_int(cfg.get("t_embed_dim", 32), 32)
    action_dim = _as_int(cfg.get("action_dim", 3), 3)

    if fallback_seed is None:
        fallback_seed = random.randint(0, 1_000_000)

    out["family_id"] = str(out.get("family_id", "00"))
    out["price_id"] = str(out.get("price_id", "SIM"))
    out["reward_id"] = str(out.get("reward_id", "pnl_linear"))
    out["seed"] = _as_int(out.get("seed", fallback_seed), int(fallback_seed))

    horizon = _as_int(out.get("horizon", 128), 128)
    out["horizon"] = max(64, min(512, horizon))

    start_idx = out.get("start_idx", None)
    if start_idx is None:
        out["start_idx"] = None
    else:
        start_idx = _as_int(start_idx, -1)
        out["start_idx"] = start_idx if start_idx >= 0 else None

    rp = dict(out.get("reward_params", {}))
    rp["pnl_w"] = _clamp(_as_float(rp.get("pnl_w", 1.0), 1.0), 0.0, 10.0)
    rp["turnover_w"] = _clamp(_as_float(rp.get("turnover_w", 0.0), 0.0), 0.0, 0.01)
    rp["dd_w"] = _clamp(_as_float(rp.get("dd_w", 0.04), 0.04), 0.0, 0.2)
    rp["dd_T_weight"] = _clamp(_as_float(rp.get("dd_T_weight", 0.0), 0.0), 0.0, 1.0)
    rp["turnover_cap"] = _clamp(_as_float(rp.get("turnover_cap", 0.02), 0.02), 0.0, 0.1)
    rp["dd_cap"] = _clamp(_as_float(rp.get("dd_cap", 0.05), 0.05), 0.0, 0.2)
    rp["reward_scale"] = _clamp(_as_float(rp.get("reward_scale", 1.0), 1.0), 0.1, 10.0)
    rp["norm_clip"] = _clamp(_as_float(rp.get("norm_clip", 5.0), 5.0), 1.0, 20.0)
    rp["inact_w"] = _clamp(_as_float(rp.get("inact_w", 0.0), 0.0), 0.0, 0.01)
    rp["vol_floor_mult"] = _clamp(_as_float(rp.get("vol_floor_mult", 1.0), 1.0), 0.1, 5.0)
    rp["clip_abs"] = _clamp(_as_float(rp.get("clip_abs", 0.0), 0.0), 0.0, 1.0)

    dd_mode = str(rp.get("dd_mode", "delta")).lower()
    rp["dd_mode"] = dd_mode if dd_mode in _DD_MODES else "delta"

    norm_mode = str(rp.get("norm_mode", "all")).lower()
    rp["norm_mode"] = norm_mode if norm_mode in _NORM_MODES else "all"

    rp["normalize"] = _as_bool(rp.get("normalize", True), True)
    rp["normalize_dd"] = _as_bool(rp.get("normalize_dd", True), True)
    out["reward_params"] = rp

    obs_dim = window_size * n_features + t_embed_dim
    out["obs_dim"] = int(obs_dim)
    out["action_dim"] = int(action_dim)

    return out
