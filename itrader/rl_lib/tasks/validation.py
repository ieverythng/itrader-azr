from __future__ import annotations

import random
from typing import Any, Dict, List

from .families import get_family_spec
from .spec import normalize_task_spec_payload


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


def _as_dict(x: Any) -> Dict[str, Any]:
    return dict(x) if isinstance(x, dict) else {}


def _as_list_of_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        value = x.strip()
        return [value] if value else []
    if isinstance(x, (list, tuple, set)):
        out = []
        for item in x:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(x).strip()
    return [text] if text else []


def validate_and_clamp_task(
    task_dict: Dict[str, Any] | None,
    env_config: Dict[str, Any] | None = None,
    *,
    fallback_seed: int | None = None,
) -> Dict[str, Any]:
    """Normalize raw proposer task dict into a safe TaskSpec-compatible payload."""
    cfg = dict(env_config or {})
    out: Dict[str, Any] = normalize_task_spec_payload(task_dict)

    window_size = _as_int(cfg.get("window_size", 30), 30)
    n_features = _as_int(cfg.get("n_features", 5), 5)
    t_embed_dim = _as_int(cfg.get("t_embed_dim", 32), 32)

    if fallback_seed is None:
        fallback_seed = random.randint(0, 1_000_000)

    out["family_id"] = str(out.get("family_id", "00"))
    family = get_family_spec(out["family_id"])
    out["price_id"] = str(out.get("price_id", "SIM"))
    out["reward_id"] = str(
        out.get("reward_id")
        or (family.default_reward_id if family is not None else "pnl_linear")
    )
    out["seed"] = _as_int(out.get("seed", fallback_seed), int(fallback_seed))

    horizon = _as_int(out.get("horizon", 128), 128)
    min_horizon = _as_int(cfg.get("min_horizon", family.horizon_range[0] if family else 64), 64)
    max_horizon = _as_int(cfg.get("max_horizon", family.horizon_range[1] if family else 512), 512)
    if max_horizon < min_horizon:
        max_horizon = min_horizon
    out["horizon"] = max(min_horizon, min(max_horizon, horizon))

    start_idx = out.get("start_idx", None)
    if start_idx is None:
        out["start_idx"] = None
    else:
        start_idx = _as_int(start_idx, -1)
        out["start_idx"] = start_idx if start_idx >= 0 else None

    rp = _as_dict(out.get("reward_params"))
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

    bar_resolution = out.get("bar_resolution", cfg.get("bar_resolution"))
    if bar_resolution is not None and str(bar_resolution).strip():
        out["bar_resolution"] = str(bar_resolution)
    else:
        out.pop("bar_resolution", None)

    regime_tags = _as_list_of_str(out.get("regime_tags"))
    if regime_tags:
        out["regime_tags"] = regime_tags
    else:
        out.pop("regime_tags", None)

    difficulty_target = _as_dict(out.get("difficulty_target"))
    if difficulty_target:
        p_solve_min = _clamp(_as_float(difficulty_target.get("p_solve_min", 0.0), 0.0), 0.0, 1.0)
        p_solve_max = _clamp(_as_float(difficulty_target.get("p_solve_max", 1.0), 1.0), 0.0, 1.0)
        lo = min(p_solve_min, p_solve_max)
        hi = max(p_solve_min, p_solve_max)
        out["difficulty_target"] = {
            "p_solve_min": lo,
            "p_solve_max": hi,
        }
    else:
        out.pop("difficulty_target", None)

    metadata = _as_dict(out.get("metadata"))
    if metadata:
        out["metadata"] = metadata
    else:
        out.pop("metadata", None)

    t_embed = out.get("t_embed")
    if t_embed is None:
        out.pop("t_embed", None)
    elif isinstance(t_embed, (list, tuple)):
        out["t_embed"] = [_as_float(v, 0.0) for v in t_embed]
    else:
        out.pop("t_embed", None)

    obs_dim = window_size * n_features + t_embed_dim
    action_dim = _as_int(
        cfg.get("action_dim", family.action_dim if family is not None else 3),
        family.action_dim if family is not None else 3,
    )
    out["obs_dim"] = int(obs_dim)
    out["action_dim"] = int(action_dim)

    return out


def validate_and_clamp_execution_intent(
    intent_dict: Dict[str, Any] | None,
    *,
    default_ttl_bars: int = 1,
) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(intent_dict or {})

    entry_logic = _as_dict(out.get("entry_logic"))
    out["entry_logic"] = {
        "type": str(entry_logic.get("type", "market")),
        "params": _as_dict(entry_logic.get("params")),
    }

    exit_logic = _as_dict(out.get("exit_logic"))
    out["exit_logic"] = {
        "type": str(exit_logic.get("type", "ttl")),
        "params": _as_dict(exit_logic.get("params")),
    }

    ttl_bars = _as_int(out.get("ttl_bars", default_ttl_bars), default_ttl_bars)
    out["ttl_bars"] = max(1, min(1024, ttl_bars))

    confidence_band = _as_dict(out.get("confidence_band"))
    min_conf = _clamp(_as_float(confidence_band.get("min", 0.0), 0.0), 0.0, 1.0)
    target_conf = _clamp(_as_float(confidence_band.get("target", min_conf), min_conf), 0.0, 1.0)
    out["confidence_band"] = {
        "min": min_conf,
        "target": max(min_conf, target_conf),
    }

    risk_envelope = _as_dict(out.get("risk_envelope"))
    out["risk_envelope"] = {
        "max_position_pct": _clamp(_as_float(risk_envelope.get("max_position_pct", 0.05), 0.05), 0.0, 1.0),
        "max_spread_bps": _clamp(_as_float(risk_envelope.get("max_spread_bps", 0.0), 0.0), 0.0, 10_000.0),
        "max_slippage_bps": _clamp(_as_float(risk_envelope.get("max_slippage_bps", 0.0), 0.0), 0.0, 10_000.0),
    }
    out["abstain_allowed"] = _as_bool(out.get("abstain_allowed", True), True)
    return out


def validate_candidate_task(
    candidate_dict: Dict[str, Any] | None,
    env_config: Dict[str, Any] | None = None,
    *,
    fallback_seed: int | None = None,
) -> Dict[str, Any]:
    raw = dict(candidate_dict or {})
    task_spec = validate_and_clamp_task(
        raw.get("task_spec"),
        env_config,
        fallback_seed=fallback_seed,
    )
    abstain_option = _as_bool(raw.get("abstain_option", False), False)

    execution_intent_raw = raw.get("execution_intent")
    execution_intent = None
    if execution_intent_raw is not None or not abstain_option:
        execution_intent = validate_and_clamp_execution_intent(
            execution_intent_raw,
            default_ttl_bars=int(task_spec["horizon"]),
        )

    expected_difficulty = raw.get("expected_difficulty", None)
    if expected_difficulty is not None:
        expected_difficulty = _clamp(_as_float(expected_difficulty, 0.0), 0.0, 1.0)

    expected_utility = raw.get("expected_utility", None)
    if expected_utility is not None:
        expected_utility = _as_float(expected_utility, 0.0)

    return {
        "task_spec": task_spec,
        "execution_intent": execution_intent,
        "expected_difficulty": expected_difficulty,
        "expected_utility": expected_utility,
        "rationale": str(raw.get("rationale", "")),
        "abstain_option": abstain_option,
        "created_at": str(raw["created_at"]) if raw.get("created_at") is not None else None,
        "ttl_generated_at": str(raw["ttl_generated_at"]) if raw.get("ttl_generated_at") is not None else None,
        "metadata": _as_dict(raw.get("metadata")),
    }


def validate_capability_profile(profile_dict: Dict[str, Any] | None) -> Dict[str, Any]:
    raw = dict(profile_dict or {})
    task_family = str(raw.get("task_family", raw.get("family_id", "00")))
    bar_resolution = str(raw.get("bar_resolution", "1m"))
    horizon_bars = _as_int(raw.get("horizon_bars", raw.get("horizon", 0)), 0)
    return {
        "task_family": task_family,
        "bar_resolution": bar_resolution,
        "regime_tags": _as_list_of_str(raw.get("regime_tags")),
        "horizon_bars": max(0, horizon_bars),
        "estimated_p_solve": _clamp(_as_float(raw.get("estimated_p_solve", 0.0), 0.0), 0.0, 1.0),
        "expected_net_utility": _as_float(raw.get("expected_net_utility", 0.0), 0.0),
        "calibration_error": max(0.0, _as_float(raw.get("calibration_error", 0.0), 0.0)),
        "sample_count": max(0, _as_int(raw.get("sample_count", 0), 0)),
        "last_updated": str(raw.get("last_updated", "")),
        "metadata": _as_dict(raw.get("metadata")),
    }


def validate_decision_record(
    record_dict: Dict[str, Any] | None,
    env_config: Dict[str, Any] | None = None,
    *,
    fallback_seed: int | None = None,
) -> Dict[str, Any]:
    raw = dict(record_dict or {})
    candidate_tasks = [
        validate_candidate_task(item, env_config, fallback_seed=fallback_seed)
        for item in raw.get("candidate_tasks", [])
    ]
    return {
        "market_summary_ref": str(raw["market_summary_ref"]) if raw.get("market_summary_ref") is not None else None,
        "candidate_tasks": candidate_tasks,
        "selected_candidate_id": str(raw["selected_candidate_id"]) if raw.get("selected_candidate_id") is not None else None,
        "wait_selected": _as_bool(raw.get("wait_selected", False), False),
        "solver_scores": _as_dict(raw.get("solver_scores")),
        "risk_decisions": _as_dict(raw.get("risk_decisions")),
        "executor_action": _as_dict(raw.get("executor_action")),
        "realized_outcome": _as_dict(raw.get("realized_outcome")),
        "created_at": str(raw["created_at"]) if raw.get("created_at") is not None else None,
        "metadata": _as_dict(raw.get("metadata")),
    }
