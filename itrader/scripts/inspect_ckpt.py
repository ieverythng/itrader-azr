# itrader/scripts/inspect_checkpoint.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def _load_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Empty JSON file: {p}")
    return json.loads(raw)


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _get_metric(rec: Dict[str, Any], metric_key: str) -> Optional[float]:
    if metric_key in rec:
        return _safe_float(rec.get(metric_key))

    if metric_key.startswith("custom_metrics/"):
        sub = metric_key.split("/", 1)[1]
        cm = rec.get("custom_metrics")
        if isinstance(cm, dict) and sub in cm:
            return _safe_float(cm.get(sub))

    sr = rec.get("sampler_results")
    if isinstance(sr, dict):
        if metric_key in sr:
            return _safe_float(sr.get(metric_key))
        cm = sr.get("custom_metrics")
        if isinstance(cm, dict) and metric_key.startswith("custom_metrics/"):
            sub = metric_key.split("/", 1)[1]
            if sub in cm:
                return _safe_float(cm.get(sub))

    return None


def _read_result_json(trial_dir: Path) -> Iterable[Dict[str, Any]]:
    f = trial_dir / "result.json"
    if not f.exists():
        return []
    out = []
    with f.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _iter_from_name(p: Path) -> Optional[int]:
    import re

    m = re.search(r"(?:checkpoint[_-]?)(\d+)", p.name)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:iter[_-]?)(\d+)", p.name)
    if m:
        return int(m.group(1))
    return None


def _list_checkpoints(trial_dir: Path) -> list[Path]:
    cands = []
    for pat in ("checkpoint_*", "checkpoint-*", "rllib_checkpoints", "rllib_checkpoint*"):
        for p in trial_dir.rglob(pat):
            if p.is_dir():
                cands.append(p)
    return sorted(set(cands))


def _pick_checkpoint_near_iter(checkpoints: list[Path], target_iter: int) -> Optional[Path]:
    scored: list[tuple[int, Path]] = []
    for p in checkpoints:
        it = _iter_from_name(p)
        if it is not None:
            scored.append((it, p))

    if not scored:
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda x: x.stat().st_mtime)

    scored.sort(key=lambda t: t[0])
    leq = [t for t in scored if t[0] <= target_iter]
    if leq:
        return leq[-1][1]
    return scored[0][1]


def _load_config_from_result(
    trial_dir: Path,
    metric_key: str,
    mode: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[float], Optional[int]]:
    best_metric = None
    best_iter = None
    best_cfg = None

    for rec in _read_result_json(trial_dir):
        mv = _get_metric(rec, metric_key)
        if mv is None:
            continue

        cfg = rec.get("config")
        if not isinstance(cfg, dict):
            continue

        it = rec.get("training_iteration") or rec.get("iteration") or rec.get("iter")
        try:
            it = int(it)
        except Exception:
            continue

        if best_metric is None:
            best_metric, best_iter, best_cfg = mv, it, cfg
            continue

        if (mode == "max" and mv > best_metric) or (mode == "min" and mv < best_metric):
            best_metric, best_iter, best_cfg = mv, it, cfg

    return best_cfg, best_metric, best_iter


def _load_config(
    trial_dir: Path,
    metric_key: str,
    mode: str,
) -> Tuple[Dict[str, Any], Optional[float], Optional[int]]:
    params_path = trial_dir / "params.json"
    if params_path.exists() and params_path.stat().st_size > 0:
        try:
            return _load_json(params_path), None, None
        except Exception:
            pass

    cfg, best_metric, best_iter = _load_config_from_result(trial_dir, metric_key, mode)
    if cfg is None:
        raise FileNotFoundError(f"No usable config found in {trial_dir} (params.json empty and result.json missing config).")
    return cfg, best_metric, best_iter


def _pick(d: Dict[str, Any], keys: list[str]) -> Dict[str, Any]:
    out = {}
    for k in keys:
        if k in d:
            out[k] = d[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--trial_dir", type=str, help="Path to a trial directory (contains params.json/result.json)")
    group.add_argument("--exp_dir", type=str, help="Path to an experiment dir (will auto-pick best trial)")
    ap.add_argument("--metric", type=str, default="custom_metrics/episode_return_mean")
    ap.add_argument("--mode", type=str, choices=("max", "min"), default="max")
    ap.add_argument("--out", type=str, default="", help="Optional output file to write a full config dump")
    args = ap.parse_args()

    best_ckpt = None
    trial_dir = None
    best_metric = None
    best_iter = None

    if args.exp_dir:
        exp_dir = Path(args.exp_dir).expanduser().resolve()
        try:
            from itrader.scripts.pbt_output import scan_experiment
            best = scan_experiment(exp_dir, args.metric, args.mode, top_k=1)
            if not best:
                raise FileNotFoundError(f"No valid trials found under: {exp_dir}")
            picked = best[0]
            trial_dir = picked.trial_dir
            best_metric = picked.metric
            best_iter = picked.iteration
            best_ckpt = picked.checkpoint
        except Exception:
            raise FileNotFoundError(
                "Failed to auto-pick best trial from exp_dir. "
                "Try running with --trial_dir directly, or ensure pbt_output.py works for this exp_dir."
            )
    else:
        trial_dir = Path(args.trial_dir).expanduser().resolve()

    cfg, cfg_metric, cfg_iter = _load_config(trial_dir, args.metric, args.mode)
    if best_metric is None and cfg_metric is not None:
        best_metric = cfg_metric
        best_iter = cfg_iter
        ckpts = _list_checkpoints(trial_dir)
        if best_iter is not None:
            best_ckpt = _pick_checkpoint_near_iter(ckpts, int(best_iter))

    print(f"\n=== Trial ===\n{trial_dir}")
    if best_metric is not None:
        metric_str = f"{best_metric:.6f}" if isinstance(best_metric, float) else str(best_metric)
        it_str = f"{best_iter}" if best_iter is not None else "?"
        print(f"Best {args.metric} ({args.mode}): {metric_str} @ iter {it_str}")
    if best_ckpt is not None:
        print(f"Checkpoint: {best_ckpt}")

    # === PPO knobs (what PBT mutated) ===
    ppo_keys = [
        "lr", "clip_param", "lambda", "gamma",
        "sgd_minibatch_size", "num_sgd_iter",
        "train_batch_size", "batch_mode",
        "vf_clip_param", "vf_loss_coeff",
        "grad_clip",
        "entropy_coeff", "entropy_coeff_schedule",
        "model",
        "num_workers", "num_gpus", "num_gpus_per_worker",
        "observation_filter",
    ]
    print("\n=== PPO / RLlib Config (key knobs) ===")
    for k, v in _pick(cfg, ppo_keys).items():
        print(f"{k}: {v}")

    # === Env knobs ===
    env_cfg = cfg.get("env_config", {})
    print("\n=== Env Config ===")
    for k in ["df_path", "window_size", "t_embed_mode", "t_scale", "ret_norm_alpha", "t_embed_dim", "task_sampler_overrides"]:
        if k in env_cfg:
            print(f"{k}: {env_cfg[k]}")
    if "env_config/task_sampler_overrides" in cfg:
        print(f"task_sampler_overrides (tune): {cfg['env_config/task_sampler_overrides']}")

    # === TaskSampler / Reward knobs ===
    ts = env_cfg.get("task_sampler", {})
    print("\n=== TaskSampler Config (task/reward ranges) ===")
    for k in [
        "horizons",
        "turnover_w_choices",
        "dd_modes",
        "dd_w_range",
        "turnover_cap",
        "dd_cap",
        "window_size",
        "n_features",
        "t_embed_dim",
        "action_dim",
        "seed",
        "p_pool",
        "task_pool_path",
    ]:
        if k in ts:
            print(f"{k}: {ts[k]}")

    # === Tune overrides (flattened keys) ===
    tune_keys = sorted(k for k in cfg.keys() if "/" in k)
    if tune_keys:
        print("\n=== Tune Overrides (flattened) ===")
        for k in tune_keys:
            print(f"{k}: {cfg[k]}")

    # Optionally dump the whole config (useful to compare vs your baseline)
    if args.out:
        outp = Path(args.out).expanduser().resolve()
        outp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"\nWrote full config dump to: {outp}")

    print("\nDone.")


if __name__ == "__main__":
    main()
