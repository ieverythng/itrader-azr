# itrader/scripts/pbt_output.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# --------------------------- helpers ---------------------------

def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _get_metric(rec: Dict[str, Any], metric_key: str) -> Optional[float]:
    """Supports both flat keys like 'custom_metrics/episode_return_mean'
    and nested dicts like rec['custom_metrics']['episode_return_mean'].
    """
    if metric_key in rec:
        return _safe_float(rec.get(metric_key))

    # common: "custom_metrics/xxx"
    if metric_key.startswith("custom_metrics/"):
        sub = metric_key.split("/", 1)[1]
        cm = rec.get("custom_metrics")
        if isinstance(cm, dict) and sub in cm:
            return _safe_float(cm.get(sub))

    # sometimes metrics get nested under "sampler_results"
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


def _iter_from_name(p: Path) -> Optional[int]:
    # checkpoint_000123, checkpoint-123, etc.
    m = re.search(r"(?:checkpoint[_-]?)(\d+)", p.name)
    if m:
        return int(m.group(1))
    # rllib_checkpoints/iter_123 style
    m = re.search(r"(?:iter[_-]?)(\d+)", p.name)
    if m:
        return int(m.group(1))
    return None


def _find_trial_dirs(exp_dir: Path) -> List[Path]:
    # Typical Tune layout: exp_dir/<trial_dir>/params.json
    direct = [d for d in exp_dir.iterdir() if d.is_dir() and (d / "params.json").exists()]
    if direct:
        return sorted(direct)

    # fallback: search shallow
    shallow = []
    for d in exp_dir.rglob("params.json"):
        shallow.append(d.parent)
    return sorted(set(shallow))


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
                # ignore malformed lines
                continue
    return out


def _read_progress_csv(trial_dir: Path) -> Optional[pd.DataFrame]:
    f = trial_dir / "progress.csv"
    if not f.exists():
        return None
    try:
        return pd.read_csv(f)
    except Exception:
        return None


def _list_checkpoints(trial_dir: Path) -> List[Path]:
    # common Ray/Tune and RLlib patterns
    cands = []
    for pat in ("checkpoint_*", "checkpoint-*", "rllib_checkpoints", "rllib_checkpoint*"):
        for p in trial_dir.rglob(pat):
            if p.is_dir():
                cands.append(p)
    # also include nested checkpoint dirs under rllib_checkpoints/*
    # (already covered by rglob)
    return sorted(set(cands))


def _pick_checkpoint_near_iter(checkpoints: List[Path], target_iter: int) -> Optional[Path]:
    scored: List[Tuple[int, Path]] = []
    for p in checkpoints:
        it = _iter_from_name(p)
        if it is not None:
            scored.append((it, p))

    if not scored:
        # fall back: choose newest modified
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda x: x.stat().st_mtime)

    scored.sort(key=lambda t: t[0])
    # best checkpoint <= target_iter
    leq = [t for t in scored if t[0] <= target_iter]
    if leq:
        return leq[-1][1]
    # else smallest iter checkpoint
    return scored[0][1]


@dataclass
class BestRow:
    trial_dir: Path
    trial_id: str
    iteration: int
    metric: float
    checkpoint: Optional[Path]
    params_path: Path


# --------------------------- main scan ---------------------------

def scan_experiment(exp_dir: Path, metric_key: str, mode: str, top_k: int) -> List[BestRow]:
    trial_dirs = _find_trial_dirs(exp_dir)
    if not trial_dirs:
        raise FileNotFoundError(f"No trial directories found under: {exp_dir}")

    best_rows: List[BestRow] = []

    for td in trial_dirs:
        params_path = td / "params.json"
        trial_id = td.name

        # 1) Prefer result.json (true source)
        recs = list(_read_result_json(td))
        best_metric = None
        best_iter = None

        if recs:
            for r in recs:
                it = r.get("training_iteration") or r.get("iteration") or r.get("iter")
                try:
                    it = int(it)
                except Exception:
                    continue

                mv = _get_metric(r, metric_key)
                if mv is None:
                    continue

                if best_metric is None:
                    best_metric, best_iter = mv, it
                else:
                    if (mode == "max" and mv > best_metric) or (mode == "min" and mv < best_metric):
                        best_metric, best_iter = mv, it

        # 2) Fallback to progress.csv
        if best_metric is None:
            df = _read_progress_csv(td)
            if df is not None and not df.empty:
                # try multiple possible column names
                col_candidates = [metric_key]
                if metric_key.startswith("custom_metrics/"):
                    col_candidates.append(metric_key.replace("custom_metrics/", "custom_metrics_"))
                col = next((c for c in col_candidates if c in df.columns), None)

                if col is not None:
                    s = pd.to_numeric(df[col], errors="coerce")
                    if mode == "max":
                        idx = int(s.idxmax())
                    else:
                        idx = int(s.idxmin())
                    best_metric = float(s.loc[idx])
                    it_col = "training_iteration" if "training_iteration" in df.columns else None
                    best_iter = int(df.loc[idx, it_col]) if it_col else int(idx)

        if best_metric is None or best_iter is None:
            continue

        # find checkpoint nearest best_iter
        ckpts = _list_checkpoints(td)
        ckpt = _pick_checkpoint_near_iter(ckpts, int(best_iter))

        best_rows.append(
            BestRow(
                trial_dir=td,
                trial_id=trial_id,
                iteration=int(best_iter),
                metric=float(best_metric),
                checkpoint=ckpt,
                params_path=params_path,
            )
        )

    # sort globally
    best_rows.sort(key=lambda r: r.metric, reverse=(mode == "max"))
    return best_rows[:top_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", type=str, default=str(Path.home() / "ray_results" / "Phase_A"))
    ap.add_argument("--metric", type=str, default="custom_metrics/episode_return_mean")
    ap.add_argument("--mode", type=str, choices=("max", "min"), default="max")
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir).expanduser().resolve()
    print(f"Using experiment dir: {exp_dir}")

    best = scan_experiment(exp_dir, args.metric, args.mode, args.top_k)
    if not best:
        print("No valid results found (no metrics in result.json/progress.csv).")
        return

    print("\nTop results:")
    for i, r in enumerate(best, 1):
        print(f"\n#{i}")
        print(f"trial_id:   {r.trial_id}")
        print(f"iter:       {r.iteration}")
        print(f"metric:     {r.metric:.6f}")
        print(f"trial_dir:  {r.trial_dir}")
        print(f"params:     {r.params_path}")
        print(f"checkpoint: {r.checkpoint if r.checkpoint else 'NOT FOUND'}")

    print("\nBest overall:")
    b = best[0]
    print(f"trial_id={b.trial_id} iter={b.iteration} metric={b.metric:.6f}")
    if b.checkpoint:
        print(f"best_checkpoint={b.checkpoint}")
        print("\nRestore hint (RLlib):")
        print("  from ray.rllib.algorithms.algorithm import Algorithm")
        print("  algo = Algorithm.from_checkpoint(r'...path_to_checkpoint...')")
        print("  # then algo.evaluate() or inspect the native module with algo.get_module().")
    else:
        print("No checkpoint directory was detected for the best row. Check your checkpointing settings/patterns.")

if __name__ == "__main__":
    main()
