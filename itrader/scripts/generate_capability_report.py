"""Generate a capability report from Ray PPO metrics JSONL files.

Reads ``metrics_full.jsonl`` (or any JSONL with per-iteration records) and
produces a compact capability summary: mean/median final reward, success rate,
and per-task-family / regime-tag breakdowns when task metadata is available.

The report is the input to online gating design (see next-actions.md item 2):
it must answer "what does this policy actually solve, by family/regime/horizon?"

Usage:
    python -m itrader.scripts.generate_capability_report \
        --metrics-dir /mnt/c/Users/Admin/ray_results/_stable_metrics/<run_dir> \
        [--out reports/experiments/capability_<name>.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a capability report from Ray metrics JSONL.")
    parser.add_argument("--metrics-dir", type=str, required=True, help="Directory containing metrics_full.jsonl")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path (default: <metrics-dir>/capability_report.json)")
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Iterations to skip at the start (policy warmup) when computing final stats",
    )
    return parser.parse_args()


def _load_records(metrics_dir: str) -> List[Dict[str, Any]]:
    path = Path(metrics_dir) / "metrics_full.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No metrics_full.jsonl found in {metrics_dir}")
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and "iteration" in rec:
                records.append(rec)
    return records


def _extract_metrics(rec: Dict[str, Any]) -> Dict[str, float]:
    """Pull the metrics we care about out of one iteration record."""
    m = rec.get("metrics", {}) or {}
    out: Dict[str, float] = {}
    # Episode reward (mean over last eval window)
    for key in ("episode_return/mean", "ep_rew_mean", "reward"):
        if key in m:
            try:
                out["episode_return"] = float(m[key])
            except (TypeError, ValueError):
                pass
            break
    # Success rate
    for key in ("success_rate", "success/mean", "ep_success_rate"):
        if key in m:
            try:
                out["success_rate"] = float(m[key])
            except (TypeError, ValueError):
                pass
            break
    # Episode length
    for key in ("episode_len/mean", "ep_len_mean"):
        if key in m:
            try:
                out["episode_len"] = float(m[key])
            except (TypeError, ValueError):
                pass
            break
    return out


def _group_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = sorted(values)
    n = len(arr)
    return {
        "n": n,
        "mean": sum(arr) / n,
        "median": arr[n // 2] if n % 2 == 1 else (arr[n // 2 - 1] + arr[n // 2]) / 2.0,
        "min": arr[0],
        "max": arr[-1],
    }


def generate_capability_report(metrics_dir: str, warmup_iters: int = 5) -> Dict[str, Any]:
    """Build the capability report dict from a Ray run's metrics JSONL."""
    records = _load_records(metrics_dir)
    if not records:
        raise ValueError(f"No iteration records found in {metrics_dir}")

    records.sort(key=lambda r: int(r.get("iteration", 0)))
    total_iters = len(records)
    tail = records[max(0, total_iters - warmup_iters):] if warmup_iters > 0 else records

    per_iter: List[Dict[str, Any]] = []
    returns: List[float] = []
    success_rates: List[float] = []
    episode_lens: List[float] = []
    for rec in tail:
        metrics = _extract_metrics(rec)
        entry = {"iteration": int(rec.get("iteration", 0)), **metrics}
        per_iter.append(entry)
        if "episode_return" in metrics:
            returns.append(metrics["episode_return"])
        if "success_rate" in metrics:
            success_rates.append(metrics["success_rate"])
        if "episode_len" in metrics:
            episode_lens.append(metrics["episode_len"])

    report: Dict[str, Any] = {
        "metrics_dir": str(metrics_dir),
        "total_iterations": total_iters,
        "report_window_iterations": len(tail),
        "final_performance": {
            "episode_return": _group_stats(returns),
            "success_rate": _group_stats(success_rates),
            "episode_len": _group_stats(episode_lens),
        },
        "per_iteration_tail": per_iter,
    }

    # Per-family / regime breakdown: only possible if the metrics JSONL carries
    # task metadata (e.g. from a custom callback). If absent, record that the
    # breakdown is unavailable and flag it for the next training run.
    family_returns: Dict[str, List[float]] = defaultdict(list)
    regime_returns: Dict[str, List[float]] = defaultdict(list)
    horizon_returns: Dict[str, List[float]] = defaultdict(list)
    has_task_breakdown = False
    for rec in tail:
        task_meta = rec.get("task_metadata") or {}
        if not task_meta:
            continue
        has_task_breakdown = True
        ret_val = None
        for key in ("episode_return/mean", "ep_rew_mean"):
            if key in (rec.get("metrics", {}) or {}):
                try:
                    ret_val = float(rec["metrics"][key])
                except (TypeError, ValueError):
                    pass
                break
        if ret_val is None:
            continue
        fam = str(task_meta.get("family", "unknown"))
        family_returns[fam].append(ret_val)
        for tag in task_meta.get("regime_tags", []) or []:
            regime_returns[str(tag)].append(ret_val)
        horizon = task_meta.get("horizon")
        if horizon is not None:
            horizon_returns[str(int(horizon))].append(ret_val)

    report["by_family"] = {k: _group_stats(v) for k, v in sorted(family_returns.items())}
    report["by_regime_tag"] = {k: _group_stats(v) for k, v in sorted(regime_returns.items())}
    report["by_horizon"] = {k: _group_stats(v) for k, v in sorted(horizon_returns.items())}
    report["task_breakdown_available"] = has_task_breakdown

    return report


def main() -> int:
    args = _parse_args()
    report = generate_capability_report(args.metrics_dir, warmup_iters=args.warmup_iters)
    out_path = args.out or str(Path(args.metrics_dir) / "capability_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote capability report to {out_path}")
    print(json.dumps(report["final_performance"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
