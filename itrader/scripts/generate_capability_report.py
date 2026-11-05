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
            if not isinstance(rec, dict):
                continue
            if "training_iteration" in rec:
                records.append(rec)
            elif "iteration" in rec:
                records.append({**rec, "training_iteration": rec["iteration"]})
    return records


def _extract_metrics(rec: Dict[str, Any]) -> Dict[str, float]:
    """Pull held-out economic metrics from flat or legacy nested records."""
    nested = rec.get("metrics", {}) or {}
    m = {**nested, **rec}
    out: Dict[str, float] = {}

    key_map = {
        "net_return": (
            "evaluation/custom_metrics/economic_net_return_mean",
            "evaluation/net_return_mean",
        ),
        "gross_return": (
            "evaluation/custom_metrics/economic_gross_return_mean",
            "evaluation/gross_return_mean",
        ),
        "net_sharpe_per_sqrt_bar": (
            "evaluation/custom_metrics/economic_net_sharpe_per_sqrt_bar_mean",
            "evaluation/net_sharpe_per_sqrt_bar_mean",
        ),
        "max_drawdown": ("evaluation/custom_metrics/economic_max_drawdown_mean",),
        "turnover": ("evaluation/custom_metrics/economic_turnover_mean",),
        "success_rate": ("evaluation/custom_metrics/success_mean",),
        "episode_len": ("evaluation/episode_len_mean", "episode_len/mean", "ep_len_mean"),
        "optimization_reward": (
            "evaluation/custom_metrics/optimization_reward_mean",
            "evaluation/optimization_reward_mean",
        ),
    }
    for output_key, candidates in key_map.items():
        for key in candidates:
            if key not in m:
                continue
            try:
                out[output_key] = float(m[key])
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

    records.sort(key=lambda r: int(r.get("training_iteration", 0)))
    total_iters = len(records)
    tail = records[max(0, int(warmup_iters)) :]
    if not tail:
        raise ValueError("warmup_iters removes every available iteration")

    per_iter: List[Dict[str, Any]] = []
    collected: Dict[str, List[float]] = defaultdict(list)
    for rec in tail:
        metrics = _extract_metrics(rec)
        entry = {"iteration": int(rec.get("training_iteration", 0)), **metrics}
        per_iter.append(entry)
        for key, value in metrics.items():
            collected[key].append(value)

    report: Dict[str, Any] = {
        "metrics_dir": str(metrics_dir),
        "total_iterations": total_iters,
        "report_window_iterations": len(tail),
        "metric_semantics": "held-out validation economic evidence; optimization_reward is separate",
        "final_performance": {key: _group_stats(values) for key, values in sorted(collected.items())},
        "per_iteration_tail": per_iter,
    }

    family_returns: Dict[str, List[float]] = defaultdict(list)
    regime_returns: Dict[str, List[float]] = defaultdict(list)
    horizon_returns: Dict[str, List[float]] = defaultdict(list)
    has_task_breakdown = False
    for rec in tail:
        flattened = {**(rec.get("metrics", {}) or {}), **rec}
        prefix = "evaluation/custom_metrics/"
        suffix = "_net_return_mean"
        for key, value in flattened.items():
            if not key.startswith(prefix) or not key.endswith(suffix):
                continue
            metric = key[len(prefix) : -len(suffix)]
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if metric.startswith("family_"):
                family_returns[metric[len("family_") :]].append(numeric)
                has_task_breakdown = True
            elif metric.startswith("regime_"):
                regime_returns[metric[len("regime_") :]].append(numeric)
                has_task_breakdown = True
            elif metric.startswith("horizon_"):
                horizon_returns[metric[len("horizon_") :]].append(numeric)
                has_task_breakdown = True

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
