"""Inventory retained evaluation-start provenance without selecting on outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from itrader.rl_lib.evaluation.provenance import is_sha256, sha256_file


def audit_exposure(roots: list[Path], dataset_digest: str) -> dict:
    if not is_sha256(dataset_digest):
        raise ValueError("Expected a nonzero lowercase dataset SHA-256")
    if not roots or any(not root.is_dir() for root in roots):
        raise ValueError("All declared exposure roots must exist as directories")
    paths = sorted({p.resolve() for root in roots for p in root.rglob("resolved_config.json")})
    records = []
    for path in paths:
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("dataset_sha256") != dataset_digest:
            continue
        native = config.get("config", {})
        env = native.get("evaluation_config", {}).get("env_config")
        # Fixed-control artifacts use env_config directly rather than an AlgorithmConfig.
        if env is None and config.get("policy_control"):
            env = config.get("env_config", native.get("env_config"))
        env = env or {}
        tasks = env.get("task_sampler", {}).get("fixed_tasks", [])
        explicit = bool(tasks) and all(type(task.get("start_idx")) is int for task in tasks)
        mode = env.get("start_idx_mode")
        metrics = path.with_name("metrics_full.jsonl")
        evaluated_reports, payload_reports = 0, 0
        if metrics.is_file():
            with metrics.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("evaluation/episodes_this_iter", 0) > 0:
                        evaluated_reports += 1
                        if any("replay_episode_payload" in key for key in row):
                            payload_reports += 1
        records.append({
            "resolved_config": str(path), "config_file_sha256": sha256_file(path),
            "evaluation_start_mode": mode,
            "declared_task_count": len(tasks),
            "explicit_starts": explicit,
            "evaluated_report_count": evaluated_reports,
            "reports_with_replay_payload_key": payload_reports,
            "metrics_file_sha256": sha256_file(metrics) if metrics.is_file() else None,
            "status": "fixed_starts_inventory_only" if explicit and mode == "task_spec"
                      else "requires_historical_start_qualification",
        })
    unresolved = [r for r in records if r["status"] == "requires_historical_start_qualification"]
    return {
        "schema_version": 1, "dataset_sha256": dataset_digest,
        "roots": [str(root.resolve()) for root in roots],
        "resolved_config_count": len(records), "unresolved_config_count": len(unresolved),
        "panel_freeze_authorized": False,
        "scope": "retained_metadata_inventory_not_complete_exposure_certificate",
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_exposure(args.root, args.dataset_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
