from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

from itrader.rl_lib.evaluation.provenance import checkpoint_sha256


REQUIRED_METRICS = (
    "evaluation/net_return_mean",
    "evaluation/net_sharpe_per_sqrt_bar_mean",
    "evaluation/custom_metrics/economic_turnover_mean",
    "evaluation/custom_metrics/economic_max_drawdown_mean",
    "evaluation/custom_metrics/act_short_frac_mean",
    "evaluation/custom_metrics/act_flat_frac_mean",
    "evaluation/custom_metrics/act_long_frac_mean",
    "training/optimization_reward_mean",
    "training/net_return_mean",
    "entropy",
    "policy_loss",
    "value_loss",
    "approx_kl",
    "explained_variance",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    invalid_lines = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows, invalid_lines
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            invalid_lines += 1
    return rows, invalid_lines


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _checkpoint_has_files(path_value: Any) -> bool:
    raw_path = str(path_value).strip()
    if not raw_path:
        return False
    try:
        checkpoint_sha256(Path(raw_path))
    except (OSError, ValueError):
        return False
    return True


def _integer(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _lineage_problem(exploit_id: str, reason: str) -> dict[str, str]:
    return {"exploit_id": exploit_id, "reason": reason}


def _strict_lineage_problems(
    *,
    exploits: list[dict[str, Any]],
    selected_events: list[dict[str, Any]],
    stable_dirs: dict[str, Path],
    report_rows: dict[str, list[dict[str, str]]],
    checkpoint_rows: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, str]], int]:
    """Validate the scheduler -> restore -> report -> checkpoint chain."""
    problems: list[dict[str, str]] = []
    verified_receipts = 0
    all_receipts: dict[str, list[tuple[str, Path, dict[str, Any] | None]]] = {}
    for trial_id, stable_dir in stable_dirs.items():
        for receipt_path in sorted((stable_dir / "restore_receipts").glob("*.json")):
            receipt = _read_json(receipt_path)
            exploit_id = str((receipt or {}).get("parent_exploit_id", ""))
            if exploit_id:
                all_receipts.setdefault(exploit_id, []).append(
                    (trial_id, receipt_path, receipt)
                )

    selected_by_identity: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for event in selected_events:
        trial = event.get("trial") or {}
        identity = (
            str(trial.get("trial_id", "")),
            _integer(event.get("checkpoint_progress_steps")),
        )
        selected_by_identity.setdefault(identity, []).append(event)

    seen_exploit_ids: set[str] = set()
    for event in exploits:
        exploit_id = str(event.get("exploit_id", ""))
        if not exploit_id:
            problems.append(_lineage_problem("<missing>", "lineage exploit_id is missing"))
            continue
        if exploit_id in seen_exploit_ids:
            problems.append(_lineage_problem(exploit_id, "lineage exploit_id is duplicated"))
            continue
        seen_exploit_ids.add(exploit_id)

        target = event.get("target") or {}
        donor = event.get("donor") or {}
        provenance = event.get("donor_checkpoint") or {}
        target_id = str(target.get("trial_id", ""))
        donor_id = str(donor.get("trial_id", ""))
        donor_steps = _integer(donor.get("pbt_progress_steps"))
        generation = _integer(event.get("target_restore_generation"))
        donor_path = str(provenance.get("donor_checkpoint_path", ""))
        donor_digest = str(provenance.get("donor_checkpoint_sha256", ""))

        expected_values = {
            "exploit_id": exploit_id,
            "target_trial_id": target_id,
            "donor_trial_id": donor_id,
            "donor_progress_steps": donor_steps,
        }
        if generation < 1:
            problems.append(
                _lineage_problem(exploit_id, "lineage restore generation is missing")
            )
        for key, expected in expected_values.items():
            if provenance.get(key) != expected:
                problems.append(
                    _lineage_problem(
                        exploit_id,
                        f"lineage donor provenance {key} does not match the exploit event",
                    )
                )
        if not donor_path or not donor_digest:
            problems.append(
                _lineage_problem(exploit_id, "lineage donor checkpoint identity is incomplete")
            )
        else:
            selected = selected_by_identity.get((donor_id, donor_steps), [])
            matching_selected = [
                item
                for item in selected
                if str(item.get("stable_checkpoint", "")) == donor_path
                and str(item.get("checkpoint_sha256", "")) == donor_digest
            ]
            if len(matching_selected) != 1:
                problems.append(
                    _lineage_problem(
                        exploit_id,
                        "lineage donor selection event does not exactly match path and digest",
                    )
                )
            try:
                observed_donor_digest = checkpoint_sha256(Path(donor_path))
            except (OSError, ValueError):
                observed_donor_digest = ""
            if observed_donor_digest != donor_digest:
                problems.append(
                    _lineage_problem(
                        exploit_id,
                        "lineage donor checkpoint bytes do not match the recorded digest",
                    )
                )

        receipts = all_receipts.get(exploit_id, [])
        if len(receipts) != 1:
            problems.append(
                _lineage_problem(
                    exploit_id,
                    f"lineage requires exactly one restore receipt; found {len(receipts)}",
                )
            )
            continue
        receipt_trial_id, _, receipt = receipts[0]
        if receipt is None:
            problems.append(_lineage_problem(exploit_id, "lineage receipt is unreadable"))
            continue
        receipt_checks = {
            "receipt_kind": "pbt_exploit",
            "exploit_id": exploit_id,
            "parent_exploit_id": exploit_id,
            "target_trial_id": target_id,
            "restore_generation": generation,
            "donor_trial_id": donor_id,
            "donor_progress_steps": donor_steps,
            "donor_checkpoint_path": donor_path,
            "expected_checkpoint_sha256": donor_digest,
            "observed_checkpoint_sha256": donor_digest,
            "digest_match": True,
        }
        receipt_mismatches = [
            key for key, expected in receipt_checks.items() if receipt.get(key) != expected
        ]
        if receipt_trial_id != target_id:
            receipt_mismatches.append("receipt_directory_target")
        for key in (
            "source_tree_sha256",
            "dataset_sha256",
            "evaluation_panel_sha256",
        ):
            if not str(receipt.get(key, "")):
                receipt_mismatches.append(key)
        if receipt_mismatches:
            problems.append(
                _lineage_problem(
                    exploit_id,
                    "lineage restore receipt mismatch: "
                    + ", ".join(sorted(set(receipt_mismatches))),
                )
            )
            continue

        restore_clock = _integer(receipt.get("target_clock_at_restore"))
        fresh_reports = [
            row
            for row in report_rows.get(target_id, [])
            if _integer(row.get("pbt_restore_generation")) == generation
            and str(row.get("pbt_parent_exploit_id", "")) == exploit_id
            and _integer(row.get("pbt_progress_steps")) > restore_clock
        ]
        if not fresh_reports:
            problems.append(
                _lineage_problem(
                    exploit_id,
                    "lineage has no fresh report from the restored generation",
                )
            )
            continue

        report_steps = {_integer(row.get("pbt_progress_steps")) for row in fresh_reports}
        matching_checkpoints = [
            row
            for row in checkpoint_rows.get(target_id, [])
            if _integer(row.get("pbt_restore_generation")) == generation
            and str(row.get("pbt_parent_exploit_id", "")) == exploit_id
            and _integer(row.get("pbt_progress_steps")) in report_steps
        ]
        valid_checkpoints = []
        for checkpoint in matching_checkpoints:
            checkpoint_path = str(checkpoint.get("stable_checkpoint", ""))
            recorded_digest = str(checkpoint.get("checkpoint_sha256", ""))
            try:
                observed_digest = checkpoint_sha256(Path(checkpoint_path))
            except (OSError, ValueError):
                observed_digest = ""
            if recorded_digest and observed_digest == recorded_digest:
                valid_checkpoints.append(checkpoint)
        if not valid_checkpoints:
            problems.append(
                _lineage_problem(
                    exploit_id,
                    "lineage has no content-addressed checkpoint from a fresh restored report",
                )
            )
            continue
        verified_receipts += 1

    return problems, verified_receipts


def _free_gb(path: Path) -> float:
    existing = path.resolve()
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    return shutil.disk_usage(existing).free / (1024**3)


def _run_status(results_root: Path, experiment_name: str) -> dict[str, Any] | None:
    run_root = results_root / "_stable_runs" / experiment_name
    statuses = sorted(run_root.glob("run_status_*.json")) if run_root.is_dir() else []
    if len(statuses) != 1:
        return None
    return _read_json(statuses[0])


def _member_statuses(results_root: Path, experiment_name: str) -> list[tuple[Path, dict[str, Any]]]:
    stable_root = results_root / "_stable_metrics"
    members: list[tuple[Path, dict[str, Any]]] = []
    if not stable_root.is_dir():
        return members
    for status_path in sorted(stable_root.glob("*/trial_status.json")):
        status = _read_json(status_path)
        if not status:
            continue
        trial_dir = Path(str(status.get("trial_dir", "")))
        if trial_dir.parent.name == experiment_name:
            members.append((status_path.parent, status))
    return members


def inspect_pbt_gate(
    results_root: Path,
    experiment_name: str,
    *,
    review_steps: int,
    expected_members: int,
    minimum_exploits: int,
    minimum_free_gb: float,
    free_gb: float | None = None,
    require_lineage_receipts: bool,
) -> dict[str, Any]:
    """Inspect stable PBT evidence without mutating or controlling the run."""
    results_root = results_root.expanduser().resolve()
    observed_free_gb = _free_gb(results_root) if free_gb is None else float(free_gb)
    failures: list[str] = []
    pending: list[str] = []

    if observed_free_gb < minimum_free_gb:
        failures.append(
            f"storage below floor: {observed_free_gb:.2f} GB < {minimum_free_gb:.2f} GB"
        )

    run_status = _run_status(results_root, experiment_name)
    run_state = str((run_status or {}).get("status", "unknown"))
    if run_state == "failed":
        failures.append(f"run status is failed: {(run_status or {}).get('error', 'unknown error')}")

    members = _member_statuses(results_root, experiment_name)
    if len(members) > expected_members:
        failures.append(
            f"member count exceeds loadout: {len(members)} > {expected_members}"
        )
    elif len(members) < expected_members:
        message = f"waiting for members: {len(members)}/{expected_members}"
        if run_state == "completed":
            failures.append(message)
        else:
            pending.append(message)

    member_progress: dict[str, int] = {}
    member_summaries: list[dict[str, Any]] = []
    stable_dirs: dict[str, Path] = {}
    report_rows: dict[str, list[dict[str, str]]] = {}
    checkpoint_rows: dict[str, list[dict[str, Any]]] = {}
    for stable_dir, status in members:
        trial_id = str(status.get("trial_id", stable_dir.name))
        stable_dirs[trial_id] = stable_dir
        if status.get("status") == "failed":
            failures.append(f"{trial_id}: trial status is failed")
        rows = _read_csv(stable_dir / "report_rows.csv")
        report_rows[trial_id] = rows
        if not rows:
            issue = f"{trial_id}: report_rows.csv is not available"
            (failures if run_state == "completed" else pending).append(issue)
            member_summaries.append(
                {"trial_id": trial_id, "status": status.get("status"), "report_count": 0}
            )
            continue

        rows_with_progress = [
            row for row in rows if _finite(row.get("pbt_progress_steps"))
        ]
        if not rows_with_progress:
            failures.append(f"{trial_id}: no finite pbt_progress_steps")
            continue
        latest = max(rows_with_progress, key=lambda row: int(float(row["pbt_progress_steps"])))
        latest_steps = int(float(latest["pbt_progress_steps"]))
        member_progress[trial_id] = latest_steps

        nonfinite = [metric for metric in REQUIRED_METRICS if not _finite(latest.get(metric))]
        if nonfinite:
            failures.append(f"{trial_id}: nonfinite required metrics: {', '.join(nonfinite)}")

        checkpoints, invalid_checkpoint_rows = _read_jsonl(
            stable_dir / "checkpoint_index.jsonl"
        )
        checkpoint_rows[trial_id] = checkpoints
        if invalid_checkpoint_rows:
            issue = (
                f"{trial_id}: checkpoint index has "
                f"{invalid_checkpoint_rows} incomplete row(s)"
            )
            (failures if run_state == "completed" else pending).append(issue)
        exact_checkpoints = [
            row
            for row in checkpoints
            if int(row.get("pbt_progress_steps", -1)) == latest_steps
        ]
        checkpoint_path = (
            str(exact_checkpoints[-1].get("stable_checkpoint", ""))
            if exact_checkpoints
            else ""
        )
        checkpoint_ok = bool(exact_checkpoints) and _checkpoint_has_files(checkpoint_path)
        if not checkpoint_ok:
            failures.append(f"{trial_id}: latest report has no non-empty exact checkpoint")

        member_summaries.append(
            {
                "trial_id": trial_id,
                "status": status.get("status"),
                "report_count": len(rows),
                "pbt_progress_steps": latest_steps,
                "at_review": latest_steps >= review_steps,
                "exact_checkpoint": checkpoint_ok,
                "net_return": float(latest["evaluation/net_return_mean"])
                if _finite(latest.get("evaluation/net_return_mean"))
                else None,
                "sharpe_per_sqrt_bar": float(
                    latest["evaluation/net_sharpe_per_sqrt_bar_mean"]
                )
                if _finite(latest.get("evaluation/net_sharpe_per_sqrt_bar_mean"))
                else None,
            }
        )

    events_path = (
        results_root
        / "_stable_metrics"
        / "_pbt_events"
        / experiment_name
        / "events.jsonl"
    )
    events, invalid_event_rows = _read_jsonl(events_path)
    if invalid_event_rows:
        issue = f"PBT event log has {invalid_event_rows} incomplete row(s)"
        (failures if run_state == "completed" else pending).append(issue)
    skipped = [event for event in events if event.get("event") == "exploit_skipped"]
    if skipped:
        failures.append(f"PBT recorded {len(skipped)} exploit_skipped event(s)")

    exploits = [event for event in events if event.get("event") == "exploit"]
    selected_events = [
        event for event in events if event.get("event") == "stable_checkpoint_selected"
    ]
    receipts: set[tuple[str, int]] = set()
    invalid_receipts: list[dict[str, Any]] = []
    for event in selected_events:
        receipt_key = (
            str((event.get("trial") or {}).get("trial_id", "")),
            int(event.get("checkpoint_progress_steps", -1)),
        )
        checkpoint_path = str(event.get("stable_checkpoint", ""))
        if _checkpoint_has_files(checkpoint_path):
            receipts.add(receipt_key)
        else:
            invalid_receipts.append(
                {
                    "trial_id": receipt_key[0],
                    "pbt_progress_steps": receipt_key[1],
                    "stable_checkpoint": checkpoint_path,
                }
            )
    if invalid_receipts:
        failures.append(
            f"{len(invalid_receipts)} exact receipt checkpoint(s) are missing or empty"
        )
    missing_receipts: list[dict[str, Any]] = []
    stale_targets: list[dict[str, Any]] = []
    lineage_problems: list[dict[str, str]] = []
    verified_lineage_receipts = 0
    if require_lineage_receipts:
        lineage_problems, verified_lineage_receipts = _strict_lineage_problems(
            exploits=exploits,
            selected_events=selected_events,
            stable_dirs=stable_dirs,
            report_rows=report_rows,
            checkpoint_rows=checkpoint_rows,
        )
        if lineage_problems:
            failures.append(
                f"{len(lineage_problems)} strict lineage verification problem(s)"
            )
    else:
        for event in exploits:
            donor = event.get("donor") or {}
            target = event.get("target") or {}
            donor_key = (
                str(donor.get("trial_id", "")),
                _integer(donor.get("pbt_progress_steps")),
            )
            if donor_key not in receipts:
                missing_receipts.append(
                    {"trial_id": donor_key[0], "pbt_progress_steps": donor_key[1]}
                )
            target_id = str(target.get("trial_id", ""))
            target_steps = _integer(target.get("pbt_progress_steps"))
            if member_progress.get(target_id, -1) <= target_steps:
                stale_targets.append(
                    {"trial_id": target_id, "exploit_steps": target_steps}
                )
        if missing_receipts:
            failures.append(
                f"{len(missing_receipts)} exploit(s) lack an exact donor receipt"
            )
        if stale_targets:
            failures.append(
                f"{len(stale_targets)} exploited target(s) lack a fresh post-restore report"
            )

    members_at_review = sum(
        progress >= review_steps for progress in member_progress.values()
    )
    gate_ready = len(members) == expected_members and members_at_review == expected_members
    if gate_ready and len(exploits) < minimum_exploits:
        failures.append(
            f"insufficient genuine exploits: {len(exploits)} < {minimum_exploits}"
        )
    elif not gate_ready:
        pending.append(
            f"review exposure incomplete: {members_at_review}/{expected_members} members"
        )

    decision = "stop" if failures else "pass" if gate_ready and not pending else "pending"
    return {
        "schema_version": 2,
        "decision": decision,
        "experiment_name": experiment_name,
        "results_root": str(results_root),
        "run_status": run_state,
        "review_steps": review_steps,
        "expected_members": expected_members,
        "observed_members": len(members),
        "members_at_review": members_at_review,
        "minimum_exploits": minimum_exploits,
        "exploit_count": len(exploits),
        "exact_receipt_count": len(receipts),
        "strict_lineage_required": require_lineage_receipts,
        "verified_lineage_receipt_count": verified_lineage_receipts,
        "lineage_problems": lineage_problems,
        "missing_receipts": missing_receipts,
        "invalid_receipts": invalid_receipts,
        "stale_targets": stale_targets,
        "free_gb": observed_free_gb,
        "minimum_free_gb": minimum_free_gb,
        "members": member_summaries,
        "reasons": failures,
        "pending": pending,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect an iTrader PBT continuation gate.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--review-steps", type=int, default=96_000)
    parser.add_argument("--expected-members", type=int, default=4)
    parser.add_argument("--minimum-exploits", type=int, default=2)
    parser.add_argument("--minimum-free-gb", type=float, default=20.0)
    parser.add_argument(
        "--allow-legacy-lineage",
        action="store_true",
        help="Permit pre-v2 runs that lack restore-generation receipts.",
    )
    args = parser.parse_args(argv)

    result = inspect_pbt_gate(
        args.results_root,
        args.experiment_name,
        review_steps=args.review_steps,
        expected_members=args.expected_members,
        minimum_exploits=args.minimum_exploits,
        minimum_free_gb=args.minimum_free_gb,
        require_lineage_receipts=not args.allow_legacy_lineage,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return {"pass": 0, "stop": 1, "pending": 2}[str(result["decision"])]


if __name__ == "__main__":
    raise SystemExit(main())
