import csv
import json
from pathlib import Path

from itrader.scripts.experiments.inspect_pbt_gate import (
    REQUIRED_METRICS,
    inspect_pbt_gate,
)
from itrader.rl_lib.evaluation.provenance import checkpoint_sha256
from itrader.rl_lib.api_stack import LEGACY_API_STACK_ID


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_member(
    root: Path,
    experiment: str,
    trial_id: str,
    progress_steps: list[int],
) -> Path:
    stable = root / "_stable_metrics" / f"training_fn_{trial_id}"
    stable.mkdir(parents=True)
    trial_dir = root / experiment / f"training_fn_{trial_id}"
    trial_dir.mkdir(parents=True)
    (stable / "trial_status.json").write_text(
        json.dumps(
            {
                "trial_id": trial_id,
                "trial_dir": str(trial_dir),
                "status": "running",
                "pbt_progress_steps": progress_steps[-1],
            }
        ),
        encoding="utf-8",
    )

    rows = []
    checkpoints = []
    for index, step in enumerate(progress_steps, start=1):
        checkpoint = stable / "checkpoints" / f"iter_{index:06d}_{step}"
        policy_dir = checkpoint / "policies" / "default_policy"
        policy_dir.mkdir(parents=True)
        (checkpoint / "algorithm_state.pkl").write_bytes(b"checkpoint")
        (checkpoint / "rllib_checkpoint.json").write_text("{}\n", encoding="utf-8")
        (policy_dir / "policy_state.pkl").write_bytes(b"policy")
        checkpoints.append(
            {
                "pbt_progress_steps": step,
                "stable_checkpoint": str(checkpoint),
            }
        )
        row = {metric: 0.1 for metric in REQUIRED_METRICS}
        row["pbt_progress_steps"] = step
        rows.append(row)

    with (stable / "report_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_jsonl(stable / "checkpoint_index.jsonl", checkpoints)
    return stable


def _write_events(root: Path, experiment: str, rows: list[dict]) -> None:
    _write_jsonl(
        root / "_stable_metrics" / "_pbt_events" / experiment / "events.jsonl",
        rows,
    )


def _stamp_restored_generation(
    stable: Path,
    *,
    exploit_id: str,
    generation: int,
) -> None:
    with (stable / "report_rows.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[-1]["pbt_restore_generation"] = str(generation)
    rows[-1]["pbt_parent_exploit_id"] = exploit_id
    fieldnames = list(rows[-1])
    with (stable / "report_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    checkpoint_path = stable / "checkpoint_index.jsonl"
    checkpoints = [
        json.loads(line) for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
    ]
    checkpoints[-1].update(
        {
            "pbt_restore_generation": generation,
            "pbt_parent_exploit_id": exploit_id,
            "checkpoint_sha256": checkpoint_sha256(
                Path(checkpoints[-1]["stable_checkpoint"]),
                api_stack_id=LEGACY_API_STACK_ID,
            ),
        }
    )
    _write_jsonl(checkpoint_path, checkpoints)


def test_gate_passes_exact_receipts_and_post_restore_reports(tmp_path: Path):
    experiment = "matched_no_cost"
    member_a = _write_member(tmp_path, experiment, "trial_a", [48_000, 96_000])
    member_b = _write_member(tmp_path, experiment, "trial_b", [48_000, 96_000])
    _write_events(
        tmp_path,
        experiment,
        [
            {
                "event": "stable_checkpoint_selected",
                "trial": {"trial_id": "trial_a", "pbt_progress_steps": 48_000},
                "checkpoint_progress_steps": 48_000,
                "stable_checkpoint": str(member_a / "checkpoints" / "iter_000001_48000"),
            },
            {
                "event": "exploit",
                "target": {"trial_id": "trial_b", "pbt_progress_steps": 48_000},
                "donor": {"trial_id": "trial_a", "pbt_progress_steps": 48_000},
            },
            {
                "event": "stable_checkpoint_selected",
                "trial": {"trial_id": "trial_b", "pbt_progress_steps": 48_000},
                "checkpoint_progress_steps": 48_000,
                "stable_checkpoint": str(member_b / "checkpoints" / "iter_000001_48000"),
            },
            {
                "event": "exploit",
                "target": {"trial_id": "trial_a", "pbt_progress_steps": 48_000},
                "donor": {"trial_id": "trial_b", "pbt_progress_steps": 48_000},
            },
        ],
    )

    result = inspect_pbt_gate(
        tmp_path,
        experiment,
        review_steps=96_000,
        expected_members=2,
        minimum_exploits=2,
        minimum_free_gb=20.0,
        free_gb=50.0,
        require_lineage_receipts=False,
    )

    assert result["decision"] == "pass"
    assert result["exploit_count"] == 2
    assert result["exact_receipt_count"] == 2
    assert result["members_at_review"] == 2
    assert result["reasons"] == []


def test_gate_remains_pending_before_every_member_reaches_review(tmp_path: Path):
    experiment = "matched_pending"
    _write_member(tmp_path, experiment, "trial_a", [48_000, 96_000])
    _write_member(tmp_path, experiment, "trial_b", [48_000])

    result = inspect_pbt_gate(
        tmp_path,
        experiment,
        review_steps=96_000,
        expected_members=2,
        minimum_exploits=2,
        minimum_free_gb=20.0,
        free_gb=50.0,
        require_lineage_receipts=False,
    )

    assert result["decision"] == "pending"
    assert result["members_at_review"] == 1


def test_gate_discovers_ray_258_driver_artifacts_members(tmp_path: Path):
    experiment = "ray258_layout"
    stable = _write_member(tmp_path, experiment, "trial_a", [48_000])
    status_path = stable / "trial_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["trial_dir"] = str(
        tmp_path / experiment / "driver_artifacts" / "training_fn_trial_a"
    )
    status_path.write_text(json.dumps(status), encoding="utf-8")

    result = inspect_pbt_gate(
        tmp_path,
        experiment,
        review_steps=48_000,
        expected_members=1,
        minimum_exploits=0,
        minimum_free_gb=20.0,
        free_gb=50.0,
        require_lineage_receipts=False,
    )

    assert result["decision"] == "pass"
    assert result["observed_members"] == 1


def test_gate_reads_ray_258_legacy_driver_artifacts_events(tmp_path: Path):
    experiment = "ray258_events"
    member_a = _write_member(tmp_path, experiment, "trial_a", [48_000, 96_000])
    _write_member(tmp_path, experiment, "trial_b", [48_000, 96_000])
    donor_checkpoint = member_a / "checkpoints" / "iter_000001_48000"
    _write_jsonl(
        tmp_path
        / "_stable_metrics"
        / "_pbt_events"
        / "driver_artifacts"
        / "events.jsonl",
        [
            {
                "event": "stable_checkpoint_selected",
                "trial": {"trial_id": "trial_a", "pbt_progress_steps": 48_000},
                "checkpoint_progress_steps": 48_000,
                "stable_checkpoint": str(donor_checkpoint),
            },
            {
                "event": "exploit",
                "target": {"trial_id": "trial_b", "pbt_progress_steps": 48_000},
                "donor": {"trial_id": "trial_a", "pbt_progress_steps": 48_000},
            },
            {
                "event": "exploit",
                "target": {"trial_id": "unrelated_b", "pbt_progress_steps": 48_000},
                "donor": {"trial_id": "unrelated_a", "pbt_progress_steps": 48_000},
            },
        ],
    )

    result = inspect_pbt_gate(
        tmp_path,
        experiment,
        review_steps=96_000,
        expected_members=2,
        minimum_exploits=1,
        minimum_free_gb=20.0,
        free_gb=50.0,
        require_lineage_receipts=False,
    )

    assert result["decision"] == "pass"
    assert result["exploit_count"] == 1


def test_gate_stops_on_skipped_exploit_or_missing_exact_receipt(tmp_path: Path):
    experiment = "matched_invalid"
    _write_member(tmp_path, experiment, "trial_a", [48_000, 96_000])
    _write_member(tmp_path, experiment, "trial_b", [48_000, 96_000])
    _write_events(
        tmp_path,
        experiment,
        [
            {
                "event": "exploit",
                "target": {"trial_id": "trial_b", "pbt_progress_steps": 48_000},
                "donor": {"trial_id": "trial_a", "pbt_progress_steps": 48_000},
            },
            {"event": "exploit_skipped", "reason": "donor_checkpoint_unavailable"},
        ],
    )

    result = inspect_pbt_gate(
        tmp_path,
        experiment,
        review_steps=96_000,
        expected_members=2,
        minimum_exploits=1,
        minimum_free_gb=20.0,
        free_gb=50.0,
        require_lineage_receipts=False,
    )

    assert result["decision"] == "stop"
    assert any("exploit_skipped" in reason for reason in result["reasons"])
    assert any("exact donor receipt" in reason for reason in result["reasons"])


def test_completed_gate_stops_when_required_member_evidence_is_missing(tmp_path: Path):
    experiment = "matched_completed_missing"
    stable = _write_member(tmp_path, experiment, "trial_a", [96_000])
    (stable / "report_rows.csv").unlink()
    run_root = tmp_path / "_stable_runs" / experiment
    run_root.mkdir(parents=True)
    (run_root / "run_status_run.json").write_text(
        json.dumps({"status": "completed", "trial_count": 1}),
        encoding="utf-8",
    )

    result = inspect_pbt_gate(
        tmp_path,
        experiment,
        review_steps=96_000,
        expected_members=1,
        minimum_exploits=0,
        minimum_free_gb=20.0,
        free_gb=50.0,
        require_lineage_receipts=False,
    )

    assert result["decision"] == "stop"
    assert any("report_rows.csv" in reason for reason in result["reasons"])


def test_gate_rejects_receipt_whose_checkpoint_is_empty(tmp_path: Path):
    experiment = "matched_empty_receipt"
    _write_member(tmp_path, experiment, "trial_a", [48_000, 96_000])
    _write_member(tmp_path, experiment, "trial_b", [48_000, 96_000])
    empty_checkpoint = tmp_path / "empty-checkpoint"
    empty_checkpoint.mkdir()
    _write_events(
        tmp_path,
        experiment,
        [
            {
                "event": "stable_checkpoint_selected",
                "trial": {"trial_id": "trial_a", "pbt_progress_steps": 48_000},
                "checkpoint_progress_steps": 48_000,
                "stable_checkpoint": str(empty_checkpoint),
            },
            {
                "event": "exploit",
                "target": {"trial_id": "trial_b", "pbt_progress_steps": 48_000},
                "donor": {"trial_id": "trial_a", "pbt_progress_steps": 48_000},
            },
        ],
    )

    result = inspect_pbt_gate(
        tmp_path,
        experiment,
        review_steps=96_000,
        expected_members=2,
        minimum_exploits=1,
        minimum_free_gb=20.0,
        free_gb=50.0,
        require_lineage_receipts=False,
    )

    assert result["decision"] == "stop"
    assert any("receipt checkpoint" in reason for reason in result["reasons"])


def test_strict_gate_rejects_lagged_step_without_restore_generation_receipt(
    tmp_path: Path,
):
    experiment = "matched_lagged_lineage"
    member_a = _write_member(tmp_path, experiment, "trial_a", [48_000, 96_000])
    _write_member(tmp_path, experiment, "trial_b", [48_000, 96_000])
    _write_events(
        tmp_path,
        experiment,
        [
            {
                "event": "stable_checkpoint_selected",
                "trial": {"trial_id": "trial_a", "pbt_progress_steps": 48_000},
                "checkpoint_progress_steps": 48_000,
                "stable_checkpoint": str(
                    member_a / "checkpoints" / "iter_000001_48000"
                ),
            },
            {
                "event": "exploit",
                "target": {"trial_id": "trial_b", "pbt_progress_steps": 44_000},
                "donor": {"trial_id": "trial_a", "pbt_progress_steps": 48_000},
            },
        ],
    )

    result = inspect_pbt_gate(
        tmp_path,
        experiment,
        review_steps=96_000,
        expected_members=2,
        minimum_exploits=1,
        minimum_free_gb=20.0,
        free_gb=50.0,
        require_lineage_receipts=True,
    )

    assert result["decision"] == "stop"
    assert any("lineage" in reason for reason in result["reasons"])


def test_strict_gate_proves_donor_restore_report_and_checkpoint_chain(tmp_path: Path):
    experiment = "matched_strict_lineage"
    member_a = _write_member(tmp_path, experiment, "trial_a", [48_000, 96_000])
    member_b = _write_member(tmp_path, experiment, "trial_b", [48_000, 96_000])
    exploit_id = "e" * 64
    generation = 1
    donor_checkpoint = member_a / "checkpoints" / "iter_000001_48000"
    donor_digest = checkpoint_sha256(
        donor_checkpoint, api_stack_id=LEGACY_API_STACK_ID
    )
    _stamp_restored_generation(
        member_b,
        exploit_id=exploit_id,
        generation=generation,
    )
    receipt = {
        "schema_version": 1,
        "receipt_kind": "pbt_exploit",
        "exploit_id": exploit_id,
        "parent_exploit_id": exploit_id,
        "target_trial_id": "trial_b",
        "restore_generation": generation,
        "target_clock_at_restore": 48_000,
        "donor_trial_id": "trial_a",
        "donor_progress_steps": 48_000,
        "donor_checkpoint_path": str(donor_checkpoint),
        "expected_checkpoint_sha256": donor_digest,
        "observed_checkpoint_sha256": donor_digest,
        "digest_match": True,
        "source_tree_sha256": "source",
        "dataset_sha256": "dataset",
        "evaluation_panel_sha256": "panel",
    }
    receipt_path = member_b / "restore_receipts" / f"{exploit_id}.json"
    receipt_path.parent.mkdir()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (receipt_path.parent / "self_resume_0001.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_kind": "tune_self_resume",
                "parent_exploit_id": exploit_id,
                "target_trial_id": "trial_b",
                "restore_generation": generation,
                "observed_checkpoint_sha256": checkpoint_sha256(
                    member_b / "checkpoints" / "iter_000002_96000",
                    api_stack_id=LEGACY_API_STACK_ID,
                ),
                "expected_checkpoint_sha256": checkpoint_sha256(
                    member_b / "checkpoints" / "iter_000002_96000",
                    api_stack_id=LEGACY_API_STACK_ID,
                ),
                "digest_match": True,
                "lineage_verified": True,
            }
        ),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": 1,
        "exploit_id": exploit_id,
        "target_trial_id": "trial_b",
        "target_progress_steps": 48_000,
        "donor_trial_id": "trial_a",
        "donor_progress_steps": 48_000,
        "donor_checkpoint_path": str(donor_checkpoint),
        "donor_checkpoint_sha256": donor_digest,
    }
    _write_events(
        tmp_path,
        experiment,
        [
            {
                "event": "stable_checkpoint_selected",
                "trial": {"trial_id": "trial_a", "pbt_progress_steps": 48_000},
                "checkpoint_progress_steps": 48_000,
                "stable_checkpoint": str(donor_checkpoint),
                "checkpoint_sha256": donor_digest,
            },
            {
                "event": "exploit",
                "exploit_id": exploit_id,
                "target_restore_generation": generation,
                "target": {"trial_id": "trial_b", "pbt_progress_steps": 48_000},
                "donor": {"trial_id": "trial_a", "pbt_progress_steps": 48_000},
                "donor_checkpoint": provenance,
            },
        ],
    )

    result = inspect_pbt_gate(
        tmp_path,
        experiment,
        review_steps=96_000,
        expected_members=2,
        minimum_exploits=1,
        minimum_free_gb=20.0,
        free_gb=50.0,
        require_lineage_receipts=True,
    )

    assert result["decision"] == "pass"
    assert result["verified_lineage_receipt_count"] == 1
    assert result["lineage_problems"] == []


def test_strict_gate_waits_for_a_just_scheduled_exploit_restore(tmp_path: Path):
    experiment = "matched_pending_lineage"
    member_a = _write_member(tmp_path, experiment, "trial_a", [48_000, 96_000])
    _write_member(tmp_path, experiment, "trial_b", [48_000, 96_000])
    exploit_id = "f" * 64
    donor_checkpoint = member_a / "checkpoints" / "iter_000002_96000"
    donor_digest = checkpoint_sha256(
        donor_checkpoint, api_stack_id=LEGACY_API_STACK_ID
    )
    provenance = {
        "schema_version": 1,
        "exploit_id": exploit_id,
        "target_trial_id": "trial_b",
        "target_progress_steps": 96_000,
        "donor_trial_id": "trial_a",
        "donor_progress_steps": 96_000,
        "donor_checkpoint_path": str(donor_checkpoint),
        "donor_checkpoint_sha256": donor_digest,
    }
    _write_events(
        tmp_path,
        experiment,
        [
            {
                "event": "stable_checkpoint_selected",
                "trial": {"trial_id": "trial_a", "pbt_progress_steps": 96_000},
                "checkpoint_progress_steps": 96_000,
                "stable_checkpoint": str(donor_checkpoint),
                "checkpoint_sha256": donor_digest,
            },
            {
                "event": "exploit",
                "exploit_id": exploit_id,
                "target_restore_generation": 1,
                "target": {"trial_id": "trial_b", "pbt_progress_steps": 96_000},
                "donor": {"trial_id": "trial_a", "pbt_progress_steps": 96_000},
                "donor_checkpoint": provenance,
            },
        ],
    )

    result = inspect_pbt_gate(
        tmp_path,
        experiment,
        review_steps=96_000,
        expected_members=2,
        minimum_exploits=1,
        minimum_free_gb=20.0,
        free_gb=50.0,
        require_lineage_receipts=True,
    )

    assert result["decision"] == "pending"
    assert result["lineage_problems"] == []
    assert any("awaiting restore receipt" in reason for reason in result["pending"])


def test_strict_lineage_rejects_iteration_rewind_within_generation():
    from itrader.scripts.experiments.inspect_pbt_gate import _strict_lineage_problems

    problems, pending, verified = _strict_lineage_problems(
        exploits=[],
        selected_events=[],
        stable_dirs={},
        report_rows={
            "trial-0": [
                {
                    "pbt_progress_steps": "73448",
                    "training_iteration": "15",
                    "pbt_restore_generation": "0",
                    "pbt_parent_exploit_id": "",
                },
                {
                    "pbt_progress_steps": "78392",
                    "training_iteration": "11",
                    "pbt_restore_generation": "0",
                    "pbt_parent_exploit_id": "",
                },
            ]
        },
        checkpoint_rows={},
    )

    assert pending == []
    assert verified == 0
    assert len(problems) == 1
    assert "training iteration rewound" in problems[0]["reason"]
