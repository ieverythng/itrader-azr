from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from itrader.tools.replay_ray_checkpoint import _checkpoint_record, _metric_value, _report_row


def test_checkpoint_record_selects_latest_or_requested_iteration(tmp_path: Path):
    records = [
        {"training_iteration": 2, "stable_checkpoint": "two"},
        {"training_iteration": 4, "stable_checkpoint": "four"},
    ]
    (tmp_path / "checkpoint_index.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    assert _checkpoint_record(tmp_path, None)["stable_checkpoint"] == "four"
    assert _checkpoint_record(tmp_path, 2)["stable_checkpoint"] == "two"
    with pytest.raises(ValueError, match="iteration 3"):
        _checkpoint_record(tmp_path, 3)


def test_report_row_and_metric_value_support_checkpoint_receipt(tmp_path: Path):
    with (tmp_path / "report_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["training_iteration", "score"])
        writer.writeheader()
        writer.writerows(
            [
                {"training_iteration": 1, "score": 0.1},
                {"training_iteration": 2, "score": 0.2},
            ]
        )

    assert _report_row(tmp_path, 2)["score"] == "0.2"
    assert _metric_value({"score__sum": 0.6, "score__count": 3}, "score") == pytest.approx(0.2)
    with pytest.raises(KeyError, match="missing"):
        _metric_value({}, "missing")


def test_replay_rejects_legacy_checkpoint_before_ray_start(tmp_path: Path, monkeypatch):
    from itrader.rl_lib.api_stack import LEGACY_API_STACK_ID
    from itrader.rl_lib.evaluation.provenance import (
        canonical_json_sha256,
        checkpoint_sha256,
    )
    from itrader.tools import replay_ray_checkpoint

    stable_trial_dir = tmp_path / "legacy-trial"
    stable_trial_dir.mkdir()
    checkpoint = tmp_path / "legacy-checkpoint"
    policy = checkpoint / "policies" / "default_policy"
    policy.mkdir(parents=True)
    (checkpoint / "algorithm_state.pkl").write_bytes(b"legacy algorithm")
    (checkpoint / "rllib_checkpoint.json").write_text("{}", encoding="utf-8")
    (policy / "policy_state.pkl").write_bytes(b"legacy policy")

    config = {"api_stack_id": LEGACY_API_STACK_ID}
    config_snapshot = stable_trial_dir / "resolved.json"
    resolved = {
        "api_stack_id": LEGACY_API_STACK_ID,
        "config": config,
        "config_hash": canonical_json_sha256(config),
        "dataset_sha256": "dataset-digest",
        "evaluation_panel_sha256": "panel-digest",
        "observation_schema": {"schema_id": "legacy-test"},
    }
    config_snapshot.write_text(json.dumps(resolved), encoding="utf-8")
    record = {
        "training_iteration": 1,
        "api_stack_id": LEGACY_API_STACK_ID,
        "stable_checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(
            checkpoint, api_stack_id=LEGACY_API_STACK_ID
        ),
        "config_snapshot": str(config_snapshot),
        "config_hash": resolved["config_hash"],
        "dataset_sha256": resolved["dataset_sha256"],
        "evaluation_panel_sha256": resolved["evaluation_panel_sha256"],
        "observation_schema": resolved["observation_schema"],
    }
    (stable_trial_dir / "checkpoint_index.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    metric_columns = (
        "evaluation/optimization_reward_mean",
        "evaluation/gross_return_mean",
        "evaluation/net_return_mean",
        "evaluation/net_sharpe_per_sqrt_bar_mean",
    )
    with (stable_trial_dir / "report_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["training_iteration", *metric_columns]
        )
        writer.writeheader()
        writer.writerow({"training_iteration": 1, **dict.fromkeys(metric_columns, "0")})

    def ray_must_not_start(**_kwargs):
        raise AssertionError("native replay must reject the stack before Ray startup")

    def digest_must_not_be_checked(*_args, **_kwargs):
        raise AssertionError("native replay must reject the stack before digest validation")

    monkeypatch.setattr(replay_ray_checkpoint.ray, "init", ray_must_not_start)
    from itrader.rl_lib.evaluation import run_checkpoint_replay

    monkeypatch.setattr(
        run_checkpoint_replay,
        "verify_checkpoint_digest",
        digest_must_not_be_checked,
    )
    with pytest.raises(ValueError, match="native.*legacy API-stack"):
        replay_ray_checkpoint.replay(stable_trial_dir, None, 1e-12)
