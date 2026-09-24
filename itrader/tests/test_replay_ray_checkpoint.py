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
    assert _metric_value({"score_mean": 0.2}, "score") == pytest.approx(0.2)
    with pytest.raises(KeyError, match="missing"):
        _metric_value({}, "missing")
