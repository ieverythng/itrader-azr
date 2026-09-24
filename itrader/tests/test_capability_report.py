import json

import pytest

from itrader.scripts.generate_capability_report import generate_capability_report


def test_capability_report_consumes_flat_training_records_and_skips_warmup(tmp_path):
    records = [
        {
            "training_iteration": 1,
            "evaluation/custom_metrics/economic_net_return_mean": -9.0,
            "evaluation/custom_metrics/family_00_net_return_mean": -9.0,
        },
        {
            "training_iteration": 2,
            "evaluation/custom_metrics/economic_net_return_mean": 0.01,
            "evaluation/custom_metrics/economic_net_sharpe_per_sqrt_bar_mean": 0.1,
            "evaluation/custom_metrics/family_00_net_return_mean": 0.01,
            "evaluation/custom_metrics/horizon_96_net_return_mean": 0.01,
        },
        {
            "training_iteration": 3,
            "evaluation/custom_metrics/economic_net_return_mean": 0.03,
            "evaluation/custom_metrics/economic_net_sharpe_per_sqrt_bar_mean": 0.3,
            "evaluation/custom_metrics/family_00_net_return_mean": 0.03,
            "evaluation/custom_metrics/horizon_96_net_return_mean": 0.03,
        },
    ]
    metrics = tmp_path / "metrics_full.jsonl"
    metrics.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    report = generate_capability_report(str(tmp_path), warmup_iters=1)

    assert report["report_window_iterations"] == 2
    assert report["final_performance"]["net_return"]["mean"] == pytest.approx(0.02)
    assert report["final_performance"]["net_sharpe_per_sqrt_bar"]["mean"] == pytest.approx(0.2)
    assert report["by_family"]["00"]["mean"] == pytest.approx(0.02)
    assert report["by_horizon"]["96"]["mean"] == pytest.approx(0.02)
    assert report["task_breakdown_available"] is True
