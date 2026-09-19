import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from itrader.scripts.experiments.run_ablation import (
    _resolve_results_root,
    _series_label,
    VariantResult,
    collect_variant_results,
    evaluate_promotion,
    main,
    write_metric_comparison,
)


def test_ablation_reporting_uses_a_headless_matplotlib_backend():
    assert matplotlib.get_backend().lower() == "agg"


def test_population_plot_labels_identify_reward_and_member_compactly():
    assert _series_label(
        "net_return_1bp_pbt_sync_120:training_fn_ITraderEnv_ab123_00001_1_seed=42"
    ) == "1 bp net · 01"
    assert _series_label(
        "pnl_linear_pbt_sync_120:training_fn_ITraderEnv_cd456_00003_3_seed=81"
    ) == "PnL linear · 03"


def test_collect_variant_results_includes_every_population_member(tmp_path: Path):
    experiment_root = tmp_path / "ray" / "experiment"
    stable_root = tmp_path / "stable"
    experiment_root.mkdir(parents=True)
    for trial_name, score in (("training_fn_a", 0.01), ("training_fn_b", 0.02)):
        (experiment_root / trial_name).mkdir()
        stable_dir = stable_root / trial_name
        stable_dir.mkdir(parents=True)
        pd.DataFrame(
            [{"training_iteration": 1, "evaluation/net_return_mean": score}]
        ).to_csv(stable_dir / "report_rows.csv", index=False)
        (stable_dir / "metrics_full.jsonl").write_text("{}\n", encoding="utf-8")

    results = collect_variant_results(
        variant_id="pbt",
        experiment_name="experiment",
        experiment_root=experiment_root,
        stable_root=stable_root,
        primary_metric="evaluation/net_return_mean",
    )

    assert [result.trial_dir.name for result in results] == ["training_fn_a", "training_fn_b"]
    assert [result.series_id for result in results] == [
        "pbt:training_fn_a",
        "pbt:training_fn_b",
    ]


def test_explicit_results_root_is_resolved_from_the_launch_directory(tmp_path: Path):
    launch_dir = tmp_path / "repo"
    launch_dir.mkdir()

    result = _resolve_results_root(
        {"results_root": "artifacts/ray-results"},
        launch_dir=launch_dir,
    )

    assert result == (launch_dir / "artifacts" / "ray-results").resolve()


def test_ablation_collects_completed_arms_without_retraining(tmp_path: Path, monkeypatch):
    root = tmp_path / "ray"
    names = {"net": "matched_net", "pnl": "matched_pnl_restart"}
    for variant, experiment in names.items():
        trial = root / experiment / f"training_fn_{variant}"
        trial.mkdir(parents=True)
        stable = root / "_stable_metrics" / trial.name
        stable.mkdir(parents=True)
        pd.DataFrame([
            {
                "pbt_progress_steps": 4800,
                "evaluation/net_return_mean": -0.001 if variant == "net" else -0.002,
                "evaluation/net_sharpe_per_sqrt_bar_mean": -0.01,
            }
        ]).to_csv(stable / "report_rows.csv", index=False)
        (stable / "metrics_full.jsonl").write_text("{}\n", encoding="utf-8")
        stable_run = root / "_stable_runs" / experiment
        stable_run.mkdir(parents=True)
        (stable_run / "run_status_run.json").write_text(
            json.dumps({"status": "completed", "trial_count": 1}), encoding="utf-8"
        )
    config = tmp_path / "loadout.json"
    config.write_text(json.dumps({
        "name": "matched", "primary_metric": "evaluation/net_return_mean",
        "common_args": {"results_root": str(root)},
        "variants": [{"id": "net", "args": {}}, {"id": "pnl", "args": {}}],
    }), encoding="utf-8")
    out = tmp_path / "comparison"
    monkeypatch.setattr(sys, "argv", [
        "run_ablation", str(config), "--collect-only", "--run-tag", "matched",
        "--variant-experiment", "net=matched_net",
        "--variant-experiment", "pnl=matched_pnl_restart",
        "--out-dir", str(out),
    ])

    main()

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert [(item["variant_id"], item["experiment_name"]) for item in summary["results"]] == [
        ("net", "matched_net"), ("pnl", "matched_pnl_restart")
    ]
    assert summary["collected_only"] is True
    assert (out / "ablation_compare_action_fractions.png").stat().st_size > 0


def test_ablation_refuses_to_score_an_interrupted_arm(tmp_path: Path, monkeypatch):
    root = tmp_path / "ray"
    stable_run = root / "_stable_runs" / "interrupted"
    stable_run.mkdir(parents=True)
    (stable_run / "run_status_run.json").write_text(
        json.dumps({"status": "starting"}), encoding="utf-8"
    )
    config = tmp_path / "loadout.json"
    config.write_text(json.dumps({
        "name": "matched", "common_args": {"results_root": str(root)},
        "variants": [{"id": "pnl", "args": {}}, {"id": "net", "args": {}}],
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "run_ablation", str(config), "--collect-only", "--run-tag", "matched",
        "--variant-experiment", "pnl=interrupted",
    ])

    with pytest.raises(RuntimeError, match="Experiment is incomplete: interrupted"):
        main()


def test_metric_comparison_prefers_monotonic_pbt_progress_steps(tmp_path: Path):
    metric = "evaluation/net_return_mean"
    results = [
        VariantResult(
            variant_id="pbt",
            series_id="member-a",
            experiment_name="experiment",
            trial_dir=tmp_path / "member-a",
            stable_report_csv=tmp_path / "member-a.csv",
            stable_plot_dir=tmp_path / "member-a-plots",
            report_df=pd.DataFrame(
                {
                    "training_iteration": [1, 2, 1, 2],
                    "pbt_progress_steps": [4800, 9600, 14400, 19200],
                    metric: [0.1, 0.2, 0.5, 0.6],
                }
            ),
        ),
        VariantResult(
            variant_id="pbt",
            series_id="member-b",
            experiment_name="experiment",
            trial_dir=tmp_path / "member-b",
            stable_report_csv=tmp_path / "member-b.csv",
            stable_plot_dir=tmp_path / "member-b-plots",
            report_df=pd.DataFrame(
                {
                    "training_iteration": [1, 2, 3, 4, 5, 6],
                    "pbt_progress_steps": [4800, 9600, 14400, 19200, 24000, 28800],
                    metric: [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
                }
            ),
        ),
    ]

    comparison = write_metric_comparison(results, metric, tmp_path / "comparison.csv")

    assert comparison["pbt_progress_steps"].tolist() == [4800, 9600, 14400, 19200, 24000, 28800]
    assert comparison.loc[2, "member-a"] == 0.5
    assert comparison.loc[3, "member-a"] == 0.6
    assert pd.isna(comparison.loc[4, "member-a"])


def test_promotion_gate_requires_baseline_relative_net_return_and_positive_sharpe(tmp_path: Path):
    def result(variant_id: str, net: float, sharpe: float) -> VariantResult:
        return VariantResult(
            variant_id=variant_id,
            series_id=variant_id,
            experiment_name="experiment",
            trial_dir=tmp_path / variant_id,
            stable_report_csv=tmp_path / f"{variant_id}.csv",
            stable_plot_dir=tmp_path / f"{variant_id}-plots",
            report_df=pd.DataFrame(
                [{
                    "evaluation/net_return_mean": net,
                    "evaluation/net_sharpe_per_sqrt_bar_mean": sharpe,
                }]
            ),
        )

    decision = evaluate_promotion(
        [result("flat", 0.0, 0.0), result("random", 0.001, 0.01), result("solver", 0.0015, -0.02)],
        {
            "baseline_variants": ["flat", "random"],
            "candidate_variants": ["solver"],
            "min_net_return_delta": 0.0001,
            "min_net_sharpe_per_sqrt_bar": 0.0,
        },
    )

    assert decision["baseline_net_return"] == pytest.approx(0.001)
    assert decision["candidates"][0]["net_return_delta"] == pytest.approx(0.0005)
    assert decision["candidates"][0]["promoted"] is False
    assert "sharpe_below_gate" in decision["candidates"][0]["failures"]
