from pathlib import Path

import matplotlib
import pandas as pd

from itrader.scripts.experiments.run_ablation import (
    _to_cli_args,
    _resolve_results_root,
    VariantResult,
    collect_variant_results,
    write_metric_comparison,
)


def test_ablation_cli_preserves_false_boolean_optional_flags():
    assert _to_cli_args(
        {"stream_jump_on_wrap": False, "auto_plot_on_end": False}
    ) == ["--no-stream_jump_on_wrap", "--no-auto_plot_on_end"]


def test_ablation_reporting_uses_a_headless_matplotlib_backend():
    assert matplotlib.get_backend().lower() == "agg"


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


def test_metric_comparison_preserves_reported_iterations_after_pbt_rewind(tmp_path: Path):
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
                    "training_iteration": [1, 2, 5, 6],
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
                    metric: [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
                }
            ),
        ),
    ]

    comparison = write_metric_comparison(results, metric, tmp_path / "comparison.csv")

    assert comparison["training_iteration"].tolist() == [1, 2, 3, 4, 5, 6]
    assert pd.isna(comparison.loc[2, "member-a"])
    assert pd.isna(comparison.loc[3, "member-a"])
    assert comparison.loc[4, "member-a"] == 0.5
    assert comparison.loc[5, "member-a"] == 0.6
