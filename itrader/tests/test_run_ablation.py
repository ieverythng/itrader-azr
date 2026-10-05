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
    ensure_minimum_free_memory,
    evaluate_promotion,
    main,
    select_variants,
    validate_active_loadout,
    write_metric_comparison,
)


def test_ablation_memory_gate_stops_before_launch_below_declared_floor():
    with pytest.raises(RuntimeError, match="below the 2.00 GiB hard stop"):
        ensure_minimum_free_memory(2.0, available_bytes=int(1.99 * 1024**3))

    assert ensure_minimum_free_memory(
        2.0,
        available_bytes=int(2.01 * 1024**3),
    ) == pytest.approx(2.01, abs=0.001)


def test_r4_v1_min_diagnostic_is_a_fixed_cost_four_cell_design():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "configs"
        / "phase_a_r4_v1_min_2x2_smoke_20260925.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    common = config["common_args"]
    assert common["window"] == 60
    assert common["seed"] == 20260925
    assert common["evaluation_seed"] == 30260943
    assert common["train_batch_size"] == 4800
    assert common["data_sessions"] == "regular"
    assert common["data_symbols"].split(",") == [
        "AAPL",
        "MSFT",
        "AMZN",
        "NVDA",
        "AMD",
        "AVGO",
        "GOOGL",
        "META",
        "SPY",
        "QQQ",
        "IWM",
        "TLT",
    ]
    assert common["start_idx_mode"] == "stream_worker"
    assert common["ret_norm_reset"] == "stream"
    assert common["observation_transform"] == "market_relative_v1"
    assert common["observation_filter"] == "NoFilter"
    assert common["evaluation_num_episodes"] == 32
    assert common["evaluation_task_panel_path"].endswith(
        "dataset_v2_r0_crossed_8x4_20260924.json"
    )
    assert common["task_reward_ids"] == "net_return"
    assert common["optimization_cost_bps"] == 1.0
    assert common["evidence_cost_bps"] == 1.0
    assert common["pbt"] is False
    assert common["workers"] == 0
    assert "optimization_cost_schedule" not in common
    assert {
        variant["id"]: variant["args"]["solver_state_mode"]
        for variant in config["variants"]
    } == {
        "masked_control": "masked",
        "position_only": "position",
        "remaining_only": "time",
        "position_remaining": "both",
    }


def test_r4_v1_blocking_loadout_is_the_frozen_paired_four_by_three_design():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "configs"
        / "phase_a_r4_v1_blocking_4x3_20260925.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    common = config["common_args"]

    assert config["minimum_free_memory_gb"] == 2.0
    assert common["iters"] == 12
    assert common["train_batch_size"] == 4800
    assert common["evaluation_seed"] == 30260943
    assert common["evaluation_num_episodes"] == 32
    assert common["evaluation_task_panel_path"].endswith(
        "dataset_v2_r0_crossed_8x4_20260924.json"
    )
    assert common["workers"] == 0
    assert common["pop"] == 1
    assert common["pbt"] is False
    assert common["checkpoint_interval"] == 12
    assert common["evaluation_interval"] == 1
    assert common["optimization_cost_bps"] == 1.0
    assert common["evidence_cost_bps"] == 1.0
    assert common["task_reward_ids"] == "net_return"
    assert common["data_sessions"] == "regular"
    assert common["window"] == 60
    assert common["observation_transform"] == "market_relative_v1"
    assert common["observation_filter"] == "NoFilter"
    assert "optimization_cost_schedule" not in common

    cells = [
        (variant["args"]["seed"], variant["args"]["solver_state_mode"])
        for variant in config["variants"]
    ]
    assert cells == [
        (seed, arm)
        for seed in (20260924, 20260925, 20260926)
        for arm in ("masked", "position", "time", "both")
    ]


def test_r4_v1_1_replication_uses_new_panel_zero_init_and_frozen_four_by_three_design():
    config_dir = (
        Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "configs"
    )
    config = json.loads(
        (config_dir / "phase_a_r0_plus_r4_v1_1_dec2025_4x3_40_20260925.json").read_text(
            encoding="utf-8"
        )
    )
    common = config["common_args"]

    assert common["iters"] == 40
    assert common["solver_state_residual_init"] == "zero_v1"
    assert common["pbt"] is False
    assert common["workers"] == 0
    assert common["evaluation_num_episodes"] == 32
    assert common["optimization_cost_bps"] == 1.0
    assert common["evaluation_task_panel_path"].endswith(
        "dataset_v2_r4_dec2025_crossed_8x4_20260925.json"
    )
    assert [
        (variant["args"]["seed"], variant["args"]["solver_state_mode"])
        for variant in config["variants"]
    ] == [
        (seed, arm)
        for seed in (20260924, 20260925, 20260926)
        for arm in ("masked", "position", "time", "both")
    ]

    prior = json.loads(
        (
            config_dir
            / "evaluation_panels"
            / "dataset_v2_r0_crossed_8x4_20260924.json"
        ).read_text(encoding="utf-8")
    )
    panel = json.loads(
        (
            config_dir
            / "evaluation_panels"
            / "dataset_v2_r4_dec2025_crossed_8x4_20260925.json"
        ).read_text(encoding="utf-8")
    )
    assert panel["selection_contract"]["selected_without_return_inspection"] is True
    assert panel["selection_contract"]["sealed_final_test"] is False
    assert len(panel["selection_contract"]["anchors"]) == 8
    assert len(panel["tasks"]) == 32
    assert {task["horizon"] for task in panel["tasks"]} == {96, 128, 160, 200}
    assert {task["start_idx"] for task in panel["tasks"]} == {
        anchor["start_idx"] for anchor in panel["selection_contract"]["anchors"]
    }
    prior_exposures = [
        (task["start_idx"] - 60, task["start_idx"] + task["horizon"])
        for task in prior["tasks"]
    ]
    next_exposures = [
        (task["start_idx"] - 60, task["start_idx"] + task["horizon"])
        for task in panel["tasks"]
    ]
    assert all(
        next_end <= prior_start or next_start >= prior_end
        for next_start, next_end in next_exposures
        for prior_start, prior_end in prior_exposures
    )


def test_r4_v1_actor_robustness_loadout_freezes_search_and_new_selection_block():
    config_dir = (
        Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "configs"
    )
    config = json.loads(
        (
            config_dir
            / "phase_a_r0_plus_r4_v1_actor_robustness_pbt_4x40_20260929.json"
        ).read_text(encoding="utf-8")
    )
    common = config["common_args"]

    assert config["minimum_free_memory_gb"] == 2.0
    assert common["workers"] == 0
    assert common["ray_num_cpus"] == 2
    assert common["iters"] == 40
    assert common["train_batch_size"] == 4800
    assert common["pbt"] is True
    assert common["pop"] == 4
    assert common["pbt_synch"] is True
    assert common["pbt_interval"] == 5
    assert common["pbt_burn_in"] == 10
    assert common["checkpoint_interval"] == 5
    assert common["pbt_selection_mode"] == "nonflat_net"
    assert common["pbt_selection_max_flat_action_fraction"] == 0.98
    assert common["pbt_lr_min"] == 0.000005
    assert common["pbt_lr_max"] == 0.00006
    assert common["solver_state_mode"] == "both"
    assert common["solver_state_residual_init"] == "zero_v1"
    assert common["evaluation_num_episodes"] == 32
    assert common["evaluation_task_panel_path"].endswith(
        "dataset_v2_r4_actor_jan2026_crossed_8x4_20260929.json"
    )
    assert common["data_sessions"] == "regular"
    assert common["task_reward_ids"] == "net_return"
    assert common["evidence_cost_bps"] == 1.0
    assert common["observation_transform"] == "market_relative_v1"
    assert common["observation_filter"] == "NoFilter"
    assert common["start_idx_mode"] == "stream_worker"
    assert common["p_pool"] == 0.0

    assert [variant["id"] for variant in config["variants"]] == [
        "both_causal_cost",
        "both_exact_1bp",
    ]
    assert config["variants"][0]["args"]["optimization_cost_schedule"] == (
        "0:0,48000:0.25,96000:0.5,144000:1"
    )
    assert "optimization_cost_bps" not in config["variants"][0]["args"]
    assert config["variants"][1]["args"]["optimization_cost_bps"] == 1.0
    assert "optimization_cost_schedule" not in config["variants"][1]["args"]

    previous_panel = json.loads(
        (
            config_dir
            / "evaluation_panels"
            / "dataset_v2_r4_dec2025_crossed_8x4_20260925.json"
        ).read_text(encoding="utf-8")
    )
    selection_panel = json.loads(
        (
            config_dir
            / "evaluation_panels"
            / "dataset_v2_r4_actor_jan2026_crossed_8x4_20260929.json"
        ).read_text(encoding="utf-8")
    )
    assert selection_panel["selection_contract"]["selected_without_return_inspection"] is True
    assert selection_panel["selection_contract"]["sealed_final_test"] is False
    assert selection_panel["selection_contract"]["role"] == "actor_robustness_selection"
    assert len(selection_panel["selection_contract"]["anchors"]) == 8
    assert len(selection_panel["tasks"]) == 32
    assert {task["horizon"] for task in selection_panel["tasks"]} == {
        96,
        128,
        160,
        200,
    }
    assert {task["start_idx"] for task in selection_panel["tasks"]} == {
        anchor["start_idx"]
        for anchor in selection_panel["selection_contract"]["anchors"]
    }
    previous_exposures = [
        (task["start_idx"] - 60, task["start_idx"] + task["horizon"])
        for task in previous_panel["tasks"]
    ]
    selection_exposures = [
        (task["start_idx"] - 60, task["start_idx"] + task["horizon"])
        for task in selection_panel["tasks"]
    ]
    assert all(
        selection_end <= previous_start or selection_start >= previous_end
        for selection_start, selection_end in selection_exposures
        for previous_start, previous_end in previous_exposures
    )


def test_ablation_reporting_uses_a_headless_matplotlib_backend():
    assert matplotlib.get_backend().lower() == "agg"


def test_variant_selection_preserves_config_order_and_rejects_invalid_requests():
    variants = [
        {"id": "no_cost", "args": {}},
        {"id": "staged_cost", "args": {}},
        {"id": "exact_1bp", "args": {}},
    ]

    assert [variant["id"] for variant in select_variants(variants, ["exact_1bp"])] == [
        "exact_1bp"
    ]
    assert [
        variant["id"]
        for variant in select_variants(variants, ["exact_1bp", "no_cost"])
    ] == ["no_cost", "exact_1bp"]
    assert select_variants(variants, []) == variants

    with pytest.raises(ValueError, match="Unknown variant"):
        select_variants(variants, ["missing"])
    with pytest.raises(ValueError, match="Duplicate variant"):
        select_variants(variants, ["no_cost", "no_cost"])
    with pytest.raises(ValueError, match="loadout contains duplicate"):
        select_variants(
            [{"id": "no_cost", "args": {}}, {"id": "no_cost", "args": {}}],
            [],
        )


def test_active_ablation_runner_rejects_legacy_rllib_stack_selector():
    with pytest.raises(ValueError, match="compatibility-only"):
        validate_active_loadout(
            {"rllib_api_stack": "legacy"},
            [{"id": "historical", "args": {}}],
        )


def test_population_plot_labels_identify_reward_and_member_compactly():
    assert _series_label(
        "net_return_0bp_pbt_sync_80:training_fn_ITraderEnv_ab123_00001_1_seed=42"
    ) == "0 bp learn / 1 bp eval · 01"
    assert _series_label(
        "net_return_staged_pbt_sync_80:training_fn_ITraderEnv_ab123_00002_1_seed=42"
    ) == "staged learn / 1 bp eval · 02"
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


def test_collect_variant_results_accepts_experiment_scoped_stable_metrics(
    tmp_path: Path,
):
    experiment_name = "controls_flat"
    trial_name = "training_fn_baseline_flat_both_zero_v1_00000"
    experiment_root = tmp_path / "ray" / experiment_name
    stable_root = tmp_path / "ray" / "_stable_metrics"
    (experiment_root / trial_name).mkdir(parents=True)
    stable_dir = stable_root / experiment_name / trial_name
    stable_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"training_iteration": 0, "evaluation/net_return_mean": 0.0}]
    ).to_csv(stable_dir / "report_rows.csv", index=False)
    (stable_dir / "metrics_full.jsonl").write_text("{}\n", encoding="utf-8")

    results = collect_variant_results(
        variant_id="flat",
        experiment_name=experiment_name,
        experiment_root=experiment_root,
        stable_root=stable_root,
        primary_metric="evaluation/net_return_mean",
    )

    assert len(results) == 1
    assert results[0].stable_report_csv == stable_dir / "report_rows.csv"


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


def test_promotion_gate_requires_a_trailing_economic_streak(tmp_path: Path):
    def result(variant_id: str, net: list[float], sharpe: list[float]) -> VariantResult:
        return VariantResult(
            variant_id=variant_id,
            series_id=variant_id,
            experiment_name="experiment",
            trial_dir=tmp_path / variant_id,
            stable_report_csv=tmp_path / f"{variant_id}.csv",
            stable_plot_dir=tmp_path / f"{variant_id}-plots",
            report_df=pd.DataFrame(
                {
                    "evaluation/net_return_mean": net,
                    "evaluation/net_sharpe_per_sqrt_bar_mean": sharpe,
                }
            ),
        )

    decision = evaluate_promotion(
        [
            result("long_only", [0.0003], [0.01]),
            result("durable", [0.0015, 0.0016, 0.0017], [0.03, 0.04, 0.05]),
            result("late_spike", [-0.001, -0.001, 0.002], [-0.02, -0.01, 0.06]),
        ],
        {
            "baseline_variants": ["long_only"],
            "candidate_variants": ["durable", "late_spike"],
            "min_net_return_delta": 0.001,
            "min_net_sharpe_delta": 0.01,
            "min_consecutive_passing_reports": 3,
        },
    )

    candidates = {row["variant_id"]: row for row in decision["candidates"]}
    assert candidates["durable"]["promoted"] is True
    assert candidates["durable"]["trailing_passing_reports"] == 3
    assert candidates["late_spike"]["promoted"] is False
    assert candidates["late_spike"]["trailing_passing_reports"] == 1
    assert "passing_streak_below_gate" in candidates["late_spike"]["failures"]


def test_promotion_gate_can_require_sharpe_uplift_over_strongest_baseline(tmp_path: Path):
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
        [result("control", 0.0003, 0.04), result("solver", 0.002, 0.045)],
        {
            "baseline_variants": ["control"],
            "candidate_variants": ["solver"],
            "min_net_return_delta": 0.001,
            "min_net_sharpe_delta": 0.01,
        },
    )

    candidate = decision["candidates"][0]
    assert decision["baseline_net_sharpe_per_sqrt_bar"] == pytest.approx(0.04)
    assert candidate["net_sharpe_delta"] == pytest.approx(0.005)
    assert candidate["promoted"] is False
    assert "sharpe_delta_below_gate" in candidate["failures"]


def test_promotion_gate_can_veto_an_economically_flat_policy(tmp_path: Path):
    def result(variant_id: str, net: float, sharpe: float, flat: float) -> VariantResult:
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
                    "evaluation/custom_metrics/act_flat_frac_mean": flat,
                }]
            ),
        )

    decision = evaluate_promotion(
        [result("flat_control", 0.0, 0.0, 1.0), result("solver", 0.002, 0.04, 1.0)],
        {
            "baseline_variants": ["flat_control"],
            "candidate_variants": ["solver"],
            "max_flat_action_fraction": 0.99,
        },
    )

    candidate = decision["candidates"][0]
    assert candidate["flat_action_fraction"] == pytest.approx(1.0)
    assert candidate["promoted"] is False
    assert "flat_action_fraction_above_gate" in candidate["failures"]


def test_promotion_streak_applies_flat_action_gate_to_every_report(tmp_path: Path):
    def result(variant_id: str, rows: list[tuple[float, float, float]]) -> VariantResult:
        return VariantResult(
            variant_id=variant_id,
            series_id=variant_id,
            experiment_name="experiment",
            trial_dir=tmp_path / variant_id,
            stable_report_csv=tmp_path / f"{variant_id}.csv",
            stable_plot_dir=tmp_path / f"{variant_id}-plots",
            report_df=pd.DataFrame(
                [
                    {
                        "evaluation/net_return_mean": net,
                        "evaluation/net_sharpe_per_sqrt_bar_mean": sharpe,
                        "evaluation/custom_metrics/act_flat_frac_mean": flat,
                    }
                    for net, sharpe, flat in rows
                ]
            ),
        )

    decision = evaluate_promotion(
        [
            result("flat_control", [(0.0, 0.0, 1.0)]),
            result("solver", [(0.002, 0.04, 1.0), (0.003, 0.05, 0.5)]),
        ],
        {
            "baseline_variants": ["flat_control"],
            "candidate_variants": ["solver"],
            "max_flat_action_fraction": 0.99,
            "min_consecutive_passing_reports": 2,
        },
    )

    candidate = decision["candidates"][0]
    assert candidate["trailing_passing_reports"] == 1
    assert candidate["promoted"] is False
    assert "passing_streak_below_gate" in candidate["failures"]


def test_cli_args_preserve_explicit_false_for_optional_booleans():
    from itrader.scripts.experiments.run_ablation import _to_cli_args

    cli = _to_cli_args(
        {
            "auto_plot_on_end": False,
            "stream_jump_on_wrap": False,
            "pbt_synch": False,
            "cpu_only": False,
        }
    )

    assert "--no-auto_plot_on_end" in cli
    assert "--no-stream_jump_on_wrap" in cli
    assert "--no-pbt_synch" in cli
    assert "--no-cpu_only" not in cli
