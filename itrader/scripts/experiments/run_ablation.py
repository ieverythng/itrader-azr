from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class VariantResult:
    variant_id: str
    series_id: str
    experiment_name: str
    trial_dir: Path
    stable_report_csv: Path
    stable_plot_dir: Path
    report_df: pd.DataFrame


def _to_cli_args(arg_map: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k, v in arg_map.items():
        key = f"--{k}"
        if isinstance(v, bool):
            if k == "stream_jump_on_wrap":
                out.append(key if v else "--no-stream_jump_on_wrap")
            elif v:
                out.append(key)
            continue
        out.extend([key, str(v)])
    return out


def _find_trial_dirs(experiment_root: Path) -> list[Path]:
    trials = sorted(
        d for d in experiment_root.iterdir() if d.is_dir() and d.name.startswith("training_fn_")
    )
    if not trials:
        raise RuntimeError(f"No trial directories found under {experiment_root}")
    return trials


def _resolve_results_root(
    arg_map: dict[str, Any],
    *,
    launch_dir: Path | None = None,
) -> Path:
    configured = str(arg_map.get("results_root", "") or "").strip()
    if not configured:
        return (Path.home() / "ray_results").resolve()
    root = Path(configured).expanduser()
    if not root.is_absolute():
        root = (launch_dir or Path.cwd()) / root
    return root.resolve()


def collect_variant_results(
    *,
    variant_id: str,
    experiment_name: str,
    experiment_root: Path,
    stable_root: Path,
    primary_metric: str,
) -> list[VariantResult]:
    trial_dirs = _find_trial_dirs(experiment_root)
    multiple = len(trial_dirs) > 1
    results: list[VariantResult] = []
    for trial_dir in trial_dirs:
        stable_trial_dir = stable_root / trial_dir.name
        stable_report_csv = stable_trial_dir / "report_rows.csv"
        stable_metrics_jsonl = stable_trial_dir / "metrics_full.jsonl"
        if not stable_report_csv.exists() or stable_report_csv.stat().st_size == 0:
            raise RuntimeError(f"Stable report CSV missing or empty: {stable_report_csv}")
        if not stable_metrics_jsonl.exists() or stable_metrics_jsonl.stat().st_size == 0:
            raise RuntimeError(f"Stable metrics JSONL missing or empty: {stable_metrics_jsonl}")
        report_df = pd.read_csv(stable_report_csv)
        if primary_metric not in report_df.columns:
            raise RuntimeError(f"Missing {primary_metric} in {stable_report_csv}")
        results.append(
            VariantResult(
                variant_id=variant_id,
                series_id=f"{variant_id}:{trial_dir.name}" if multiple else variant_id,
                experiment_name=experiment_name,
                trial_dir=trial_dir,
                stable_report_csv=stable_report_csv,
                stable_plot_dir=stable_trial_dir / "plots",
                report_df=report_df,
            )
        )
    return results


def _run_variant(
    python_exe: str,
    module: str,
    common_args: dict[str, Any],
    variant_id: str,
    variant_args: dict[str, Any],
    run_tag: str,
    primary_metric: str,
) -> list[VariantResult]:
    experiment_name = f"{run_tag}_{variant_id}"
    args = dict(common_args)
    args.update(variant_args)
    args["experiment_name"] = experiment_name

    cmd = [python_exe, "-m", module] + _to_cli_args(args)
    print(f"\n[AB] Running variant '{variant_id}'")
    print("[AB] Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    results_root = _resolve_results_root(args)
    experiment_root = results_root / experiment_name
    results = collect_variant_results(
        variant_id=variant_id,
        experiment_name=experiment_name,
        experiment_root=experiment_root,
        stable_root=results_root / "_stable_metrics",
        primary_metric=primary_metric,
    )
    for result in results:
        result.stable_plot_dir.mkdir(parents=True, exist_ok=True)
        plot_cmd = [
            python_exe,
            str(Path(__file__).resolve().parents[2] / "tools" / "plot_ray_results.py"),
            "--metrics-jsonl",
            str(result.stable_plot_dir.parent / "metrics_full.jsonl"),
            "--out",
            str(result.stable_plot_dir),
        ]
        subprocess.run(plot_cmd, check=True)
    return results


def write_metric_comparison(
    results: list[VariantResult],
    metric: str,
    out_csv: Path,
) -> pd.DataFrame:
    """Align metrics by monotonic exposure, falling back for legacy runs."""
    progress_column = (
        "pbt_progress_steps"
        if results and all("pbt_progress_steps" in result.report_df.columns for result in results)
        else "training_iteration"
    )
    series: list[pd.Series] = []
    for result in results:
        if metric not in result.report_df.columns or progress_column not in result.report_df.columns:
            continue
        values = result.report_df[[progress_column, metric]].copy()
        values[progress_column] = pd.to_numeric(
            values[progress_column], errors="coerce"
        )
        values[metric] = pd.to_numeric(values[metric], errors="coerce")
        values = values.dropna(subset=[progress_column])
        values = values.drop_duplicates(subset=[progress_column], keep="last")
        series.append(values.set_index(progress_column)[metric].rename(result.series_id))

    if series:
        comparison = pd.concat(series, axis=1).sort_index().reset_index()
    else:
        comparison = pd.DataFrame(columns=[progress_column])
    comparison.to_csv(out_csv, index=False)
    return comparison


def _series_label(series_id: str) -> str:
    variant, separator, trial = series_id.partition(":")
    reward = (
        "1 bp net" if variant.startswith("net_return")
        else "PnL linear" if variant.startswith("pnl_linear")
        else variant
    )
    member = re.search(r"_(\d{5})_\d+_seed=", trial) if separator else None
    return f"{reward} · {int(member.group(1)):02d}" if member else reward


def _plot_metric(df: pd.DataFrame, title: str, ylabel: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    progress_column = "pbt_progress_steps" if "pbt_progress_steps" in df.columns else "training_iteration"
    if progress_column not in df.columns:
        plt.close(fig)
        return
    for c in [col for col in df.columns if col != progress_column]:
        y = pd.to_numeric(df[c], errors="coerce")
        valid = y.notna()
        if valid.any():
            ax.plot(df.loc[valid, progress_column], y[valid], marker="o", markersize=2,
                    linewidth=1.1, label=_series_label(c))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Sampled environment steps" if progress_column == "pbt_progress_steps" else progress_column)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.15 if handles else 0, 1, 1))
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _plot_multi_metrics(metric_dfs: dict[str, pd.DataFrame], out_png: Path):
    action_metric = next(
        (metric for metric in sorted(metric_dfs) if metric.startswith("custom_metrics/act_") and metric.endswith("_mean")),
        None,
    )
    metric_titles = [
        ("evaluation/net_return_mean", "Held-out Net Return Mean"),
        ("entropy", "Entropy"),
        ("episode_len_mean", "Episode Length Mean"),
        (action_metric, "Action Fraction") if action_metric else (None, "Action Fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (metric, title) in zip(axes, metric_titles):
        df = metric_dfs.get(metric)
        progress_column = (
            "pbt_progress_steps"
            if df is not None and "pbt_progress_steps" in df.columns
            else "training_iteration"
        )
        if metric is None or df is None or progress_column not in df.columns:
            ax.set_title(f"{title} (no data)")
            ax.axis("off")
            continue
        x = df[progress_column].to_numpy()
        plotted = False
        for c in [col for col in df.columns if col != progress_column]:
            y = pd.to_numeric(df[c], errors="coerce")
            valid = y.notna()
            if valid.any():
                ax.plot(x[valid.to_numpy()], y[valid].to_numpy(), linewidth=1,
                        label=_series_label(c))
                plotted = True
        if plotted:
            ax.set_title(title)
            ax.legend(fontsize=6, ncol=2)
        else:
            ax.set_title(f"{title} (no data)")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def _plot_action_fractions(metric_dfs: dict[str, pd.DataFrame], out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, action in zip(axes, ("short", "flat", "long")):
        metric = f"evaluation/custom_metrics/act_{action}_frac_mean"
        frame = metric_dfs.get(metric)
        if frame is not None and not frame.empty:
            x_key = "pbt_progress_steps" if "pbt_progress_steps" in frame else "training_iteration"
            for column in frame.columns.drop(x_key):
                y = pd.to_numeric(frame[column], errors="coerce")
                valid = y.notna()
                if valid.any():
                    ax.plot(frame.loc[valid, x_key], y[valid],
                            label=_series_label(column), linewidth=1.3)
        ax.set_title(f"Held-out {action} fraction")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Sampled environment steps")
    axes[0].set_ylabel("Action fraction")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", fontsize=8, ncol=4)
    fig.tight_layout(rect=(0, 0.19 if handles else 0, 1, 1))
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def evaluate_promotion(results: list[VariantResult], contract: dict[str, Any]) -> dict[str, Any]:
    """Apply explicit terminal gates relative to the strongest named baseline."""
    net_metric = "evaluation/net_return_mean"
    sharpe_metric = "evaluation/net_sharpe_per_sqrt_bar_mean"
    baseline_ids = set(contract.get("baseline_variants", []))
    candidate_ids = set(contract.get("candidate_variants", []))
    minimum_delta = float(contract.get("min_net_return_delta", 0.0))
    minimum_sharpe = float(contract.get("min_net_sharpe_per_sqrt_bar", 0.0))
    require_positive = bool(contract.get("require_positive_net_return", True))

    def terminal(result: VariantResult, metric: str) -> float:
        if metric not in result.report_df or result.report_df.empty:
            return float("nan")
        return float(pd.to_numeric(result.report_df[metric], errors="coerce").iloc[-1])

    baseline_scores = [
        terminal(result, net_metric)
        for result in results
        if result.variant_id in baseline_ids
    ]
    baseline_scores = [score for score in baseline_scores if math.isfinite(score)]
    if not baseline_scores:
        raise RuntimeError("Promotion contract produced no finite baseline terminal scores")
    baseline_net = max(baseline_scores)

    candidates = []
    for result in results:
        if result.variant_id not in candidate_ids:
            continue
        net_return = terminal(result, net_metric)
        sharpe = terminal(result, sharpe_metric)
        delta = net_return - baseline_net
        failures = []
        if not math.isfinite(net_return) or delta < minimum_delta:
            failures.append("net_return_delta_below_gate")
        if require_positive and (not math.isfinite(net_return) or net_return <= 0.0):
            failures.append("net_return_not_positive")
        if not math.isfinite(sharpe) or sharpe <= minimum_sharpe:
            failures.append("sharpe_below_gate")
        candidates.append(
            {
                "variant_id": result.variant_id,
                "series_id": result.series_id,
                "net_return": net_return,
                "net_sharpe_per_sqrt_bar": sharpe,
                "net_return_delta": delta,
                "promoted": not failures,
                "failures": failures,
            }
        )

    return {
        "schema_version": 1,
        "baseline_variants": sorted(baseline_ids),
        "candidate_variants": sorted(candidate_ids),
        "baseline_net_return": baseline_net,
        "min_net_return_delta": minimum_delta,
        "min_net_sharpe_per_sqrt_bar": minimum_sharpe,
        "require_positive_net_return": require_positive,
        "candidates": candidates,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run reproducible start-index ablations.")
    ap.add_argument("config_json", help="Path to ablation config JSON")
    ap.add_argument("--out-dir", help="Optional output directory for comparison artifacts")
    ap.add_argument("--collect-only", action="store_true", help="Report completed experiments without training")
    ap.add_argument("--run-tag", help="Existing ablation run tag (required with --collect-only)")
    ap.add_argument(
        "--variant-experiment", action="append", default=[], metavar="ID=EXPERIMENT",
        help="Override a variant's experiment name when collecting a restarted arm",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config_json).expanduser().resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    python_exe = str(cfg.get("python_exe", "python"))
    module = str(cfg.get("train_module", "itrader.rl_lib.train.train_solo_V2"))
    run_name = str(cfg.get("name", "ablation"))
    if args.collect_only != bool(args.run_tag) or (args.variant_experiment and not args.collect_only):
        ap.error("--collect-only requires --run-tag; experiment overrides require --collect-only")
    run_tag = args.run_tag if args.collect_only else f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}"
    common_args = dict(cfg.get("common_args", {}))
    primary_metric = str(cfg.get("primary_metric", "evaluation/net_return_mean"))
    variants = list(cfg.get("variants", []))
    if len(variants) < 2:
        raise RuntimeError("Config must provide at least two variants")

    experiment_overrides: dict[str, str] = {}
    for assignment in args.variant_experiment:
        variant_id, separator, experiment_name = assignment.partition("=")
        if not separator or not variant_id or not experiment_name or variant_id in experiment_overrides:
            ap.error(f"Invalid or duplicate --variant-experiment: {assignment}")
        experiment_overrides[variant_id] = experiment_name
    variant_ids = {str(variant["id"]) for variant in variants}
    if not experiment_overrides.keys() <= variant_ids:
        ap.error("--variant-experiment references an unknown variant")

    results: list[VariantResult] = []
    for v in variants:
        variant_id = str(v["id"])
        variant_args = dict(v.get("args", {}))
        if args.collect_only:
            combined_args = {**common_args, **variant_args}
            experiment_name = experiment_overrides.get(variant_id, f"{run_tag}_{variant_id}")
            root = _resolve_results_root(combined_args)
            statuses = sorted((root / "_stable_runs" / experiment_name).glob("run_status_*.json"))
            if len(statuses) != 1:
                raise RuntimeError(f"Expected exactly one stable run status for {experiment_name}")
            status = json.loads(statuses[0].read_text(encoding="utf-8"))
            expected_count = int(variant_args.get("pop", common_args.get("pop", 10))) if (
                variant_args.get("pbt", common_args.get("pbt", False))
            ) else 1
            if status.get("status") != "completed" or status.get("trial_count") != expected_count:
                raise RuntimeError(f"Experiment is incomplete: {experiment_name}")
            collected = collect_variant_results(
                variant_id=variant_id,
                experiment_name=experiment_name,
                experiment_root=root / experiment_name,
                stable_root=root / "_stable_metrics",
                primary_metric=primary_metric,
            )
            if len(collected) != expected_count:
                raise RuntimeError(f"Trial count differs from completed run: {experiment_name}")
            results.extend(collected)
        else:
            results.extend(_run_variant(
                python_exe,
                module,
                common_args,
                variant_id,
                variant_args,
                run_tag,
                primary_metric,
            ))

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (cfg_path.parent / "outputs" / run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    action_metric_keys = sorted(
        {
            column
            for result in results
            for column in result.report_df.columns
            if column.startswith("custom_metrics/act_") and column.endswith("_mean")
        }
    )
    metric_keys = [
        primary_metric,
        "training/optimization_reward_mean",
        "training/net_return_mean",
        "evaluation/gross_return_mean",
        "evaluation/net_sharpe_per_sqrt_bar_mean",
        "entropy",
        "episode_len_mean",
        "policy_loss",
        "value_loss",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
    ] + action_metric_keys + [
        f"evaluation/custom_metrics/act_{action}_frac_mean"
        for action in ("short", "flat", "long")
    ]
    metric_dfs: dict[str, pd.DataFrame] = {}
    for m in metric_keys:
        metric_safe = m.replace("/", "__")
        metric_dfs[m] = write_metric_comparison(
            results,
            m,
            out_dir / f"ablation_compare_{metric_safe}.csv",
        )

    compare_df = metric_dfs[primary_metric]
    compare_csv = out_dir / "ablation_compare.csv"
    compare_df.to_csv(compare_csv, index=False)
    compare_png = out_dir / "ablation_compare.png"
    _plot_metric(compare_df, "Ablation: Held-out Net Return", primary_metric, compare_png)

    sharpe_png = out_dir / "ablation_compare_validation_sharpe.png"
    _plot_metric(
        metric_dfs["evaluation/net_sharpe_per_sqrt_bar_mean"],
        "Ablation: Held-out Net Sharpe per sqrt(bar)",
        "evaluation/net_sharpe_per_sqrt_bar_mean",
        sharpe_png,
    )

    entropy_png = out_dir / "ablation_compare_entropy.png"
    _plot_metric(metric_dfs["entropy"], "Ablation: Entropy", "entropy", entropy_png)

    ep_len_png = out_dir / "ablation_compare_episode_len.png"
    _plot_metric(metric_dfs["episode_len_mean"], "Ablation: Episode Length Mean", "episode_len_mean", ep_len_png)

    diagnostics_png = out_dir / "ablation_compare_diagnostics.png"
    _plot_multi_metrics(metric_dfs, diagnostics_png)
    action_png = out_dir / "ablation_compare_action_fractions.png"
    _plot_action_fractions(metric_dfs, action_png)

    summary = {
        "run_tag": run_tag,
        "collected_only": args.collect_only,
        "config": str(cfg_path),
        "results": [
            {
                "variant_id": r.variant_id,
                "series_id": r.series_id,
                "experiment_name": r.experiment_name,
                "trial_dir": str(r.trial_dir),
                "stable_report_csv": str(r.stable_report_csv),
                "stable_plot_dir": str(r.stable_plot_dir),
                "final_metric": (
                    float(r.report_df[primary_metric].iloc[-1])
                    if len(r.report_df) else float("nan")
                ),
            }
            for r in results
        ],
        "primary_metric": primary_metric,
        "compare_csv": str(compare_csv),
        "compare_png": str(compare_png),
        "sharpe_compare_png": str(sharpe_png),
        "entropy_compare_png": str(entropy_png),
        "episode_len_compare_png": str(ep_len_png),
        "diagnostics_compare_png": str(diagnostics_png),
        "action_compare_png": str(action_png),
    }
    promotion_contract = cfg.get("promotion")
    if isinstance(promotion_contract, dict):
        summary["promotion"] = evaluate_promotion(results, promotion_contract)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[AB] Completed")
    print("[AB] Compare CSV:", compare_csv)
    print("[AB] Compare PNG:", compare_png)
    print("[AB] Summary:", summary_path)


if __name__ == "__main__":
    main()
