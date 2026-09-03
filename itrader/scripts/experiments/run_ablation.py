from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class VariantResult:
    variant_id: str
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


def _find_trial_dir(experiment_root: Path) -> Path:
    trials = [d for d in experiment_root.iterdir() if d.is_dir() and d.name.startswith("training_fn_")]
    if not trials:
        raise RuntimeError(f"No trial directories found under {experiment_root}")
    return max(trials, key=lambda d: d.stat().st_mtime)


def _run_variant(
    python_exe: str,
    module: str,
    common_args: dict[str, Any],
    variant_id: str,
    variant_args: dict[str, Any],
    run_tag: str,
) -> VariantResult:
    experiment_name = f"{run_tag}_{variant_id}"
    args = dict(common_args)
    args.update(variant_args)
    args["experiment_name"] = experiment_name

    cmd = [python_exe, "-m", module] + _to_cli_args(args)
    print(f"\n[AB] Running variant '{variant_id}'")
    print("[AB] Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    experiment_root = Path.home() / "ray_results" / experiment_name
    trial_dir = _find_trial_dir(experiment_root)
    stable_report_csv = Path.home() / "ray_results" / "_stable_metrics" / trial_dir.name / "report_rows.csv"
    if not stable_report_csv.exists():
        raise RuntimeError(f"Stable report CSV not found: {stable_report_csv}")
    stable_metrics_jsonl = Path.home() / "ray_results" / "_stable_metrics" / trial_dir.name / "metrics_full.jsonl"
    stable_plot_dir = Path.home() / "ray_results" / "_stable_metrics" / trial_dir.name / "plots"
    stable_plot_dir.mkdir(parents=True, exist_ok=True)

    report_df = pd.read_csv(stable_report_csv)
    if "custom_metrics/episode_return_mean" not in report_df.columns:
        raise RuntimeError(f"Missing custom_metrics/episode_return_mean in {stable_report_csv}")

    # Guarantee per-variant plot outputs in stable metrics folder.
    plot_cmd = [
        python_exe,
        str(Path(__file__).resolve().parents[2] / "tools" / "plot_ray_results.py"),
        "--metrics-jsonl",
        str(stable_metrics_jsonl),
        "--out",
        str(stable_plot_dir),
    ]
    subprocess.run(plot_cmd, check=True)

    return VariantResult(
        variant_id=variant_id,
        experiment_name=experiment_name,
        trial_dir=trial_dir,
        stable_report_csv=stable_report_csv,
        stable_plot_dir=stable_plot_dir,
        report_df=report_df,
    )


def _metric_series(df: pd.DataFrame, metric: str) -> list[float]:
    if metric not in df.columns:
        return []
    return [float(x) for x in pd.to_numeric(df[metric], errors="coerce").tolist()]


def _write_metric_compare(results: list[VariantResult], metric: str, out_csv: Path) -> pd.DataFrame:
    max_len = max((len(_metric_series(r.report_df, metric)) for r in results), default=0)
    rows: list[dict[str, Any]] = []
    for i in range(max_len):
        row: dict[str, Any] = {"training_iteration": i + 1}
        for r in results:
            vals = _metric_series(r.report_df, metric)
            row[r.variant_id] = vals[i] if i < len(vals) else float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def _plot_metric(df: pd.DataFrame, title: str, ylabel: str, out_png: Path):
    plt.figure(figsize=(9, 4.5))
    if "training_iteration" not in df.columns:
        return
    x = df["training_iteration"].to_numpy()
    for c in [col for col in df.columns if col != "training_iteration"]:
        y = pd.to_numeric(df[c], errors="coerce")
        if y.notna().any():
            plt.plot(x, y.to_numpy(), marker="o", label=c)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("training_iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def _plot_multi_metrics(metric_dfs: dict[str, pd.DataFrame], out_png: Path):
    action_metric = next(
        (metric for metric in sorted(metric_dfs) if metric.startswith("custom_metrics/act_") and metric.endswith("_mean")),
        None,
    )
    metric_titles = [
        ("custom_metrics/episode_return_mean", "Episode Return Mean"),
        ("entropy", "Entropy"),
        ("episode_len_mean", "Episode Length Mean"),
        (action_metric, "Action Fraction") if action_metric else (None, "Action Fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (metric, title) in zip(axes, metric_titles):
        df = metric_dfs.get(metric)
        if metric is None or df is None or "training_iteration" not in df.columns:
            ax.set_title(f"{title} (no data)")
            ax.axis("off")
            continue
        x = df["training_iteration"].to_numpy()
        plotted = False
        for c in [col for col in df.columns if col != "training_iteration"]:
            y = pd.to_numeric(df[c], errors="coerce")
            if y.notna().any():
                ax.plot(x, y.to_numpy(), marker="o", label=c)
                plotted = True
        if plotted:
            ax.set_title(title)
            ax.legend(fontsize=8)
        else:
            ax.set_title(f"{title} (no data)")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run reproducible start-index ablations.")
    ap.add_argument("config_json", help="Path to ablation config JSON")
    ap.add_argument("--out-dir", help="Optional output directory for comparison artifacts")
    args = ap.parse_args()

    cfg_path = Path(args.config_json).expanduser().resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    python_exe = str(cfg.get("python_exe", "python"))
    module = str(cfg.get("train_module", "itrader.rl_lib.train.train_solo_V2"))
    run_name = str(cfg.get("name", "ablation"))
    run_tag = f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}"
    common_args = dict(cfg.get("common_args", {}))
    variants = list(cfg.get("variants", []))
    if len(variants) < 2:
        raise RuntimeError("Config must provide at least two variants")

    results: list[VariantResult] = []
    for v in variants:
        variant_id = str(v["id"])
        variant_args = dict(v.get("args", {}))
        results.append(_run_variant(python_exe, module, common_args, variant_id, variant_args, run_tag))

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
        "custom_metrics/episode_return_mean",
        "entropy",
        "episode_len_mean",
        "policy_loss",
        "value_loss",
    ] + action_metric_keys
    metric_dfs: dict[str, pd.DataFrame] = {}
    for m in metric_keys:
        metric_safe = m.replace("/", "__")
        metric_dfs[m] = _write_metric_compare(results, m, out_dir / f"ablation_compare_{metric_safe}.csv")

    compare_df = metric_dfs["custom_metrics/episode_return_mean"]
    compare_csv = out_dir / "ablation_compare.csv"
    compare_df.to_csv(compare_csv, index=False)
    compare_png = out_dir / "ablation_compare.png"
    _plot_metric(compare_df, "Ablation: Episode Return Mean", "custom_metrics/episode_return_mean", compare_png)

    entropy_png = out_dir / "ablation_compare_entropy.png"
    _plot_metric(metric_dfs["entropy"], "Ablation: Entropy", "entropy", entropy_png)

    ep_len_png = out_dir / "ablation_compare_episode_len.png"
    _plot_metric(metric_dfs["episode_len_mean"], "Ablation: Episode Length Mean", "episode_len_mean", ep_len_png)

    diagnostics_png = out_dir / "ablation_compare_diagnostics.png"
    _plot_multi_metrics(metric_dfs, diagnostics_png)

    summary = {
        "run_tag": run_tag,
        "config": str(cfg_path),
        "results": [
            {
                "variant_id": r.variant_id,
                "experiment_name": r.experiment_name,
                "trial_dir": str(r.trial_dir),
                "stable_report_csv": str(r.stable_report_csv),
                "stable_plot_dir": str(r.stable_plot_dir),
                "final_metric": (
                    float(r.report_df["custom_metrics/episode_return_mean"].iloc[-1])
                    if len(r.report_df) else float("nan")
                ),
            }
            for r in results
        ],
        "compare_csv": str(compare_csv),
        "compare_png": str(compare_png),
        "entropy_compare_png": str(entropy_png),
        "episode_len_compare_png": str(ep_len_png),
        "diagnostics_compare_png": str(diagnostics_png),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[AB] Completed")
    print("[AB] Compare CSV:", compare_csv)
    print("[AB] Compare PNG:", compare_png)
    print("[AB] Summary:", summary_path)


if __name__ == "__main__":
    main()
