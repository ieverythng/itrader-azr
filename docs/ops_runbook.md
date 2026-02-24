# Operations Runbook

## Purpose

This runbook defines the stable workflow for local/main-PC execution and reproducibility.

## Canonical Entry Points

Training:

- `python -m itrader.rl_lib.train.train_solo_V2 ...`

Health check:

- `python itrader/tools/healthcheck.py`

Tests:

- `python -m pytest -q itrader/tests`

Ablation:

- `python -m itrader.scripts.experiments.run_ablation <config.json>`

## Environment Setup (Main PC)

1. Create conda env:
- `conda env create -f itrader/environment.yml`
- `conda activate itrader`

2. Validate runtime:
- `python itrader/tools/healthcheck.py`
- `python -m pytest -q itrader/tests`

3. Freeze lock after successful validation:
- `conda env export --from-history > env.lock.yml`
- Optional full pin:
  - `pip freeze > itrader/constraints.lock.txt`

## Security/Secrets

Never hardcode API keys in source files.

Required:

- `POLYGON_API_KEY`

Optional:

- `POLYGON_WS_URL` (defaults to Polygon stocks websocket)

Example:

```bash
export POLYGON_API_KEY="your_key_here"
```

## Metrics and Artifacts

Ray results:

- `~/ray_results/<experiment_name>/...`

Stable mirror:

- `~/ray_results/_stable_metrics/<trial_dir>/metrics_full.jsonl`
- `~/ray_results/_stable_metrics/<trial_dir>/report_rows.csv`
- `~/ray_results/_stable_metrics/<trial_dir>/plots/`

## Common Recovery Actions

1. Missing dependencies:
- rerun env creation, then rerun healthcheck/tests.

2. Broken checkpoint restore:
- verify restore path exists and points to checkpoint directory.

3. Plot generation errors:
- run manually with metrics jsonl:
  - `python itrader/tools/plot_ray_results.py --metrics-jsonl <path> --out <dir>`

## Pre-Run Checklist

1. `git status` clean or intentionally dirty.
2. `healthcheck` passes required packages.
3. tests pass or expected skips are understood.
4. dataset path exists and matches intended snapshot.
5. experiment name and seed are explicitly set.
