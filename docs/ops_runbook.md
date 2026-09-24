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

Immutable Alpaca snapshot:

- `python -m itrader.inputs.market_data.cli build --help`

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

Required for the legacy Polygon collector:

- `POLYGON_API_KEY`

Required for Alpaca dataset snapshots:

- `APCA_API_KEY_ID`
- `APCA_API_SECRET_KEY`

Optional:

- `POLYGON_WS_URL` (defaults to Polygon stocks websocket)

Example:

```bash
export POLYGON_API_KEY="your_key_here"
```

On Windows, keep the secrets in the ignored repository `.env` file using the
names in `.env.example`. Do not commit `.env`, paste credentials into run JSON,
or add keys to snapshot manifests.

## Dataset-v2 snapshot workflow

Build to a new directory only; a completed snapshot is immutable:

```text
.venv\Scripts\python.exe -m itrader.inputs.market_data.cli build \
  --env-file .env \
  --output itrader/data/datasets/dataset_v2_YYYYMMDD_YYYYMMDD_alpaca_sip_split \
  --symbols AAPL,MSFT,AMZN,NVDA,AMD,AVGO,GOOGL,META,SPY,QQQ,IWM,TLT \
  --start 2022-01-03 \
  --end YYYY-MM-DD \
  --feed sip
```

Use the last fully complete all-session date as `--end`. An interrupted build
may resume only with the identical build request; per-symbol/year completion
markers force partial partition writes to be rebuilt. Once `manifest.json`
exists, start a new snapshot ID instead of mutating the old one.

The snapshot stores licensed raw pages and Parquet under the ignored
`itrader/data/datasets/` tree. Track only the build request, manifest, aggregate
quality/evaluation evidence, and experiment loadouts. Dataset-v2 Parquet is the
historical authority; `ITRADER_DUCKDB_PATH` remains a legacy forward-collection
setting and does not identify the canonical snapshot.

Phase A loadouts must specify:

```text
--data_sessions regular
--episode_boundary_cols session_id
--expected_bar_interval 1min
```

Also freeze `--data_symbols`, split fractions/purge, evaluation seed, task
horizons, evidence cost, dataset hash, and random/generated proposer settings.

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
