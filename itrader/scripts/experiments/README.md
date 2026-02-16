# Experiment Harness

Use this folder for reproducible ablations with config-driven runs.

## Run an A/B experiment

```bash
python -m itrader.scripts.experiments.run_ablation \
  itrader/scripts/experiments/configs/ablation_start_idx.example.json
```

Outputs are written under:

- `itrader/scripts/experiments/configs/outputs/<run_tag>/`
  - `ablation_compare.csv`
  - `ablation_compare.png`
  - `ablation_compare_entropy.png`
  - `ablation_compare_episode_len.png`
  - `ablation_compare_diagnostics.png`
  - `summary.json`

Each variant also generates individual plots in:

- `~/ray_results/_stable_metrics/<trial_dir_name>/plots/`

## Stable training metrics path

`train_solo_V2` now writes per-iteration sidecar metrics to:

- `~/ray_results/_stable_metrics/<trial_dir_name>/metrics_full.jsonl`
- `~/ray_results/_stable_metrics/<trial_dir_name>/report_rows.csv`

This avoids relying only on Ray trial-root `progress.csv` files.

## Hyperparameter knobs via config JSON

You can pass most train knobs directly in `common_args` or per-variant `args`, e.g.:

- `gamma`, `lambda`, `lr`
- `train_batch_size`, `sgd_minibatch_size`, `num_sgd_iter`
- `clip_param`, `vf_clip_param`, `vf_loss_coeff`

See `schemas/ablation.schema.json` for the supported shape and defaults.
