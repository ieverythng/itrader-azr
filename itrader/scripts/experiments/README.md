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
  - `ablation_compare_validation_sharpe.png`
  - `ablation_compare_entropy.png`
  - `ablation_compare_episode_len.png`
  - `ablation_compare_diagnostics.png`
  - `summary.json`

Each variant also generates individual plots in the configured `results_root`
(or `~/ray_results` when omitted):

- `<results_root>/_stable_metrics/<trial_dir_name>/plots/`

Multi-member variants contribute one series per trial. The primary metric is
explicit in the loadout; Phase A performance studies use
`evaluation/net_return_mean`, not optimization reward.

## Stable training metrics path

`train_solo_V2` now writes per-iteration sidecar metrics to the explicit
`--results_root` when configured, otherwise to `~/ray_results`:

- `~/ray_results/_stable_metrics/<trial_dir_name>/metrics_full.jsonl`
- `~/ray_results/_stable_metrics/<trial_dir_name>/report_rows.csv`
- `~/ray_results/_stable_metrics/<trial_dir_name>/resolved_config.json`
- `~/ray_results/_stable_metrics/<trial_dir_name>/config_history.jsonl`
- `~/ray_results/_stable_metrics/<trial_dir_name>/trial_status.json`
- `~/ray_results/_stable_metrics/<trial_dir_name>/checkpoint_index.jsonl`
- `~/ray_results/_stable_metrics/<trial_dir_name>/checkpoints/`

Run-wide manifests and status live under
`~/ray_results/_stable_runs/<experiment_name>/`. These repository-owned paths
are authoritative because Ray 2.9 may truncate Tune-owned files on Windows.

This avoids relying only on Ray trial-root `progress.csv` files.

## Hyperparameter knobs via config JSON

You can pass most train knobs directly in `common_args` or per-variant `args`, e.g.:

- `gamma`, `lambda`, `lr`
- `train_batch_size`, `sgd_minibatch_size`, `num_sgd_iter`
- `clip_param`, `vf_clip_param`, `vf_loss_coeff`
- `pbt_interval`, `pbt_burn_in`, `checkpoint_interval`
- `evaluation_interval`, `evaluation_num_episodes`
- `evidence_cost_bps`, `evidence_periods_per_year`
- `split_fractions`, `split_purge_rows`
- `observation_transform`, `observation_filter`
- `results_root` for Ray, stable metrics, and run-manifest colocation
- `cpu_only` for resources, or `pbt`/`cpu_pbt` to activate population search

For PBT, checkpoint and perturbation intervals must match. Training batch stays
fixed while `training_iteration` is the search clock; validation tasks are
matched across members. The random/generated no-LLM seam is configured by an
empty `task_pool_path` and `p_pool: 0.0`.

On Ray 2.9, Phase A PBT additionally fails closed unless both training and
evaluation use `observation_transform: market_relative_v1` with
`observation_filter: NoFilter`. This makes the market normalization deterministic
and checkpoint-free. After training or restore, the trainer forces the evaluation
policy back to `explore=False`, synchronizes learner weights, and limits metric
smoothing to the frozen-panel episode count; Ray 2.9 otherwise restores the
training policy's stochastic exploration setting into evaluation. At every
checkpoint boundary, the trainer copies a complete RLlib checkpoint into the
stable path before reporting a separate checkpoint to Tune; AIR may consume the
reported source directory on Windows. Promotion runs must fresh-process replay
the stable copy before treating its metric as a PBT selection score.

See `schemas/ablation.schema.json` for the supported shape and defaults.
