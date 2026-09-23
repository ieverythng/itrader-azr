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
- `ablation_compare_action_fractions.png`
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
- `~/ray_results/_stable_metrics/<trial_dir_name>/pbt_progress.json`
- `~/ray_results/_stable_metrics/<trial_dir_name>/checkpoint_index.jsonl`
- `~/ray_results/_stable_metrics/<trial_dir_name>/best_checkpoint.json`
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
- `pbt_synch` to checkpoint upper-quantile donors before lower-quantile exploit
- `evaluation_interval`, `evaluation_num_episodes`
- `evidence_cost_bps`, `optimization_cost_bps`, `net_return_reward_scale`,
  `evidence_periods_per_year`
- `split_fractions`, `split_purge_rows`
- `observation_transform`, `observation_filter`
- `results_root` for Ray, stable metrics, and run-manifest colocation
- `cpu_only` for resources and `ray_num_cpus` for an explicit local-cluster cap,
  or `pbt`/`cpu_pbt` to activate population search

For PBT, checkpoint and perturbation intervals must match. Training batch stays
fixed while trial-owned cumulative sampled environment steps are the search
clock; restoring donor RLlib state cannot move this clock backwards. Validation
tasks are matched across members and balanced across configured horizons. The
random/generated no-LLM seam is configured by an
empty `task_pool_path` and `p_pool: 0.0`.

Promotion loadouts on Ray 2.9 should set `pbt_synch: true`. Asynchronous PBT can
rank a lower trial before the selected upper donor has a PBT-owned checkpoint,
which skips that exploit even when repository-owned cadence checkpoints exist.
Synchronous PBT orders upper-donor saves before lower-trial exploitation while
the trial-owned environment-step clock preserves equal target exposure.
On Ray 2.9/Windows, a scheduled save can still resolve without a checkpoint
after the donor is paused. The iTrader scheduler retries from the latest
repository-owned checkpoint at or before the donor clock, reattaches a paused
upper donor to its own stable checkpoint before resuming it, and does not mutate
after `pbt_target_steps`. Durable event receipts distinguish exploits,
fallbacks, and fail-closed skips. On a 16 GB host, use `workers: 1` and
`ray_num_cpus: 4` for a four-member population unless a capacity test proves a
higher setting safe.

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
the stable copy referenced by `best_checkpoint.json` before treating its metric
as a PBT selection score. Ray 2.9 exposes approximate KL and explained variance;
clip fraction stays `NaN` until the separate RLlib migration or a custom learner
statistic makes it observable.

The original Phase A P0 entry loadouts were:

- `configs/phase_a_p0_frozen_policy_baselines_20260914.json` for the identical-
  start flat, long-only, short-only, random, and seeded untrained-FiLM controls;
- `configs/phase_a_p0_net_return_pbt_probe_20260914.json` for a bounded historical
  `pnl_linear` reference versus the explicit 1 bp `net_return` objective.

`net_return` is a TaskSpec reward, not broker semantics. It charges action
turnover plus terminal liquidation at the configured basis-point rate and uses
one constant numerical scale. Dividing episode optimization reward by that scale
recovers the same net-return arithmetic emitted independently by EpisodeEvidence.
PPO still discounts step rewards when `gamma < 1`; the exact equality applies to
the undiscounted episode sum.

P0 measurement and PBT mechanics are now closed enough to define P1. New
promotion experiments should use independent training lineages and genuinely
new chronological validation blocks, then apply selection-aware block
uncertainty, regime/action attribution, and fresh per-step 0.5/1/2 bp return and
Sharpe replay. The final chronological test remains sealed until selection is
frozen.

See `schemas/ablation.schema.json` for the supported shape and defaults.

## Collect a completed paused-arm restart without retraining

If an interrupted arm cannot be resumed at a mutually consistent model,
trial-owned step/report, and scheduler checkpoint, preserve the interrupted
experiment and restart only that arm under a distinct experiment name. Once
every required member of both selected arms has completed, generate one
comparison from those specific experiments without modifying their trials:

```bash
python -m itrader.scripts.experiments.run_ablation \
  itrader/scripts/experiments/configs/phase_a_p0_reward_bridge_sync_120_20260917.json \
  --collect-only \
  --run-tag phaseA_p0_reward_bridge_sync_120_20260917_001810 \
  --variant-experiment pnl_linear_pbt_sync_120=phaseA_p0_reward_bridge_sync_120_20260917_001810_pnl_linear_pbt_sync_120_restart
```

The collector fails closed on incomplete status or missing population members.
It aligns metric series on the monotonic trial-owned environment-step clock,
not Ray's checkpoint-restored `training_iteration`. Preserve the original
interrupted experiment as provenance and identify the restart in the report.

For a selected stable best-checkpoint pointer, run deterministic fresh-process
replay before interpreting a held-out score:

```bash
python -m itrader.rl_lib.evaluation.run_checkpoint_replay \
  --stable-trial-dir <results_root>/_stable_metrics/<trial_dir_name> \
  --out-dir <ablation_output>/replays
```

Replay verifies the dataset hash and frozen validation split, writes episode
receipts, and fails if mean net return diverges from the selected score. A
selection-panel block bootstrap or repricing of the same actions is only a
diagnostic; neither establishes selection-adjusted generalization or a
different-cost Sharpe without fresh per-step replay.
