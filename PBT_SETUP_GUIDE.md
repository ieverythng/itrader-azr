# PBT Setup Guide — Ray 2.9 Phase A

Last updated: 2026-09-12

This guide is the operational contract for the pinned Ray/RLlib 2.9 stack. PBT
optimizes held-out economic evidence; shaped solver reward is a diagnostic only.

## Non-negotiable contract

- Primary objective: `evaluation/net_return_mean`, maximized.
- Training data: per-symbol chronological `train` partition only.
- PBT evaluation: `explore=False` on the purged `validation` partition.
- Final promotion: one post-selection evaluation on the untouched `test`
  partition; never tune on it.
- Costs: always state `evidence_cost_bps`; use multiple cost scenarios for
  promotion.
- Checkpoint interval must equal the perturbation interval for this trainable.
- Population members start from distinct deterministic seeds and sampled initial
  hyperparameters.
- `custom_explore_fn` validates every mutation; `gamma` must remain at most
  `0.999`.
- `train_batch_size` stays fixed while `training_iteration` is the PBT clock so
  population members receive comparable environment exposure.
- All members must appear in the ablation summary. A “best trial” without the
  rest of the population is incomplete evidence.
- Use `semantic_v2` task encoding for new runs. `legacy_v1` is historical
  checkpoint compatibility only.

Ray's current [PBT guide](https://docs.ray.io/en/latest/tune/examples/pbt_guide.html)
recommends random initial hyperparameters and aligning checkpoint and perturbation
cadence. The [scheduler API](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html)
documents mutation domains, burn-in, and `custom_explore_fn`. These are practice
references; the installed Ray 2.9.0 source and tests remain the API authority.

## Required smoke test

Run the checked-in loadout through the same orchestration used for long studies:

```powershell
.venv\Scripts\python.exe -m itrader.scripts.experiments.run_ablation `
  itrader\scripts\experiments\configs\phase_a_performance_validity_probe_20260912.json `
  --out-dir itrader\scripts\experiments\configs\outputs\phase_a_performance_validity_20260912\repaired_stack_probe
```

Acceptance:

1. fixed control completes three iterations;
2. two PBT members each expose iterations 1–12 exactly once;
3. their initial seed/config hashes differ;
4. validation seed and train batch are identical;
5. each ends with the same total exposure;
6. at least one restore/mutated config is present in `config_history.jsonl`;
7. every `gamma` is within bounds;
8. run/trial status, config history, checkpoint index, metrics, plots, and
   capability reports are non-empty;
9. `summary.json` contains the fixed trial and both PBT members.

The 2026-09-12 accepted probe is
`phase_a_performance_validity_probe_20260912_031358`. It validates seams only;
12 iterations and four validation episodes per iteration do not validate
learnability or performance.

## Long-run configuration

Start from the checked-in JSON loadout, create a new dated config, and change only
declared study factors. Recommended initial overnight shape:

- 6–10 population members if CPU/RAM allow;
- at least three independent population-level seeds;
- perturb/checkpoint interval chosen so several exploit windows occur after
  burn-in;
- enough iterations to exceed the historical 200-iteration comparison;
- a materially larger fixed validation schedule than the four-episode smoke;
- fixed `train_batch_size` across the study;
- random/generated proposer path (`task_pool_path=""`, `p_pool=0`) unless the
  proposer itself is the named factor.

Before committing an overnight run, estimate `population × workers`, memory,
checkpoint storage, and validation overhead with one full perturbation window.
Do not silently lower evaluation coverage to increase training throughput.

## Outputs and provenance

Tune trial/checkpoint directories remain under:

```text
~/ray_results/<experiment_name>/training_fn_*/
```

On the current Windows/Ray 2.9 installation, Tune-owned manifests and PBT text
logs may be zero bytes. They are not provenance authority. Use:

```text
~/ray_results/_stable_runs/<experiment_name>/
  run_manifest_*.json
  run_status_*.json

~/ray_results/_stable_metrics/<trial_name>/
  resolved_config.json
  config_history.jsonl
  trial_status.json
  checkpoint_index.jsonl
  metrics_full.jsonl
  report_rows.csv
  capability_report.json
  plots/
```

`config_history.jsonl` is append-only audit history. When PBT clones an older
checkpoint, the active CSV/JSONL trajectory rewinds to that checkpoint before
continuing, so superseded rows do not bias summaries.

Generate or regenerate a capability report with:

```powershell
.venv\Scripts\python.exe itrader\scripts\generate_capability_report.py `
  --metrics-dir <stable-trial-directory> `
  --warmup-iters <iterations-to-skip>
```

## Selection and promotion

Do not select on the last point alone. For each training seed, retain PBT lineage
and candidate checkpoints, then evaluate candidates on an identical deterministic
validation schedule. Compare against flat, long-only, short-only, random, and
untrained-policy baselines with the same starts and costs. Report gross/net
return, turnover, drawdown, exposure, positive-net-bar fraction, and Sharpe per
square-root bar. Add block-aware uncertainty across non-overlapping time units.

Only after the choice is frozen may it be evaluated on `test`. A promotion
artifact must include commit/dataset/config/checkpoint hashes, all training seeds,
baseline deltas, cost assumptions, and failure/abstention behavior.

## Recovery and failure handling

- A repeated command is a new run; it is not automatic recovery.
- Resume only from an explicit verified checkpoint and preserve the original
  stable provenance alongside the resumed run.
- If a PBT boundary logs “no checkpoint,” reject the run rather than treating it
  as a successful search.
- If any stable manifest/status/config/history/checkpoint/metrics file is absent
  or empty, the run is incomplete evidence even if Ray reports `TERMINATED`.
- Optional dashboard absence is supported: the trainer retries without it.
- Prometheus is disabled for PBT to avoid multi-trial port collisions.
- CUDA is not required for CPU exploration; hardware mode must remain a named
  factor when comparing results.

## What not to conclude

- Positive shaped reward does not imply positive return.
- A single positive four-episode validation point does not imply learnability.
- PBT exploitation makes trial endpoints correlated; population rows are not
  independent replications.
- The repaired smoke probe validates the stack, not the historical winners and
  not Phase A profitability.
