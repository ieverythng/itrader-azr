# Phase A Performance Validity Audit

**Date:** 2026-09-12; updated 2026-09-16
**Repository state:** `agent/Watson-dev` at `bec5c775d3adaafe758202c7175791f6e8b323a8`
**Decision:** the Phase A results support optimizer activity, reward sensitivity,
and pipeline learnability, but they do **not** support a claim of validated
trading performance or baseline-relative economic uplift.

## Executive conclusion

The two historical PBT checkpoints are real and restorable. They contain trained `SolverMLPFiLM` policies and observation-filter state after roughly 0.9–1.0 million environment steps. The historical curves also show finite reward signals, non-collapsed training, and non-trivial action distributions. That is useful evidence that PPO can interact with the Phase A environment and optimize the shaped objective.

It is not evidence of economic performance. The metric named `custom_metrics/episode_return_mean` is the episode's accumulated **optimization reward**, while the metric named `episode_return` is the normalized reward component called `pnl_delta`. Neither is raw portfolio return. PBT selected on that shaped metric using the same data used for training, without a held-out evaluation environment. One selected checkpoint has an invalid PBT-mutated discount factor of `gamma=1.188`; the other has `gamma=0.6272`, whose discount half-life is about 1.49 steps despite 96–200 step tasks.

A deterministic, symbol-safe diagnostic replay of 288 episodes per checkpoint produced the following independent arithmetic-return measurements:

| Historical target | Gross mean | Net mean at 1 bp per turnover unit | Positive net episodes | Mean turnover | Flat actions | Mean excess vs long-only |
|---|---:|---:|---:|---:|---:|---:|
| final winner `00002` | +0.0163% | **−0.0728%** | 22.6% | 8.91 | 93.6% | −0.0334% |
| peak winner `00004` | **−0.0112%** | **−0.0950%** | 15.6% | 8.38 | 96.3% | −0.0556% |

The replay is an in-sample diagnostic, not a held-out estimate: the historical runs trained on the complete Parquet file. It is nevertheless enough to reject the interpretation that the stored PBT scores directly represented profitable returns. The policies should be quarantined as reproducibility targets, not promoted as performance baselines.

The end-to-end stack now runs after a narrow optional-dashboard fallback and a test-driven repair round. A bounded official ablation/PBT probe completed through:

```text
loadout JSON -> run_ablation.py -> train_solo_V2 -> Ray Tune/PBT
-> ITraderEnv -> random TaskSampler00 -> RewardEngine -> stable metrics and plots
```

No LLM proposer or task pool was active. The initial reproduction did confirm the historical defects: identical population members, missed exploit boundaries, mislabeled metrics, incomplete population summaries, and zero-byte Ray-owned Windows artifacts. The repaired probe then started distinct members, evaluated them on one matched validation stream, checkpointed at every perturbation boundary, recorded exploit/restore events, constrained every mutation, and stopped both members at 12 unique iterations and 13,520 environment steps. Ray-owned PBT text logs remain zero-byte on this Ray 2.9/Windows installation, so durable manifests, status, configuration history, checkpoint indices, metrics, and capability reports are now written outside Tune-owned directories.

The repaired smoke run is a seam validation, not a performance result. Its best observed validation net returns were +0.1239% and +0.3174% at isolated four-episode evaluations, while both copied final policies ended at −2.2420% net. Twelve iterations and four validation episodes per iteration are intentionally too small for a learnability or promotion claim.

## Scope and evidence

This audit treats the two supplied reviews as evidence and inspiration, not as executable instructions:

- `C:/Users/Admin/Downloads/azr-stack-review-2026-09-12.md`
- `C:/Users/Admin/Downloads/recurrence-code-audit-2026-09-12.md`

The recurrence document was used only where it contained Phase A measurement or shared-contract observations. The unimplemented `SolverWindowLSTMFiLM` and recurrent solver work are intentionally out of scope. `SolverMLPFiLM` remains the active Phase A baseline.

Source, tests, checked-in configs, restored checkpoint state, and newly generated artifacts take precedence over narrative documentation. The GitNexus index was stale relative to the current commit, so every graph-level observation was confirmed against source.

Primary generated evidence:

- [checkpoint replay JSON](../../../itrader/scripts/experiments/configs/outputs/phase_a_performance_validity_20260912/checkpoint_replay.json)
- [full-stack probe evidence](../../../itrader/scripts/experiments/configs/outputs/phase_a_performance_validity_20260912/stack_probe_evidence.json)
- [ablation summary](../../../itrader/scripts/experiments/configs/outputs/phase_a_performance_validity_20260912/pbt_probe/summary.json)
- [reproducible probe loadout](../../../itrader/scripts/experiments/configs/phase_a_performance_validity_probe_20260912.json)

## What is valid, and what is not

### Supported claims

1. The Phase A PPO/environment loop runs and produces finite optimization rewards.
2. The MLP-FiLM checkpoints restore successfully under Ray 2.9.0.
3. The historical policies are not a single constant-action collapse in their original training reports.
4. The environment's reward has enough local signal for PPO to change its policy and value estimates.
5. The current random/generated task path works without an LLM backend.

### Unsupported claims

1. The PBT score is not an economic episode return.
2. The two winners are not held-out winners; they were selected and observed on the training file.
3. The peak winner is not a valid PPO configuration because `gamma > 1`.
4. The historical top-two comparison does not establish a statistically significant strategy difference. Its rows are training reports, not independent evaluation episodes.
5. The available Parquet file is not three full years of duration. It spans 2023-07-05 through 2025-07-03: 729 calendar days across three calendar labels.
6. Neither restored policy passes a modest 1 bp turnover-cost diagnostic.

The precise answer to “was the reward learnable?” is therefore: **learnable as an optimizer objective, yes; validated as a proxy for economic performance, no.**

## Historical winner reconstruction

The checked-in comparison config points to these targets:

- final winner: `...83c77_00002.../checkpoint_000019`
- peak winner: `...83c77_00004.../checkpoint_000019`

Both checkpoints report training iteration 200 and use `SolverMLPFiLM`, a 60-row OHLCV window, random family-00 tasks, `MeanStdFilter`, and horizons 96/128/160/200.

| Property | final winner `00002` | peak winner `00004` |
|---|---:|---:|
| Environment steps | 985,088 | 912,944 |
| Historical final shaped score | 2.2731 | 1.8674 |
| Historical best shaped score | 3.3793 | 3.8235 |
| `gamma` in checkpoint | 0.6272 | **1.1880** |
| `lambda` | 0.9 | 0.9 |
| Policy-state SHA-256 | `ca658b81…fc87332` | `9c50e974…0ac31` |
| Promotion status | quarantine and reevaluate | diagnostic only; invalid config |

Ray PBT's arithmetic perturbation of numeric categorical choices can turn `0.99` into `1.188` (`0.99 × 1.2`) when no constrained exploration function validates the mutated config. It can likewise drive `0.98` down through repeated multiplication. For `gamma=0.6272`, weights beyond a few steps are negligible: `gamma^96 ≈ 3.56e−20` and `gamma^200 ≈ 3.03e−41`. For `gamma=1.188`, they explode: `gamma^96 ≈ 1.52e7` and `gamma^200 ≈ 9.19e14`.

## Checkpoint replay protocol and result

The diagnostic evaluator restored each complete `Algorithm` checkpoint, including its observation filter. It then used `explore=False` on fixed family-00 tasks while preserving the historical task-embedding fields.

The 719,855-row dataset contains AAPL, AMZN, and MSFT. For every symbol it divided ordered rows into early/middle/late temporal thirds, then selected eight deterministic starts for each of four horizons. All 288 windows per policy were asserted to remain inside one symbol. Economic measurements were calculated independently of `RewardEngine`:

- gross arithmetic strategy return;
- turnover from every target-position change;
- terminal liquidation turnover;
- net return at 1 bp per turnover unit;
- arithmetic max drawdown;
- matched long-only result.

The final winner's small positive gross mean disappears under costs. It is negative on all three symbols net of costs, and its late-segment mean is −0.1854%. The peak winner is negative gross and net. Both policies are overwhelmingly flat in deterministic inference. Because horizons reuse related time regions, these 288 rows are diagnostic observations rather than independent statistical samples; no IID confidence interval is claimed.

## Full-stack probe

The full-stack loadout uses the repository's ablation schema and runner. It explicitly sets:

- `task_pool_path: ""` and `p_pool: 0.0`;
- `task_reward_ids: "pnl_linear"`;
- family-00 random generation through `TaskSampler00`;
- no proposer backend and no LLM endpoint;
- the historical window, horizons, stream mode, and reward-normalization mode;
- CPU execution with two rollout workers.

The first attempt exposed an unconditional Ray dashboard dependency. A test-first fallback now retries without the optional dashboard only when Ray reports that the dashboard package is absent. The official stack then completed a 3-iteration fixed control and a 2-member, 12-iteration PBT probe.

The probe's shaped score ranged from +0.9846 at iteration 1 to −2.2267 at iteration 4 and finished at +0.1370. That volatility confirms that a positive final point is not a learning-curve validation. More importantly, both PBT members had the same normalized metric hash, `8c444cff…8c463`, across all 12 reports. No exploitation or mutation was observed.

This is caused by three interacting seams:

1. `num_samples` clones the same fixed base config and the same seed; PBT does not sample diverse initial hyperparameters.
2. PBT's perturbation interval is 3, but `checkpoint_interval` is 10. Ray logged “no checkpoint” and skipped exploit events.
3. `run_ablation.py` selects only the most recently modified trial directory for a variant, so a multi-trial population is summarized as one trial.

Before repair, `cpu_pbt` was only a resource-mode flag: scheduler/population activation depended on `pbt`, so a loadout with `cpu_pbt: true` but no `pbt: true` silently ran one FIFO trial. It now activates CPU PBT; the separate `cpu_only` flag selects resources without activating a scheduler.

## Implementation round and repaired-stack verdict

The implementation followed the repository architecture guardrails and one red/green/refactor slice at a time. `ITraderEnv` remains the deep market/task/accounting module; `RewardEngine` still owns optimization reward; the solver does not acquire broker, risk, or execution semantics.

| Seam | Implemented contract | Repaired-stack evidence |
|---|---|---|
| Episode accounting | terminal `episode_evidence` schema v1 separates optimization reward from gross/cost/net arithmetic returns; includes liquidation turnover, drawdown, exposure, position changes, positive-bar fraction, and Sharpe per square-root bar | known-trace arithmetic tests and live callback metrics pass |
| Data partitioning | symbol-specific contiguous `train`/`validation`/`test` intervals with lookback/outcome purge | 24 seeded multi-symbol episodes stayed inside one symbol and split |
| Reset/task semantics | Gym reset seed reseeds task/start streams; sampler honors declared ranges; overrides normalize before encoding | deterministic replay and bounds tests pass |
| Task observability | checkpoint-compatible `legacy_v1`; new runs use `semantic_v2`, which hashes solver-relevant task semantics and excludes the episode seed/start shortcut | alias/seed invariance tests pass |
| PBT objective | Ray ranks held-out `evaluation/net_return_mean` at an explicit 1 bp cost, never shaped reward | live Ray table and ablation summary expose gross, net, and Sharpe separately |
| PBT search | distinct deterministic trial seeds and sampled initial domains; categorical mutations; `custom_explore_fn` clamps continuous values and keeps `gamma <= 0.999` | initial config hashes differ; live exploit/explore and bounded `gamma` observed |
| Fair exposure | `train_batch_size=1024` stays fixed; validation seed `30260931` is shared; restored workers train only remaining iterations | both members ended at 12 unique iterations and 13,520 timesteps |
| Lineage | active metric/checkpoint rows rewind to the restored iteration; append-only configuration history retains every start/restore | both reports have exactly iterations 1–12; mutated restore configs are non-empty |
| Provenance | atomic run manifests/status outside Tune, plus per-trial resolved config, status, history, checkpoint index, metrics, plots, and capability report | all repository-owned acceptance artifacts are non-empty despite Ray-owned zero-byte files |
| Reporting | ablation enumerates every member; capability reporter accepts the trainer's flat JSONL and skips warmup correctly | one fixed plus two PBT series summarized; all three capability reports generated |

Sharpe is reported as a dimensionless per-square-root-bar diagnostic because no annualization convention is inferred from irregular market rows. Annualized Sharpe is emitted only when `evidence_periods_per_year` is explicitly configured. This prevents a plausible-looking but undocumented scaling factor from entering promotion evidence.

The repaired run is recorded in [acceptance.json](../../../itrader/scripts/experiments/configs/outputs/phase_a_performance_validity_20260912/repaired_stack_probe/acceptance.json), with the complete all-member [summary.json](../../../itrader/scripts/experiments/configs/outputs/phase_a_performance_validity_20260912/repaired_stack_probe/summary.json) and comparison outputs beside it.

### Ray 2.9-compatible PBT practice adopted

The current Ray guidance still recommends random initial hyperparameters, frequent checkpointing at perturbation boundaries, explicit mutation spaces, and replayable PBT policy schedules. Ray 2.9 supports `burn_in_period`, `custom_explore_fn`, `log_config`, and categorical list mutations, but it rejects duplicate metric ownership between the scheduler and `tune.run`. The repaired seam therefore gives the scheduler sole ownership of the PBT objective, aligns checkpoint and perturbation intervals, uses a burn-in, validates post-mutation configs, and records an independent history because Ray's Windows policy log is not durable here.

Primary references: Ray's [PBT guide](https://docs.ray.io/en/latest/tune/examples/pbt_guide.html), [PopulationBasedTraining API](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html), and [PB2 PPO example](https://docs.ray.io/en/latest/tune/examples/includes/pb2_ppo_example.html). Version-specific behavior was also confirmed against the installed Ray 2.9.0 signatures and source; later documentation was used as practice guidance, not as proof that newer APIs exist in 2.9.

Batch size is deliberately not mutated in this Phase A study. With `training_iteration` as the Ray 2.9 PBT clock, mutating it would compare members after unequal environment exposure. A later upgrade may instead use a timestep clock and a trainable/checkpoint contract designed around that unit.

## Confirmed findings

| Severity | Finding | Direct consequence |
|---|---|---|
| Critical | Optimization reward is labeled and selected as episode return | PBT ranks reward shaping, not raw or net strategy performance |
| Critical | PBT permits invalid `gamma` mutations | one historical “winner” has `gamma=1.188` |
| Critical | No held-out evaluation path is configured | winner status is in-sample and cannot support generalization |
| Critical | PBT population starts identical and checkpoint cadence misses perturbations | bounded PBT can perform no search while appearing successful |
| High | Ray root manifests, `params.json`, `progress.csv`, and `result.json` are zero bytes | historical and current run provenance is incomplete |
| High | Capability report expects nested records with `iteration`, while trainer emits flat records with `training_iteration` | the report raises “No iteration records found” on real trainer output |
| High | `ITraderEnv` samples global row indices from a symbol-concatenated dataframe | rare episodes can cross AAPL→AMZN or AMZN→MSFT boundaries |
| High | Reward parameter ranges declared by the trainer are ignored by `TaskSampler00` | 51.7% of a 1,000-task probe exceeded the declared `dd_w <= 0.04` range |
| High | Task embedding omits reward parameters, regime, difficulty, bar resolution, and start | materially different tasks can be observationally aliased |
| High | Task seed affects the sampled start and is also encoded | the model can learn a seed/location shortcut rather than task semantics |
| Medium | `reset(seed=...)` does not reseed the task/start samplers | Gymnasium's seed contract is not reproducible across resets |
| Medium | `task_sampler_overrides` are merged after embedding and without normalization | invalid values bypass `RewardEngine` bounds and are hidden from the solver |
| Medium | No terminal liquidation cost is charged | episode economics favor open terminal positions |
| Medium | invalid mode strings silently fall back and NaN/Inf values are silently sanitized | evidence corruption can be hidden during audit runs |
| Medium | capability-report “warmup” keeps the final N records rather than skipping the first N | report semantics disagree with CLI help |
| Medium | callback reads the first sub-environment's private `_last_info` | vectorized environments would misattribute terminal evidence |

### Metric conflation

`PSMetrics` assigns accumulated reward to `episode_return_mean` and assigns the accumulated normalized `pnl_delta` reward component to `episode_return`. `RewardEngine` multiplies raw arithmetic PnL deltas by a dynamic `ret_unit`. A deterministic price probe produced raw arithmetic return `0.00377537` but the reported `pnl_delta` component was `3.78058`, roughly 1,000 times larger. The [ablation runner](../../../itrader/scripts/experiments/run_ablation.py) then labels and plots this field as “Episode Return Mean”, and [train_solo_V2.py](../../../itrader/rl_lib/train/train_solo_V2.py) makes it the PBT selection objective.

The shaped reward is allowed to be scaled; the defect is presenting it as a return and using it as the only promotion score.

### Data and split safety

The Parquet file is grouped contiguously by symbol, while [itrader_env.py](../../../itrader/envs/itrader_env.py) indexes the reset window and future returns without a symbol boundary. The theoretical fraction of invalid global starts is low—about 0.043% to 0.072% depending on horizon—but non-zero. In a 2,000-reset simulation per worker, a worker produced an AAPL→AMZN episode. Low frequency does not make the accounting correct.

A chronological split must be constructed by timestamp **within each symbol**, then purged so that neither the 60-row observation window nor the 96–200 row outcome horizon overlaps a neighboring split.

### Task semantics

[encoder.py](../../../itrader/rl_lib/tasks/encoder.py) encodes family, `price_id`, reward ID, horizon, dimensions, and `seed % 997`. [samplers.py](../../../itrader/rl_lib/tasks/samplers.py) separately samples reward weights and start positions. Since reward parameters are not encoded, the same observation/task embedding can ask the solver to optimize different objectives. Since the seed is encoded and also controls start selection, the task representation leaks a location proxy.

For Phase A, either freeze reward parameters to one declared objective or introduce a versioned task encoding that represents every solver-relevant degree of freedom. Do not overload `TaskSpec` with broker, risk, or execution semantics.

## Target design

The architecture should make optimization evidence and economic evidence distinct by construction.

```text
TaskSpec + market split -> ITraderEnv -> EpisodeEvidence
                                 |             |
                                 v             +-> EvidenceWriter -> evaluator/capability report
                           RewardEngine
                                 |
                                 +-> optimization_reward + named components -> PPO/PBT
```

### `ITraderEnv` remains the deep accounting module

`ITraderEnv` should continue to own market-window assembly, task-conditioned reset, action-to-position transitions, and episode accounting. At terminal step it should emit a public, versioned `EpisodeEvidence` record containing:

- raw gross arithmetic return;
- explicit cost schedule and net return;
- turnover including terminal liquidation;
- max drawdown and per-step count;
- action counts;
- symbol, start/end indices and timestamps;
- split ID and dataset fingerprint;
- task identity and canonical reward configuration;
- optimization reward and finite named components.

This is a deeper seam than having callbacks scrape private environment state. Two consumers justify it immediately: the RLlib callback adapter and a standalone checkpoint evaluator.

### `RewardEngine` remains the optimization module

`RewardEngine` should continue to own finite solver reward calculation. Its output should be named `optimization_reward`, never `episode_return`. Reward normalization and clipping must be recorded, but economic return must be computed independently from raw position and price transitions.

### Evaluation stays outside `TaskSpec`

Introduce an `EvaluationPlan` or equivalent experiment contract for symbol-specific train/validation/test ranges, purge widths, cost scenarios, episode starts, baselines, seeds, and block-bootstrap units. These are experiment and evidence semantics, not broker or online risk semantics. `MarketFramer`, `DecisionGate`, `RiskEngine`, and `Executor` remain Phase C seams.

### One evidence schema

The trainer, stable-metrics writer, checkpoint evaluator, and capability reporter should consume a single versioned record contract. Persist atomically to a run-owned evidence directory that Tune cannot truncate. A completed run is invalid if its manifest, resolved config, dataset hash, checkpoint index, or evaluation record is absent or empty.

## TDD implementation sequence

Implement one vertical behavior slice at a time. Do not begin by adding every proposed test.

### Slice 1 — truthful terminal evidence

**Public seam:** `ITraderEnv.reset/step`, `RewardEngine.compute`, and terminal `info["episode_evidence"]`.

1. Red: a known price/action trace asserts raw gross return, optimization reward, turnover, liquidation cost, net return, and max drawdown independently.
2. Green: add the smallest immutable/versioned evidence payload.
3. Refactor: make `PSMetrics` consume the public payload; rename the shaped metric.

Acceptance: changing reward normalization changes only optimization fields, not economic return.

### Slice 2 — symbol-safe evaluation

**Public seam:** reset sampling and `EvaluationPlan`.

1. Red: no observation or outcome row may cross a symbol or split boundary.
2. Green: sample from precomputed per-symbol valid intervals.
3. Refactor: share boundary validation between training and checkpoint evaluation.

Acceptance: every emitted episode carries one symbol and a declared split; purge overlap is impossible.

### Slice 3 — bounded PBT domains

**Public seam:** scheduler/loadout builder.

1. Red: every initial and mutated config satisfies `0 < gamma <= 1`, `0 <= lambda <= 1`, batch divisibility, and declared ranges.
2. Green: seed population members independently, sample diverse initial values, and use a validating `custom_explore_fn`.
3. Refactor: align checkpoint cadence to perturbation cadence and expose both in the JSON schema.

Acceptance: a 2-member smoke run must show different resolved configs, at least one recorded mutation/exploit when intended, and no out-of-domain value.

### Slice 4 — durable run evidence

**Public seam:** resolved-run manifest and evidence writer.

1. Red: the current Windows/Ray integration test fails if any required artifact is zero bytes.
2. Green: write manifests and resolved configs atomically outside Tune-owned paths, then copy/link them into summaries.
3. Refactor: make `run_ablation.py` enumerate every population member and preserve per-trial results.

Acceptance: interruption, success, and failure each leave a non-empty status record; multi-trial summaries include all members.

### Slice 5 — task semantics and reproducibility

**Public seam:** `TaskSampler00`, `TaskEncoder`, and `ITraderEnv.reset(seed=...)`.

1. Red: declared reward bounds govern all sampled and overridden tasks; the same reset seed reproduces task and start; task aliases cannot differ in hidden solver-relevant parameters.
2. Green: normalize overrides before encoding, respect configured sampling domains, and split semantic task identity from episode RNG.
3. Refactor: version task encoding and remove the seed shortcut.

Acceptance: two tasks with different reward objectives either have different embeddings or are rejected by a fixed-objective Phase A profile.

## Promotion gate for a Phase A winner

A checkpoint may be called a Phase A performance winner only when all gates pass:

1. **Provenance:** non-empty manifest, commit, dirty-tree marker, resolved config, dataset SHA-256, Ray/Python versions, and checkpoint hash.
2. **Valid search:** independently seeded population; diverse initial configs; every mutation validated; exploit events and lineage recorded.
3. **Strict data separation:** per-symbol chronological train/validation/test splits with window/horizon purge; the test split unavailable to PBT and proposer updates.
4. **Frozen inference:** restored training-only observation filter, `explore=False`, deterministic episode schedule.
5. **Truthful economics:** gross return, net return under multiple cost assumptions, turnover, drawdown, exposure, and terminal liquidation.
6. **Baselines:** flat, long-only, short-only, random policy, and the untrained policy under the identical schedule.
7. **Replication:** multiple independent training seeds and block-aware uncertainty over non-overlapping time units.
8. **Stratification:** symbol, time regime, horizon, task family, and reward configuration.
9. **Separation of concerns:** optimization reward reported next to—but never named as—economic return.
10. **Capability report:** generated from the same versioned episode-evidence schema and rejected on missing fields.

The historical checkpoints remain regression fixtures after the gate was
implemented; they are not performance candidates:

- `00002` is the primary compatibility target because its `gamma` is finite and below 1, although it is too short-horizon for the stated tasks.
- `00004` is a negative test proving that PBT domain validation and promotion checks are necessary.

## Implementation status and next priorities

| Status | Deliverable | Remaining boundary |
|---|---|---|
| Implemented | `EpisodeEvidence` plus truthful metric names | add checkpoint/dataset fingerprint at evaluation-run aggregation level |
| Implemented | symbol-safe purged train/validation/test partition | keep final test inaccessible to PBT; frozen baseline schedule has run |
| Implemented | constrained/diverse PBT with matched evaluation and aligned checkpoints | parity and synchronous bridges were negative; best-checkpoint replay and controls ran, but economic promotion still fails |
| Implemented | atomic stable manifests, status, config history, checkpoint index, and all-member summaries | Ray-owned zero-byte files remain a known platform defect |
| Implemented | flat-schema capability report and warmup semantics | report baseline deltas and selection-aware uncertainty in promotion output |
| Implemented | versioned semantic task encoding, reset seed, normalized overrides, and sampler bounds | retain `legacy_v1` solely for historical checkpoint compatibility |
| Implemented | episode-correlated callback evidence | test at higher vectorization before changing workers-per-environment topology |
| Completed P0 | deterministic flat/long/short/random/untrained baselines on identical validation starts | rerun on independent/new blocks for any promoted candidate |
| Completed, not promoted | rerun four-member 200-iteration parity PBT | all terminal net returns and Sharpes were negative; exact fresh-process replay passed; untouched final-test evaluation is still required |
| Exploratory P1 diagnostic | paired 14-block selection-panel check and 0.5/1/2 bp same-action repricing | independent seeds, new blocks, selection-aware uncertainty, and per-step different-cost Sharpe remain |

No recurrence/LSTM port is required for any of these corrections. They strengthen the current MLP Phase A contract and create cleaner evidence for later solver comparisons.

## Verification record

- Restored both historical checkpoints successfully with Ray 2.9.0.
- Replayed 576 total symbol-safe diagnostic episodes and persisted the complete JSON.
- Validated the new ablation JSON against `ablation.schema.json`.
- Reproduced the optional-dashboard failure, added a failing unit test, implemented the narrow fallback, and observed the test pass.
- Completed the official fixed-control and CPU PBT probe with random/generated tasks and no LLM backend.
- Reproduced zero-byte Ray root/trial artifacts and replaced their evidentiary role with non-empty stable run/trial artifacts outside Tune.
- Completed the final repaired probe with two distinct members, matched validation seed, fixed batch exposure, real exploit/restore events, and exactly 12 unique iterations per member.
- Generated a capability report from each live trainer JSONL file and verified every population member is present in the ablation summary.
- Repository healthcheck passes required dependencies; CUDA is unavailable and optional LLM packages are absent.
- Final Windows validation: 97 tests passed with one warning; repository healthcheck passed on Python 3.11.13 and Ray 2.9.0; the ablation loadout passed its JSON schema.
- Pre-commit's YAML/whitespace checks passed, but its Bash hooks selected the Windows `.venv/Scripts/python.exe` through WSL and failed with `Exec format error` before healthcheck/test execution. The identical checks passed directly with the Windows interpreter; this is a launcher/shell integration failure, not a test failure.
- A 2026-09-14 post-launch pre-commit retry inside the managed Codex sandbox was blocked before hook execution because the global `C:\Users\Admin\.cache\pre-commit` SQLite store is read-only to this task. This is a sandbox/cache constraint; re-run pre-commit from a normal writable shell after training.
- After the PBT run and reporting correction, `pre-commit run --all-files` passed merge-conflict, YAML, EOF, trailing-whitespace, repository-healthcheck, and core-test hooks.

## 2026-09-14 parity-run update

The run-test-patch loop found and corrected two final orchestration/reporting
seams before the long run:

- Ray, stable trial evidence, and ablation collection now share an explicit
  `results_root`; the default remains `~/ray_results` for compatibility.
- The aggregate ablation reporter now forces a headless Matplotlib backend and
  emits `ablation_compare_validation_sharpe.png` in addition to the held-out
  net-return plot.

The first parity-shaped burst deliberately used the production batch size
(4,800), four concurrent CPU members, two rollout workers per member, and 16
matched validation episodes. Six iterations proved training and checkpointing
but ended exactly on an exploit boundary, so it did not prove a mutation. The
follow-up ran nine iterations, leaving iteration 6 non-terminal. All four members
completed approximately 44.3k–44.5k steps, three recorded checkpoint-restore
events, and the reporter produced a non-empty 207 KiB held-out Sharpe plot. The
best final point remained negative (-0.115% net; Sharpe -0.0081), which is
expectedly inconclusive at this horizon.

The parity loadout is
`itrader/scripts/experiments/configs/phase_a_pbt_parity_200_20260913.json`.
Experiment `pbtA26_parity200_20260914_000736_pbt200` completed with four members,
200 iterations, fixed 4,800-step batches (about 0.99M actual steps per member),
16 held-out episodes each iteration, 1 bp turnover cost, perturb/checkpoint
cadence 10, and burn-in 20. The proposer was random/generated and no LLM backend
was involved. Historical winners were not restored because their `legacy_v1`
encoding is incompatible with the corrected `semantic_v2` task contract.

All four final economic observations were negative: net returns were -1.185%,
-0.678%, -0.901%, and -0.598%; corresponding per-square-root-bar Sharpes were
-0.1983, -0.1712, -0.1475, and -0.1415. Optimization reward remained finite and
sometimes positive, so this run neither establishes nor disproves that the reward
is learnable. It does show that the current objective did not yield a positive
held-out economic candidate at historical exposure.

### 2026-09-14 checkpoint-boundary correction

The apparent iteration-40 discontinuity was a PBT exploit boundary: two losing
members cloned one selected member and their evaluation turnover jumped from
single digits to roughly 127. A frozen 16-task evaluation panel reproduced the
same class of jump after an iteration-6 restore, ruling out changing evaluation
tasks as the sole cause. A fresh restore of the selected checkpoint produced
about 118 turnover even before another PPO update, while the score used to select
it had about 2 turnover. The selection metric and restorable state were therefore
not the same artifact.

Weight hashes proved that the selected and restored policies were identical.
The discontinuity came from evaluation state: Ray 2.9 `Algorithm.restore()`
copied the training policy configuration onto the evaluation policy and silently
changed its `explore` flag from `false` to `true`. The restored policy was then
sampled stochastically, producing roughly random-policy turnover. This is not
negative grokking or a weight collapse.

Two secondary Ray 2.9 seams obscured the diagnosis. `Algorithm.evaluate()`
returns the evaluation payload directly on this installation, while the first
wrapper expected a nested `evaluation` key and silently retained the stale
built-in result. RLlib also smooths over 100 episodes by default, which blended
old and current policy evaluations instead of reporting the frozen 16-episode
panel. Finally, Ray AIR consumes/truncates the directory passed to
`session.report` on this Windows installation. The corrected contract is now:

1. training remains `start_idx_mode=stream_worker`;
2. evaluation uses one frozen task panel and deterministic starts;
3. market windows use checkpoint-free `market_relative_v1` features;
4. PBT requires `observation_filter=NoFilter` in both train and evaluation;
5. after every training/restore boundary, the evaluation policy is forced back
   to `explore=False`, learner weights are synchronized, and the direct Ray 2.9
   evaluation payload replaces the built-in result;
6. the metric-smoothing window equals the 16-episode frozen panel;
7. each checkpoint is copied to stable, run-owned storage before Tune receives
   its checkpoint object; and
8. the checkpoint used for selection must reproduce its score in a fresh process.

The final official two-member, six-iteration probe
`pbtA26_checkpoint_snapshot_probe_20260914_041234_pbt_checkpoint_snapshot`
exercised a non-terminal restore. At iteration 4, the selected source reported
net -0.3539%, Sharpe -0.04580, and turnover 56.625. The clone's first post-restore
update reported net -0.1016%, Sharpe -0.03783, and turnover 56.25; iteration 6
reported turnover 65.125. The spurious jump to roughly 127 is gone. A separate
fresh process restored the durable source checkpoint and reproduced all three
selection metrics exactly: net -0.0035387177838755394, Sharpe
-0.04580387065499236, and turnover 56.625.

Checkpoint/evaluation parity is now green. The corrected four-member,
200-iteration PBT
`pbtA26_parity200_checkpoint_valid_20260914_041907_pbt200_checkpoint_valid`
completed on 2026-09-14 with `stream_worker` training, random/generated tasks, the
frozen 16-task held-out panel, and 10-iteration checkpoint/perturbation cadence.
It is candidate generation and cannot support a performance or reward-
learnability claim until frozen policy baselines, multi-seed uncertainty, cost
scenarios, and the untouched final test are complete.

The live corrected run crossed the formerly suspicious iteration-40 boundary.
Member `c55da_00003` was cloned there and reported held-out turnover 11.25 and
8.50 on its first two post-restore updates, rather than the former roughly-127
jump. Other restored members likewise remained in the 6–13 range immediately
after cloning. This directly confirms in the production-length run that the
iteration-40 cliff was restore-induced stochastic evaluation, not negative
grokking.

### 2026-09-14 corrected 200-iteration completion

The corrected run completed all four members at iteration 200 in 49 minutes 57
seconds, with approximately 990k retained environment steps per member. The
official stable run status is `completed`; the proposer remained
random/generated, training remained `stream_worker`, and evaluation remained the
same deterministic 16-task held-out panel. Terminal observations were:

| Member | Gross return | Net return, 1 bp | Sharpe / sqrt(bar) | Turnover | Flat action | Evaluation optimization reward |
|---|---:|---:|---:|---:|---:|---:|
| `00000` | -0.0268% | **-0.0556%** | **-0.00060** | 2.875 | 91.32% | -0.6031 |
| `00001` | +0.0856% | -0.1069% | -0.03145 | 19.250 | 36.20% | -0.5585 |
| `00002` | +0.2522% | -0.1866% | -0.04089 | 43.875 | 13.83% | +0.8291 |
| `00003` | +0.0532% | -0.3906% | -0.05376 | 44.375 | 15.27% | -1.0828 |

The least-negative terminal member converged mostly to flat and had zero
positive-net observations over its final 20 retained iterations. Other members
showed transient positive peaks, including +0.3495% net at iteration 107, but
none persisted to the terminal checkpoint. Member `00002` is especially useful
evidence of objective/economic separation: its evaluation optimization reward
was positive (+0.8291) while its cost-adjusted return and Sharpe were negative.

The two-iteration control ended at +0.0340% net but Sharpe -0.01354. It is too
short and single-seed to establish superiority, yet it prevents claiming PBT
uplift: the trained population did not beat this minimal control at the terminal
net-return point. Reward learnability therefore remains `not_scored` in the
strong sense. The runs prove finite PPO interaction and changing policy behavior;
they do not yet prove improvement over an untrained policy or economic alignment.

The terminal durable checkpoint for member `00000` was restored in a fresh Ray
process. Net return (-0.0005557122979728761), Sharpe
(-0.000600490529072014), and turnover (2.875) reproduced exactly within
`1e-15`. Checkpoint/evaluation parity therefore remains green at full exposure.

One post-run reporting defect was caught before sign-off. The aggregate reporter
aligned metric values by row position, so PBT lineage rewinds misplaced members
`00002` and `00003` terminal values at iterations 190 and 180. A red/green test
now requires alignment by the recorded `training_iteration`; the regenerated
CSVs and plots contain all four true iteration-200 values. Late exploit rewinds
also show why a future PBT implementation should use a monotonic environment-step
clock rather than restored `training_iteration` for scheduling and progress.
The final trial's actor-local auto-plot subprocess reported Windows error 87,
but the authoritative outer ablation reporter completed and produced every
comparison CSV and PNG. Keep outer reporting authoritative until actor-local
plotting is removed or made platform-safe.

Decision: the repaired run/test/checkpoint/reporting seams are sufficient to
move to the audit's **next P0 baseline study**, but not to Phase A performance
promotion or a Phase B dependency on learned R0 behavior. Run deterministic
flat, long, short, random, and untrained-policy controls on the identical frozen
starts; then add independent training seeds, block-aware uncertainty, and cost
scenarios. Keep the untouched final test sealed until selection is complete.

Completion artifacts:

- [all-member summary](../../../itrader/scripts/experiments/configs/outputs/pbtA26_parity200_checkpoint_valid_20260914_041907/summary.json)
- [held-out net-return plot](../../../itrader/scripts/experiments/configs/outputs/pbtA26_parity200_checkpoint_valid_20260914_041907/ablation_compare.png)
- [held-out Sharpe plot](../../../itrader/scripts/experiments/configs/outputs/pbtA26_parity200_checkpoint_valid_20260914_041907/ablation_compare_validation_sharpe.png)
- [diagnostic plot](../../../itrader/scripts/experiments/configs/outputs/pbtA26_parity200_checkpoint_valid_20260914_041907/ablation_compare_diagnostics.png)
- [best terminal checkpoint replay](../../../itrader/scripts/experiments/configs/outputs/pbtA26_parity200_checkpoint_valid_20260914_041907/best_terminal_checkpoint_replay.json)

### Next-P0 implementation round (staged 2026-09-14)

The late restore pattern is now fixed at its cause rather than merely documented.
New trials report a target-owned cumulative sampled-environment-step clock in
`pbt_progress.json`. Tune PBT uses `pbt_progress_steps` for burn-in and mutation
cadence, checkpoints are emitted when that clock crosses the matching step
boundary, and the target exposure budget no longer inherits a donor checkpoint's
RLlib iteration. Historical report files remain readable through an iteration
fallback, while new comparison plots align on exposure.

The earlier reward dominance is addressed with a separate, replay-safe objective,
not by silently changing `pnl_linear`. `net_return` computes arithmetic PnL minus
the configured cost in basis points for position turnover and terminal
liquidation. A fixed scale of 1,000 supplies a practical PPO magnitude without
changing the economic ordering; with `cost_bps=1`, summed episode reward divided
by 1,000 equals independently derived 1 bp net return. This is a task/evaluation
assumption and does not introduce broker execution into `TaskSpec`. PPO still
discounts step rewards when `gamma < 1`, so equality of the undiscounted episode
sum does not claim perfect equivalence to an undiscounted portfolio objective.

The next-P0 evidence seams now include:

- frozen validation panels balanced over configured horizons;
- deterministic flat, long-only, short-only, seeded random, and seeded untrained
  `SolverMLPFiLM` controls with per-episode receipts;
- Ray 2.9 PPO columns for approximate KL, value explained variance, total loss,
  gradient norm, learning rate, and KL coefficient; clip fraction is explicitly
  unavailable rather than inferred;
- a best-held-out cadence-checkpoint pointer that is marked as requiring fresh-
  process replay; and
- optional terminal promotion gates relative to the strongest named baseline,
  requiring positive net return and positive held-out Sharpe by default.

Two official loadouts are staged: a 64-episode frozen-policy baseline panel and a
bounded 20-update `pnl_linear` reference versus 40-update, four-member
`net_return` PBT probe. Four-episode flat and untrained-policy smoke executions
passed. The full baseline panel and bounded PBT probe have not yet been run; this
section is implementation evidence, not a new performance claim. Block bootstrap
uncertainty, multiple cost scenarios, independent training seeds, and the sealed
final test remain later gates.

### 2026-09-16 matched 120-update reward bridge

The matched bridge ran the repository stack end to end with `stream_worker`, a
random/generated proposer path, no LLM backend or task pool, four PBT members per
arm, two rollout workers per member, identical member seeds, the same balanced
32-task validation panel, and a target of 576,000 trial-owned environment steps
per member. It compared only the optimization reward: exact scaled 1 bp
`net_return` versus historical `pnl_linear`. Both arms used the same 1 bp
economic evaluator and the same MLP-FiLM R0 solver. Every member crossed the
target exposure; variable complete-episode batches produced 117 report rows and
roughly 578--579k sampled steps rather than exactly 120 RLlib report rows.

The exact matched frozen controls on those 32 starts were:

| Policy | Net return at 1 bp | Sharpe / sqrt(bar) | Turnover |
|---|---:|---:|---:|
| flat | 0.00000% | 0.00000 | 0.0000 |
| long-only | **+0.03034%** | **+0.01077** | 2.0000 |
| short-only | -0.07034% | -0.01608 | 2.0000 |
| seeded random | -1.46079% | -0.20074 | 130.6875 |
| seeded untrained MLP-FiLM | -0.31137% | -0.03210 | 3.6250 |

Terminal bridge evidence was:

| Reward arm / member | Gross | Net at 1 bp | Sharpe | Turnover | Short / flat / long |
|---|---:|---:|---:|---:|---:|
| `net_return` / `00000` | +0.05409% | -0.04716% | -0.02171 | 10.1250 | 37.55 / 62.45 / 0.00% |
| `net_return` / `00001` | -0.14382% | -0.78257% | -0.12148 | 63.8750 | 25.97 / 50.08 / 23.95% |
| `net_return` / `00002` | 0.00000% | **0.00000%** | 0.00000 | 0.0000 | 0.00 / 100.00 / 0.00% |
| `net_return` / `00003` | +0.01671% | -0.49392% | -0.09533 | 51.0625 | 24.72 / 59.03 / 16.25% |
| `pnl_linear` / `00000` | -0.14193% | -0.80756% | -0.10836 | 66.5625 | 24.62 / 29.52 / 45.86% |
| `pnl_linear` / `00001` | -0.15515% | -1.03202% | -0.13641 | 87.6875 | 52.22 / 22.92 / 24.86% |
| `pnl_linear` / `00002` | +0.09126% | -0.85499% | -0.12711 | 94.6250 | 31.73 / 36.24 / 32.03% |
| `pnl_linear` / `00003` | -0.09277% | -0.84527% | -0.12093 | 75.2500 | 49.94 / 27.52 / 22.54% |

Mean terminal net return was approximately -0.3309% for `net_return` and
-0.8850% for `pnl_linear`. The net-return arm therefore reduced economic damage,
but its best terminal member did so by becoming exactly flat. This is the feared
easy-gradient failure mode: an exact 1 bp objective teaches abstention faster
than it discovers gross edge. The PnL-linear arm maintained diverse actions but
did not learn turnover discipline, so explicit costs overwhelmed its gross
returns.

The best replayable net-return cadence checkpoint was member `00000` at progress
480,456: +0.01941% net, +0.02503% gross, 0.005625% cost, Sharpe -0.00175,
turnover 0.5625, and 75.02% flat. The best replayable PnL-linear checkpoint was
member `00001` at progress 49,944: +0.04948% net, +0.47885% gross, 0.429375%
cost, Sharpe -0.01032, turnover 42.9375, and 33.71% flat. The latter exceeded
long-only net by only 0.01914 percentage points while losing on Sharpe and
turnover. Neither checkpoint clears the positive-Sharpe and baseline-relative
promotion gates, so no final-test evaluation or promotion replay was authorized.

PPO remained active: policies, value diagnostics, action fractions, and
optimization behavior changed materially. The bridge does not reproduce the
corrected 200-update member `00001` positive window, however, and it does not
establish robust reward learnability in the economic sense. The correct reading
is reward-sensitive optimization with two different failure modes, not a dead
optimizer and not a validated trading policy.

The long bridge also exposed a Ray 2.9 asynchronous PBT limitation. Only one true
donor exploit occurred in each arm, and Ray could select an upper-quantile donor
whose scheduler-owned checkpoint was not yet available. Application-owned stable
checkpoints were present; the failure was scheduler ordering. An explicit
`pbt_synch` loadout seam now opts promotion-oriented studies into synchronous
PBT while preserving asynchronous mode as the legacy reproduction default. A
red/green 20-update A/B probe removed the missing-donor warning and produced
real configuration changes. It then exposed a pause boundary inside
`session.report`; checkpoint indices are now written before reporting. A final
12-update synchronous probe produced three durable indices and three stable
checkpoint directories per member, monotonic progress, two pause/resume
boundaries per member, true mutations in two of four members, and no missing-
donor warning. In synchronous mode, raw restore counts include self-resume and
must not be reported as exploit counts; configuration history remains the
authority for actual mutations.

Bridge artifacts:

- [matched loadout](../../../itrader/scripts/experiments/configs/phase_a_p0_net_return_pbt_bridge_120_20260916.json)
- [all-member summary](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_reward_bridge_120_20260916_062129/summary.json)
- [compact net/Sharpe/turnover/action dashboard](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_reward_bridge_120_20260916_062129/phase_a_bridge_dashboard.png)
- [held-out net-return plot](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_reward_bridge_120_20260916_062129/ablation_compare.png)
- [held-out Sharpe plot](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_reward_bridge_120_20260916_062129/ablation_compare_validation_sharpe.png)
- [PPO diagnostic plot](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_reward_bridge_120_20260916_062129/ablation_compare_diagnostics.png)
- [action-fraction evolution and terminal composition](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_reward_bridge_120_20260916_062129/ablation_compare_action_fractions.png)
- [matched frozen controls](../../../itrader/scripts/experiments/configs/phase_a_p0_frozen_policy_baselines_32_bridge_20260916.json)
- [synchronous checkpoint probe](../../../itrader/scripts/experiments/configs/phase_a_pbt_sync_checkpoint_probe_20260916.json)
- [synchronous index probe](../../../itrader/scripts/experiments/configs/phase_a_pbt_sync_index_probe_20260916.json)

### Ray upgrade decision

Ray 2.9 should remain only as a locked reproduction environment. The current
stable release is Ray 2.58.0, but upgrading in place would combine two independent
changes: runtime/checkpoint behavior and RLlib's default-new API stack. The new
stack replaces `ModelV2` with `RLModule`, rollout workers with EnvRunners/Learners,
and old trajectory/callback plumbing with Episodes and ConnectorV2. That is the
right destination for iTrader, but it is not a patch-level dependency bump.

The repository currently has declaration drift that must be removed during the
migration: `pyproject.toml` and `itrader/environment.yml` pin Ray 2.9.0, while
`itrader/constraints.txt` pins 2.47.1 and `itrader/infra/cluster.yaml` installs
2.47. This is not a supported matrix. The locked 2.9 reproduction and the new
2.58 compatibility environment each need one authoritative, fully resolved lock.

Use a separate branch and environment with this sequence:

1. lock the current Ray 2.9 environment and preserve one known reproduction;
2. create a Ray 2.58.0/Python 3.11 compatibility environment;
3. first disable the new API stack explicitly and prove dataset, task, reward,
   checkpoint, evaluation, and PBT parity on the current MLP baseline;
4. then port `SolverMLPFiLM` to `RLModule`, metrics to `RLlibCallback`, rollout
   ownership to EnvRunner/Learner, and normalization to an explicit ConnectorV2
   or the environment; and
5. run the same fixed-panel checkpoint parity and short PBT acceptance loadout
   before any new long search.

Prefer Linux or WSL2 for overnight PBT: Ray documents Windows support as beta,
with untested multi-node behavior, slower file operations, and path caveats.
Unsloth is useful for LLM fine-tuning, not as a replacement for RLlib PPO, Tune
PBT, or numerical-policy checkpoint semantics. It may later accelerate an LLM
proposer experiment, but it cannot repair this solver-side state contract.

Primary references: [Ray 2.58.0 release](https://pypi.org/project/ray/2.58.0/),
[RLlib new API migration](https://docs.ray.io/en/master/rllib/new-api-stack-migration-guide.html),
[RLModule documentation](https://docs.ray.io/en/latest/rllib/rl-modules.html), and
[Ray Windows installation caveats](https://docs.ray.io/en/latest/ray-overview/installation.html).

That 200-update run answered whether corrected held-out economic behavior
emerged at roughly the historical training exposure. It did not, by itself,
establish solver uplift. Matched flat/long/short/random/untrained controls and
exploratory block/cost diagnostics have since run; independent seeds, genuinely
new blocks, selection-aware inference, and the untouched final test remain
mandatory for a performance claim.

## 2026-09-17 synchronous 120-update bridge and recovery

The [frozen loadout](../../../itrader/scripts/experiments/configs/phase_a_p0_reward_bridge_sync_120_20260917.json)
kept four R0 members per arm, `stream_worker`, generated/random family-00
tasks, no LLM backend, Ray/RLlib 2.9, the same 32-task frozen validation
identities, 1 bp economic reporting cost, two rollout workers per member,
`pbt_synch`, 10-update perturbation/checkpoint cadence, 20-update burn-in, and
a 576,000 trial-owned sampled-step target. Dataset SHA-256 was
`5F2E449548532FF71F419431DF86962A2AEDF7275D7BFF5592729BBB8A900F73`.
The initial population seeds and validation task identities match between arms;
the reward ID and reward parameters intentionally differ. Exploitation may
subsequently clone a donor's policy/configuration, so a terminal member's
checkpoint seed need not equal its initial seed.

The first `net_return_1bp` arm completed. The user paused the machine while
the second arm was only 9–13 updates in: three members had first checkpoints,
one had none, and trial-owned progress/report files ran ahead of the durable
models. Ray saved an experiment snapshot, but a direct resume without
reconciling model, evidence, and scheduler state would misstate trained
exposure; the uncheckpointed trial would restart. We preserved that incomplete
experiment as provenance, restarted **only** `pnl_linear` from update zero
under a new `_restart` experiment name, and left the completed 1 bp arm
untouched. This lost only the short partial second-arm prefix, not the first
arm. The official trainer exited successfully on 2026-09-17: all four restart
members have completed statuses, 116–117 report rows, 12 durable checkpoint
indices each, and 576,408–579,632 trial-owned sampled steps. The final
[eight-member summary](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_reward_bridge_sync_120_20260917_001810/summary.json)
was assembled with the ablation CLI's tested `--collect-only` path, which
rejects incomplete arms. It contains the restarted, not interrupted, second
arm. Ray's own terminal `training_iteration` can rewind after PBT cloning;
the comparison charts align the monotonic trial-owned environment-step clock.

| Objective / member | Terminal net return | Gross return | Net Sharpe / sqrt(bar) | Turnover | Short / flat / long | Genuine config changes |
|---|---:|---:|---:|---:|---:|---:|
| 1 bp net 00 | -0.17059% | -0.05559% | -0.04140 | 11.50 | 25.9 / 66.1 / 7.9% | 1 |
| 1 bp net 01 | -0.11031% | +0.00406% | -0.02683 | 11.44 | 26.4 / 64.6 / 9.1% | 3 |
| 1 bp net 02 | -0.14602% | -0.02977% | -0.02783 | 11.63 | 26.4 / 65.9 / 7.7% | 3 |
| 1 bp net 03 | -0.14185% | -0.03310% | -0.03029 | 10.88 | 25.9 / 65.1 / 8.9% | 3 |
| PnL linear 00 | -0.96870% | +0.12380% | -0.12748 | 109.25 | 49.8 / 18.4 / 31.9% | 3 |
| PnL linear 01 | -0.97490% | +0.10197% | -0.12730 | 107.69 | 47.1 / 19.4 / 33.5% | 2 |
| PnL linear 02 | -0.92244% | +0.11069% | -0.12374 | 103.31 | 48.6 / 18.0 / 33.4% | 2 |
| PnL linear 03 | -1.13590% | -0.01278% | -0.15445 | 112.31 | 46.1 / 21.9 / 31.9% | 1 |

Mean terminal net return was **-0.14219%** under exact 1 bp `net_return`
versus **-1.00048%** under `pnl_linear`. Every terminal Sharpe was negative;
the matched 32-start flat and long-only controls remained 0 and +0.03034%
net, respectively (long-only Sharpe +0.01077, turnover 2). The 1 bp policies
limited cost mostly through reduced exposure but were not fully flat; the
PnL-linear policies kept trading and paid 1.03–1.12% of equity in mean
episode costs. The latter's terminal training optimization rewards were
finite/positive and KL, entropy, and value explained variance remained
reportable: PPO was active, but its shaped objective and economic return
diverged. For example, terminal PnL-linear evaluation component means were
roughly `rc_pnl_delta=+0.015–0.017` in three members (the fourth +0.00255),
`rc_dd=-0.0041–0.0043`, and
`rc_turnover=-0.00082–0.00088` in their **own normalized reward units**;
these are not interchangeable with the independently computed 1 bp cost
return of 0.0103–0.0112. The source normalizes PnL by `ret_unit` while using
an abstract, capped turnover weight. Clip fraction is unavailable (`NaN`) on
this legacy Ray learner, not evidence of zero clipping.

Synchronous PBT performed **10 genuine configuration changes** in the 1 bp
arm and **8** in the restarted PnL-linear arm, measured by changes to the
supported PPO mutation knobs between stable configuration-history records.
Unchanged self-resumes were excluded. Across all records gamma remained
0.98–0.995; there was no repeat of the historical invalid gamma or late
iteration-130 scheduling rewind. The exact donor identity is less durable
than the stable config/checkpoint history because Ray-owned Windows policy
logs may be empty; count actual changes, not raw restores.

Fresh-process deterministic episode replay of each arm's **best retained
cadence checkpoint** reproduced the selected 32-start mean net return to
floating-point precision. The 1 bp member-00 checkpoint had -0.04260% net,
Sharpe -0.00491, and turnover 6.4375. The early PnL-linear member-01
checkpoint at 49,944 steps had +0.06718% net, **Sharpe -0.02301**, turnover
26.1875, and short/flat/long fractions 16.6/30.1/53.3%. It is a genuine
positive *point estimate*, not a stable or Sharpe-positive winner. Later
terminal policies all lost money. The replayed episode receipts and exact
selection-parity JSON are under
[replays](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_reward_bridge_sync_120_20260917_001810/replays/).

Paired by symbol, start/end rows, horizon, and task seed against the 32-start
long-only control, the PnL-linear checkpoint's 1 bp uplift was +0.03684
percentage points. An exploratory symbol-stratified chronological 5,000-row
block bootstrap (10,000 draws, seed 20260917, **14 effective blocks**) gave a
95% interval of **[-0.20893, +0.33746] percentage points**. The 1 bp
checkpoint's uplift was -0.07294 points, interval [-0.34631, +0.17073].
Intervals with 2,500-row (22 blocks) and 10,000-row (9 blocks) choices also
crossed zero. These are *not* selection-adjusted confidence intervals: PBT
selected checkpoints on the same validation panel, episodes overlap, and only
three symbols are represented. They are diagnostic stability checks, not P1
promotion or a substitute for independent-seed validation and the sealed test.
Repricing the **same checkpoint actions**, without changing learned policy,
gave the following mean arithmetic net returns:

| Cost | Best 1 bp checkpoint | Best PnL-linear checkpoint | Long-only | Flat |
|---:|---:|---:|---:|---:|
| 0.5 bp | -0.01041% | +0.19811% | +0.04034% | 0 |
| 1 bp | -0.04260% | +0.06718% | +0.03034% | 0 |
| 2 bp | -0.10697% | -0.19470% | +0.01034% | 0 |

Only mean return is repriced here; a different-cost Sharpe requires replay of
per-step net returns. No independent training seed, selection-adjusted
inference, or untouched final-test run has occurred. The next **P0 reward
ablation** remains no-cost learning, staged-cost curriculum, and exact 1 bp
learning under identical frozen tasks/seeds, with every candidate evaluated
at 1 bp. Gate first on gross edge over long-only and then on positive net
return, positive Sharpe, controlled turnover, checkpoint parity, and
independent-seed stability. Define **P1** only after a P0 candidate clears
those gates: block-aware uncertainty on genuinely new chronological blocks,
0.5/1/2 bp policy and per-step Sharpe replay, and selection-safe final-test
procedure. The untouched final test stays sealed until selection is complete.

The pause exposed a separate operational seam: exact resume requires a
coordinated final checkpoint for **every** member and recovery of trial-owned
progress/report rows to each checkpoint, plus Ray scheduler/search state.
`--collect-only` repairs reporting without pretending to repair training
resume. Before another long overnight job, add a tested coordinated pause and
resume protocol or explicitly budget a fresh-arm restart. An NVMe versus
current-storage A/B can be run later with identical short loadouts, separated
Ray temp/results paths, and matched dataset/cache conditions; no storage move
was made during this study.

Completion plots: [net return](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_reward_bridge_sync_120_20260917_001810/ablation_compare.png),
[Sharpe](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_reward_bridge_sync_120_20260917_001810/ablation_compare_validation_sharpe.png),
and [held-out action fractions](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_reward_bridge_sync_120_20260917_001810/ablation_compare_action_fractions.png).

## Final judgment

The historical work was not meaningless: it established that the Phase A
machinery can train a policy against a finite shaped reward. The mistake is in
the strength and naming of the conclusion. The recorded “episode return” is not
a portfolio return, the selected configurations were not constrained correctly,
and the historical dataset was not held out. The corrected 200-update run and
the matched 120-update reward bridge now score R0 against explicit controls; both
fail economic promotion. The solver is active and reward-sensitive, but neither
reward proves repeatable uplift over long-only with positive Sharpe.

The correct current status is:

> **Phase A economic measurement, checkpoint/evaluation parity, exposure-aligned
> reporting, frozen controls, and synchronous Ray 2.9 donor/checkpoint ordering
> are repaired and full-run validated. The matched synchronous bridge confirms active
> PPO but exposes an inactivity attractor under exact 1 bp net reward and
> cost-blind turnover under PnL-linear reward. An early PnL-linear checkpoint
> replayed at a positive mean net return, but its Sharpe was negative, its
> block-aware selection-panel interval crossed zero, and all terminal policies
> lost money. Neither arm passes the matched long-only plus Sharpe gate.
> Historical winners remain quarantined;
> do not claim economic reward learnability, solver uplift, or Phase A promotion
> until a bounded challenger passes the controls, independent seeds, block-aware
> uncertainty, multiple costs, replay, and the untouched final test.**
