# R4-v1-min solver-state implementation and smoke audit — 2026-09-25

## Status

R4-v1-min passes its implementation and bounded runtime gate. The accepted
observation contract reaches Ray, the environment, actor/value heads, manifests,
checkpoints, and fresh-process replay without a schema or provenance
contradiction.

The three-update smoke is **mechanistic evidence only**. It does not promote an
economic candidate: every retained update-3 checkpoint lost money after fixed
1 bp cost, had negative held-out Sharpe, and failed to beat flat, short-only, or
the seeded untrained control. No PBT or R1 run was started.

## Implemented contract

The optional observation layout is:

```text
[market_window_flat | task_embedding | solver_state]
```

`solver_state_mode` selects one fixed two-value block:

```text
masked   -> [0, 0]
position -> [current_position, 0]
time     -> [0, remaining_horizon_fraction]
both     -> [current_position, remaining_horizon_fraction]
```

The default is `off`, preserving the historical R0 observation width, base
parameter shapes, and seeded base outputs. Every diagnostic cell uses the same
two-dimensional block and therefore the same parameter count. `previous_action`
is intentionally unsupported.

The accepted runtime identity for every diagnostic cell is:

```text
market_window_flat = 300
task_embedding     = 32
base_obs_dim       = 332
solver_state_dim   = 2
agent_obs_dim      = 334
observation_schema_id = r4_v1_min
```

`TaskSpec.obs_dim` remains `332` as the legacy/base task-conditioned width so
`semantic_v2` task embeddings and historical task panels do not change merely
because an auxiliary environment-state block is activated. Ray still receives
one flat `Box(334)`; the model splits the three blocks explicitly. Run, trial,
resolved-config, best-checkpoint, and checkpoint-index evidence record the
schema and all component dimensions.

- `ITraderEnv` owns actual position, elapsed steps, reset, causal normalization,
  and observation assembly.
- `TaskSpec` remains static and unchanged.
- `RewardEngine` remains the only reward calculator.
- `SolverMLPFiLM` splits market, task, and state blocks explicitly. The market
  MLP, task encoder/FiLM, and base actor/value heads are unchanged. A bias-free
  `Linear(2, 8) -> ReLU` encoder feeds bias-free actor/value residuals, so
  `[0, 0]` maps exactly to zero and reproduces the base path. Paired positive and
  negative hidden units can represent the symmetric `abs(position)` turnover
  effect that a one-layer odd Tanh path could not.
- Every environment reset, including `stream_worker`, restores solver state to
  `[position=0, remaining=1]`.
- A nonterminal next observation reports the newly selected position and the
  decremented remaining fraction.

## Terminal observation choice

The accepted terminal observation reports the final selected target position
and `remaining_horizon_fraction=0`. The existing
liquidation amount remains a terminal reward/evidence charge; the environment
does not silently mutate its held position to flat before returning the terminal
observation. No subsequent decision consumes the terminal observation. The next
reset returns `[position=0, remaining=1]`, including under `stream_worker`, which
continues market sampling but never carries portfolio or recurrent state.

## Matched smoke design

`phase_a_r4_v1_min_2x2_smoke_20260925.json` ran four matched, sequential cells:

1. parameter-matched masked-state control;
2. position only;
3. remaining horizon only;
4. position plus remaining horizon.

All cells used the same 60x5 dataset-v2 RTH input, 12 symbols, Family 00,
crossed 32-task validation panel, `stream_worker`, random/generated training
tasks, Ray local-worker topology, fixed exact 1 bp `net_return`, seed
`20260925`, evaluation seed `30260943`, and 14,664 sampled steps. PBT and global
cost scheduling were disabled.

Runtime identity was Python 3.11.13, Ray 2.58.0, Gymnasium 1.2.2, PyArrow 17.0.0,
Torch 2.3.1, and Pandas 2.1.4. Every manifest resolved storage to
`D:\iTRADER-offload\active\ray_results\phase-a-r4-v1-min-20260925`.

The first launch attempt stopped before Ray or training because the ablation
runner's child `python` resolved an obsolete `C:\Users\Admin\itrader`
environment with incompatible NumPy/Pandas/PyArrow binaries. The loadout now
pins the repository `.venv` interpreter. The failed attempt produced no run
evidence or model update and is excluded from the comparison.

## Update-3 held-out results

Returns are arithmetic mean episode returns on the identical 32-task panel.
Action fractions are equal-weight means of per-episode fractions.

| Cell | Net return | Sharpe / sqrt(bar) | Gross return | Turnover | Position changes | Liquidation | Short / flat / long | Entropy | Explained variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| masked | -0.2004% | -0.03873 | -0.0616% | 13.875 | 12.906 | 0.969 | 37.62% / 12.38% / 50.00% | 1.0653 | 0.0163 |
| position | -0.3977% | -0.07080 | +0.0004% | 39.812 | 38.312 | 0.938 | 42.66% / 15.36% / 41.99% | 1.0644 | 0.0135 |
| time | -0.0934% | -0.02474 | +0.0329% | 12.625 | 9.438 | 1.000 | 47.76% / 3.65% / 48.58% | 1.0456 | 0.0126 |
| position + time | -0.1843% | -0.04036 | -0.0162% | 16.812 | 15.406 | 0.906 | 34.75% / 15.45% / 49.80% | 1.0649 | 0.0115 |

Time-only was the least-bad terminal cell, but it remains below flat and is
economically ineligible. Position-only reacted most strongly and
pathologically: mean turnover rose to 39.8, so an almost flat gross return
became the worst net result.

Each cell showed a short-lived positive return before update 3: masked
`+0.0459%` at update 1, position `+0.0466%` at update 2, time `+0.0617%` at
update 1, and both `+0.0305%` at update 1. These are not retained winners. With
checkpoint interval 3, only update 3 was checkpoint-authoritative. A longer R4
probe must checkpoint every evaluation improvement if transient behavior is
part of the question.

## Matched controls

| Control | Net return | Sharpe / sqrt(bar) | Turnover | Short / flat / long |
|---|---:|---:|---:|---:|
| flat | 0.0000% | 0.00000 | 0.000 | 0% / 100% / 0% |
| long-only | -0.0979% | +0.00827 | 2.000 | 0% / 0% / 100% |
| short-only | +0.0579% | -0.01558 | 2.000 | 100% / 0% / 0% |
| seeded random | -1.2144% | -0.24465 | 130.625 | 32.86% / 34.29% / 32.85% |
| seeded untrained R0 | +0.0618% | -0.00976 | 2.000 | 75% / 0% / 25% |

The panel has a modest directional short advantage in mean return, although the
mean of per-episode Sharpe values remains negative. The untrained R0 policy
exploits almost exactly the same 75%-short/25%-long tendency seen in the first
update of the trained cells. The early positive values therefore do not
demonstrate learning beyond initialization or directional exposure. Flat
remains the non-negotiable promotion floor.

## Exact checkpoint replay

All four retained checkpoints restored in fresh processes and reproduced their
selected mean net return:

- masked: replay delta `0.0`;
- position: replay delta `0.0`;
- time: replay delta `1.08e-19`;
- position + time: replay delta `2.17e-19`.

Dataset hash, evaluation-panel hash, source-tree fingerprint, observation
schema, config snapshot, checkpoint hash, and lineage remained present. This
closes the runtime/checkpoint parity part of the smoke.

## State-conditioned behavior

Diagnostic replay recorded 4,672 decisions per cell and computed action
probabilities from `action_dist_inputs`, value predictions, actual prior
position, fixed remaining-time quarters, switching, and turnover. Its aggregate
fractions are step-weighted; the training report above averages episode
fractions equally, so mixed horizons legitimately produce different values.
Direct same-observation checks confirmed normal and `full_fetch=True` inference
choose identical actions.

| Cell | Step-weighted short / flat / long | Mean probabilities short / flat / long | Switching | Inference entropy | Value mean +/- std |
|---|---:|---:|---:|---:|---:|
| masked | 30.27% / 13.57% / 56.16% | 34.43% / 30.36% / 35.21% | 8.84% | 1.0669 | -3.245 +/- 0.329 |
| position | 35.72% / 18.94% / 45.33% | 36.55% / 32.94% / 30.51% | 26.24% | 1.0699 | -3.294 +/- 0.360 |
| time | 41.07% / 4.00% / 54.92% | 35.80% / 26.83% / 37.37% | 6.46% | 1.0474 | -3.262 +/- 0.336 |
| position + time | 27.08% / 16.93% / 55.99% | 34.66% / 29.88% / 35.47% | 10.55% | 1.0675 | -3.287 +/- 0.348 |

The position cell is measurably state-sensitive. When already short, its mean
probabilities were `46.46% / 31.57% / 21.97%`; when already long they were
`30.37% / 32.60% / 37.03%`. When flat it switched into exposure on 67.65% of
decisions, explaining the turnover failure. In the combined cell, long-action
probability fell from 37.61% in the earliest quarter to 33.53% in the terminal
quarter while flat probability rose from 28.52% to 31.16%. This is a small but
visible horizon-conditioned response.

These slices prove that the connector can influence actor/value outputs and
that the checkpoint can be interrogated by causal state. They do not prove that
state use is economically useful. Entropy remains near the categorical maximum
`ln(3)=1.0986`, explained variance is only about 0.01, and value fit is weak.
`clip_fraction` remains `NaN` in the legacy-stack report and is still a
diagnostics gap.

## Verdict and next gate

R4-v1-min is implementation-valid and replay-valid. The prior observation was
partially observable because it omitted position and remaining horizon; the new
contract supplies that causal state without changing TaskSpec, reward, base
MLP-FiLM, or historical R0 identity.

The smoke does **not** show economic uplift in three PPO updates. The correct
next experiment is a longer matched masked-versus-both probe with checkpointing
at every evaluation improvement, the same fixed 1 bp panel and seeds, and the
state-conditioned diagnostic retained. Only durable state use without turnover
explosion can justify making R4-v1 the R1 observation contract or entering it
into PBT. PBT, recurrent-state work, causal C-series work, and new task families
remain outside this result.

## Evidence index

- [four-cell comparison](solver-state-r4-v1-min-implementation-note-2026-09-25-evidence/smoke_4cell/summary.json)
- [net-return plot](solver-state-r4-v1-min-implementation-note-2026-09-25-evidence/smoke_4cell/ablation_compare.png)
- [Sharpe plot](solver-state-r4-v1-min-implementation-note-2026-09-25-evidence/smoke_4cell/ablation_compare_validation_sharpe.png)
- [action-fraction plot](solver-state-r4-v1-min-implementation-note-2026-09-25-evidence/smoke_4cell/ablation_compare_action_fractions.png)
- [fresh checkpoint replays](solver-state-r4-v1-min-implementation-note-2026-09-25-evidence/replay/)
- [state-conditioned diagnostics](solver-state-r4-v1-min-implementation-note-2026-09-25-evidence/state_diagnostics/)
- [matched controls](solver-state-r4-v1-min-implementation-note-2026-09-25-evidence/controls/)

## Validation record

- prelaunch full repository suite: 268 passed;
- prelaunch healthcheck and scoped pre-commit: passed;
- focused state-diagnostic suite after the evidence seam: 19 passed;
- final full repository suite: 275 passed; final healthcheck and scoped
  pre-commit passed;
- all four runs completed at 14,664 steps with one authoritative trial each;
- free RAM was about 8.0 GiB before/between cells and about 5.1 GiB during
  training, above the 2 GiB hard stop;
- no PBT, R1, commit, staging, deletion, or final-test access occurred.
