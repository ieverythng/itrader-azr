# Ray 2.58 R0 platform-parity gate — 2026-09-24

## Decision

The current dataset-v2 R0 source passes the Ray 2.58.0 platform gate without
replacing the repaired environment, sampled-step PBT, checkpoint, lineage, or
reporting contracts. A fresh local-worker two-member run completed 12 updates
per member, crossed the 24,000-step cadence, performed one genuine synchronous
exploit/restore, passed strict generation-bound lineage inspection, and replayed
its retained best checkpoint exactly in a fresh process.

Ray 2.58 is accepted as the forward R-series runtime. Ray 2.9 remains a frozen
historical reproduction profile, not the root dependency authority. The two
versions do not produce seed-identical PPO trajectories, so every result must
remain runtime-labelled and R1 must be compared with a Ray-2.58 R0 control.

## Review boundary

The separate migration branch commit
`1033dc4433abea325cba56229537e6fffe4bf7d0` was reviewed against current
`agent/Watson-dev` at `089411666e7105b79d891f2edefcd7c5fa9792de`.
It was not cherry-picked because it was based on dataset v1, `pnl_linear`, four
evaluation episodes, asynchronous stock PBT, and a broader dependency rewrite.
The current tree instead received only compatibility adapters needed to preserve
the newer dataset-v2 R0 contract.

The coordinated root dependency declarations now target Ray 2.58.0, Gymnasium
1.2.2, PyArrow 17.0.0, and pandas 2.1.4. The old repository `.venv` remains the
live Ray 2.9 reproduction environment, and its direct versions are recorded in
`itrader/compat/ray29/`. The isolated Ray 2.58 environment used for this gate is
`C:/Users/Admin/.codex/worktrees/3054/itrader-azr/.venv`.

## Compatibility work

- Kept the legacy `ModelV2`/RolloutWorker stack explicit on Ray 2.58.
- Adapted Tune storage, trial context, incoming checkpoint, report, trial-output,
  runner-group, resource-key, and nested metric APIs by runtime capability.
- Preserved deterministic post-sync evaluation, dataset/panel/source hashes,
  target-owned sampled-step clocks, stable checkpoints, generation receipts, and
  current economic metrics.
- Added runtime versions to every run manifest.
- Accepted both the Ray 2.9 PBT controller facade and Ray 2.58 direct controller.
- Added a completion guard: Tune's `TERMINATED` label is insufficient when a
  trial has not reached `target_environment_steps`. Such a run is now recorded
  as `incomplete` and cannot be consumed as completed ablation evidence.

## Verification

| Check | Ray 2.9 | Ray 2.58 | Result |
|---|---:|---:|---|
| Python | 3.11.13 | 3.11.13 | matched |
| Torch / NumPy / pandas | 2.3.1 / 1.26.4 / 2.1.4 | same | matched |
| Gymnasium / PyArrow | 0.28.1 / 13.0.0 | 1.2.2 / 17.0.0 | intentional migration |
| Repository tests | 254 passed | 254 passed | green |
| Healthcheck | green | green | green |
| Initial policy SHA-256 | `ceba986d…3140` | `ceba986d…3140` | exact |
| Initial deterministic panel policy | 100% short | 100% short | exact |
| Initial panel net return | +0.05785% | +0.05785% | exact |
| Initial panel Sharpe/√bar | -0.01558 | -0.01558 | exact |

## Same-seed result divergence

Two repeated one-update controls were bit-stable inside each runtime but not
across runtimes:

| Runtime | Net return | Sharpe/√bar | Short | Flat | Long |
|---|---:|---:|---:|---:|---:|
| Ray 2.9 | -0.26884% | -0.05481 | 89.856% | 0.671% | 9.473% |
| Ray 2.58 | -0.00523% | -0.01640 | 74.473% | 0.000% | 25.527% |

This is not dataset, panel, model-initialization, or deterministic-evaluation
drift. The initial policy weights and initial fixed-panel evaluation were exact.
The first stochastic rollout diverged:

| Runtime | action 0 / 1 / 2 counts | rollout reward sum |
|---|---:|---:|
| Ray 2.9 | 1,897 / 1,432 / 1,543 | -364.0094 |
| Ray 2.58 | 1,766 / 1,330 / 1,776 | -399.6049 |

Therefore a shared seed does not imply a bit-identical PPO trajectory across
these RLlib versions. The migration gate must compare contracts, deterministic
checkpoint replay, and matched multi-seed outcome distributions. A one-seed
economic delta must not be attributed to the platform.

## Bounded synchronous-PBT probe

Run `v2r0_ray258_parity_20260924_202620_sync_exploit_2x12` used the current
dataset hash `20aac96d…04ce9`, crossed-panel hash `2314a55c…36b9f`, 1 bp
`net_return`, `stream_worker`, `NoFilter`, `market_relative_v1`, synchronous PBT,
and a 24,000-step perturbation cadence.

Both members completed and retained their first report/checkpoint:

- member `76727_00000`: 4,944 steps, -0.09199% net, -0.06033 Sharpe/√bar,
  71.34% flat;
- member `76727_00001`: 4,848 steps, +0.03268% net, -0.00566 Sharpe/√bar,
  74.98% short.

The process was stopped before OOM when free memory reached 1.04 GiB. Both
members were below the 24,000-step burn-in/cadence, so there were zero exploit
arms and no restore receipt to validate. The experiment status is correctly
`incomplete`; its metrics are startup diagnostics only.

Ray 2.58 also emits Windows access-violation stack messages during worker
disconnect/logging. The tested runs still exited and persisted evidence, but the
warning remains an operational risk to monitor during longer R-series runs; it
did not prevent the completed local-worker acceptance probe.

## Completed local-worker acceptance probe

The remote-worker topology was not required for this R0 parity gate and copied
the full dataset into two additional worker processes. With all foreground apps
closed it still fell to 1.23 GiB free after the first update. The replacement
topology used `workers=0` for both Ray versions, changing topology as an explicit
platform factor rather than changing it silently between comparison arms.

Run `v2r0_ray258_parity_20260925_001454_sync_exploit_local_2x12` completed at
approximately 58.6k sampled environment steps per member. It recorded one
genuine exploit, one exact donor/target restore receipt, a post-restore report,
and strict lineage closure. Fresh replay of retained member `63247_00001` at its
best first report was exact (`abs(delta)=0`): -0.013105% net,
-0.02327 Sharpe/sqrt(bar), 48.21% short, 51.79% flat, and 0% long. Both terminal
members were -0.108146% net and -0.02327 Sharpe/sqrt(bar). This is a mechanical
acceptance result, not an economic promotion.

The matched Ray-2.9 local-worker run
`v2r0_ray29_parity_20260925_002225_sync_exploit_local_2x12` also completed with
one exploit and strict lineage closure. Its retained best member replayed
exactly at +0.026606% net but -0.01213 Sharpe/sqrt(bar), with 74.80% short. The
runtime trajectories differ, as expected from the earlier sampling-stream test.

The Ray-2.58 run took approximately 6m13s and bottomed near 2.34 GiB free RAM;
Ray 2.9 took approximately 5m23s and bottomed near 2.09 GiB. On this small
Windows local-worker probe Ray 2.58 used about 0.25 GiB less memory but was about
16% slower. These are topology-specific observations, not general benchmarks.

## Evaluation-dropout correction

The first fresh replay attempt exposed a real source bug rather than a checkpoint
bug: `SolverMLPFiLM` conditioned task-embedding dropout on PyTorch module mode.
RLlib evaluation keeps the module in training mode and supplies
`input_dict["is_training"]`; therefore fixed-panel evaluation was stochastic
whenever `t_keep_p < 1`. The model now honors RLlib's inference signal, and both
runtime acceptance runs replay exactly.

This narrows the meaning of older results. Their checkpoints and provenance
remain valid for the source that produced them, but in-loop evaluation values
from dropout-enabled runs are not deterministic evidence unless a fresh replay
already verified the selected point. The corrected short gate being negative
also removes any basis for claiming that the Ray upgrade itself improved R0.

## Three-seed fixed-hyperparameter bridge

Three one-member, three-update controls held source, dataset, panel, seed,
hyperparameters, costs, and local-worker topology fixed. This is a smoke-scale
distribution check, not enough power for an equivalence claim.

| Runtime | Seed | Best net | Best Sharpe/sqrt(bar) | Terminal net | Terminal Sharpe/sqrt(bar) |
|---|---:|---:|---:|---:|---:|
| Ray 2.58 | 20260924 | +0.05882% | -0.01553 | -0.00628% | -0.02414 |
| Ray 2.58 | 20260925 | +0.06177% | -0.00976 | -0.15454% | -0.03128 |
| Ray 2.58 | 20260926 | +0.00168% | -0.01895 | -0.12769% | -0.01205 |
| Ray 2.9 | 20260924 | -0.01449% | -0.00407 | -0.17064% | -0.02864 |
| Ray 2.9 | 20260925 | -0.02947% | -0.00648 | -0.24119% | -0.04489 |
| Ray 2.9 | 20260926 | -0.07157% | -0.02146 | -0.07157% | -0.02146 |

Both runtimes ended negative for all three seeds, and every best checkpoint had
negative Sharpe. Ray 2.58 showed early positive net-return transients in this
tiny sample; Ray 2.9 did not. That is encouraging only as a search diagnostic.
It neither proves economic uplift nor permits cross-version seed pairing.

## Storage decision

`D:/iTRADER-offload` must not be deleted. It holds approximately 7.46 GiB and is
the authoritative target of `C:/Users/Admin/ray_results` plus 37 experiment
junctions. At least one selected raw-24k policy checkpoint is unique there
(`C663C5E9…EB020`). The drive had about 150.8 GiB free during the audit, so there
is no immediate capacity reason to risk removing it. A later cleanup must copy
cold trees to another drive, verify hashes and fresh replay, retarget junctions,
and only then remove verified old copies.

## Acceptance gate and next actions

1. **Passed:** complete a fresh two-member 12-update Ray 2.58 synchronous probe
   with at least one exact exploit/restore lineage and fresh deterministic replay.
2. **Passed:** run the same local-worker mechanical gate on Ray 2.9 and retain
   runtime-labelled evidence rather than requiring identical PPO trajectories.
3. **Passed at smoke scale:** run three fixed-hyperparameter seeds per runtime.
   Both remain economically negative; do not interpret Ray 2.58's early positive
   transients as promotion.
4. **Accepted:** make the coordinated Ray 2.58 pins the forward dependency
   authority and retain `itrader/compat/ray29/` for historical reproduction.
5. **Next after review:** start R1 with a matched Ray-2.58 R0 control, holding
   dataset-v2 RTH, family 00, crossed panel, costs, proposer mode, PBT cadence,
   local-worker topology, compute, and evidence fixed. Do not start the native
   RLModule/EnvRunner modernization in the same cell.

Compact machine-readable acceptance evidence is retained in
`ray-258-r0-parity-gate-2026-09-24-evidence/platform_acceptance_receipt.json`.
Raw Tune/stable metrics and checkpoints remain on the authoritative
`D:/iTRADER-offload` junction target; the compact three-seed summaries remain in
the ignored experiment-output directories named in this audit.

The roadmap remains:

```text
frozen R0
  -> Ray 2.58 R0 exploit/restore + multi-seed parity gate
  -> R1 reset-window representation control
  -> R2 stateful recurrence
  -> causal C-series
  -> System-One/CLM/Jev S-series
```
