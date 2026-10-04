# R0 + R4-v1 causal-cost and PBT forensic review

Date: 2026-09-29
Status: completed analysis; mechanics pass, curriculum effect retained, no economic checkpoint promoted

## Executive decision

The completed search is a meaningful positive result for **training behavior**, but not yet for
conditional trading performance.

- Causal `0 -> 0.25 -> 0.5 -> 1 bp` training prevented the immediate all-flat basin seen when
  exact 1 bp was active from update zero.
- Jointly positive held-out net-return/Sharpe reports increased from `0/40` at 0 bp to `8/40`,
  `33/40`, and `40/40` at the subsequent cost stages.
- That improvement was inseparable from directional collapse: all 81 jointly positive reports
  were at least 98.014% short, and 73 were exactly short-only.
- Every terminal causal member equalled the matched short-only control episode for episode.
  Every terminal exact-cost member equalled flat.
- PBT searched and restored correctly. The remaining failure is the objective/representation
  basin and tied-basin selection, not another checkpoint-provenance failure.

Do **not** run another full R0 + R4-v1 PBT population against the same January panel. Preserve
causal cost staging as an explicit matched factor for R-series attribution and move to the clean
R1 boundary after the cheap transient/logit diagnostic documented below. Keep R0 immutable and
keep `R0 + R4-v1` as the architecture-qualified feed-forward baseline.

## Evidence contract

The paired completed experiments were:

- causal curriculum:
  `phase_a_r0_plus_r4_v1_actor_robustness_pbt_4x40_20260929_065425_both_causal_cost`;
- exact 1 bp from update zero:
  `phase_a_r0_plus_r4_v1_actor_robustness_pbt_4x40_20260929_074623_both_exact_1bp`.

Both used dataset-v2 RTH, Family 00, random/generated tasks, `stream_worker` training,
`R0 + R4-v1` position/remaining-time state, Ray 2.58 local-worker topology, the same four seeds,
the same January 2026 crossed 8-anchor x 4-horizon development panel, exact 1 bp evaluation,
four members, 40 updates, and approximately 192k target-owned sampled environment steps per
member.

The two arms have 320 total report rows and 320 indexed stable checkpoints. Each arm completed
six exploit events, 18 exact restore receipts, and six verified generation-bound lineages with
no stale, missing, or pending provenance. This result is not contaminated by the earlier PBT
restore defects.

## Curve anatomy by cost stage

The table groups each report by the cost actually used for the just-completed PPO update. The
held-out evaluation cost remained exactly 1 bp at every stage.

| Training cost | Step span | Reports | Positive net + Sharpe | Mean net | Mean Sharpe / sqrt(bar) | Mean turnover | Short / flat / long |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 bp | 4,808--48,896 | 40 | 0 | -0.295182% | -0.033379 | 10.681 | 7.41% / 41.34% / 51.25% |
| 0.25 bp | 53,464--98,144 | 40 | 8 | -0.066297% | -0.037054 | 16.798 | 37.16% / 46.74% / 16.10% |
| 0.5 bp | 102,040--146,944 | 40 | 33 | +0.301633% | +0.000512 | 9.075 | 88.39% / 8.49% / 3.12% |
| 1 bp | 150,600--195,560 | 40 | 40 | +0.441611% | +0.024455 | 2.000 | 100% / 0% / 0% |

This is stronger than saying the run merely became positive. Cost staging kept the actor out of
the immediate flat solution long enough for PPO/PBT to retain an active direction. It also did
not produce a balanced or market-conditional action rule. The move in the population mean is a
move toward the fixed panel's short-only return.

The final 1 bp transition did not itself cause the directional collapse. Member `00002` first
became all-short at 82,672 steps; copied descendants followed by 103,072 and 127,592 steps.
Member `00001` retained only one or two non-short actions through 131,336. The population was
therefore effectively short-only before the 1 bp stage began at the next update after 144k.

Current matched plots remain authoritative for the raw trajectories:

- [net-return trajectories](solver-state-r4-v1-actor-robustness-2026-09-29-evidence/matched_cost_comparison/ablation_compare.png)
- [Sharpe trajectories](solver-state-r4-v1-actor-robustness-2026-09-29-evidence/matched_cost_comparison/ablation_compare_validation_sharpe.png)
- [action-fraction trajectories](solver-state-r4-v1-actor-robustness-2026-09-29-evidence/matched_cost_comparison/ablation_compare_action_fractions.png)
- [PPO diagnostics](solver-state-r4-v1-actor-robustness-2026-09-29-evidence/matched_cost_comparison/ablation_compare_diagnostics.png)

## PBT exploit and mutation anatomy

PBT did not remain inert. Its useful events were concentrated in the middle of the run:

| Approximate boundary | Target <- donor | Before | After / interpretation |
|---:|---|---|---|
| 48.8k | `00003 <- 00000` | both economically weak | target remained negative/mostly flat; no material gain |
| 72.9k | `00001 <- 00002` | target 50% flat / 50% long, -0.2498% net | next report 98.97% short, +0.4021% net, 0.01865 Sharpe |
| 98.1k | `00000 <- 00002` | target 72% flat, -0.5002% net | next report exactly short-only, +0.4416% / 0.02445 |
| 122.7k | `00003 <- 00002` | target -0.6045% net / -0.1619 Sharpe | next report exactly short-only |
| 151.6k and 175.8k | tied short lineages | identical short basin | configuration mutation without new economics |

The strongest causal inference available from the lineage is narrow but useful: member `00002`
reached the short basin without a restore, and PBT then propagated it. The basin is therefore not
an artifact of donor contamination or a missing exploit. Once all members tie at flat or
short-only, however, Ray's quantile tie-breaking can continue to emit technically valid exploit
events that add no information. Future PBT summaries should call this **search exhaustion**, add
an all-equal/all-ineligible stop, and not count tied-basin mutations as economic progress.

Lower learning rates dominated. The useful unexploited member-`00002` recipe was approximately:

- learning rate `7.1517e-6`;
- entropy schedule `[[0, 0.005], [96k, 0.002], [240k, 0.0005]]`;
- gamma `0.99`, GAE lambda `0.90`, clip `0.1`;
- minibatch `128`, 20 SGD passes;
- value clip `20`, value-loss coefficient `2.0`.

Freeze this recipe for the first matched R0 / clean R1 / R1 + R4-v1 representation cells.
Do not let a second PBT search confound the representation attribution.

## Exact-cost basin

Exact 1 bp from update zero was not simply a weaker version of the causal result. Members became
pure flat at 9,880 / 19,472 / 24,280 / 24,344 steps, all before the first exploit boundary. Its
first ten-update population mean was -0.050374% net, -0.007476 Sharpe, turnover 2.261, and
90.10% flat; every later report was exactly zero and flat. Entropy fell through approximately
`0.946 -> 0.405 -> 0.180 -> 0.096` by ten-update quarter.

The six later exploit events therefore mutated an all-flat tie. This is a real objective basin,
not failed PBT machinery. Exact 1 bp from update zero remains rejected as the default training
curriculum for this baseline.

One reporting defect was found during this analysis. Static-cost runs omitted
`training/optimization_cost_bps` from report rows, and the aggregate CSV represented the missing
field as zero. The resolved exact-arm training and evaluation configs both contain
`net_return.cost_bps = 1.0`; the reward behavior and replay are consistent with exact-cost
training. A test-first patch now resolves the report value from either the schedule or the static
training task configuration. This was a provenance-label defect, not a reward-delivery defect,
and does not change the completed economic result.

## Candidate checkpoint review

No checkpoint is economically promotable.

- 81 of 160 causal reports had positive net and Sharpe.
- 73 were exactly short-only.
- The other eight were between 98.014% and 99.943% short.
- No jointly positive report had a short fraction below 98%.
- Only member `00001` at 107,056 steps exceeded short-only on both aggregate metrics, and only
  marginally: +0.444556% net / 0.024954 Sharpe / 99.943% short.
- Fresh replay changed one near-tie flat action to short and moved the search score by
  `7.4030326e-06`, so the checkpoint fails the strict exact-parity gate.

The earlier member-`00002` checkpoint at 77,832 steps was retained and replayed during this
forensic pass. It reported +0.402257% net / 0.018752 Sharpe / 98.999% short in-run. Fresh replay
produced +0.402997% / 0.018824 / 99.023% short and the same `7.4030326e-06` score delta. The
repeated one-action delta confirms a decision boundary problem rather than hidden economic edge.

Its new per-step state diagnostic is especially informative:

| Diagnostic | Value |
|---|---:|
| greedy short fraction | 99.358% |
| mean action probabilities, short / flat / long | 40.18% / 27.30% / 32.52% |
| mean categorical entropy | 1.08234 (`ln(3) = 1.09861`) |
| top-two logit margin p1 / p5 / median | 0.00592 / 0.02057 / 0.19693 |
| steps with top-two logit margin < 0.01 | 1.755% |
| masked-state counterfactual argmax changes | 2.440% |
| mean explained variance at its report | weak; not a critic-quality winner |

The deterministic evaluator looks collapsed while the stochastic policy is still broad. Tiny
argmax margins on a small fraction of states are sufficient to change an action across fresh
processes. This explains why a one- or two-action aggregate uplift is not robust enough to call
exploitable.

Evidence:

- [77,832 replay report](solver-state-r4-v1-causal-forensics-2026-09-29-evidence/replay_77832/)
- [77,832 state/logit diagnostic](solver-state-r4-v1-causal-forensics-2026-09-29-evidence/state_diag_77832/)

## What “regime change” can and cannot mean here

The January evaluation panel is fixed at every training update. Its market regime did not change
as the curve advanced. Curve changes therefore measure changes in the learned policy under PPO,
cost stages, mutations, and copied lineages—not movement through different held-out regimes.

The training stream does move through symbols and market states, but report rows do not retain a
training-block symbol or causal regime summary. A statement that a particular training regime
caused a later jump is currently unscoreable. Add per-report training-block symbol/session and
causal, train-fitted trailing-regime attribution before making that claim. Future-derived
`regime_tags` remain ex-post reporting only.

The fixed panel itself is mixed but magnitude-short-biased. Short-only was positive on 19/32
episodes and negative on 13/32:

| Anchor symbol | Mean short-only net | Positive horizons |
|---|---:|---:|
| AMZN | +0.6744% | 4/4 |
| AMD | +0.4319% | 4/4 |
| AVGO | +2.1405% | 4/4 |
| NVDA | +1.1124% | 4/4 |
| IWM | -0.3884% | 0/4 |
| MSFT | -0.4514% | 0/4 |
| QQQ | +0.0396% | 2/4 |
| SPY | -0.0261% | 1/4 |

A hindsight per-episode constant-direction oracle would average +0.66035%, versus +0.44161%
for short-only. The oracle is not a causal target, but it quantifies the unused room for
market-conditional direction selection. The population captured the panel aggregate while
losing the bullish anchors.

## PPO and critic interpretation

There was no PPO blow-up. Approximate KL remained small. Causal learner entropy stayed high
(`1.0849 -> 1.0682 -> 1.0267 -> 0.8709` by cost stage) while greedy evaluation became short-only.
Mean explained variance improved only `0.0065 -> 0.0126 -> 0.0278 -> 0.0376`, which is still weak
critic fit. The exact arm's falling value loss mostly reflects learning the trivial zero-return
flat basin, not economic understanding.

`clip_fraction` is NaN in all 320 reports and absent from the captured learner-stat keys. This is
a real diagnostics gap to close before R1: approximate KL alone is not enough to characterize
PPO update pressure.

## Historical direction of travel

Raw net return and Sharpe cannot be connected as one model-improvement curve across these runs:
dataset, evaluation panel, panel direction, runtime, cost curriculum, compute budget, and evidence
quality changed. The defensible autoresearch view is matched-control margin, with architecture
factor deltas kept separate.

| Evidence cell | Net margin over strongest matched deterministic control | Sharpe margin | Qualification |
|---|---:|---:|---|
| v1 discovery `00001` | +0.16980 pp | +0.01040 | exact replay; shared ancestry; independent replications failed |
| v1 discovery `00003` | +0.18610 pp | +0.00993 | descendant of `00001` |
| v1 raw-24k transient | +0.22392 pp | +0.01391 | exact replay; transient; block interval crosses zero |
| v2 Gate D transient | +0.02146 pp | +0.00819 | three positive reports, then regression |
| v2 crossed staged R0 | 0 pp | -0.02385 | exact short-only terminal |
| December R0 + R4-v1 mean | -0.2111 pp | -0.0573 | +0.0885 pp paired net uplift over masked R0 |
| January causal terminal | 0 pp | 0 | exact short-only tie |
| January exact terminal | 0 pp | 0 | exact flat tie |

The monotone improvement is in **evidence quality**: arithmetic return; deterministic panels;
fresh replay; target-owned PBT clocks; generation-bound lineage; corrected OHLCV; crossed
start/horizon panels; block uncertainty; and agent-visible control state. Absolute alpha has not
improved monotonically. The durable
[interactive dashboard](solver-state-r4-v1-causal-forensics-2026-09-29-evidence/dashboard.html)
shows cost-stage net return, Sharpe, turnover, action fractions, PPO/critic diagnostics, and the
control-relative historical view without drawing a misleading raw connected line. Labels that
name December 2025 or January 2026 refer to market evaluation panels, not experiment run dates;
the R4-v1 experiments ran in September 2026.

## Decision: one more R0 + R4-v1 run or R1?

Proceed to R1. Another same-panel fixed-schedule PBT population would optimize a known shortcut
and make the January development panel even more selection-contaminated. Extending to 384k,
changing the panel, adding an independent population, and changing the cost timing would no
longer be a narrow replication.

Before R1 training, complete one bounded, no-PPO diagnostic:

1. replay unexploited member `00002` checkpoints around 77.8k, 92.4k, 102.0k, 107.0k, and
   terminal exposure on a predeclared non-overlapping chronological panel;
2. persist per-anchor net/actions and normalized top-two logit margins;
3. require two-process action and reward identity;
4. reject any apparent directional-control uplift whose score-bearing decisions sit inside the
   measured cross-process logit-drift margin.

### G0 closure result — completed 2026-10-03

That diagnostic is complete. The panel was predeclared before outcome inspection:
eight November-2025 RTH anchors crossed with horizons 96, 128, 160, and 200. It
is cardinality-matched to the January source panel, belongs to the validation
partition, and has no overlapping observation/outcome interval with the source
panel. It is development evidence, not a final-test opening and not a new
selection surface.

| Steps | Training cost | January source net | Fresh net | Fresh Sharpe / √bar | Fresh short / flat / long | p1 normalized margin |
|---:|---:|---:|---:|---:|---:|---:|
| 77,832 | 0.25 bp | +0.40226% | -0.34567% | -0.00803 | 99.77 / 0.00 / 0.23% | 0.00307 |
| 92,384 | 0.25 bp | +0.40477% | -0.31781% | -0.00527 | 99.71 / 0.00 / 0.29% | 0.00401 |
| 102,040 | 0.5 bp | +0.20748% | -0.22822% | -0.00620 | 75.07 / 0.00 / 24.93% | 0.00784 |
| 106,968 | 0.5 bp | +0.44161% | -0.33873% | -0.00681 | 100.00 / 0.00 / 0.00% | 0.02354 |
| 194,368 | 1 bp | +0.44161% | -0.33873% | -0.00681 | 100.00 / 0.00 / 0.00% | 0.44358 |

The normalized margin is the first percentile of the top-two softmax probability
gap; raw logit margins are persisted alongside it. Every episode/reward JSONL
and every per-step action/logit JSONL matched byte-for-byte across two clean
Ray-2.58 processes. The original 77,832-step one-action mismatch therefore does
not reproduce on this fresh panel.

The 102,040-step checkpoint is the least negative and the only materially mixed
policy, but its approximately 75/0/25 episode-weighted action mix is almost
identical at every anchor. GOOGL, META, and MSFT were positive while five anchors
were negative and NVDA dominated downside. The lack of anchor-conditioned action
change means this is not portable conditional edge. The 106,968-step and terminal
checkpoints have identical pure-short actions and economics even though their
logit margins continue to separate.

G0 therefore passes determinism/provenance and fails economic promotion. It
closes R0 + R4-v1 without invalidating the architectural qualification or the
evidence that causal cost staging delayed the flat basin. Causal staging remains
an explicit matched R-series curriculum factor; it is not silently promoted to
the R1 default. Clip fraction remains unavailable in these legacy learner rows
and is recorded as `not scored`, not zero.

Post-review provenance qualification (2026-10-03): the repository artifacts do
prove exact byte parity for the paired episode and state-step files, and the
receipt metrics/panel hashes agree with the closure summary. The historical
receipts do not record distinct producer/run identifiers, however, so they do
not independently authenticate that the two directories came from separate
processes. The canonical experiment browser therefore reports artifact-pair
parity and leaves independent-process provenance unknown. This does not change
the negative economic result; future replay receipts must stamp producer
identity at creation time before claiming an independently verified process
replication.

Artifacts:

- [G0 summary](solver-state-r4-v1-causal-forensics-2026-09-29-evidence/g0_fresh_panel/g0_closure_summary.json)
- [G0 fresh panel](../../../itrader/scripts/experiments/configs/evaluation_panels/dataset_v2_r4_g0_nov2025_crossed_8x4_20261003.json)
- [G0 evidence directory](solver-state-r4-v1-causal-forensics-2026-09-29-evidence/g0_fresh_panel/)

Then run the frozen member-`00002` recipe as a matched representation study:

```text
Ray-2.58 R0 control
clean R1
R1 + R4-v1
```

Use the same dataset, panel, seeds, tasks, proposer, compute, evaluation cost, and one explicitly
named curriculum factor across all cells. Causal staging is now a standard **ablation axis**, not
an unqualified new default. PBT follows only after the fixed-recipe representation attribution.

If a final feed-forward training probe is ever mandated, it must be an edge-gated no-cost
discovery experiment, not another monotonic schedule: cost cannot increase until a checkpoint
beats the strongest directional control on net and Sharpe for three consecutive exact-replay
reports. No gate by 384k is a hard stop. That design requires a new conditional-schedule seam and
is lower priority than R1.

## Keep, fix, and defer

| Item | Decision |
|---|---|
| R0 + R4-v1 control state | keep as architecture-qualified FF baseline |
| causal cost staging | keep as an explicit matched R-series factor |
| exact 1 bp from update zero | reject as default curriculum for this baseline |
| current January checkpoints | forensic only; none promoted |
| lower-LR member-`00002` recipe | freeze for fixed-recipe R1 attribution |
| tied-basin PBT exploits | report as search exhaustion; add early stop |
| static optimization-cost reporting | fixed test-first for future runs |
| top-two logit margins | add to candidate/per-report diagnostics |
| training-block causal attribution | add before regime-causality claims |
| `clip_fraction` | repair/capture before R1 performance interpretation |
| R4-v2 late fusion | defer; do not fold into R1 |
| causal C-series | defer until R-series representation/lifecycle closure |

## Validation performed in this forensic pass

- Added exact progress-checkpoint selection to deterministic replay; focused replay tests pass.
- Replayed member `00002` at 77,832 steps and reproduced the same strict-parity failure magnitude.
- Added arbitrary progress selection to state diagnostics and generated 4,672 per-step records.
- Added a red/green regression for static optimization-cost reporting; schedule/static focused
  tests pass (`6 passed`).
- No training run, commit, push, staging operation, deletion, or final-test access occurred.

Primary upstream evidence remains the
[completed actor-robustness audit](solver-state-r4-v1-actor-robustness-2026-09-29.md) and its
[terminal mechanical gate](solver-state-r4-v1-actor-robustness-2026-09-29-evidence/terminal_gate_summary.json).
