# R0 + R4-v1 actor-robustness audit

Date: 2026-09-29
Status: complete; mechanically valid search, no economic promotion

## Decision under test

The accepted feed-forward baseline remains `R0 + R4-v1`: the MLP-FiLM market/task path plus the
minimal agent-visible position and remaining-horizon state residual. State-free R0 remains the
immutable parity control. The combined state arm is architecture-qualified but economically
unpromoted; this audit tests whether its actor is more robust under lower learning rates,
bounded entropy pressure, and a causal transaction-cost curriculum.

This is not an R4-v2, R1, task-family, proposer, data, or reward-function experiment. No
market/task x state interaction is added. Family 00, the random/generated proposer, dataset-v2
RTH partitions, the 60x5 market input, semantic-v2 task encoding, exact 1 bp evidence, and the
Ray 2.58 legacy PPO stack remain fixed.

## Frozen experiment contract

The canonical loadout is
[`phase_a_r0_plus_r4_v1_actor_robustness_pbt_4x40_20260929.json`](../../../itrader/scripts/experiments/configs/phase_a_r0_plus_r4_v1_actor_robustness_pbt_4x40_20260929.json).
Its primary arm uses:

- combined `position + remaining_horizon_fraction` state with `zero_v1` residual initialization;
- four synchronous PBT members, two resident local-worker trials at a time, and a 40-update /
  approximately 192k-step cap per member;
- `nonflat_net` donor ranking with the absolute economic gate kept separate;
- learning rate bounded to `5e-6 .. 6e-5`;
- the four frozen entropy schedules already covered by the PBT contract;
- causal cost stages `0 -> 0.25 -> 0.5 -> 1 bp` at target-owned sampled-environment steps
  `0 / 48k / 96k / 144k`;
- a five-update PBT/checkpoint cadence, ten-update burn-in, retained-best checkpoints, and exact
  generation-bound provenance;
- a matched exact-1-bp arm that may run only after the causal-cost arm is reviewed.

The masked, position-only, and time-only cells from the completed December 2025 4x3x40
replication remain the attribution reference. A winning combined-state recipe must later be
frozen and replayed against those arms; spending four independent PBT populations on the four
state masks would confound state attribution with search luck.

## New chronological selection block

The PBT search does not select again on the December 2025 attribution panel. The predeclared
January 2026 panel is
[`dataset_v2_r4_actor_jan2026_crossed_8x4_20260929.json`](../../../itrader/scripts/experiments/configs/evaluation_panels/dataset_v2_r4_actor_jan2026_crossed_8x4_20260929.json).
It contains eight outcome-uninspected RTH anchors from January 2, 5, and 6, crossed with the
frozen 96/128/160/200 horizons. All 32 tasks were validated against the symbol-safe validation
partition before launch. It is a development selection block, not the sealed final test.

## Gates and bounded accounting

The 96k-step review is the continuation gate. Mechanical continuation requires all four members,
finite/non-empty reports, at least two genuine exploit attempts after burn-in, exact target
receipts, monotonic target-owned clocks, and no stale or pending lineage. The host must retain at
least 2 GiB free RAM.

Economic promotion remains stricter than PBT donor selection:

1. positive 1 bp net return and positive Sharpe per square-root bar;
2. no effectively all-flat policy (`flat < 98%`) and explicit turnover reporting;
3. sustained passing reports rather than one transient point;
4. exact fresh-process checkpoint replay;
5. comparison with flat, long-only, short-only, random, and untrained controls;
6. paired chronological-block uncertainty and 0.5/1/2 bp replay;
7. a frozen masked/position/time/both replication and an independent population seed before
   economic promotion.

`Sharpe / sqrt(bar) >= 0.02` and a `+0.005` margin over the strongest directional control remain
high-signal challenger targets, not substitutes for the full promotion gate.

## Run ledger

| Run | Status | Keep/reject reason |
|---|---|---|
| `..._062326_both_causal_cost` | invalidated at update zero | Ray 2.58 `EnvRunnerGroup` has no `foreach_worker`; causal cost delivery had not been exercised by the prior fixed-cost platform gate. No PPO update completed. |
| `..._062604_both_causal_cost` | invalidated at the 120k boundary | It exposed a contradiction between exact post-sync checkpoint retention and restore validation. The scheduler correctly selected an immutable post-sync sidecar, while the validator incorrectly required `reported_to_tune=true`. The evidence before the failure remains diagnostic only. |
| `..._065425_both_causal_cost` | completed; mechanically valid; economically rejected | Test-first restore validation accepts either a Tune-reported checkpoint or an exact post-sync retained sidecar. All four members exceeded 192k target-owned steps; six exploits, 18 exact receipts, and six verified lineages closed without stale or invalid provenance. All terminal members collapsed to the matched short-only policy. |
| `..._074623_both_exact_1bp` | completed; mechanically valid; economically rejected | Identical Python fingerprint, seeds, panel, topology, and budget. All four members exceeded 192k target-owned steps; six exploits, 18 exact receipts, and six verified lineages closed without stale or invalid provenance. Every terminal policy was exactly flat. |

## Causal-cost result

The repaired causal-cost arm passed its 120k, 144k, and terminal gates. At the terminal gate all
four members reported the same held-out result:

| Metric | Terminal value |
|---|---:|
| 1 bp net return | `+0.441611%` |
| Sharpe / sqrt(bar) | `0.024455` |
| turnover | `2.0` |
| short / flat / long | `100% / 0% / 0%` |

This is a valid PBT search result but not solver promotion. The identical January-panel controls
were flat `0%`, long-only `-0.481611%`, short-only `+0.441611%`, random `-1.385487%`, and
untrained `-0.127701%`. The learned population therefore tied the strongest directional control
exactly; its apparent edge is the panel's short direction, not demonstrated conditional trading.

PBT itself did search. Six exploit decisions converged the population onto the best available
short lineage while lower learning rates and entropy schedules continued to mutate. The causal
cost stages were delivered at target-owned `0 / 48k / 96k / 144k` steps, and all members
continued after the final exact-1-bp transition. This keeps causal cost staging as a mechanically
valid curriculum factor, but does not establish economic uplift.

Across members, positive-net reports numbered `20 / 25 / 29 / 15`; the first occurred at
approximately `103k / 78k / 58k / 128k` target-owned steps. The strongest recorded report was
member `00001` at 107,056 steps: `+0.444556%` net and `0.024954` Sharpe with action fractions
`99.9430% short / 0.0244% flat / 0.0326% long`. That apparent increment over short-only was not
checkpoint-stable and was far below the declared `+0.005` Sharpe margin.

## Matched exact-1-bp result

The exact-cost arm completed the same four-member, 40-update budget on the same Python source
fingerprint. Its terminal lineage gate passed with four complete members, six genuine exploits,
18 exact restore receipts, six verified generation chains, and no missing, invalid, stale, or
pending provenance. Every terminal checkpoint reported and freshly replayed:

| Metric | Terminal value |
|---|---:|
| 1 bp net return | `0.000000%` |
| Sharpe / sqrt(bar) | `0.000000` |
| turnover | `0.0` |
| short / flat / long | `0% / 100% / 0%` |

The only positive report in the entire exact-cost population occurred in member `00001` at
9,776 steps: `+0.003367%` net and `0.002014` Sharpe while `99.9844%` flat. It failed the explicit
near-flat gate and never recurred. PBT continued to perturb learning rate, entropy schedule,
discount, GAE lambda, minibatch size, SGD passes, clipping, and value-loss settings; it did not
recover a non-flat policy. The result therefore separates a valid search from an economic
failure: exact 1 bp from update zero makes inactivity the stable basin for this search space.

Because every trained exact-cost report was ineligible, its retained forensic `best` pointers
refer to update-zero checkpoints. All four replayed exactly but remained economically negative.
The four terminal checkpoints also replayed exactly at the all-flat zero result. There is no
hidden trained winner behind checkpoint selection.

## State-arm attribution carried forward

The December fixed-recipe replication remains the clean state attribution study; this PBT search
does not replace it with four independently searched populations. Its terminal means were:

| Arm | Mean 1 bp net | Mean Sharpe / sqrt(bar) | Interpretation |
|---|---:|---:|---|
| masked | `-0.1243%` | `-0.0370` | immutable R0-like attribution control |
| position | `-0.0532%` | `-0.0262` | consistent relative net improvement, no absolute promotion |
| time | `-0.0433%` | `-0.0201` | strongest mean gross return and critic fit |
| position + time | `-0.0359%` | `-0.0164` | architecture-qualified combined baseline; primary arm here |

The combined branch remains the correct forward feed-forward control-state representation. The
new search changes the actor conclusion, not the observability conclusion: lower learning rates,
entropy pressure, and cost curriculum did not produce conditional economic edge.

## Retained-best replay

Three of four retained-best checkpoints reproduced their selected score exactly in fresh
processes. Member `00001` was rejected by the strict replay gate. Its recorded best report was
almost short-only and contained one flat and one long action; native RLlib and manual replay
consistently changed the flat action to short while preserving the long action. The delta was
`7.4030326e-06` in search score (`0.0007403` percentage points of net return).

This is not evidence of meaningful edge. It is evidence that a decision near an action-logit tie
was not checkpoint-stable. Exact parity remains the gate; future candidate reports must include
the minimum top-two action-logit margin so restore-sensitive ties are visible before promotion.

## Harness defect found by controls

The first control-panel invocation completed the flat evaluation but the outer collector rejected
its artifact. Baseline runs namespace stable metrics by experiment, while the outer collector
only understood the legacy flat stable directory. A red regression reproduced the mismatch and a
narrow compatibility repair accepts the experiment-scoped layout while retaining the legacy
fallback. The repaired control panel then completed all five arms. This reporting-only patch is
kept outside the source-matched PBT pair until the exact-1-bp comparator closes.

The failed run is retained as provenance and cannot contribute economic evidence. The repair is
narrow: `ITraderEnv` still owns mutable training task overrides, the evaluation task panel remains
unchanged, and no solver/reward/task semantics changed.

After the matched PBT pair closed, the tested collector repair was restored to the working tree.
It now resolves experiment-scoped baseline metrics first and retains the legacy flat-layout
fallback. The source fingerprints of the two PBT arms therefore remain identical.

## Controls, uncertainty, and cost-sensitivity decision

The causal arm's terminal action receipt is exactly the short-only control, episode for episode;
its paired chronological-block delta is therefore identically zero. The exact-cost terminal
receipt is exactly flat, so its paired delta to the flat control is also identically zero. These
are stronger rejection statements than overlapping confidence intervals.

No checkpoint survives all of non-flatness, directional-control uplift, and exact replay. The
predeclared 0.5/1/2 bp replay and independent-population replication are therefore not spent on a
non-candidate. They remain mandatory for the first future checkpoint that clears those earlier
gates; omitting them here is a bounded-search stop, not a waived promotion requirement.

## Autoresearch keep/reject ledger

| Factor or seam | Decision | Evidence |
|---|---|---|
| R0 + R4-v1 combined state | keep as architecture baseline | required control-state completion and prior paired attribution |
| lower learning-rate search | keep available | exercised validly; insufficient by itself |
| entropy-pressure mutation | keep available | exercised validly; did not prevent either basin |
| causal cost staging | keep as a curriculum factor | preserved activity through 1 bp, unlike exact-cost-from-start |
| exact 1 bp from update zero | reject as the default curriculum for this baseline | all four terminal policies became flat |
| causal-arm economic candidate | reject | equals short-only and one apparent incremental best fails replay parity |
| exact-cost economic candidate | reject | flat, zero turnover, no eligible trained checkpoint |
| new masked/position/time PBT populations | reject for this goal | would confound state attribution with independent search luck |
| independent population and 0.5/1/2 bp stress | stop before launch | no candidate cleared the earlier gates |
| R4-v2 or additional solver-state features | defer | outside the frozen factor boundary |

## Validation record

- New-panel schema and all 32 symbol/horizon eligibility checks: pass.
- Frozen-loadout contract: red before files existed, then green.
- Focused run/config/PBT/environment/checkpoint suite before launch: 155 passed.
- Reward-override compatibility regression: red on `foreach_worker`, then green on the group-level
  environment seam; neighboring training/PBT/environment suite: 136 passed.
- Experiment-scoped baseline collector regression: red on the legacy-only lookup, then green on
  scoped-first/legacy-fallback lookup; focused experiment/training/PBT tests: 150 passed.
- Both terminal PBT gates: pass with four members, six exploits, 18 exact receipts, and six
  verified restore lineages each.
- Exact-cost retained-best and terminal replay: four of four exact in each selection mode.
- Repository healthcheck: Ray 2.58.0, Gymnasium 1.2.2, PyArrow 17.0.0, PyTorch 2.3.1 CPU.
- GitNexus: stale relative to the dirty experimental tree and used only as advisory evidence.

## Final decision and next boundary

This goal closes without an economic winner. PBT, sampled-step scheduling, causal reward
delivery, retention, lineage, replay, and aggregate reporting are now mechanically valid. The
negative result is model/search behavior: the causal arm finds the January panel's short-only
direction, while exact cost from update zero finds flat.

Preserve R0 unchanged for historical parity and preserve `R0 + R4-v1` as the
architecture-qualified feed-forward baseline. Do not claim that R4-v1 passed the economic gate,
that the market is fully observed, or that causal staging created conditional alpha. After review
of this dirty tree, the next representation boundary is the predeclared clean `R1` versus
`R1 + R4-v1` comparison, with a matched Ray-2.58 R0 control and the same attribution rules.
R4-v2 remains deferred.

Key evidence:

- [checkpoint-level causal/PBT forensics](solver-state-r4-v1-causal-forensics-2026-09-29.md)

- [terminal mechanical gates](solver-state-r4-v1-actor-robustness-2026-09-29-evidence/terminal_gate_summary.json)
- [matched causal-versus-exact comparison](solver-state-r4-v1-actor-robustness-2026-09-29-evidence/matched_cost_comparison/summary.json)
- [net-return trajectories](solver-state-r4-v1-actor-robustness-2026-09-29-evidence/matched_cost_comparison/ablation_compare.png)
- [Sharpe trajectories](solver-state-r4-v1-actor-robustness-2026-09-29-evidence/matched_cost_comparison/ablation_compare_validation_sharpe.png)
- [action fractions](solver-state-r4-v1-actor-robustness-2026-09-29-evidence/matched_cost_comparison/ablation_compare_action_fractions.png)
- [PPO diagnostics](solver-state-r4-v1-actor-robustness-2026-09-29-evidence/matched_cost_comparison/ablation_compare_diagnostics.png)
- [causal retained-best replay](solver-state-r4-v1-actor-robustness-2026-09-29-evidence/causal_cost_pbt/replay_best/)
- [exact-cost retained-best replay](solver-state-r4-v1-actor-robustness-2026-09-29-evidence/exact_1bp_pbt/replay_best/)
- [exact-cost terminal replay](solver-state-r4-v1-actor-robustness-2026-09-29-evidence/exact_1bp_pbt/replay_terminal/)
