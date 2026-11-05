# Phase A MLP-FiLM Economic-Performance Autoresearch

**Date:** 2026-09-19
**Branch:** `agent/Watson-dev`
**Scope:** Phase A `SolverMLPFiLM`, PPO/GAE, synchronous PBT, environment-step
scheduling, frozen validation controls, checkpoint replay, and economic evidence.
Recurrent/LSTM solvers, live execution, broker semantics, and deterministic risk
shells remain out of scope.

## Decision standard

This study separates three claims that were previously easy to conflate:

1. **PPO activity:** finite gradients, nonzero policy updates, value learning, and
   policy changes.
2. **Candidate economic learning:** sustained positive 1 bp net return and
   Sharpe on a frozen validation panel, with non-flat behavior and a meaningful
   margin over matched controls.
3. **Validated economic performance:** independent population replication,
   exact fresh-process replay, positive 2 bp repricing, block-aware uncertainty,
   and an untouched final-test result after selection is frozen.

Only the first claim was established before this loop. A Phase A checkpoint is
now called a **candidate** when it has at least five consecutive evaluation
reports with positive 1 bp net return and Sharpe and beats the strongest matched
directional control on both metrics. The aspirational **profound** gate is ten
such reports, fresh replay, positive 2 bp net return, a distinct policy hash and
lineage, reproduction under an independent PBT population seed, and supportive
block evidence. Directional action bias is diagnostic rather than disqualifying:
only an effectively all-flat policy is an automatic action-based rejection.
Regime-conditioned behavior is evaluated separately.

## Architecture and measurement guardrails

- `TaskSpec` remains the task contract.
- `ITraderEnv` still owns task-conditioned reset, observation assembly, and
  reward context. It now exposes a narrow normalized override method for future
  resets; it is not a backtester or execution shell.
- `RewardEngine` remains the only reward calculator.
- The solver learns policy/value behavior. No broker, risk, or live-execution
  semantics moved into the solver or environment.
- Training uses `stream_worker`, random/generated tasks, and no LLM backend.
- Evaluation uses 32 deterministic validation episodes balanced across horizons
  96, 128, 160, and 200. The final chronological test remains sealed.
- Economic evidence is always reported independently at 1 bp per turnover unit,
  even when the optimization objective uses 0 or fractional basis points.
- PBT burn-in, perturbation, checkpoints, and curricula use target-owned
  cumulative sampled environment steps. Restored RLlib iteration numbers do not
  control scheduling.

## New seams implemented test-first

The loop added eight bounded seams:

1. A strict `step:value` optimization-cost schedule, resolved from monotonic
   target-owned environment steps.
2. Training-worker reward overrides that take effect on subsequent resets while
   leaving the frozen evaluation task panel unchanged.
3. A separate `pbt_population_seed`, allowing new initialization/search
   randomness without changing validation tasks or starts.
4. Durable promotion diagnostics: trailing passing-report counts,
   baseline-relative Sharpe, a near-all-flat veto, 0.5/1/2 bp repricing, paired
   block intervals, and ex-post bullish/bearish/sideways action summaries.
5. An opt-in lexicographic PBT donor score: positive-net, positive-Sharpe,
   non-all-flat
   members form the eligible tier and are ranked by net return; when no member
   is eligible, ordinary net-return ordering is preserved. Directional action
   bias is never penalized by this score.
6. A strict fail-closed donor mode that refuses to overwrite a lineage when the
   upper quantile contains no promotion-eligible checkpoint.
7. Source-tree fingerprints and repository-owned, fsynced exploit/progress/
   checkpoint evidence so dirty-code runs and zero-byte Tune state are visible.
8. Explicit checkpoint recovery inputs for target-owned progress, frozen-panel
   seed, and checkpoint-consistent evaluation, plus a Ray-2.9-local synchronous
   scheduler adapter for stranded or orphan-staged paused trials.

The main configs are:

- [`phase_a_p0_cost_curriculum_sync_80_20260919.json`](../../../itrader/scripts/experiments/configs/phase_a_p0_cost_curriculum_sync_80_20260919.json)
- [`phase_a_p0_frozen_policy_baselines_32_sync_20260919.json`](../../../itrader/scripts/experiments/configs/phase_a_p0_frozen_policy_baselines_32_sync_20260919.json)
- [`phase_a_p0_no_cost_replication_sync_70_20260919.json`](../../../itrader/scripts/experiments/configs/phase_a_p0_no_cost_replication_sync_70_20260919.json)
- [`phase_a_p0_promotion_eligible_sync_40_20260919.json`](../../../itrader/scripts/experiments/configs/phase_a_p0_promotion_eligible_sync_40_20260919.json)
- [`phase_a_p0_fail_closed_replication_sync_80_20260919.json`](../../../itrader/scripts/experiments/configs/phase_a_p0_fail_closed_replication_sync_80_20260919.json)
- [`phase_a_p0_fail_closed_cadence_24k_sync_80_20260919.json`](../../../itrader/scripts/experiments/configs/phase_a_p0_fail_closed_cadence_24k_sync_80_20260919.json)
- [`phase_a_p0_raw_net_cadence_24k_sync_80_20260919.json`](../../../itrader/scripts/experiments/configs/phase_a_p0_raw_net_cadence_24k_sync_80_20260919.json)
- [`phase_a_p0_sync_exploit_recovery_smoke_20260919.json`](../../../itrader/scripts/experiments/configs/phase_a_p0_sync_exploit_recovery_smoke_20260919.json)

The primary visual evidence is retained with the run outputs:

- [matched-arm net-return comparison](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_cost_curriculum_sync_80_20260919_135303/ablation_compare.png)
- [matched-arm Sharpe comparison](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_cost_curriculum_sync_80_20260919_135303/ablation_compare_validation_sharpe.png)
- [matched-arm action fractions](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_cost_curriculum_sync_80_20260919_135303/ablation_compare_action_fractions.png)
- [independent-replication net return](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_no_cost_replication_sync_70_20260919_152152/ablation_compare.png)
- [independent-replication Sharpe](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_no_cost_replication_sync_70_20260919_152152/ablation_compare_validation_sharpe.png)
- [eligibility-aware probe](../../../itrader/scripts/experiments/configs/outputs/phaseA_p0_promotion_eligible_sync_40_20260919_162149/ablation_compare.png)

## Matched 2026-09-19 validation controls

An initially reused control value came from seed `20260913`; it was not a
matched gate for this run's seed `20260919`. The mismatch was caught before
promotion and the complete control panel was rerun on the exact current tasks
and starts.

| Control | Mean net return at 1 bp | Sharpe per sqrt bar | Mean turnover |
|---|---:|---:|---:|
| flat | 0.00000% | 0.00000 | 0.0000 |
| long-only | -0.20866% | -0.02087 | 2.0000 |
| short-only | **+0.16866%** | **+0.01568** | 2.0000 |
| seeded random | -1.39662% | -0.17718 | 128.4375 |
| seeded untrained MLP-FiLM | -0.04160% | -0.00993 | 1.5000 |

The panel is directionally bearish. Using a transparent ex-post regime label
based on long-only gross return and a 2 bp absolute sideways band, it contains
21 bearish and 11 bullish episodes and no sideways episodes. Therefore a short
bias is not itself evidence of collapse. The relevant questions are whether the
policy improves on short-only and whether its actions vary appropriately across
the bullish and bearish episodes.

## Synchronous 80-update bridge

All arms use four members, identical training/evaluation seeds and starts,
32 evaluation episodes, 4,800 target steps per nominal update, synchronous PBT,
a 48,000-step mutation cadence, and a 96,000-step burn-in. They differ only in
the cost embedded in the optimization reward:

| Arm | Optimization cost | Independent evidence cost | Purpose |
|---|---:|---:|---|
| discovery | 0 bp | 1 bp | determine whether PPO can first discover gross edge |
| staged | 0 → 0.1 → 0.25 → 0.5 → 1 bp | 1 bp | test gradual cost pressure |
| exact | 1 bp from step zero | 1 bp | direct net-return objective |

### No-cost discovery arm

The arm completed 78 reports per member and approximately 386k–388k sampled
steps per target. Two terminal policies are economically interesting:

| Member | Terminal 1 bp net | Sharpe | Short / flat / long | Positive net+Sharpe tail | Tail beating short-only on both | Lineage |
|---|---:|---:|---:|---:|---:|---|
| `00001` | **+0.33846%** | **+0.02608** | 73.0% / 1.4% / 25.5% | 6 | 5 | untouched root |
| `00003` | **+0.35476%** | **+0.02561** | 72.8% / 2.1% / 25.1% | 8 | 5 | descendant of `00001` |
| `00002` | +0.23583% | +0.01422 | 70.6% / 4.3% / 25.1% | 2 | 0 | descendant |
| `00000` | -0.01240% | +0.00235 | 96.8% / 0.9% / 2.3% | 0 | 0 | descendant |

Member `00001` exceeds short-only by +0.16980 percentage points of mean net
return and +0.01040 Sharpe. Member `00003` exceeds it by +0.18610 percentage
points and +0.00994 Sharpe. Both remain positive under arithmetic 2 bp
repricing before fresh replay: approximately +0.25034% and +0.24163%,
respectively.

Both policies are positive at every configured horizon:

| Member | 96 | 128 | 160 | 200 |
|---|---:|---:|---:|---:|
| `00001` | +0.13295% | +0.26828% | +0.54868% | +0.40392% |
| `00003` | +0.15118% | +0.31512% | +0.56647% | +0.38626% |

This is the first repaired Phase A run in this audit sequence with sustained
positive 1 bp economics and positive Sharpe above the matched directional
control. It is not yet two independent winners: `00003` cloned the `00001`
lineage at approximately 149k target steps and later mutated. Their retained
policy-state hashes are different, but their ancestry is shared.

PBT behaved as configured. Across the four discovery members, six boundaries
changed lineage and/or mutable hyperparameters. The winning `00001` root itself
was never exploited or mutated; its starting recipe was sampled by the search:

```text
lr=0.0036320659, clip=0.1, gamma=0.99, lambda=0.9,
sgd_minibatch_size=512, num_sgd_iter=30,
vf_clip=20, vf_loss_coeff=1, train_batch_size=4800
```

PPO remained active over the winner's final 20 reports: approximate KL ranged
from 0.00859 to 0.03815, explained variance from 0.17257 to 0.35701, entropy
from 1.01755 to 1.05378, and gradients stayed finite. Ray 2.9 still reports
`clip_fraction` as unavailable/NaN, so that diagnostic cannot be used as a gate
on this stack.

The reward interpretation is important. With 0 bp embedded cost,
`net_return` is a scaled gross-PnL discovery objective. The terminal training
batch remained costly under independent 1 bp evidence because exploratory
actions generated high turnover, while the deterministic evaluation policy had
mean gross return +0.42658%, turnover 8.8125, and net return +0.33846%.
Optimization reward therefore did not silently substitute for the economic
metrics used for selection.

### Staged-cost arm

The schedule transitioned at the intended target-owned steps: 0 bp through
192k, then 0.1 bp, 0.25 bp at 240k, 0.5 bp at 288k, and 1 bp at 360k. Four
configuration-changing PBT boundaries occurred. The best retained score was
only +0.06941% net at the 0.1 bp stage; every terminal member was negative or
effectively flat at 1 bp. The arm fails promotion.

The most likely explanation is timing rather than a broken curriculum seam.
The discovery winner's durable improvement appeared after roughly 300k steps,
whereas the staged arm introduced cost pressure at 192k and increasingly
favored inactivity before the gross edge emerged. Future cost curricula should
start from a retained gross-edge checkpoint or delay pressure until after an
explicit edge gate; they should not repeat this early monotonic schedule.

### Exact-1-bp arm

The exact-cost arm also fails. PBT propagated a checkpoint that briefly reached
+0.10210% net and +0.01069 Sharpe with 75% flat actions, but it remained below
matched short-only on both metrics. Every terminal member converged to roughly
-0.065% to -0.081% net, non-positive Sharpe, and about 48--50% flat actions.
Embedding the full evidence cost from step zero therefore neither found a
stronger edge nor merely collapsed to 100% flat; it learned a weaker mixed
policy whose early best checkpoint was still inferior to the trivial
directional control.

### Fresh replay, costs, blocks, and regimes

Fresh Ray processes restored the retained `00001` and `00003` checkpoints. Both
reproduced selected net return exactly (`replay_delta=0`) across all 32 frozen
episodes. Their policy-state SHA-256 hashes differ, confirming distinct final
weights despite shared ancestry.

| Evidence | `00001` | `00003` |
|---|---:|---:|
| 0.5 bp repriced net | +0.38252% | +0.41132% |
| 1 bp exact replay net | +0.33846% | +0.35476% |
| 2 bp repriced net | +0.25033% | +0.24163% |
| paired uplift vs flat, 95% block interval | +0.113% to +0.596% | +0.157% to +0.593% |
| paired uplift vs short-only, 95% block interval | -0.199% to +0.619% | -0.168% to +0.627% |

The 18-effective-block exploratory intervals are positive versus flat, long,
random, and the untrained policy. They cross zero versus short-only. These are
selection-panel diagnostics, not selection-adjusted confidence intervals, so
they strengthen candidate status without authorizing promotion.

The policies are profitable in both ex-post regimes and change behavior in the
expected direction:

| Candidate | Regime | Episodes | Net return | Short / flat / long |
|---|---|---:|---:|---:|
| `00001` | bullish | 11 | +0.26767% | 52.3% / 2.1% / 45.5% |
| `00001` | bearish | 21 | +0.37554% | 83.9% / 1.1% / 15.1% |
| `00003` | bullish | 11 | +0.29234% | 52.2% / 2.9% / 44.9% |
| `00003` | bearish | 21 | +0.38745% | 83.6% / 1.6% / 14.7% |

This resolves the aggregate short-bias concern: the policy is not simply always
short. It increases long exposure by about 30 percentage points in bullish
episodes and remains net profitable in both groups. Because regimes are labeled
ex post from the same selection panel, this is behavioral attribution rather
than a deployable regime detector.

## Independent-population replication

The broad `pbt_population_seed=90210` population changed both initialization
and training/search randomness while preserving the exact frozen validation
panel. All four members completed about 336k--338k target-owned steps. It was a
clean negative replicate:

| Member | Terminal 1 bp net | Sharpe | Best observed net / Sharpe | Terminal short / flat / long |
|---|---:|---:|---:|---:|
| `00000` | -0.520% | -0.080 | -0.420% / -0.070 | 26.4% / 24.3% / 49.3% |
| `00001` | -0.740% | -0.120 | -0.170% / -0.030 | 24.1% / 17.9% / 58.0% |
| `00002` | -0.480% | -0.080 | +0.150% / +0.010 | 26.3% / 24.7% / 49.0% |
| `00003` | -0.930% | -0.140 | -0.440% / -0.040 | 32.0% / 18.9% / 49.1% |

The isolated `00002` early point remained below matched short-only on both net
return and Sharpe and did not persist. Four genuine configuration changes
occurred across the population (`00000`: two, `00003`: two); the other
checkpoint restores were synchronous scheduler bookkeeping with unchanged
configs. PBT therefore searched correctly, but this random draw did not sample
or evolve the high-learning-rate discovery recipe in the available budget.
The negative result rules out a claim that the frozen panel or PBT mechanics
automatically manufacture positive scores.

The `pbt_population_seed=314159` arm then started all four members from the
untouched discovery-winner recipe while assigning independent optimization
seeds. It also failed replication:

| Member | Terminal 1 bp net | Sharpe | Best observed net / Sharpe | Longest positive net+Sharpe run | Genuine config changes |
|---|---:|---:|---:|---:|---:|
| `00000` | -0.010% | -0.010 | +0.180% / -0.010 | 1 | 0 |
| `00001` | -0.080% | -0.010 | +0.100% / +0.010 | 7 | 0 |
| `00002` | -0.090% | -0.010 | +0.280% / approximately 0 | 1 | 3 |
| `00003` | -0.160% | -0.010 | +0.16989% / +0.01027 | 5 | 2 |

Member `00003`'s closest point barely exceeded short-only net return
(+0.16989% versus +0.16866%) but remained below short-only Sharpe (+0.01027
versus +0.01568) and did not persist. Member `00001` is more informative about
continuity: its untouched root lineage held positive net return and Sharpe for
seven reports, but the net returns were only about +0.04% to +0.10% and its
policy was about 94--98% short during that stretch. It was not rejected for
directionality; it failed because its economics remained below the matched
short-only control. All terminal members were negative.

The replication therefore confirms recipe sensitivity and repeated learning
glimpses, not an independent Phase A winner. The original `00001`/`00003`
checkpoints remain valid selection-panel candidates, but their shared ancestry
and the failed independent populations block promotion.

### Eligibility-aware 40-update probe

The replication exposed a contract mismatch: promotion required positive
Sharpe and rejected effectively all-flat behavior, while PBT donors were ranked
only by point net return. A test-first, opt-in selection metric now makes
positive net return, positive Sharpe, and non-all-flat behavior lexicographic
without changing reward, reported economics, or the
default legacy PBT objective. Ineligible members remain in a finite lower tier
ordered by net return; eligible members are still ordered by net return.

A four-member, 192k-step synchronous probe from the winner recipe exercised the
new contract. Two genuine config changes occurred. Member `00003` generated two
eligible reports; its best was +0.09902% net, +0.00691 Sharpe, 4.38% flat at
128,040 steps. The eligibility did not persist to the 144k PBT boundary, and
all four terminal policies were negative:

| Member | Terminal 1 bp net | Sharpe | Short / flat / long |
|---|---:|---:|---:|
| `00000` | -0.300% | -0.030 | 1.7% / 2.9% / 95.4% |
| `00001` | -0.080% | -0.050 | 0.3% / 98.3% / 1.4% |
| `00002` | -0.080% | -0.060 | 3.0% / 96.4% / 0.6% |
| `00003` | -0.040% | -0.030 | 3.3% / 94.8% / 2.0% |

The scorer works, but the probe does not improve the economic conclusion. It
also shows that a 48k-step donor cadence can miss short eligible intervals.
Future work should compare a 24k cadence against the existing 48k control and
retain cadence checkpoints for every eligible report; it should not lower the
economic gate or penalize directional policies to manufacture persistence.

## Fail-closed replication and 24k cadence study

The eligibility-aware probe still allowed ordinary ordering when every member
was ineligible. That was useful for backward compatibility but not strict
promotion search: a negative donor could overwrite an independent lineage. The
promotion scheduler now fails closed. A lower-quantile member may clone only an
upper-quantile donor whose selection score is positive; otherwise the scheduler
retains the member and continues training. Raw-net PBT remains available as an
explicit diagnostic control.

### Matched 48k fail-closed replication

Two four-member populations ran for an 80-update target with the winner recipe,
the same frozen 32-episode panel, 96k burn-in, and 48k synchronization cadence.
The `271828` population produced no eligible donor and no exploit. The `314159`
population produced one valid exploit at approximately 298k target-owned steps:
member `00003` cloned member `00001`, after which the bounded mutable values were
`lr=0.00435848`, `clip=0.2`, `gamma=0.995`, `lambda=0.9`, minibatch `256`,
20 SGD epochs, value clip `10`, and value-loss coefficient `0.5`.

The best independent point was member `00002` at 184,024 steps: +0.18438% net,
+0.01761 Sharpe, +0.32500% gross, turnover 14.0625, and 95.74% / 1.72% /
2.53% short / flat / long. It exceeded the matched short-only point estimate by
only +0.01572 percentage points of net return and +0.00193 Sharpe, and it lasted
one report. Exact fresh-process replay matched to numerical precision. Arithmetic
same-action repricing remained positive at 0.5 bp (+0.25469%) and 1 bp
(+0.18438%), and fell to +0.04375% at 2 bp. The paired block interval versus
short-only crossed zero (-0.12743 to +0.17228 percentage points).

Regime attribution rejects a stronger interpretation. The checkpoint used about
95% short actions in both groups; it lost -0.59381% across 11 bullish episodes
and gained +0.59200% across 21 bearish episodes. It learned a validation-panel
directional exposure, not the earlier shared-lineage candidates' regime-varying
behavior. The run therefore proves exact replay and valid fail-closed PBT, but
does not provide an independent winner.

### Eligibility-aware 24k arm and recovery evidence

The same `314159` population was then rerun with only the synchronization and
checkpoint cadence changed from 48k to 24k. There were no exploit events:
member `00001` was eligible for two consecutive reports, peaking at 139,376
steps with +0.09433% net, +0.00453 Sharpe, and 92.16% / 6.75% / 1.09% short /
flat / long, but no positive eligible donor existed at a synchronization
decision that could improve the population. The other members produced no
eligible reports.

Ray 2.9 then exposed two separate platform defects. First, synchronous PBT could
leave the last paused member beyond a stale boundary while reporting an idle
cluster. Second, after interruption, Tune's newest Windows experiment-state
files could be truncated to zero bytes. Repository-owned metrics, progress,
checkpoint indices, and immutable RLlib checkpoints remained intact. An explicit
recovery arm restored member `00003` from its last stable checkpoint at 361,272
steps, seeded the target-owned clock at 73 updates, and reused evaluation seed
`30260938`. It completed at 386,008 steps with +0.07032% net, +0.00089 Sharpe,
and 92.89% / 5.02% / 2.09% short / flat / long—again below short-only.

Automatic Tune resume is therefore not an evidence authority on this
Ray-2.9/Windows stack. Recovery must name an immutable repository-owned
checkpoint, its exact progress boundary, and the frozen evaluation seed. The
training adapter now also releases a Ray 2.9 orphaned staged marker after a
synchronous exploit; a four-member 30-update smoke crossed the previously
deadlocked boundary and completed all members at approximately 144k--149k
target-owned steps. This is deliberately localized in the PBT adapter rather
than leaking Ray internals into the solver, environment, or task contract.

### Raw-net 24k diagnostic control

The first raw-net control attempt recorded one genuine exploit near 99k steps,
then reproduced the post-exploit staged-trial deadlock before the adapter fix.
Its prefix is retained as failure provenance and is not a completed comparison.
The corrected matched control is the authoritative raw-net result described
below.

<!-- RAW24K_RESULT -->

The corrected four-member control advanced every member beyond 194k sampled
steps and recorded two genuine exploits before a platform resource failure:

| Event | Target | Donor | Target / donor steps | Mutated result |
|---|---|---|---:|---|
| 1 | `00002` | `00000` | 99,248 / 99,792 | `lr=0.00435848`, `gamma=0.995`, `lambda=0.9`, clip `0.2`, minibatch `256`, 20 epochs, value clip `10`, value coefficient `0.5` |
| 2 | `00000` | `00001` | 144,456 / 154,384 | `lr=0.00290565`, `gamma=0.995`, `lambda=0.9`, clip `0.2`, minibatch `256`, 20 epochs, value clip `10`, value coefficient `0.5` |

The post-exploit staged-member repair worked: target `00002` resumed and all
members continued. At about 194k--250k steps, however, four concurrent trials
with two rollout workers each exhausted the 16 GB Windows host. Three trials
failed worker construction after four retries with `MemoryError` or
`ArrowMemoryError`; the remaining trial made no further progress. This was not
an Ubuntu/WSL crash and not a storage failure. The active watchdog found no new
WSL crash file, C retained more than 76 GB free, and every repository-owned
checkpoint/index remained readable.

An explicit recovery loadout restored each member from its latest immutable
checkpoint and completed the target exposure sequentially on evaluation seed
`30260938`. This freezes the four lineages after the interruption; it is valid
checkpoint-continuation evidence, but not a substitute for the missing second
half of cross-member PBT decisions.

| Member tail | Best step | Best 1 bp net | Best Sharpe | Best short / flat / long | Terminal interpretation |
|---|---:|---:|---:|---:|---|
| `00000` lineage | 239,120 | +0.09413% | +0.00925 | 95.19% / 3.05% / 1.76% | ended 99.82% flat; rejected |
| `00001` lineage | 324,352 | +0.18403% | +0.01226 | 97.04% / 2.48% / 0.48% | point net beat short-only, Sharpe did not; ended 98.70% flat |
| `00002` lineage | 372,216 | **+0.39258%** | **+0.02959** | 74.71% / 0.29% / 25.01% | two consecutive reports beat short-only, then regressed toward near-always-short |
| `00003` lineage | 375,296 | +0.16917% | +0.01520 | 99.68% / 0.31% / 0.02% | economically indistinguishable from short-only |

Member `00002` is the strongest new transient challenger. Its preceding report
at 367,208 steps was also above short-only at +0.27155% net and +0.01851
Sharpe. Fresh-process replay of the retained 372,216-step checkpoint reproduced
the selected score to `8.7e-19`. Same-action net return remained +0.43883% at
0.5 bp, +0.39258% at 1 bp, and +0.30008% at 2 bp.

The policy was direction-sensitive rather than fixed-short on the frozen panel.
Across 11 bullish episodes it used 54.32% short and 45.35% long actions and
returned +0.40181% net. Across 21 bearish episodes it used 85.39% short and
14.35% long actions and returned +0.38775% net. The exploratory paired block
delta versus short-only was +0.22392 percentage points, but its interval still
crossed zero (-0.13824 to +0.68115 percentage points). It therefore fails the
study's five-report persistence and independent/block-support gates despite
being the clearest new evidence that MLP-FiLM PPO can jointly produce policy
movement and positive economics.

The checkpoint's PPO diagnostics were active rather than frozen: approximate KL
was `0.00651`, entropy `1.0772`, and explained variance `0.1383`. Its training
batch was negative while the frozen validation panel was positive. Together
with abrupt short/flat/long mode switches, that supports selection sensitivity
and policy instability; it does not establish “negative grokking.”

Evidence:

- [tail net-return plot](phase-a-mlp-film-autoresearch-2026-09-19-evidence/raw24k-tail-ablation_compare.png)
- [tail Sharpe plot](phase-a-mlp-film-autoresearch-2026-09-19-evidence/raw24k-tail-ablation_compare_validation_sharpe.png)
- [tail action-fraction plot](phase-a-mlp-film-autoresearch-2026-09-19-evidence/raw24k-tail-ablation_compare_action_fractions.png)
- [replay receipt](phase-a-mlp-film-autoresearch-2026-09-19-evidence/raw24k-member02-best-replay.json)
- [cost sensitivity](phase-a-mlp-film-autoresearch-2026-09-19-evidence/raw24k-member02-cost_sensitivity.png)
- [regime actions](phase-a-mlp-film-autoresearch-2026-09-19-evidence/raw24k-member02-regime_action_fractions.png)
- [block, cost, and regime JSON](phase-a-mlp-film-autoresearch-2026-09-19-evidence/raw24k-member02-checkpoint_evidence.json)

### Memory-safe paused-donor verification

The run exposed two additional orchestration facts. CPU mode silently clamped a
requested single rollout worker back to two, and a paused Ray 2.9 donor could
have no Tune-owned checkpoint even though the durable checkpoint index already
contained the exact boundary state. Test-first fixes now respect one worker,
allow the local Ray node's declared CPUs to cap concurrent trial execution,
recover a paused donor from the latest indexed checkpoint at or before its
scheduler clock, record exploit/fallback/skip receipts, and suppress actor-local
plotting when `auto_plot_on_end` is false.

<!-- SYNC50_RESULT -->

The first memory-safe four-member diagnostic used one rollout worker per trial,
declared four Ray CPUs, and therefore ran two trials at a time. All members
reached the approximately 97k-step synchronization boundary without memory or
storage pressure. Ray's scheduled donor save nevertheless resolved to `None`,
so the scheduler emitted an explicit `donor_checkpoint_unavailable` skip. The
run was stopped at that boundary rather than spending another 100k steps on a
known missing-exploit path.

A test-first retry now replaces that empty Ray future with the latest durable
checkpoint at or before the donor's trial-owned clock and repeats the exploit
transaction. The 25-update verification recorded:

- durable fallback for donor `00001` at 97,368 steps;
- one real `00002 <- 00001` exploit at 97,384 / 97,368 steps;
- mutation to `lr=0.00435848`, `gamma=0.98`, clip `0.2`, minibatch `512`,
  20 SGD epochs, value clip `20`, and value coefficient `0.5`; and
- successful target restoration and continuation through 121,584 steps.

That probe then localized a second Ray-2.9 tail defect: the paused upper donor
was later released with another trial's transient Tune checkpoint. Three
members reached update 25, while donor `00001` remained at its intact
97,368-step durable checkpoint. This is failure provenance, not a completed
economic comparison.

The final adapter repair reattaches a paused upper donor's own stable checkpoint
before Ray schedules it. Exploited lower members are not overwritten because
Ray clears their scheduler-owned checkpoint before cloning. A separate terminal
guard also prevents PBT from starting a new mutation after a member has reached
the declared target exposure. The corresponding red/green regression suite is
75 tests across the scheduler, training config, and ablation runner.

The compressed post-fix live proof
`phaseA_p0_sync_donor_tail_6_20260920_001655_p314159` completed all four members
with status `TERMINATED` and exit code 0. It recorded exactly one legitimate
exploit at the 24k boundary (`00002 <- 00000`, 24,256 / 24,552 steps), then
completed every one-update tail. There were no terminal exploit/fallback events,
no errored trials, and no orphaned member. The source fingerprint was
`5d0ef0e1...a57bbd49f`; the dataset fingerprint remained
`5f2e4495...a900f73`. These six-update economics are intentionally negative and
are not a performance result; the run exists solely to validate synchronized
checkpoint/exploit/resume behavior under the memory-safe topology.

During both live probes, free RAM bottomed near 6.0 GB, C stayed above 76 GB
free, and the independent watchdog observed zero new Ubuntu/WSL crash files.
The earlier long-run interruption is therefore attributed to excess concurrent
Windows Ray actors, not Ubuntu or storage exhaustion.

Verification evidence:

- [event receipt](phase-a-mlp-film-autoresearch-2026-09-19-evidence/sync-donor-tail-event-summary.json)
- [run summary](phase-a-mlp-film-autoresearch-2026-09-19-evidence/sync-donor-tail-summary.json)
- [net-return plot](phase-a-mlp-film-autoresearch-2026-09-19-evidence/sync-donor-tail-net-return.png)
- [Sharpe plot](phase-a-mlp-film-autoresearch-2026-09-19-evidence/sync-donor-tail-sharpe.png)
- [action-fraction plot](phase-a-mlp-film-autoresearch-2026-09-19-evidence/sync-donor-tail-action-fractions.png)

## Current interpretation

The repaired evidence no longer supports the earlier blanket statement that
the Phase A reward was merely learnable in an optimizer-only sense. It now
supports a stronger but still bounded result: the current MLP-FiLM PPO solver
can discover policies on the validation selection panel that retain positive
economics after 1–2 bp costs, beat the matched directional control, remain
positive across all four horizons, and persist for two successive evaluations.

It does **not** yet prove validated economic performance. The two best policies
share ancestry, the validation panel selected them, paired intervals are not
selection-adjusted, and the final test remains sealed. Exact replay and
regime-conditioned receipts passed, but two independent populations failed to
reproduce the candidates. Independent replication and the final-test gate
remain mandatory.

## Verification record

- 158 repository tests pass.
- Repository healthcheck passes on Python 3.11.13, Ray 2.9.0, Torch 2.3.1 CPU,
  NumPy 1.26.4, Pandas 2.1.4, and Gymnasium 0.28.1.
- Optional `llama_cpp` and `transformers` remain absent and unused.
- All new loadouts validate against `ablation.schema.json`.
- Random/generated proposer semantics and `stream_worker` sampling are explicit.
- The prior completed-work boundary was preserved in pushed commits
  `f045fde6` (docs/scripts/artifacts) and `88361f05` (code/tests). This dated
  loop remains a separate review surface.
- The full repository suite passes with one upstream Ray deprecation warning.
  Merge-conflict, YAML, trailing-whitespace, healthcheck, core-test, and
  all-files hooks pass. The first all-files pass normalized the missing final
  newline in three historical generated summary JSON files; the second pass is
  clean. A required GitNexus refresh was then aborted when its WSL Node process
  crashed and emitted an 82.46 GB dump. The dump was moved to E after its lock
  released, restoring C to 76.84 GB free. The existing graph remains explicitly
  stale and is not an execution oracle; the training evidence predates and is
  independent of that indexing failure.

## Storage remediation and deferred A/B

After this evidence pass, a storage incident audit found that a 77.94 GB WSL
Node crash dump—not PBT artifacts alone—caused the abrupt C-drive exhaustion.
Active `ray_results` and 37 experiment directories are now junctioned to the D
NVMe, cold crash diagnostics are on E, and original report paths remain valid.
See `phase-a-storage-incident-audit-2026-09-19.md` for byte counts, junction
receipts, and the deferred WSL-distro move.

NVMe placement may improve Parquet startup, checkpoint copying, Ray metadata,
and plot/report generation, but it is unlikely to multiply PPO throughput if
the workers remain CPU-bound after the dataset is cached. Measure it rather
than assuming it.

The ready A/B protocol is one short fixed-config run per storage target with the
same commit, data hash, seed, workers, 32-task panel, update budget, checkpoint
cadence, and no concurrent workload. Put Ray temporary files, `results_root`,
stable checkpoints, and the Parquet copy on the selected target together. Run
each target twice in alternating order and compare median
`time_this_iter_s`, total wall time, checkpoint-boundary latency, and effective
sampled environment steps per second. Keep the economic metrics out of the
storage decision except as a determinism check. The active NVMe output root is
now `D:\iTRADER-offload\active`; moving an active Ray run remains explicitly
prohibited.

## Evidence still required before opening the final test

1. **Complete:** exact-1-bp arm and retained-checkpoint review.
2. **Complete:** fresh-process replay with exact selection parity.
3. **Complete:** paired block intervals, 0.5/1/2 bp sensitivity, and
   regime-conditioned action/return plots against matched controls.
4. **Complete, failed:** one broad population and one winner-recipe population
   preserved the exact evaluation panel but produced no independent winner.
5. **P1 defined, not passed:** train independent lineages against genuinely new,
   non-overlapping validation blocks; use selection-aware block uncertainty,
   regime-conditioned action attribution, and fresh per-step 0.5/1/2 bp replay.
6. Require an independent lineage to reproduce sustained positive net return and
   Sharpe before selecting a final candidate.
7. Freeze selection, then open the untouched final chronological test once.

No Phase B or recurrent-solver performance claim should depend on this result
until those gates close.
