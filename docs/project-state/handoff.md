# Handoff

## 2026-09-25 — R0 + R4-v1 architecture qualification and replication

- R0 remains the immutable state-free parity control. `R0 + R4-v1` is the
  forward Phase A feed-forward training baseline because position and remaining
  horizon complete the known endogenous control state. This is not economic
  promotion and does not claim full market observability.
- The fresh December-2025 4x3x40 replication is complete. All 12 terminal
  checkpoints replay exactly. Combined state improved net over masked in every
  seed and passed the pooled paired net gate, but no arm beat long-only or had
  positive net and Sharpe across seeds; seed 20260926 collapsed near-flat.
- The decision is architecture promotion without economic promotion. The next
  bounded P0 searches actor robustness through learning rate, entropy pressure,
  and causal cost staging with retained-best replay. R4-v2 remains deferred.
- The completed random-initialized 12-update artifacts are frozen. New runs use
  the named `zero_v1` residual initialization and verify update-zero policy/value
  parity before training.
- The analyzer now makes the turnover ceiling (`20.0`) and near-flat veto
  (`flat < 0.98`) executable. Baselines are state-arm aware, and checkpoint
  diagnostics include identical-input state interventions with logit/value
  sensitivity.
- The return-uninspected December-2025 RTH panel with eight disjoint anchors and
  four crossed horizons has now served the completed fixed 4x3x40 gate. PBT was
  off and the final test remains sealed.
- Future episode PnL, accounting, holding, drawdown, turnover-budget, and session
  clock features are registered but not added. R4-v2 nonlinear late fusion and
  recurrence remain separate factors.

## 2026-09-25 — Ray 2.58 R0 platform gate accepted

- Current dataset-v2 source passes 254 tests and healthcheck on Ray 2.9 and 2.58.
- Fresh Ray-2.58 `workers=0` 2x12 synchronous PBT crossed the cadence, completed
  one genuine exploit/restore, passed strict lineage inspection, and replayed
  its retained best checkpoint exactly. Matched Ray 2.9 mechanics also passed.
- Three fixed-hyperparameter seeds per runtime all ended negative; Ray 2.58's
  early positive-net transients all retained negative Sharpe. This is platform
  acceptance, not solver promotion.
- Fresh replay found and closed an inference-dropout defect. Older in-loop
  evaluation with task dropout is stochastic unless verified by exact replay.
- Root dependency declarations now target Ray 2.58; `itrader/compat/ray29/`
  records the historical reproduction profile.
- Validated topology for this 16-GiB host is `workers=0`. Ray 2.58 used about
  0.25 GiB less memory but ran about 16% slower in the matched small probe.
- Next after user review: matched Ray-2.58 R0 versus R1. Do not start native
  RLModule/EnvRunner modernization or R2 in that same attribution cell.

## 2026-09-24 — Ray 2.58 R0 platform gate initial attempt

- Reviewed migration commit `1033dc4`; did not cherry-pick its dataset-v1,
  asynchronous-PBT experiment boundary.
- Added current-tree compatibility adapters while preserving dataset-v2 RTH,
  sampled-step synchronous PBT, checkpoint lineage, and economic reporting.
- Full suites pass on Ray 2.9 and isolated Ray 2.58. Initial weights and initial
  deterministic evaluation match exactly; stochastic rollout sequences differ
  across RLlib versions under the same seed.
- A two-member 12-update Ray-2.58 sync probe was stopped safely after one update
  each because concurrent game use reduced free memory to 1.04 GiB. No exploit
  cadence was reached. Evidence is retained and marked `incomplete`.
- Do not delete `D:/iTRADER-offload`; it is the authoritative junction target for
  active Ray results and contains unique checkpoint evidence.
- Superseded by the completed 2026-09-25 gate above.

## 2026-09-24 — crossed-panel R0 cost study closed

- Final valid arm:
  `v2p0_nonflat_search_sync30_20260924_173537_exact_1bp`.
- Mechanical gate: pass at >=96k sampled environment steps for all four members;
  three exploit events, three verified generation-bound restore lineages, eight
  exact checkpoint selections, and no pending/stale/rewind evidence.
- Economic gate: fail. Exact-cost population mean was -0.05317% net,
  -0.03815 Sharpe/sqrt(bar), 67.33% flat/32.67% short, and turnover 6.609; no
  report combined positive net return and positive Sharpe.
- Matched staged cost converged to 100% short at +0.05785% net but -0.01558
  Sharpe. It is a directional shortcut, not a winner.
- Root cause closed: synchronous boundary targets can no longer be pre-staged
  while live peers are below the barrier; post-exploit orphan staging is removed
  and an already-created exploited actor fails closed.
- Decision: freeze R0 and stop further same-panel cost search. Next review and
  parity-test the separately developed Ray-2.58 migration with an R0-only bridge;
  then test bounded R1 under identical dataset/tasks/seeds/cost/PBT/evidence contracts;
  later attribute memory only with matched state carry/reset controls. Keep
  future-derived regime tags ex-post; causal C0/C1/shuffled-C1 is a separate
  later factor. Keep the CLM/Jev/System-One S-series research-only until the
  R-series solver/platform/evidence contract is frozen.

Last updated: 2026-09-24
Branch: `agent/Watson-dev`
Baseline HEAD: `0894116`; this dataset-v2 implementation/evidence round is the
next reviewable delivery on `agent/Watson-dev`

## 2026-09-24 dataset-v2 reset and R0 bridge

Dataset v1 is now understood: authenticated Alpaca SIP overlap confirms the
close path was real, but OHLC was collapsed to the close, volume was sometimes
summed across duplicate inserts, and local timestamps were DST-ambiguous. Do not
use v1 for new promotion work.

The local ignored snapshot
`itrader/data/datasets/dataset_v2_20220103_20260922_alpaca_sip_split` contains
10,370,064 split-adjusted SIP minute bars for twelve symbols, all three sessions,
and content hash
`fac41f6d42f8c5b1eac739755667bb38407461f82c168d7955de26bb2e5fb79c`.
Phase A selects RTH only. Credentials remain in ignored `.env`; licensed raw and
Parquet data remain ignored. The tracked manifest, configs, aggregate evidence,
plots, conclusions, and exact next gate are in
`docs/artifacts/research/dataset-v2-r0-audit-2026-09-24.md`.

The original 32-task controls and two matched 12-update fresh R0 bridges are complete.
Long-only is +0.05970% net. No-cost and exact-1-bp terminal net are -0.80750% and
-0.29511%. The cost-aware arm cut turnover but became 95.23% short. This proves
PPO/reward responsiveness on full OHLCV, not positive economic performance.
The follow-up Gate D completed four synchronous-PBT members at
170,152--170,416 trial-owned steps with 140 reports/checkpoints, five genuine
exploits, five exact-step donor receipts, and no errors or skips. Its strongest
transient (`00000`) reached +0.08116% 1 bp net and +0.01576 Sharpe for three
consecutive positive-net/positive-Sharpe reports, then regressed; all terminal
rows were negative. The Ray 2.9 checkpoint/search seam passes, but no checkpoint
is promoted. A forensic audit matched all five Gate D restore copies to their
selected donor full-tree hashes; new arms additionally require a generation-bound
selection/receipt/report/checkpoint chain.

The crossed development panel uses eight fixed RTH starts with horizons
96/128/160/200, removing the old horizon/start shortcut. Its bounded staged and
exact-cost arms are now complete through the 96k review gate. Staged cost became
the short-only control; exact cost ended negative with mostly flat/short actions
and no jointly positive net/Sharpe report. Freeze this R0 evidence, review Ray
2.58 through a bounded R0-only parity bridge, then start R1 as a matched
representation-only challenger on the accepted platform. Move any candidate to new
chronological blocks, block uncertainty, 0.5/1/2 bp replay, and independent
seeds. Keep the final test sealed.

## 2026-09-17 synchronous bridge — pause and completed recovery

The following interruption snapshot is retained as chronology; the completed
restart and current decision are recorded immediately below it.

The official matched synchronous 120-update bridge was launched with
`itrader/scripts/experiments/configs/phase_a_p0_reward_bridge_sync_120_20260917.json`
and run tag `phaseA_p0_reward_bridge_sync_120_20260917_001810`. The four-member
`net_return_1bp` arm completed. The subsequent `pnl_linear` arm was interrupted
at the user's request to release the PC. At interruption, members 00000, 00001,
and 00003 had durable first checkpoints; member 00002 had nine updates of
metrics but no first checkpoint. Ray wrote an experiment-state snapshot at
01:00:42 local time. The run's processes are gone (9.55 GiB physical RAM free
after stopping), and the outer ablation comparison is incomplete. Do not
describe this as a completed matched comparison or promise exact four-member
continuation: Ray Tune resume and the uncheckpointed member must be verified
before restarting. Preserve the output tree and do not rerun the completed
arm inadvertently. At that pause point, full sync analysis, paired checkpoint
replay, block-aware uncertainty, and cost scenarios were pending; all were
subsequently completed as described in the recovery section below.

### Restart decision and completed run

The saved Ray experiment state did not make an exact continuation defensible:
member 00002 had no model checkpoint, while the other members' durable progress
sidecars and report rows ran beyond their last checkpoint. Restoring them
without rewinding both model and evidence would overstate trained exposure.
The interrupted arm is preserved unchanged. After healthcheck, 41 focused
tests, loadout-schema validation, and dataset SHA-256
`5F2E449548532FF71F419431DF86962A2AEDF7275D7BFF5592729BBB8A900F73`,
only `pnl_linear` was restarted from update zero through the official trainer.
Its experiment name is
`phaseA_p0_reward_bridge_sync_120_20260917_001810_pnl_linear_pbt_sync_120_restart`;
the completed 1 bp arm remains untouched. Both arms use the checked loadout,
identical dataset and frozen evaluation task identities, training seed list,
split, 1 bp reporting cost, stream-worker sampling, and sync-PBT cadence. Their
task reward IDs/parameters intentionally differ. The 2026-09-17 restart
finished successfully with four completed statuses,
116–117 reports and 12 indexed checkpoints per member, and 576,408–579,632
trial-owned steps. Across the combined eight members, PBT produced 18 genuine
configuration changes (10 in `net_return`, 8 in `pnl_linear`), distinguished
from unchanged synchronous self-resumes. All observed gammas remained
0.98–0.995. The completed 1 bp arm averaged -0.14219% terminal held-out net
return; the restarted PnL-linear arm averaged -1.00048%. All eight terminal
Sharpes were negative.

The ablation CLI now has a tested `--collect-only --run-tag ...` path and
`--variant-experiment ID=EXPERIMENT` override. It requires completed stable
run status and expected member count, then creates the comparison plots,
including held-out action fractions, without training again. After all four
restart members completed, the comparison was collected with run tag
`phaseA_p0_reward_bridge_sync_120_20260917_001810`, overriding
`pnl_linear_pbt_sync_120` to the `_restart` experiment above. Exclude the
incomplete original `pnl_linear` arm from scoring. The comparison summary and
net-return, Sharpe, and held-out action-fraction plots are under
`itrader/scripts/experiments/configs/outputs/phaseA_p0_reward_bridge_sync_120_20260917_001810/`.

Fresh-process deterministic replay of the best checkpoint in each arm reproduced
its selected 32-episode mean net return to machine precision. The early
`pnl_linear` member-00001 checkpoint reached +0.06718% at 1 bp, but Sharpe was
-0.02301 and turnover 26.1875; later terminals were all negative. The best
1 bp checkpoint scored -0.04260%, Sharpe -0.00491, turnover 6.4375. Exploratory
paired 5,000-row symbol-stratified block resampling against long-only gave
`pnl_linear` checkpoint uplift +0.03684 percentage points with an interval
[-0.20893, +0.33746] points over 14 effective blocks; it crosses zero and is
not selection-adjusted. Repricing its same actions at 0.5/1/2 bp gave
+0.19811%/+0.06718%/-0.19470% mean net return. No fresh independent
training seed or untouched final-test evaluation was run; the Phase A promotion
gate remains closed. The audit and masterplan hold the full decision.

## 2026-09-16 P0 execution update

The official control panel
`phaseA_p0_frozen_policy_baselines_20260916_043314` completed over 64 identical
validation tasks. Mean 1 bp net return / per-square-root-bar Sharpe was: flat
0/0; long-only +0.04329%/+0.01475; short-only -0.08329%/-0.01950; random
-1.33475%/-0.17783; seeded untrained FiLM -0.28988%/-0.02282. The 32-task prefix
used by the bounded PBT probe is identity-matched; its long-only bar is
+0.03034% net and +0.01077 Sharpe.

The official bounded probe
`phaseA_p0_net_return_pbt_probe_20260916_043423` then completed. The historical
`pnl_linear` reference reached 98,640 sampled environment steps; the four 1 bp
`net_return` members reached 192,896-193,248 trial-owned steps, except member
`00001`, which correctly ran 44 RLlib iterations and 217,736 local sampled steps
to reach 193,128 target-owned steps after restores. PBT recorded two exploits,
both from member `00003` into `00001`; the target progress clock remained
monotonic and the second mutation did not reproduce the old iteration-130
scheduling defect.

All terminal PBT rows were negative. Member `00001` retained the best cadence
checkpoint at progress 168,256: net return -0.16643%, Sharpe -0.06043, turnover
13.875, and action fractions 7.76% short / 91.05% flat / 1.18% long. Gross
return was -0.02768% and explicit 1 bp costs were 0.13875%, so both weak gross
edge and trading costs contributed. A fresh Ray process reproduced every listed
metric exactly. The bounded seam acceptance is green; economic promotion is
red. No 200-update or multi-seed overnight run was started because the bounded
challenger failed flat and long-only.

The requested matched 120-update bridge subsequently completed. Both arms used
`stream_worker`, random/generated tasks with no LLM backend, four members, two
workers per member, identical seeds and validation starts, and approximately
578--579k trial-owned steps per member. Terminal member-level action fractions
are recorded in the performance audit. The exact 1 bp `net_return` arm averaged
-0.3309% terminal net and included one exactly-flat member. The `pnl_linear` arm
averaged -0.8850% terminal net and finished with 66.6--94.6 turnover. The best
retained PnL-linear checkpoint briefly reached +0.04948% net, but its Sharpe was
-0.01032 and turnover was 42.94; it therefore failed the matched long-only gate
despite showing a real transient gross edge. No candidate warranted sealed-test
evaluation or promotion replay.

The bridge had only one true asynchronous exploit per arm and reproduced Ray
2.9's missing-donor-checkpoint ordering warning. The application-owned
checkpoints were present. An explicit `pbt_synch` flag now enables synchronous
PBT for promotion studies, while false remains the historical default. A TDD
integration round also persists checkpoint indices before `session.report`,
because synchronous Tune may pause the function trainable inside that call. The
final 12-update sync probe completed four members with three indexed/stable
checkpoints each, two pause/resume boundaries each, monotonic exposure, true
configuration changes in two members, and no missing-donor warning.

## Resume point

The Phase A measurement, PBT checkpoint/evaluation, and iteration-aligned
reporting seams are implemented in the dirty working tree. The corrected
four-member, 200-iteration run
`pbtA26_parity200_checkpoint_valid_20260914_041907_pbt200_checkpoint_valid`
completed in 49 minutes 57 seconds. It used R0 `SolverMLPFiLM`, `stream_worker`
training, random/generated tasks with no LLM backend, two rollout workers per
member, fixed batch 4,800, one deterministic 16-task held-out panel, 1 bp cost,
validation net return as the PBT objective, and checkpoint/perturbation cadence
10 after burn-in 20.

All members reached iteration 200 and about 990k retained environment steps.
Terminal net returns were -0.0556%, -0.1069%, -0.1866%, and -0.3906%; terminal
Sharpes per square-root bar were -0.00060, -0.03145, -0.04089, and -0.05376.
The least-negative member was 91.3% flat. Its durable terminal checkpoint
reproduced net return, Sharpe, and turnover exactly in a fresh process. This is
valid negative candidate-generation evidence, not proof of solver uplift or
reward learnability.

PBT recorded 23 population exploit/restore mutations: member `00000` restored
3 times, `00001` 2 times, and `00002`/`00003` 9 times each. Member `00001` was
strongest before its second restore: mean held-out net return was +0.1211% over
iterations 80-100 and +0.0636% over 101-120, with a +0.3495% peak at iteration
107. It moved from about 60% flat/39% long over 80-100 to 72% flat/24% long over
101-120 at 6.2-7.9 turnover. Over 121-140 it flipped toward short and fell to
-0.1200% mean net. Its iteration-130 exploit changed lambda 0.90 to 0.95,
minibatch 256 to 128, SGD passes 10 to 20, value-loss coefficient 1 to 2, and
learning rate 1.79e-5 to 9.47e-5. The following segment therefore cannot be
interpreted as uninterrupted learning by the same optimizer configuration.

The fixed panel contains 16 `pnl_linear` tasks: horizons 96/128/160/200 occur
7/3/1/5 times, and turnover weights 0/0.0005/0.001/0.002 occur 6/4/3/3 times.
During member `00001`'s positive 80-120 window, mean reported PnL reward terms
were about +0.0106, drawdown terms -0.0023 to -0.0036, and turnover terms only
about -0.00003. PnL dominates the shaped reward, drawdown is material, and the
task-level turnover penalty is negligible relative to both; separate 1 bp
economic costs still make high-turnover policies unattractive.

The reviewable implementation/evidence delivery includes the monotonic sampled-
environment-step PBT clock, exact scaled `net_return` reward, crossed panels,
PPO/economic diagnostics, immutable checkpoint indexing, strict generation-
bound restore lineage, fixed/untrained controls, and the bounded cost-search
loadouts. Raw Ray trial/checkpoint trees and licensed market data remain local.
No recurrent-solver implementation, Ray upgrade, or dependency migration is
included.

The final architecture/deslop pass also closed three future-run provenance gaps:
snapshot hashes exclude interrupted atomic-write temporaries, manifests record
resolved custom PBT learning-rate bounds as stable JSON, and best-checkpoint
economic eligibility comes from the explicit net/Sharpe/flat promotion gate
rather than the active ranking metric. Current validation passes the repository
healthcheck, the complete 236-test suite, focused 95-test affected-module slice,
`git diff --check`, and a file-scoped pre-commit pass over the six affected
Phase A source/test/handoff files. The closeout docs then passed HTML parsing,
local-link resolution, diff check, and a second file-scoped pre-commit pass over
the complete Phase A closeout set. The earlier all-file pre-commit pass predates
these three fixes; do not target the concurrently owned Jev/CLM research while
it is active. GitNexus is readable but stale and should be refreshed separately
after delivery; the previously observed WSL dump risk still applies.

## User-review boundary

Phase B includes offline/curriculum work and online replay/shadow validation;
Phase C is later operationalization. The shared recurrent trunk/family-specific
policy-value head proposal, detailed FiLM placement/order, and any R-ID
reconciliation are not accepted decisions. Families `00`–`04` are the roadmap;
`00`–`02` sharing `Discrete(3)` width does not settle inputs, task/reward
semantics, or state ownership. `RewardEngine` remains the environment-owned
reward seam, not a learned reward-head requirement.

## Current truth

- R0 `SolverMLPFiLM` is the active/default model.
- Its implemented FiLM is `tanh(gamma)*hidden + beta` with task dropout.
- Non-`film` conditioning in that model does not implement concat.
- Legacy `SolverLSTMModel` is shape-incompatible with appended task embeddings
  and does not carry state.
- R1–R3 remain design/plan cells. R4-v1-min has an implemented FF seam plus
  replay-valid blocking and fresh 4x3x40 gates. Combined state passed relative
  net attribution on the fresh block, but every arm failed absolute economic
  promotion; no recurrent cross is economically authorized by this result.
- Original R2a/R2b/R2c/R2d meanings are preserved from `4732a98`.
- R2c per-step FiLM is the active candidate; R2b init-only and R2d value-only
  are deferred; R4-v1 is matched explicit solver-state observability.
- `R2c-CARRY` versus `R2c-RESET` is the required same-input control.
- R1 versus newest-bar R2 remains an unequal-information engineering comparison.

## AZR boundary

The local AZR v2 paper synthesis is in
`docs/artifacts/research/azr-stack-review-2026-09-12.md`. Eq. 4 is zero at
observed `p=0` or `p=1`, otherwise `1-p`; it is not a symmetric 50% target or
measured learning uplift. iTrader has historical OHLCV, a language-model
proposer, and numerical PPO. `train_AZR.py` is a bootstrap handoff, not a
co-adaptive AZR loop. Keep proposer frozen/adaptive reference-buffer studies
separate from solver recurrence.

## Remaining Phase A performance sequence

### 2026-09-19 P0 autoresearch update

The no-cost discovery arm produced two exact-replay shared-lineage candidates:
`00001` at +0.33846% net / +0.02608 Sharpe and `00003` at +0.35476% /
+0.02561 on the frozen 32-task 1 bp panel. Both remained positive at 2 bp and
changed action mix between ex-post bullish and bearish episodes. The staged and
exact-1-bp objectives failed. A broad independent population and a second
population initialized from the winning recipe also failed; no independent
member sustained the matched short-only net-plus-Sharpe margin.

An opt-in `economic_selection_score` now makes positive net return, positive
Sharpe, and less than 98% flat exposure eligibility constraints for PBT donors,
then ranks eligible
members by net return. Direction is not scored. Its 40-update probe found two
eligible reports in one member but missed them at the 48k-step synchronization
boundary and ended negative. Next compare 24k and 48k cadence under matched
raw-net and eligibility-aware selection. Keep the final test sealed.

That cadence comparison is now complete. Eligibility-aware 24k search produced
no exploit; raw-net 24k search produced two exploits and a replay-exact
member-02 transient at 372,216 steps (+0.39258% 1 bp net, +0.02959 Sharpe,
+0.30008% at 2 bp). It cleared short-only for two consecutive reports and was
positive in bullish and bearish episode groups, but its paired block interval
against short-only crossed zero and it later regressed. Retain it as the P1
challenger, not a promoted winner.

The long diagnostic also isolated Windows memory pressure: four concurrent Ray
trials with two rollout workers each exhausted the 16 GB host; storage and WSL
did not fail. The validated topology is `workers=1`, `ray_num_cpus=4`, two
concurrent trials. Ray-2.9 synchronous PBT now recovers an empty scheduled donor
save from the durable index, reattaches a paused upper donor's own checkpoint,
unstages orphaned trials, and skips mutation at terminal exposure. The final
four-member proof completed all members with exactly one legitimate exploit,
no terminal exploit, and no errored trial. C remained above 76 GB free and the
watchdog recorded zero Ubuntu/WSL crash files.

The zero-crash result applies to training. The subsequent GitNexus refresh
crashed its WSL Node analyzer, emitted an 82.46 GB dump that briefly filled C,
and was aborted. The dump was preserved on E after its writer released it; C
recovered to 76.84 GB. Ubuntu and the completed run evidence remained intact.
Do not retry GitNexus until a coordinated WSL restart activates the configured
E-drive dump route; the graph is stale.

P1 is now defined as independent training on genuinely new chronological
validation blocks, selection-aware block uncertainty, regime/action
attribution, and fresh per-step 0.5/1/2 bp return plus Sharpe replay. Keep the
sealed final test closed until a sustained independent challenger passes.

1. Preserve R0 and the repaired evidence/PBT contracts.
2. Treat the frozen controls, bounded probe, and matched 120-update bridge as
   completed negative gates. Neither exact 1 bp `net_return` nor historical
   `pnl_linear` is promoted.
3. Preserve the completed no-cost/staged/exact-1-bp and 24k/48k P0 studies as
   the selection record; do not rerun them merely to search the same panel.
4. Require gross edge over long-only, positive net return and Sharpe, controlled
   turnover, independent seeds, and block-aware uncertainty before spending the
   multi-seed overnight budget or opening the untouched final test.
5. Keep alignment/reward controls independent; do not call reward changes
   necessary for recurrence or claim longer horizons automatically win.
6. Resolve the user's open placement/memory-conditioning decision before choosing
   or implementing R2a/R2c.
7. Implement the selected recurrent control using the native state lifecycle and
   sequence contract.
8. Run `R2c-CARRY` versus `R2c-RESET` only if R2c remains selected; preserve the
   same-input/same-conditioning attribution control.
9. Preserve both completed R4-v1 gates as failed economic promotions. The next
   bounded P0 may use the predeclared fresh block only for its declared
   actor-robustness comparison; it must retain best eligible checkpoints and may
   not silently add state features. Keep future recurrent controls separately
   named and chronologically predeclared.
10. Treat R3 as a later cell-type comparison. Proposer and online/replay/shadow
   work remain within Phase B and can proceed alongside the offline track;
   recurrence experiments do not impose a blanket gate on that work.

Ray/RLlib 2.9.0 is validated in the Windows-native environment for the Phase A
path. Recurrent APIs still require their own pinned-runtime lifecycle tests; the
earlier WSL executable mismatch is not the current validation environment.

## Validation commands for a compatible environment

```text
./itrader/tools/run_repo_python.sh itrader/tools/healthcheck.py
./itrader/tools/run_repo_python.sh -m pytest -q itrader/tests
./itrader/tools/run_repo_python.sh -m pre_commit run --all-files
git diff --check
```

Use `uv.lock`/`pyproject.toml` as authority; do not use the historical Ray 2.47.1
plan line to choose a recurrent API.

## Storage remediation, 2026-09-19

The apparent PBT storage block was dominated by one 77.94 GB WSL Node crash
dump. It is preserved on E for now. `ray_results` and 37 experiment run
directories are verified NTFS junctions to D; original paths and report plots
remain valid. C/D/E free space is 78.25/163.85/259.79 GB respectively. A new
user-level `.wslconfig` redirects future dumps to E and retains one; it becomes
active on the next normal WSL restart. Do not force that restart or move the
Ubuntu VHDX until Hermes, OpenClaw, ZeroTier, and the browser harness can be
stopped in a coordinated window. See
`docs/artifacts/research/phase-a-storage-incident-audit-2026-09-19.md`.
