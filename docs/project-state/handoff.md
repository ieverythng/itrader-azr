# Handoff

Last updated: 2026-09-17
Branch: `agent/Watson-dev`
HEAD before the current staged implementation round: `42d1b3a`

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

The aggregate outputs, plots, replay receipt, final loadout, and earlier
documentation are committed as evidence. The raw Ray trial/checkpoint tree
remains local. A new implementation round is staged for user review: a monotonic
sampled-environment-step PBT clock, exact scaled `net_return` task reward at 1 bp,
balanced fixed panels, PPO diagnostics, best cadence-checkpoint indexing,
baseline-relative promotion gates, a fixed-policy/untrained-policy evaluator,
and two next-P0 JSON loadouts. No recurrent-solver implementation, Ray upgrade,
or dependency migration is included. The later 120-update matched bridge and
synchronous PBT probes are local experiment evidence plus staged loadouts; the
large raw Ray trial/checkpoint trees remain ignored.

Validation passed: repository healthcheck, 97-test full suite, focused reporting
tests, JSON-schema validation, `git diff --check`, and the complete pre-commit
suite. GitNexus was readable but stale relative to the dirty implementation and
should be refreshed after the code commit.

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
- R1–R3 and R4-v1 are design/plan cells only.
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
9. Run R4-v1 with identical explicit state fields on FF/recurrent controls.
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
