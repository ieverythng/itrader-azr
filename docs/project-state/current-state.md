# Current State

Last updated: 2026-09-24
Branch: `agent/Watson-dev`
HEAD: `891fcaa` plus the uncommitted dataset-v2 implementation/evidence round

## Dataset-v2 and first R0 bridge

An authenticated, immutable Alpaca SIP dataset-v2 snapshot now covers twelve
symbols from 2022-01-03 through 2026-09-22. It contains 10,370,064 unique bars,
including 5,515,061 RTH rows, with zero duplicate keys, OHLC invariant failures,
or negative volumes. Parquet partitions are authoritative; DuckDB performs
filtered discovery/query; Ray loads configured sessions and symbols. Phase A is
RTH-only and session-safe. Dataset v1 is retained solely as historical evidence:
its close path was real, but sampled OHLC was entirely flat, volume was often
duplicated/summed, and timestamps were not DST-safe.

The original fixed 32-episode v2 controls make long-only the strongest directional
baseline at +0.05970% net and +0.00757 Sharpe/sqrt(bar). Two matched fresh
12-update bridges show active PPO but no economic winner. No-cost ended at
-0.80750% net. Exact-1-bp training reduced turnover to 18.06 but converged to
95.23% short and -0.29511% net. The sealed final test remains untouched. The next
P0 searches cost schedules and PPO settings on v2. It no longer selects on that
panel because each horizon used different starts while `semantic_v2` exposes
horizon, creating a start/horizon shortcut. The replacement development panel
crosses eight explicit starts with all four horizons. Its controls are flat
0.00000%, long-only -0.09785% net/+0.00827 Sharpe, short-only +0.05785% net/
-0.01558 Sharpe, random -1.31143% net/-0.25857 Sharpe, and a seeded untrained
FiLM that exactly matches short-only. This crossed panel is selection-informed,
not an independent or final test. P1 moves to new chronological blocks and
independent populations.

The first 60-update, four-member synchronous v2 P0 attempt was stopped during
the no-cost arm after four real exploits. Member exposure ranged from 121,648 to
141,176 sampled environment steps. Ray 2.9 attempted `Trainable.save()` on an
exploited actor before that fresh actor had emitted its first post-restore
result, raising its `_last_result` assertion and invalidating the four-member
synchronous comparison. Stable metrics, 22 checkpoint-index records, and the
four exploit receipts were preserved; staged-cost and exact-cost arms were not
started. A subsequent 35-update verification gate completed three exploits but
showed that the first repair protected exploit-target restores only: member
`00002` failed at ~97k when its own stable checkpoint was reattached before
resume. A second fresh gate then proved both restore paths through new reports
and brought all four members to ~97k with three exploits. It stopped when Ray
attempted a redundant actor save while an exact same-step durable donor fallback
was being resolved. Synchronous upper donors now select that exact immutable
sidecar checkpoint directly; missing exact evidence fails closed. Gate C reached
~97k for all four members with two clean exploits and two exact selections, then
the guard exposed Ray's one-update in-flight overshoot: member `00002` reported
102,080 steps while its last cadence snapshot was 97,248. Synchronous PBT now
retains an immutable snapshot at every report but reports only cadence/terminal
checkpoints to Tune, preserving exact metric/model identity without changing the
search cadence. Gate D completed all four members at 170,152--170,416
trial-owned steps. Its 140 reports map one-to-one to 140 non-empty stable
checkpoints; five genuine exploits map to five exact-step donor receipts with
zero lag, including the former 102,080-step Gate C failure. A direct forensic
audit matched all five retained restore copies to their selected donor full-tree
hashes. New-arm lineage is stricter and generation-bound: selection -> one target
restore receipt -> target report -> retained target checkpoint. No exploit skip,
trial error, missing evidence, required-metric nonfinite, or storage stop
occurred. The 35-update gate therefore closes the Ray 2.9 synchronous checkpoint
seam. It does not close economic promotion: all terminal 1 bp net returns and
Sharpes were negative.

The canonical no-cost/staged/exact-cost P0 is now fully preflighted with the
crossed panel, `stream_worker`, synchronous four-member PBT, family 00,
random/generated tasks, 60 updates per arm, a narrower `5e-6..3e-4` learning-rate
range, and bounded decaying entropy schedules. No arm has been launched. The
goal is paused immediately before `--variant no_cost` at the user's request.
All earlier interrupted arms remain diagnostic evidence, not cost-arm results.

## Working-tree boundary

This dirty branch now includes the Phase A measurement/PBT implementation round.
The previously staged Sharpe challenger is preserved as an experimental reward;
`pnl_linear` remains the sole default. The supplied AZR and recurrence audits are
source-grounded evidence, not executable instructions. Recurrent/LSTM solver work
remains out of scope for this implementation.

## Current decision boundary

The user's Phase B scope includes both offline/curriculum work and online
replay/shadow work; Phase C is not the first replay boundary. Families `00`–`04`
remain a roadmap. `00`–`02` may share `Discrete(3)` width without settling input
schemas, task/reward semantics, or state ownership. Shared recurrent trunk plus
family-specific policy/value heads, recurrence placement, and restored R-ID
reconciliation remain proposed until the ongoing grill-me review confirms them.
`RewardEngine` remains the environment-owned reward seam, not a mandatory learned
reward head.

## Implemented baseline

- `train_solo_V2.py` selects `SolverMLPFiLM` directly.
- `ITraderEnv` emits chronological flattened OHLCV window plus task embedding.
- `TaskSpec` is canonical; `RewardEngine` owns reward calculation.
- `pnl_linear` is the only proven/default reward; Sharpe variants remain
  experimental.
- `train_AZR.py` generates/validates a task pool and hands it to solver training.
  This is bootstrap, not a co-adaptive AZR loop.
- Only source family `00` is enabled. Families `01`/`02` are bootstrap contracts;
  `03` is trade management; `04` is abstention/WAIT semantics.

## Phase A performance evidence

The 2026-09-19 P0 autoresearch is now the latest evidence. A matched no-cost
discovery arm produced two positive 1 bp checkpoints (`00001` +0.33846%,
Sharpe +0.02608; `00003` +0.35476%, Sharpe +0.02561). Exact replay, every
horizon, 2 bp repricing, and bullish/bearish action attribution passed, but the
two checkpoints share ancestry. A broad independent population and an
independent four-member winner-recipe population failed to reproduce the
matched short-only net-plus-Sharpe margin. An opt-in eligibility-aware PBT
score was then implemented test-first and exercised for 40 updates; it observed
two eligible reports but no boundary-time or terminal winner. Phase A therefore
has credible selection-panel candidates and active PPO/PBT, not validated
economic promotion. The sealed final test remains untouched. See the dated
2026-09-19 MLP-FiLM autoresearch report for controls, lineage, blocks, costs,
regimes, plots, and the next cadence experiment.

The follow-up raw-net 24k cadence run recorded two genuine PBT exploits before
the original four-trial/two-worker topology exhausted the 16 GB Windows host.
Immutable-checkpoint recovery completed every lineage. Member `00002` produced
the strongest new transient at 372,216 steps: +0.39258% 1 bp net return,
+0.02959 Sharpe, exact fresh-process replay, +0.30008% at 2 bp, and positive
returns in both ex-post bullish and bearish episode groups. It beat short-only
for only two consecutive reports, its paired block interval against short-only
crossed zero, and it later regressed. This is bounded evidence of economic
learnability on the selection panel, not independent promotion.

Ray-2.9 PBT is now live-validated in a memory-safe topology: one rollout worker
per trial, four declared Ray CPUs, and two concurrent trials on this 16 GB host.
The adapter retries an empty scheduled donor save from the stable checkpoint
index, reattaches paused upper donors to their own durable checkpoint, releases
orphaned staged trials, and suppresses mutations at terminal exposure. The
post-fix four-member proof terminated all members with one legitimate boundary
exploit, no terminal exploit, and no errored trial. The watchdog observed no
Ubuntu/WSL crash and C stayed above 76 GB free.

That statement is specific to the PBT probes. A later required GitNexus refresh
crashed its WSL Node analyzer and created an 82.46 GB dump that briefly drove C
to 0 GB. The analyzer was stopped and the dump was moved to E after the writer
released it, restoring C to 76.84 GB; Ubuntu and the completed PBT evidence
remained intact. Do not retry indexing before a coordinated WSL restart
activates the already configured E-drive crash-dump route. GitNexus remains
explicitly stale.

P0 measurement, controls, candidate discovery, replay, and PBT mechanics are
therefore closed enough to define P1, while economic promotion remains blocked.
P1 is independent training on genuinely new chronological validation blocks,
selection-aware block uncertainty, regime/action attribution, and fresh
per-step 0.5/1/2 bp return plus Sharpe replay before the sealed final test.

The 2026-09-17 synchronous matched bridge is the preceding negative bridge.
Its completed 1 bp net-return arm and restarted-from-zero PnL-linear arm each
have four completed members, 576k–580k trial-owned environment steps per member,
and 12 durable checkpoint indices per member. The interrupted PnL-linear prefix
was preserved but excluded: one member had no model checkpoint, and the other
trial-owned progress/report records were ahead of their model checkpoints, so
exact Ray experiment resume could not guarantee comparable exposure. Only that
arm was rerun. Ten and eight genuine configuration changes occurred in the
respective synchronous arms; gamma stayed in the valid 0.98–0.995 range.

Mean terminal 1 bp net return was -0.14219% (`net_return`) and -1.00048%
(`pnl_linear`); all eight terminal Sharpes were negative. The best retained
PnL-linear member-01 cadence checkpoint replayed at +0.06718% net, but its
Sharpe was -0.02301, turnover 26.1875, and its exploratory paired 14-block
uplift interval over long-only crossed zero. The same actions lost -0.19470%
when repriced at 2 bp. Both best checkpoints reproduced their selection scores
in a fresh process. PPO and synchronous PBT are active; neither objective has
established robust economic reward learnability. The sealed final test remains
untouched. See the dated Phase A performance-validity audit for per-member
actions, plots, replay receipts, and bootstrap caveats. Next P0 compares no-cost
training, staged cost, and exact 1 bp training on matched starts while scoring
all at 1 bp. P1 requires independent seeds, new blocks, selection-aware
uncertainty, and 0.5/1/2 bp per-step replay after a P0 candidate passes the
gross-edge, net, Sharpe, and turnover gates.

The following paragraphs preserve earlier chronology rather than superseding
the latest synchronous-bridge decision.

The current R0 implementation is a compatibility baseline, not a validated
performance winner. The 2026-09-12 audit restored both historical iteration-200
PBT checkpoints and measured each on 288 symbol-safe deterministic diagnostic
episodes. Checkpoint `00002` averaged +0.0163% gross and -0.0728% net at 1 bp per
turnover unit; checkpoint `00004` averaged -0.0112% gross and -0.0950% net. The
policies were 93.6% and 96.3% flat, respectively.

This does not disprove that `pnl_linear` is learnable as a PPO optimization
signal. It invalidates the stronger claim that the current training metric proves
economic performance: `custom_metrics/episode_return_mean` is accumulated shaped
reward, evaluation was in-sample, and the peak target has invalid `gamma=1.188`.

The initial official random/generated probe reproduced identical population
members, missed exploit boundaries, and zero-byte Ray-owned files. The repaired
probe now uses distinct training seeds/configs, one matched validation seed,
fixed rollout exposure, held-out net return at 1 bp as the PBT objective, bounded
mutations, aligned checkpoint/perturbation cadence, and restore-aware lineage.
Both PBT members completed exactly 12 unique iterations and 13,520 timesteps;
real exploit/restore events and mutated configs were recorded. Repository-owned
manifests, status, config history, checkpoint indices, metrics, plots, and
capability reports are non-empty outside Tune-owned paths.

The smoke probe does not validate performance. Four validation episodes per
iteration and two training seeds are deliberately insufficient; isolated
positive observations reversed by the final iteration. The overnight baseline-
controlled, multi-seed study and untouched final test remain the promotion gate.

On 2026-09-14, a parity-shaped preflight completed with four PBT members, nine
iterations, 16 held-out episodes per iteration, and approximately 44.3k–44.5k
environment steps per member. Three members recorded checkpoint-restore events
after the non-terminal iteration-6 exploit boundary. Aggregate net-return and
per-square-root-bar Sharpe plots completed headlessly. The best final burst point
was still negative (-0.115% net; Sharpe -0.0081), so it is seam evidence only.

The 200-iteration parity run
`pbtA26_parity200_20260914_000736_pbt200` completed all four members at roughly
0.99M environment steps each. Final held-out net returns ranged from -1.185% to
-0.598%; final per-square-root-bar Sharpes ranged from -0.1983 to -0.1415. The
run is valid evidence of negative economic outcomes under its reported policy
state, but not of solver uplift or reward learnability.

A checkpoint-boundary audit froze the 16-task evaluation panel and traced the
abrupt turnover jumps to Ray 2.9 restore semantics. Policy hashes were identical,
but `Algorithm.restore()` changed the evaluation policy's `explore` flag from
`false` to `true`; PBT therefore ranked stochastic post-restore actions. The
trainer now forces deterministic evaluation, explicitly synchronizes weights,
accepts Ray 2.9's direct evaluation payload, and limits smoothing to the 16-panel
episodes. PBT additionally uses checkpoint-free `market_relative_v1` observations
with `NoFilter` and copies checkpoints outside Tune ownership.

The final two-member, six-iteration official probe exercised a non-terminal
restore. Source turnover 56.625 became 56.25 after the clone's first update, not
roughly 127, and a fresh process reproduced the stable checkpoint's net return,
Sharpe, and turnover exactly. Checkpoint/evaluation parity is now green. The
four-member, 200-iteration run
`pbtA26_parity200_checkpoint_valid_20260914_041907_pbt200_checkpoint_valid`
completed on 2026-09-14 using `stream_worker`, random/generated tasks, the frozen
16-task held-out panel, deterministic evaluation, and 10-iteration
checkpoint/perturbation cadence. All members reached iteration 200 and roughly
990k retained environment steps. Terminal net returns were -0.0556%, -0.1069%,
-0.1866%, and -0.3906%; terminal Sharpes were -0.00060, -0.03145, -0.04089,
and -0.05376. The best member was 91.3% flat and its durable terminal checkpoint
reproduced net return, Sharpe, and turnover exactly in a fresh process. This is
valid candidate-generation evidence, not performance promotion.

The aggregate reporter initially aligned PBT histories by row position, which
misplaced terminal points after lineage rewinds. The completed evidence was
repaired against recorded `training_iteration`. New runs instead report and
plot a trial-owned cumulative `pbt_progress_steps` clock. It survives donor
checkpoint restore, drives perturbation/checkpoint cadence in sampled environment
steps, and fixes the repeated late-schedule mutation seen around member `00001`'s
iteration-130 restore.

The next-P0 implementation is staged for review. It adds a separate `net_return`
reward whose scaled episode sum matches 1 bp `EpisodeEvidence` arithmetic,
horizon-balanced frozen validation tasks, available Ray 2.9 PPO diagnostics, a
best cadence-checkpoint replay pointer, explicit baseline-relative promotion
gates, and an official flat/long/short/random/untrained baseline runner.

The official 64-episode control panel and bounded 40-update/192k-step PBT probe
completed on 2026-09-16. On the 32-task subset used by PBT, flat scored 0 while
long-only scored +0.03034% net with Sharpe +0.01077. All PBT terminal rows were
negative. Member `00001` retained the best cadence checkpoint at -0.16643% net,
Sharpe -0.06043, turnover 13.875, and 91.05% flat. Its checkpoint reproduced
those values exactly in a fresh Ray process. Two donor-to-`00001` exploits were
recorded, and its trial-owned progress remained monotonic through both restores;
the member required 44 RLlib iterations to reach its own 193k-step budget. The
measurement/PBT seams therefore pass the bounded acceptance check, but the
economic promotion gate fails. Do not launch the multi-seed overnight study or
let Phase B consume R0 as a performance winner until a bounded challenger beats
flat and the matched long-only control.

The follow-up matched 120-update bridge also completed on 2026-09-16. It used
`stream_worker`, random/generated tasks, no LLM backend, four population members
per arm, two rollout workers per member, identical seeds, and roughly 578--579k
trial-owned environment steps per member. Exact 1 bp `net_return` ended at a
mean -0.3309% net; one member converged to exactly flat. Historical `pnl_linear`
ended at a mean -0.8850% net while retaining substantially more trading. Its best
cadence checkpoint reached +0.04948% net, but Sharpe was -0.01032 and turnover
42.94 versus the matched long-only control's +0.03034% net, +0.01077 Sharpe, and
2.0 turnover. PPO is active and reward-sensitive, but neither arm passes
economic promotion or establishes repeatable reward learnability.

The bridge exposed underpowered asynchronous exploitation on Ray 2.9: each arm
recorded only one true donor exploit, and Ray could attempt to exploit from an
upper trial before its scheduler-owned checkpoint was ready. Promotion loadouts
now opt into `pbt_synch`; the legacy asynchronous default remains available for
reproduction. A synchronous A/B probe removed the missing-donor warning, and a
second TDD slice moved stable checkpoint-index persistence before
`session.report` so a synchronous pause cannot lose the index row. The final
12-update integration probe produced three indexed/stable checkpoints per member,
monotonic progress, real mutations in two of four members, and no missing-donor
warning. Raw synchronous restore counts include self-resume and are not exploit
counts; stable configuration histories are authoritative.

Ray migration is now an explicit platform follow-up, not an in-place edit to the
active experiment. The repository currently pins 2.9.0 in `pyproject.toml` and
`itrader/environment.yml`, but 2.47.x in `itrader/constraints.txt` and
`itrader/infra/cluster.yaml`. Preserve one authoritative 2.9 reproduction lock,
then test Ray 2.58.0 in a separate environment before migrating the legacy
`ModelV2` path to the new RLlib API stack. Unsloth is relevant only to a future
LLM proposer fine-tune, not the numerical PPO/PBT solver.

## Solver truth

R0 is the only implemented/default solver. `SolverMLPFiLM` uses
`tanh(gamma(task))*hidden + beta(task)` and task dropout; non-`film` conditioning
currently ignores task features rather than concatenating them.

The checked-in `SolverLSTMModel` infers feature width from the full observation,
reshapes incorrectly when task embedding is appended, discards LSTM state, and
returns incoming state unchanged. It is not a stateful recurrent solver.

R-series design identities are preserved from commit `4732a98`:

- R1 reset-window LSTM + FiLM — unimplemented;
- R2a concat — unimplemented;
- R2b init-only FiLM — deferred because current initial-state hook has no task;
- R2c per-step FiLM — active candidate, unimplemented;
- R2d init+per-step value-head-only — deferred;
- R3 GRU per-step — later;
- R4-v1 matched explicit solver-state feature study — planned and versioned.

The old `bec5c77` per-step label R2b maps to original R2c; the old init R4 maps
to original R2b. `R2c-CARRY`/`R2c-RESET` are named same-input controls, not new
IDs.

## Required evidence before recurrence claims

- explicit `[price_window_flat | task_embedding]` split and newest-bar order;
- cold-start versus causal initial-window warmup documented;
- separate input-window, unroll/fragment, and episode/reward-stat windows;
- native pinned Ray/RLlib sequence/state contract with valid-length packing,
  padding-loss masks, and last-valid h **and c** preservation;
- real termination versus truncation bootstrap and fragment detach semantics;
- train/inference state parity, vector-worker ownership, checkpoint/replay state;
- same-input/same-conditioning train-from-scratch carry/reset control;
- R4-v1 fields added identically to FF and recurrent controls;
- chronological held-out validation/final test and cost-aware metrics.

A zero-state probe is an OOD diagnostic, not a causal memory proof. Any
post-cell/readout FiLM claim is readout-only only when no other task path (such as
concat) reaches the cell; with concat, the transition is already task-conditioned.
The placement and memory-conditioning order remain open. OHLCV-only recurrence
cannot reconstruct sampled action/fill/position.

## Task encoding contract

Historical checkpoints retain `legacy_v1`. New training uses `semantic_v2`, which
represents solver-relevant task semantics and excludes episode seed/start as a
location shortcut. Reset seeding, declared reward ranges, and normalized
overrides are covered by regression tests.

## Dependency and validation status

`pyproject.toml`/`uv.lock` are authoritative: Python 3.11, Ray/RLlib 2.9.0,
Gymnasium 0.28.1, and setuptools 69.5.1. The old Phase B 2.47.1 line is
historical. No dependency files were changed.

The working Windows `.venv` runs the current suite and Ray 2.9 stack. No recurrent
training result is reported here. The bounded Phase A probe validates effective
PBT search and evidence wiring, not reward learnability or trading performance.
Ray-owned PBT text files still truncate to zero bytes on this platform; stable
repository-owned provenance is authoritative.

## Next priority

Resume the prepared `no_cost` arm only when the user reactivates the goal. At
the 20-update/~96k-step boundary, apply the strict lineage/population/evidence/
storage gate before allowing the arm to continue. Then return for authorization
before staged-cost or exact-cost; never chain arms automatically. Any candidate
must be replayed at 0.5/1/2 bp and compared by chronological/regime blocks before
independent P1 lineages. Keep the final test sealed. Preserve R0 while this
happens; recurrence work may continue separately but must use the repaired data,
panel, and evidence contracts.

## 2026-09-19 storage incident

The abrupt C-drive exhaustion was caused by a 77,937,594,360-byte WSL crash
dump for `/usr/bin/node`, not by Ray checkpoints alone. Phase experiment outputs
and user-level `ray_results` contributed about 5.82 GB. The dump archive is now
on E, active Ray results and 37 experiment directories are junctioned to the D
NVMe, and C has 78.25 GB free. Future WSL dumps are configured for E with a
one-dump retention cap, effective after the next normal WSL restart. Moving the
76.95 GB Ubuntu VHDX to D is deferred because the distro currently hosts live
Hermes, OpenClaw, ZeroTier, and browser-harness services. Full evidence is in
`docs/artifacts/research/phase-a-storage-incident-audit-2026-09-19.md`.
