# Current State

Last updated: 2026-10-03
Branch: `agent/Watson-dev`
Baseline HEAD: `c13af18`; the current R4-v1 experiment/code/docs round remains
unstaged and uncommitted for review on `agent/Watson-dev`

Research-wing identities and cross-wing dependencies are indexed in the
[iTrader Research Dashboard](../research/itrader_research_dashboard.md); this
file remains the checkout/experiment-state authority.
Concrete run/member/checkpoint evidence is searchable in the generated
[Experiment Browser](../research/itrader_experiment_browser.html). Its
[registry](../research/experiment_registry.json) stores comparison metadata and
source pointers; authoritative summaries remain the metric source of truth.

## R0 + R4-v1 actor-robustness P0 closed

Next preparation is specified in [R1 development readiness](r1-development-readiness.md):
learner-produced clipping diagnostics, process-attributed verified replay,
explicit model selection and reset-per-decision window encoding. The planned
study includes matched R0 + R4-v1 alongside R0, R1 and R1 + R4-v1. These are
preparation requirements, not an implemented R1 or an authorized economic promotion.

The January-2026 combined-state search completed two matched four-member,
40-update synchronous-PBT arms on one Ray-2.58 source fingerprint. Both passed
strict mechanics: six exploits, 18 exact restore receipts, six verified
generation lineages, four complete members, and no stale or invalid provenance.

The causal `0 -> 0.25 -> 0.5 -> 1 bp` arm ended with all four members exactly
short-only at `+0.441611%` net and `0.024455` Sharpe per square-root bar. That is
identical to the matched short-only control, so it is directional exposure rather
than demonstrated conditional edge. One marginally better retained report failed
exact replay by `7.4e-06` in search score near an action-logit tie.

The exact-1-bp-from-update-zero arm ended with all four members exactly flat at
zero return, Sharpe, and turnover. All four terminal replays were exact. PBT
mutated throughout, so this is a model/curriculum basin rather than a dead search
harness. No checkpoint cleared non-flatness, directional-control uplift, and
exact replay; independent-seed and 0.5/1/2-bp promotion spending therefore
stopped at the declared gate.

Preserve state-free R0 for parity and `R0 + R4-v1` as the
architecture-qualified feed-forward baseline, without economic promotion. The
forensic follow-up found 81 jointly positive causal reports, but all were at
least 98.014% short. At the 77,832-step checkpoint, greedy actions were 99.36%
short even though mean action probabilities remained 40.18% / 27.30% / 32.52%
and entropy was 1.0823; 1.76% of steps had a top-two logit margin below 0.01.
That checkpoint repeated the same one-action replay mismatch as the marginal
best. Causal staging therefore remains an explicit matched curriculum factor,
not proof of conditional alpha. Do not run another full population on the same
January panel. The bounded G0 closure diagnostic is now complete: five exact
member-`00002` checkpoints at 77,832, 92,384, 102,040, 106,968, and 194,368
steps were replayed twice on a predeclared non-overlapping November-2025 8x4
RTH panel. Every episode/reward trace and every per-step action/logit trace
matched byte-for-byte across the retained artifact pairs. All five fresh-panel mean net
returns and Sharpes were negative. The least-negative checkpoint was 102,040
steps at -0.22822% net and -0.00620 Sharpe; its approximately 75/0/25
short/flat/long episode-weighted mix was nearly identical at every anchor.
The 106,968-step and terminal checkpoints were pure short and economically
identical. This closes R0 + R4-v1 as architecture-qualified but economically
unpromoted. Open clean Ray-2.58 R1 beside matched R0, R0 + R4-v1 and a separately named
`R1 + R4-v1` cell using the frozen lower-LR recipe. Keep fixed 1 bp and causal
staging as explicit matched curriculum factors. PBT follows fixed-recipe
representation attribution. R4-v2 remains deferred.

The canonical browser independently verifies the paired artifact hashes,
receipt metrics, and panel hashes. It does not elevate the historical
`two_process_identity` source field into independent producer provenance,
because those receipts did not record distinct producer/run identifiers. Future
R1 replay receipts must stamp producer identity at creation time; this
qualification does not alter G0's negative economic decision.

Static exact-cost reports also exposed a provenance-label defect: the training
task configuration correctly used 1 bp, but report rows omitted the static cost
field and aggregate output displayed zero. A test-first repair now resolves
both scheduled and static optimization costs for future reports; completed
reward delivery and economic results are unchanged.

## R0 + R4-v1 architecture gate closed

The fresh December-2025 4x3x40 replication is complete on the zero-initialized
R4-v1 evidence profile. All 12 terminal checkpoints replay exactly. Combined
position-plus-time state improved terminal net return over masked in every seed,
and the pooled seed-then-anchor interval excluded zero. The actor/critic
counterfactuals confirm real state use, with remaining horizon affecting the
critic most strongly.

No arm is economically promoted: all mean returns and Sharpes remain below the
long-only control, Sharpe uplift is not seed-consistent, and seed 20260926
collapsed above the 98% flat ceiling. R0 remains the immutable parity control;
`R0 + R4-v1` is the architecture-qualified forward FF baseline. Its bounded
actor-robustness search is now complete and economically rejected. R4-v2 remains
deferred.

## Ray 2.58 platform gate accepted

The separately committed migration was reviewed but not cherry-picked because it
carried a dataset-v1, asynchronous-PBT experiment boundary. Runtime-capability
adapters were applied to current dataset-v2 source instead. The full 254-test
suite and healthcheck pass on Ray 2.9 and isolated Ray 2.58; both runtimes produce
the exact same initial MLP-FiLM weights and deterministic crossed-panel control.
Their stochastic PPO trajectories differ under the same seed, so results must
remain runtime-labelled.

The fresh Ray-2.58 local-worker two-member/12-update gate completed at about
58.6k steps per member with one genuine synchronous exploit, one verified
generation-bound restore lineage, and exact fresh replay. The matched Ray-2.9
mechanical gate also passed. A three-seed/fixed-hyperparameter smoke bridge ended
negative for every runtime/seed terminal and for every best-checkpoint Sharpe;
it validates execution breadth, not economic equivalence or uplift.

During the gate, fresh replay exposed and fixed evaluation-time task-embedding
dropout: `SolverMLPFiLM` now honors RLlib's `is_training` signal. Older in-loop
evaluation with `t_keep_p < 1` is stochastic unless its selected checkpoint has
an exact fresh replay. The corrected acceptance runs replay at zero delta.

The coordinated root pins now target Ray 2.58/Gymnasium 1.2.2/PyArrow 17. Ray
2.9 remains a frozen profile in `itrader/compat/ray29/`. R1 must eventually run
beside a matched Ray-2.58 R0 control on the validated `workers=0` topology, after
the bounded R4-v1 actor-robustness P0. See the dated Ray-2.58 R0 parity-gate audit.

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

The canonical crossed-panel P0 has now completed its bounded staged-cost and
exact-1-bp reviews with `stream_worker`, synchronous four-member PBT, family 00,
random/generated tasks, lower learning rates, and bounded entropy schedules.
The final exact-cost arm passed its 96k mechanical gate with four members, three
verified generation-bound exploits, and no pending/stale lineage. It failed the
economic gate: population mean net return was -0.05317%, Sharpe/sqrt(bar)
-0.03815, actions were 67.33% flat/32.67% short/0% long, and turnover was 6.609.
Only 3/84 reports had positive net return, 3/84 had positive Sharpe, and none had
both. Positive gross return was erased by costs; mean turnover plus liquidation
penalties were approximately 3.55 times the positive PnL component.

The staged-cost arm instead converged all four members to the matched short-only
control: +0.05785% net, -0.01558 Sharpe, and turnover 2. This confirms that cost
curriculum changes policy persistence, but neither arm demonstrates risk-adjusted
edge. R0 is now frozen as the compatibility/economic control. More selection on
the same crossed panel is stopped; the next high-information step is a bounded
R1 representation challenger under identical data, tasks, seeds, costs, PBT
cadence, compute, and gates. Earlier contaminated probes remain diagnostic only.

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
- `pnl_linear` remains the historical/default reward; `net_return` is an
  implemented challenger, while Sharpe variants remain experimental. None is an
  economically promoted solver recipe.
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

The completed next-P0 implementation added a separate `net_return`
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

Ray 2.58 is now the coordinated root dependency authority; Ray 2.9 remains an
explicit historical reproduction profile. The accepted bridge retains the
legacy `ModelV2`/RolloutWorker stack through narrow compatibility adapters. Port
to RLModule/EnvRunner only as a later named factor. Unsloth remains relevant only
to a future LLM proposer fine-tune, not the numerical PPO/PBT solver.

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
- R4-v1-min position/remaining-horizon FF seam — implemented; blocking 4x3 and
  fresh 4x3x40 three-seed gates plus exact terminal replay are complete. Combined
  state passed paired net attribution on the fresh block, but all arms failed
  absolute promotion and one seed collapsed near-flat. No recurrent cross is
  economically authorized from this result.

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
- preserve the failed R4-v1 economic promotion decision while using
  `R0 + R4-v1` as the architecture-qualified forward FF baseline. R0 remains the
  immutable parity control. The December-2025 `zero_v1` replication is complete;
  the bounded actor-robustness search and causal forensics are also complete and
  economically rejected. The fresh-panel checkpoint/logit diagnostic is closed;
  open fixed-recipe matched R0, R0 + R4-v1, clean R1, and `R1 + R4-v1` cells next. Keep these
  separate from R2 carry/reset attribution;
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

`pyproject.toml`/`uv.lock` are authoritative: Python 3.11, Ray/RLlib 2.58.0,
Gymnasium 1.2.2, PyArrow 17.0.0, and setuptools 69.5.1. The old Phase B 2.47.1
line is historical; `itrader/compat/ray29/` records the reproduction profile.

The existing repository `.venv` remains the live Ray-2.9 reproduction surface;
the accepted Ray-2.58 environment passes the same 254-test suite. No recurrent
training result is reported here. The bounded Phase A probe validates effective
PBT search and evidence wiring, not reward learnability or trading performance.
Ray-owned PBT text files still truncate to zero bytes on this platform; stable
repository-owned provenance is authoritative.

A final read-only architecture/deslop pass found and closed three future-run
provenance gaps without altering completed run artifacts: snapshot content hashes
ignore interrupted atomic-write `.tmp` files, run manifests serialize the
resolved per-study PBT learning-rate bounds rather than the global/domain object,
and `best_checkpoint.json` records economic eligibility from the explicit
net/Sharpe/flat promotion score instead of inferring it from the active ranking
metric. The healthcheck and complete 236-test suite pass after these changes.

## Next priority

Freeze the completed R0 staged/exact-cost evidence and stop selecting against
the same crossed 32-task panel. The Ray-2.58 R0-only platform gate is accepted;
keep Ray 2.9 as a historical reproduction profile and never change native RLlib
stack plus solver in one attribution cell. After user review, the next controlled
cell is R1: a bounded,
reset-every-step chronological window encoder with the same dataset, task panel,
seeds, cost schedules, PBT cadence, compute, and economic gates. R1 is a
representation/shape control, not persistent-memory evidence. Native R2 work
requires lifecycle, mask, reset, checkpoint/replay, and carry-versus-reset tests.
Move economic selection to predeclared chronological blocks, 0.5/1/2 bp replay,
block uncertainty, and independent lineages. Keep the final test sealed.
After core R-series representation/lifecycle closure, run causal C0/C1/shuffled-
C1 as a separate factor. Keep System-One/CLM/Jev S-series work research-only
until the solver interface, platform, and evidence contract are frozen.

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
