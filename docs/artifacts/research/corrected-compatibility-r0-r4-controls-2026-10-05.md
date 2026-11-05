# Corrected-compatibility R0 / R0 + R4-v1 controls

Date: 2026-10-05 (Europe/Madrid). Status: corrected-control replay closeout complete; eight training cells and 256 witnessed terminal replay episodes independently accepted; 416 tests and pre-commit pass. No economic promotion. All earlier attempts preserved.
Supervisor: GPT-6 Luna at xhigh, delegated within the current research chat.
Source branch remains dirty; no staging, commit, push or artifact deletion.

## Authorized longer development bridge

### Attempt04: eight witnessed terminal replays complete

#### Final sign-off (16:38)

The independent evidence audit accepted **8/8 terminal replays**, with **0/8
economic promotions**. Every acceptance summary agrees with raw episodes;
checkpoint bytes were independently rehashed, and config, panel, runtime,
schema, source and producer declarations match their indexed counterparts.
The reviewer did not independently reload/rehash the full dataset; that remains
an explicit scope limit rather than a second dataset audit. See the
[independent review](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/replay_closeout_review.md).
All confirmed launcher, score, receipt-version and operator-reporting findings
have been fixed and independently re-reviewed. Required dependency healthcheck,
416 tests, all pre-commit hooks and `git diff --check` pass. The corrected-control
replay sprint is closed; claims of zero possible undiscovered defects, economic
generalization, fresh-panel qualification or effective PBT search are not made.

The monitor is being paused; no training/replay remains running. WSL is confirmed
stopped and approximately 5.8 GiB RAM is free at closeout. Keep the branch dirty
for human review. The next work is the accepted panel/completion/reporting
preparation for R1, not another R0 sweep or a native-platform integration.

#### Completion record (16:32; current status)

All eight terminals now have successful fresh-process receipts, exact indexed
checkpoint and episode-payload hashes, matching selection scores, live producer
identity witnesses, and Windows job accounting with zero active processes after
cleanup. Four receipts are in the first job-scope batch (cells 4--7); the other
four are in retry01 (cells 8,1,2,3). Each contains 32 episodes: **256 accepted
replay episodes**. The rejected first cell8 attempt is not included in that count.

| Seed | Representation / training cost | Terminal net return | Sharpe per sqrt-bar | Flat actions |
|---|---|---:|---:|---:|
| 20260924 | R0 / exact 1 bp | 0% | 0 | 100% |
| 20260924 | R0 / staged | -0.0002303% | -0.0017581 | 99.9805% |
| 20260924 | R0 + R4-v1 / exact 1 bp | 0% | 0 | 100% |
| 20260924 | R0 + R4-v1 / staged | +0.3927612% | +0.0188915 | 0% |
| 20260925 | R0 / exact 1 bp | 0% | 0 | 100% |
| 20260925 | R0 / staged | -0.1710680% | -0.0598841 | 76.5234% |
| 20260925 | R0 + R4-v1 / exact 1 bp | 0% | 0 | 100% |
| 20260925 | R0 + R4-v1 / staged | 0% | 0 | 100% |

Evaluation cost is fixed at 1 bp for every row. The positive seed is 95.08%
short, below the known short-only panel control (+0.441611% net / +0.024455
Sharpe). Staging remains a useful matched experimental factor, not a proven
generalizable edge. Five terminals are flat, two negative, one positive; no
replicated economic promotion, no PBT-search claim (PBT was disabled), and no
claim that 12 updates establish an architecture ceiling. Historical policy
outcomes remain evidence, while the corrected compatibility recipe is the
forward comparison authority.

Accepted replay minimum sampled free RAM was 2.134 GiB; actual Ray session logs
confirm CPU2. This does not isolate the cause of improvement over earlier RAM
pressure. The below-buffer rejected attempt and earlier 2 GiB-floor breach are
still recorded. WSL/Hermes was shut down at the human's request; no distributions
or Vmmem/wslhost remained, and subsequent replay used Windows only.

Portable evidence: [first batch](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/replay_jobscope_20261005/)
and [retry01](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/replay_jobscope_retry01_20261005/).
All 53 copied files were byte-hash checked against external run artifacts.
Serialized Windows pre-commit passed all hooks, including the required-dependency
healthcheck and **416 tests in 99.37 seconds**. `git diff --check` also passed.
The validation job retained 4.220 GiB minimum free RAM and ended with zero owned
processes. Source remains the separately recorded 105-file launcher/consumer
fingerprint `a8bf027a2f2f0f7e412c5f0f5fcf921642f0a29d0baa3d83039ec816fe9b6321`;
the evaluator remains the original 104-file snapshot. GitNexus is stale at
`1d5ba82` versus `d3eb4c5`; indexing is deferred, not used as authority.
Optional llama_cpp/transformers are absent and irrelevant to random/generated
Family00 replay. No GPU execution was introduced (installed Torch is CPU-only).
The final independent aggregate evidence review subsequently accepted all eight
terminals (sign-off above). No code was staged, committed or pushed.

Validation evidence: [pre-commit/full-suite log](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/closeout_precommit_20261005.log)
and [final operator source](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/replay_jobscope_closeout_final.py).

**Next boundary:** close final validation, then implement the accepted Q21
all-cell completion/panel-opening gate, resolve Q23 historical exposure before
calling the 128-task panel fresh, and test Q24 equal-symbol aggregation and Q25
per-actual-bar diagnostics. Keep corrected compatibility for the subsequent
fixed-recipe R0 / R0+R4-v1 / R1 / R1+R4-v1 comparison; native API migration
remains isolated and later. No R1 encoder was implemented in this closeout.

#### Windows-only completion pass (16:28; latest operational status)

Fresh job-scope replays of terminals4--7 passed, with minimum free RAM ranging
from 2.615 to 2.951 GiB. The positive staged R4-v1 seed reproduced
`net=0.0039276117295857`, `Sharpe=0.018891505246234376` per sqrt-bar, turnover
2.9375 and short/flat/long fractions 0.950846/0/0.049154. This is exact replay,
not replicated economic promotion. All four raw summaries agree with training.

Cell8 stopped at the proactive memory guard: 1,286,672,384 bytes (1.198 GiB)
free, above the human's hard1GiB floor but below the1.25GiB safety buffer.
The Windows job reported zero active processes and no cleanup errors. All95
observed process identities across cells4--8 were subsequently absent. Preserve
the failure; no acceptance was emitted. A concurrent WSL startup following a
repository knowledge-status check likely contributed, but no isolated causal
memory comparison was performed. The human then explicitly requested stopping
WSL/Hermes: `wsl --shutdown` completed, no distributions or Vmmem/wslhost remained,
and available RAM recovered to5.42GiB. No environment/data was deleted. No
Windows Hermes startup task/service was found; Linux auto-start configuration
was not inspected by restarting WSL. All further work is Windows-only.

The human reaffirmed completion. One controlled retry sequence8,1,2,3 uses a
new non-overwriting output root, `study_attempt04_terminal_replay_jobscope_retry01_20261005`.
Cell8 has now passed all32episodes with2.134GiB minimum free RAM and flat
terminal metrics. The three witnessed repeats are still pending. The old
cell8 failure remains at its original path. The output-root CLI guard was
independently tested against outside paths, traversal, root itself and existing
cell folders; all reject before launch and preserve existing evidence.

The final operator review also caught a reporting-only defect: copying an
unverified receipt summary after validating raw episodes could publish a
tampered summary. The driver now recomputes acceptance summaries from episodes.
The reviewer exercised its actual acceptance path with a false999.0 net claim:
the fixed output correctly contained0.0039276117295857. Existingcell4 predates
this fix, but its saved summary independently matches all raw fields and is
preserved unchanged. Later cells stamp the changed operator hash separately.
Both score/version and process-scope fixes now have independent approval; the
operator fixes also have fresh approval. Full sprint closure still awaits all
eight witnessed receipts and serialized repository validation.

Portable first-batch evidence: [replay_jobscope_20261005](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/replay_jobscope_20261005/)
(29 files copied and byte-hash verified against the run root).

#### Reviewed replay renewal (16:10; supersedes earlier operational deadlines)

The human removed the fixed session deadline and requested completion as soon
as practical. The new finite scope is eight sequential fresh terminal replays,
each capped at 300 seconds, ordered 4,5,6,7,8,1,2,3. Repeating terminal3 as well
gives every cell the same complete lifetime witness; its original resource
breach remains historical. Hard available-RAM reserve is 1 GiB, with proactive
abort at 1.25 GiB and 5 GiB measured startup headroom. CPU arithmetic, Ray node
CPU2 and Torch8 remain unchanged; no new training or GPU parity factor.

At 15:37 the replacement external PowerShell watchdog failed with a hashtable
type error. It stopped only its recorded launcher/helper subset; its saved
`cleanup_zero_survivors` field must **not** be interpreted as full descendant
containment. Later replay descendants were found and exited naturally. The
producer failed waiting for its Ray GCS port file, leaving only a running
receipt and no episodes. Sampled free RAM remained above 6.62 GiB: this was
not a demonstrated memory-floor failure. Preserve the entire failed attempt
under `study_attempt04_terminal_replay_resumed20261005/replay_04_dfa55`.

The durable launcher now creates the Windows child suspended, assigns it to a
kill-on-close job before executing, and queries job accounting for zero active
processes after cleanup. Timeout/resource failure and missing or inconclusive
cleanup fields cannot pass the launch-success gate. Both independent reviewers
(GPT-6.1 Sol/MAX and GPT-6 Luna/xhigh) approved real fast-root-exit, timeout,
child/grandchild and injected-fault probes. Unrelated sentinels survived.
Frozen-baseline fast-exit probes leaked descendants; current Windows probes
contained them. POSIX fallback is not runtime-validated on this workstation.

Before repair, 174 runtime/config files were copied into
`frozen_source_944d7d30`; its 104 Python files match the original training source
fingerprint exactly. New replays execute that frozen evaluator, while recording
the repaired launcher's current fingerprint separately. The main dirty tree
has changed and must not be labelled with the old fingerprint. Outputs go to
`study_attempt04_terminal_replay_jobscope20261005`, never overwriting originals.

The terminal consumer also requires literal parity booleans and independently
reconstructs the indexed selection score; receipt version must be integer 1,
not boolean true or float 1.0. Luna independently approved valid frozen-metric
equivalence and adversarial actual-receipt substitutions; second-model review
is pending. These repairs alter evidence acceptance, not trained policy weights.
Full replay/sprint acceptance remains open until the fresh receipts and final
validation pass. The requested native Bugbot launcher remains unavailable;
these are the repository's independent REVIEW checks, not a Bugbot result.

#### Renewed replay authorization (15:16)

**Human resource-policy revision, 15:31:** minimum 1 GB available RAM is now
explicitly authorized; operational supervision uses the slightly more
conservative **1 GiB** hard runtime floor. The earlier 7 GiB startup cushion
and 2 GiB runtime floor below are superseded prospectively, not retroactively.
Use measured replay headroom (5 GiB startup under the two-CPU Ray allocation),
unchanged eight PyTorch threads and the same CPU execution path; do not silently
introduce GPU arithmetic into this parity study. The original 15:46 deadline
remains. The external watchdog's synthetic root-exit/child-survival/identity-bound
cleanup probe passed at 15:30 with zero recorded survivors; evidence is under
`study_attempt04_terminal_replay_resumed20261005/watchdog_probe/`.
Parent cleared Luna to begin the seven sequential jobs with that watchdog.

The human explicitly requested completion after freeing ChatGPT memory. Parent
read 7.44 GiB available, no Python/Ray processes and the unchanged 104-file
source hash. Luna/xhigh owns seven sequential replay jobs: remaining terminals
4--8 plus witnessed repeats of 1--2, with fresh outputs under
`study_attempt04_terminal_replay_resumed20261005`. Terminal 3 is not repeated;
its valid parity and witnessed producer are retained alongside its failed
resource gate. New startup floor is conservatively **7 GiB**, runtime floor
remains **2 GiB**, and this replay-only authorization ends at **15:46 Madrid**.
The existing launch helper records live driver PID/start/launch-ID witnesses;
resource checks must run concurrently. Stop on any critical gate failure.
No runtime source or checkpoint configuration changes precede these replays.
At 15:23 the parent verified the installed Ray resource resolver accepts a
child-local `RAY_OVERRIDE_RESOURCES={"CPU":2}` override without starting Ray.
Training already used two Ray CPUs; the replay entry point otherwise auto-detects
the whole machine. Resumed launches explicitly carry this operational factor,
record it, and check actual node resources in logs. This may reduce idle worker
startup pressure; causality and memory savings remain unproven until measured.
The 7/2 GiB gates and deadline are unchanged. At 15:23 no new replay had started.
The five-minute heartbeat is active again for this bounded phase only.

GRILL confirmed no newer accepted design decision is missing from readiness:
Q21 runner/panel opening, Q23 historical exposure, and Q24/Q25 aggregation
implementation remain outstanding. This closeout does not implement R1.
The requested Bugbot-specific launcher is unavailable despite its skill being
installed; record that limitation, not a successful Bugbot run. The independent
repository review ladder is proceeding with its normal code-review model and
a separate Luna/xhigh evidence check. Parent owns any necessary implementation.

At 15:25 the fresh Sol/MAX reviewer reproduced a blocking launcher-safety
defect: `launch_replay` times out its root process but can leave observed child
processes alive. The synthetic probe used a sleeping child, not Ray or market
data; the reviewer stopped only its exact witnessed test descendants. New
replays were held while the independent supervisor cleanup was checked. This
does not establish leakage in prior runs or invalidate episode parity, but
sprint closure requires correction and fresh review of this timeout path.

#### Independent closeout review findings (before fixes)

The fresh GPT-6.1 Sol/MAX code review and separate GPT-6 Luna/xhigh evidence
review both returned `VERDICT: CHANGES`. These are the repository REVIEW
fallback, **not** a native Bugbot run. Confirmed findings:

1. **BLOCKING, launcher lifetime and timeout acceptance.** A real sleeping-child
   subprocess survived the helper's root timeout. Separately, adding
   `timed_out=true` to an otherwise successful witnessed launch still passed
   process replication. Expected: stop all exact witnessed owned instances and
   reject timeout independently of exit code. No real replay timeout/leak was
   established by these synthetic probes.
2. **BLOCKING, contradictory score evidence.** An in-memory substitution in the
   actual `01e66` receipt set `selection_score_parity=false` and changed
   `replayed_score` by +0.01; the terminal analyzer still accepted its unchanged
   indexed episode payload. Expected: reject contradiction and independently
   bind reported selection scores to the index and replay summary. Actual
   receipts were not altered and all three existing receipts are consistent.
3. **NIT, strict receipt version.** JSON boolean `true` was accepted as integer
   version 1. Require an integer type as well as its value.

Sol executed 9 non-training diagnostic tests, 4 replay-integrity tests, 24
selected reporting/schedule/loadout/resource tests and 9 checkpoint-selection
tests; six controlled PPO loss cases matched the installed original loss,
statistics and RNG. Four Adam/schedule probes at steps 0/40/100/200 preserved
moments, parameters, KL and native clocks while reconciling target rates.
Luna independently validated the three real receipts, episode contracts,
checkpoint bytes, unchanged source, and the third replay's live producer
witness. Dataset bytes were not rehashed by that reviewer. Neither reviewer
launched Ray/training/full-data evaluation. Base comparison used `d3eb4c5`;
new receipt/launcher modules are absent there, and its trainer has a pre-existing
indentation error, so executable consumer comparisons were isolated.
All five design principles were reported: separation/cohesion/coupling OK;
intention and encapsulation violations are the findings above. No finding
permits erasing historical results or claiming the current controls promoted.

#### Historical safety hold (13:46--13:48; renewal above)

Replay 3 started at 13:44:50 after its startup gate passed, but Luna observed
available RAM drop to **1.47 GiB**, below the hard **2 GiB runtime floor**.
It finished naturally before the exact-process stop check; **no process was
killed**. Parent and supervisor subsequently found no active Python/Ray/replay
process, and RAM recovered to approximately 5.30 GiB. This recovery does not
erase the resource-gate breach. All further launches are held, and heartbeat
`monitor-corrected-r0-r4-attempt04` was explicitly set to `PAUSED`.

Three terminals have complete 32-episode receipts with exact checkpoint,
episode-payload and selection-score parity (96 replay episodes total).
Replay 2's selected/replayed score is `-2.3030470476790425e-06`; an informal
supervisor message omitted its exponent, but the saved receipt is correct.
Replays 1 and 2 lack a saved live launcher/driver witness; preserve that limitation
rather than inferring parentage. Replay 3 uses the existing launch helper:
`launch_03_01e66.json` records return code 0 and observed descendants; actual
receipt producer **PID 6412** and its start time match that record and launch ID.
The launcher PID is 37216, not the receipt producer. Its identity checks pass,
but its resource gate failed. These are separate verdicts.

No fourth replay or repeat has been authorized to proceed after this stop.
Before resuming, define a safer replay resource envelope and persist launch
witnesses for every accepted terminal. Five terminals remain unevaluated;
the first two also need witnessed repeats for closure. Independent review
still awaits agent capacity. All eight trained checkpoints and original evidence
remain intact, source fingerprint is unchanged, and no training rerun is needed.

Read-only resource lead: `run_checkpoint_replay.py` calls
`ray.init(include_dashboard=False)` without an explicit CPU cap, unlike the
training launcher's configured two-CPU node. The witnessed replay tree contains
many Ray descendants. This is a candidate explanation to investigate, not a
measured causal attribution or permission to edit the frozen runtime here.
No OOM was observed; the safety floor, not an OOM or economic failure, stopped
further replay execution. Chronological progress below predates this hold.

The official runner completed all eight cells and wrote its comparison summary
at **13:30:16 Europe/Madrid**, approximately 49 minutes after launch and before
the 14:41:24 deadline. All stable trial statuses are completed. Total sampled
exposure is **467,584 steps** versus 460,800 nominal (6,784 complete-episode
overshoot): 58,256 per first-seed cell and 58,640 per second-seed cell.
The owned launcher/trainer processes exited; no training restart is authorized.

| Representation / training cost | Seed 20260924: net / Sharpe | Seed 20260925: net / Sharpe |
|---|---:|---:|
| R0 / exact 1 bp | 0% / 0 | 0% / 0 |
| R0 / staged cost | -0.000230305% / -0.001758110 | -0.171067951% / -0.059884074 |
| R0 + R4-v1 / exact 1 bp | 0% / 0 | 0% / 0 |
| R0 + R4-v1 / staged cost | +0.392761173% / +0.018891505 | 0% / 0 |

These are terminal, unweighted episode means on the **inspected January 32-task
development panel**, evaluated at 1 bp. Sharpe is per sqrt-bar, not annualized.
All four exact-cost cells and the second-seed staged R4 cell end fully flat.
The other staged R0 cells end 99.980469% and 76.523438% flat respectively.
The first-seed staged R4 result is non-flat but below the known short-only
control; it does not replicate on the second seed. Staging remains a useful
explicit comparison factor, not a demonstrated robust economic improvement.
This PBT-disabled bridge says nothing new about exploit/mutation effectiveness.

Parent read-only verification executed across **all 96 report rows and 33 indexed
checkpoints**: raw/public train and evaluation gross/net/Sharpe/reward agreement,
gross-minus-cost identities, fixed-1-bp evaluation reward scaling, finite PPO
diagnostics, clipped/valid-visit ratios, original checkpoint digests and matched
source/dataset/task-panel identities all pass. Exact-cost training reward scaling
also passes; the staged optimizer objective remains distinct from fixed-cost
training economics. The frozen 104-file source fingerprint is unchanged.
Observed in-run available RAM remained above 2 GiB (about 2.48 GiB in the final
cell), recovering to 5.53 GiB after training. This is sampled, not continuous,
resource evidence; no OOM or resource-policy relaxation was observed.

Luna owns **eight sequential fresh-process terminal replays only**, through
the official replay tool, under `study_attempt04_terminal_replay`. Keep source
and configuration frozen until those receipts pass checkpoint, episode-payload,
score and producer-identity checks. Retained-best checkpoints cannot substitute
for terminals. Independent evidence review and dashboard indexing follow replay;
training completion alone does not close the mechanics gate. No fresh 128-task
panel, final test, native integration or R1 implementation was opened.

**13:37 replay preflight hold:** all eight official terminal-input selections
passed checkpoint/index/source/dataset/panel checks, but available RAM fell to
3.56 GiB and recovered only to approximately 4.12 GiB, below the unchanged
5 GiB startup gate. No replay has launched; no owned Ray/trainer/replay process
remains to clean up. Luna will recheck headroom without stopping unrelated apps
or lowering the gate. There is no replay-failure receipt and no lost training.
Independent closure review is also pending: the attempted fresh-reviewer spawn
was rejected by the agent-thread limit, not waived or counted as approval.

**13:42 replay update:** RAM recovered to 5.37 GiB and terminal replay 1
(R0/exact cost, seed 20260924) ran at 13:38:37--13:39:00. Its 32-episode receipt
passes checkpoint, episode-payload and score parity, with matching protected
identities. Launcher PID 29584 differs from receipt producer PID 35740. The
existing replay harness explicitly supports Windows venv launcher children,
but no live descendant witness was saved for this launch; do not claim observed
parentage after the processes exited. Preserve the successful replay evidence
with that limitation. Luna will use the existing `launch_replay` helper for
each remaining terminal, persist its observed-process launch record, then repeat
only terminal 1 once in a distinct output directory with the same witness.
No source or configuration change is needed. Study closure remains pending.

Official outputs were copied byte-for-byte into the dated evidence directory;
all 29 copied files were SHA-256 checked against their originals:

- [Eight-cell summary](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/study_attempt04_summary/summary.json).
- [Net-return curves](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/study_attempt04_summary/ablation_compare.png).
- [Sharpe curves](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/study_attempt04_summary/ablation_compare_validation_sharpe.png).
- [Action fractions](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/study_attempt04_summary/ablation_compare_action_fractions.png).
- [PPO and behavior diagnostics](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/study_attempt04_summary/ablation_compare_diagnostics.png).

Earlier chronological observations are retained below, not current-status claims.

#### What the positive staged-state trajectory does and does not show

For seed 20260924, staged R4-v1 has positive 1-bp evaluation net at **11 of
12 updates**, but positive per-sqrt-bar Sharpe at only **4 of 12** (1, 2, 9, 12).
Updates 1--2 are exactly short-only: +0.441611% net / +0.024455 Sharpe.
Updates 6--8 and 10--11 repeat +0.207477% net with slightly negative Sharpe
(-0.000678467), 75% short / 25% long and turnover 2. Thus positive mean return
does not imply consistently positive risk-adjusted performance. At update 12,
gross +0.422136% less cost 0.029375% produces net +0.392761%; turnover 2.9375
and actions 95.084635% short / 0% flat / 4.915365% long. That terminal is about
0.048850 percentage points below the known short-only net control.

For seed 20260925, the same staged representation shows positive net **and**
Sharpe at updates **2--6**. At updates 3--5 it holds +0.196209% net /
+0.016931 Sharpe, 75% short, 25% long, turnover 2. After the optimizer-cost
schedule enters 0.5 bp, update 7 has turnover 16.5625, 40.253906% flat and
-0.144746% net; update 8 has 77.359375% flat and -0.171239% net. It is fully
flat at update 9, before the first reported 1-bp stage at update 10. This timing
is consistent with cost-sensitive policy collapse, but schedule stage, training
exposure and optimizer changes co-vary: it does not isolate cost as the cause.
Evaluation stays at 1 bp throughout both trajectories.

The useful result is transient economic activity on both seeds and terminal
activity on one, not a replicated edge. All exact-cost terminals are flat.
Terminal critic explained variance remains small (approximately 0.00924 and
0.02344 for the staged R4 seeds); finite PPO updates and positive economic
points alone do not establish a strong critic. PBT is off, so there are no
exploits or mutations to attribute these transitions to. Historical PBT results
remain qualified by their effective-optimizer audit, not directly interchangeable
with this short fixed-recipe comparison. Do not increase the search budget or
open the fresh panel merely because one terminal is positive.

#### Experiment Browser publication

The browser now indexes this bridge separately from the earlier smoke, under
`corrected-compat-r0-r4-bridge-attempt04-20261005` (six experiments / 33 records).
Its existing ablation adapter reads the official terminal net-return field only;
Sharpe/action/PPO trajectories are linked through this report rather than invented
as missing browser metrics. All eight displayed net values were checked against
the original summary, and re-rendering is byte-identical to the generated HTML.
The 44 generic browser tests pass. One existing repository-snapshot test has a
hard-coded five-experiment ID list and was deliberately deselected: that list
needs the new ID after the Python-source freeze ends. The equivalent production
render/source checks were executed separately; this is not a claim that all
45 tests pass unchanged. No Python or runtime/config file was changed.
Fresh GPT-6.1 Sol/MAX scoped display review found no blocking or nit findings.
It checked schema/base-record preservation, native-byte production render equality,
all eight terminal net values, all 96 net-curve points, 29 copied-file SHA-256
pairs, local links and virtual-DOM behavior over 36 ordered comparison selections.
All five REVIEW.md design principles passed. This approves the provisional
display only; pixel layout, resource acceptance and full replay/study closure
remain unverified. The referenced browser-tab mention was stale; reopening was
queued in the right panel, not claimed as a visually verified refresh.

Official full eight-cell restart launched **12:41:24 Europe/Madrid**, launcher
PID **10480**, startup available RAM **5.64 GiB**. Run tag:
`phase_a_corrected_ray258_r0_r4v1_fixed_2x2_cost_20261005_20261005_124125`.
First cell is `r0_exact1bp_seed20260924`; initial in-run RAM was 3.24 GiB.
At update 3 / 14,712 sampled steps, available RAM was 2.61 GiB and derived
training net return exactly matched raw runner net return
(`-0.014484303960688373`). Evaluation net `-0.004204373237776498` and clip fraction
`0.007106633771929825` reproduced attempt03 at the same update. This is a
bounded consistency observation, not full trajectory or study parity.
At the 12:47 heartbeat, cell 1 completed **12 updates / 58,256 steps** and
cell 2 (`r0_causal_cost_seed20260924`) started. Independent read-only checks of
all 12 cell-1 rows confirm exact raw/derived gross, net and Sharpe agreement,
reward-scale accounting within 1e-12 and finite PPO diagnostics. Terminal
evaluation is all-flat, net/Sharpe/turnover zero. Terminal checkpoint digest is
`9aa552d88930dafe409975a137f923868041a83d5f04b3b23a983d7b715e1705`;
indexed episode-payload digest is
`45f6454dd793a2970b645a3c1c198d303c607499140ece839dcf8492a0145366`.
Fresh terminal replay remains pending. All eight variants remain included;
one completed cell does not close the matched comparison.

At the 12:52 heartbeat, cell 2 (staged-cost R0, seed 20260924) also completed
12 updates / 58,256 steps. Its four indexed checkpoints (updates 1/4/8/12)
hash-match. All 12 raw/derived economics and fixed-cost gross-minus-cost identities
pass. Training stage costs are exactly `[0,0,0,0.25,0.25,0.25,0.5,0.5,0.5,1,1,1]`,
applied from each update's starting exposure; evaluation always remains at 1 bp.
Terminal net is **-0.000230305%**, Sharpe **-0.001758110 per sqrt-bar**, flat
fraction **99.980469%**. Staging retained activity longer than exact-cost R0
(0% flat through update 5 versus 100% flat at exact-cost update 5), but this
paired-seed trajectory still ended near-flat, not economically promoted.
Its update-1 +0.441611% net / +0.024455 Sharpe matches the known directional
control and must not be presented as demonstrated alpha. Fresh replay is pending.

Monitoring correction: scheduled training optimization reward divided by 1,000
must **not** be equated with the always-1-bp training economic metric. Verify
each against its own raw callback metric; fixed-cost evaluation reward/1,000
still matches evaluation net. A supervisor's provisional parity warning came
from applying the exact-cost invariant to the staged arm, not a runtime defect.
Final-stage aggregate differences may include earlier-cost episodes; inspect
the aggregation window before claiming every aggregate reflects only the current
stage. No reward/report source was changed during this check.

At 12:57, cell 3 (R0 + R4-v1 / exact 1 bp, seed 20260924) was at update 9 /
43,752 steps with 2.65--2.67 GiB available RAM. Evaluation net stayed positive
through updates 1--8, while Sharpe was mixed (positive at 1/2/3/6/8). At update 9,
net became -0.033860%, Sharpe -0.019643 per sqrt-bar and flat fraction 97.6172%.
All nine raw/derived economic and reward-scale checks passed, PPO metrics were
finite, and indexed checkpoint hashes at 1/4/8 matched. This is an interim
trajectory, not a retained economic improvement or terminal verdict.

At 13:02, cell 3 completed **12 updates / 58,256 steps**, ending fully flat:
evaluation net, Sharpe and turnover all zero. All 12 reporting/accounting rows
and four checkpoint hashes (updates 1/4/8/12) pass. Terminal checkpoint SHA-256:
`547334eada7c702198d19a79036bc958aace5c970b6a45ffa097d1368bf81b30`.
Fresh terminal replay remains pending. Cell 4, R0 + R4-v1 with staged cost on
the same seed, is active: at update 6 / 29,160 steps its optimizer stage is
0.25 bp, while evaluation remains fixed at 1 bp. Evaluation net is +0.207477%,
Sharpe -0.000678467 per sqrt-bar, actions 75% short / 0% flat / 25% long.
The six current report rows and checkpoint hashes at 1/4 pass, source is unchanged,
and RAM is about 2.68 GiB. This unfinished positive-return point is not promotion.

### First-seed terminal comparison (13:07 heartbeat; replay pending)

All four seed-20260924 cells completed 12 updates / 58,256 sampled steps each.
These are inspected-development-panel results, not fresh validation.

| Representation / training cost | Terminal net return | Sharpe per sqrt-bar | Flat actions |
|---|---:|---:|---:|
| R0 / exact 1 bp | 0% | 0 | 100% |
| R0 / staged cost | -0.000230305% | -0.001758110 | 99.980469% |
| R0 + R4-v1 / exact 1 bp | 0% | 0 | 100% |
| R0 + R4-v1 / staged cost | +0.392761173% | +0.018891505 | 0% |

The combined-state staged arm ends with 95.084635% short / 4.915365% long,
turnover 2.9375, gross +0.422136173% less cost 0.029375%. All 12 report rows
pass raw/derived and fixed-cost economic-accounting checks; all four indexed
checkpoint hashes match. Terminal checkpoint SHA-256:
`102ad678c3b39b57458de7c4e0811ee8036a84c869ebb440b54e883fef1890b5`.
This is better than the other three terminal cells for this seed, but below
the known short-only development control (+0.441611%, Sharpe +0.024455).
Do not claim alpha, causal interaction significance, or promotion. Complete the
paired second seed and fresh-process replay first; later fresh-panel gates remain.

Cell 5 (R0 / exact 1 bp, seed 20260925) is active: update 4 / 19,576 steps,
RAM about 2.56 GiB. Its current raw/derived accounting, finite PPO diagnostics
and indexed checkpoint hashes at 1/2/4 pass; source remains unchanged.

At 13:12, cell 5 completed **12 updates / 58,640 steps** (+1,040 complete-episode
overshoot). Second-seed R0/exact-cost terminal evaluation is all-flat: net,
Sharpe and turnover zero. All 12 reporting/accounting rows and finite PPO
diagnostics pass; all six retained checkpoint hashes (updates 1/2/4/7/8/12)
match their indices. Terminal SHA-256:
`13d9badbcb2137823e88c280f21279cd88d5d77442adc836c874453c8490ea5b`.
Cell 6, staged-cost R0 on seed 20260925, is active at update 2 / 9,800 steps,
optimizer cost 0 bp and fixed evaluation cost 1 bp. Both current reporting rows
and its first checkpoint hash pass. Source remains frozen; RAM is about 2.60 GiB.
Five of eight training cells are complete; terminal fresh replays remain pending.

At 13:17, cell 6 (second-seed staged-cost R0) completed **12 / 58,640 steps**.
Terminal evaluation net **-0.171067951%**, Sharpe **-0.059884074 per sqrt-bar**,
turnover **20.0625** and actions **17.773438% short / 76.523438% flat /
5.703125% long**. All 12 reporting/accounting and finite-PPO checks pass; the
four indexed checkpoint hashes at 1/4/8/12 match. Terminal checkpoint SHA-256:
`4f30d4ae5cb6a591fb2c25cdd00f894c9d46fa197ca076adcfd46d833b378ee0`.
Cell 7 (R0 + R4-v1 / exact 1 bp, seed 20260925) started at 13:17:58, trial
`training_fn_ITraderEnv_6b27c_00000_0_2026-10-05_13-17-58`.
Memory recovered between cells and is about 2.49 GiB in the new cell, above
the unchanged runtime floor; source is unchanged. Six of eight complete,
all terminal replays still pending. Staging is not uniformly beneficial across
the currently completed controls; do not generalize from the first-seed R4 result.

**Cost-boundary observability limitation:** cell 2's reward/1,000 versus fixed-cost
net difference at the first 1-bp update disappears at updates 11/12. The effective
metric smoothing setting is 32 episodes; stage overrides apply on subsequent
environment resets, so an episode already active at the boundary may retain its
previous cost. This is boundary-consistent, but per-episode optimizer-cost stamps
are not persisted, so exact causal attribution cannot be recovered from these
aggregates. Keep it as a reporting improvement for the next unfrozen pass; it
does not contradict the observed raw/derived or fixed-cost economic identities.

- Source: `944d7d308654b93b090fec83a8097728157405671ae8aff858d2d00658fcf233`.
- Summary: `C:/Users/Admin/ray_results/corrected-compat-r0-r4v1-20261005/study_attempt04_summary`.
- Logs: `C:/Users/Admin/ray_results/corrected-compat-r0-r4v1-20261005/study_attempt04_logs`.
  Files are `run_ablation.stdout.log` and `run_ablation.stderr.log`.
- Hard training deadline: **14:41:24 Europe/Madrid**; 5 GiB per-cell startup
  and 2 GiB runtime floors retained. The user permitted lower headroom, but
  recovery made a resource-policy change unnecessary.
- Luna/xhigh remains live supervisor. Thread heartbeat
  `monitor-corrected-r0-r4-attempt04` checks every five minutes as a recovery and
  notification mechanism; it must coordinate and never launch duplicate work.
- Compare raw and derived economic metrics during every cell, then replay
  terminal checkpoints sequentially. No fresh 128-task panel or R1 implementation.

### Preserved attempt03

Official launch: **2026-10-05 12:09:28 Europe/Madrid**, launcher PID **8672**,
startup available RAM **5.34 GiB**. Outputs are isolated under
`C:/Users/Admin/ray_results/corrected-compat-r0-r4v1-20261005/study_attempt03_summary`
and `study_attempt03_logs`. Launch is not a completed or healthy-cell verdict;
record reports and resource observations as they arrive.

Run tag:
`phase_a_corrected_ray258_r0_r4v1_fixed_2x2_cost_20261005_20261005_120928`.
First cell `r0_exact1bp_seed20260924` launched at 12:09:38, trainer PID 40188,
an owned descendant of launcher 8672. Initial configuration readback confirms
native 30 epochs / minibatch 128, approved LR/PPO settings, PBT off, fixed 1 bp
and 57,600 target steps. Available RAM after startup was 2.46 GiB, above the
declared 2 GiB runtime floor; ongoing supervision is required.

**Reporting hold at 12:18:32:** cell 1 completed 12 updates / 58,256 sampled
steps (+656 over nominal). The launcher had already started cell 2, which stopped
at two updates / 9,768 steps, before its first cadence checkpoint. Only the
owned launcher/descendant tree was stopped; all 24 captured processes exited,
and available RAM recovered to 5.59 GiB. Both complete and partial artifacts
remain preserved. This was a conservative correctness stop, not OOM or proof
that training was contaminated. The full eight-cell study is incomplete.

**Open reporting finding during this run:** an empty top-level `custom_metrics`
dictionary shadows populated `env_runners/custom_metrics` in
`_extract_custom_metrics`. Consequently derived training economic fields can
default to zero while optimization reward falls back to Ray's populated metric.
At update 10 the persisted raw values are optimization reward
`-10.491749017929932` and economic net return `-0.010491749017929933`, consistent
with the x1,000 scale, while derived `training/net_return_mean` is zero. Do not
interpret those derived zeros as measured flat-policy behavior. Runtime source
is unchanged pending terminal replay. Independent Sol/max review reproduced
the discrepancy in all 12 training reports and also at base `d3eb4c5`.
All 12 derived evaluation gross/net/Sharpe values match raw evaluation values;
checkpoint hashes at updates 4/8/12 match their indices. PPO reward divided
by 1,000 matches raw training net return throughout. Raw metrics permit derived
report recovery without rewriting originals. Finish old-source terminal replay,
repair and independently review reporting, then declare a fresh source snapshot
before restarting the matched study. Do not mix old/new source cells silently.

### Attempt03 terminal replay and repair validation

The completed R0/exact-cost terminal replay passed in a fresh process on the
unchanged source (`5015ddc9...e9a2181f`). All 32 episode payloads and action/reward
traces match; selected/replayed score is 0. Every episode is genuinely all-flat,
with zero gross/net return, cost, turnover and Sharpe. This establishes the
terminal control outcome, not absence of learning at every earlier update.

| Attempt03 R0 / exact 1 bp update | Sampled steps | Evaluation net return | Sharpe per sqrt-bar | Flat actions |
|---|---:|---:|---:|---:|
| 1 | 4,944 | -0.059056% | -0.044665 | 0% |
| 2 | 9,768 | -0.172701% | -0.050027 | 26.1563% |
| 3 | 14,712 | -0.420437% | -0.075081 | 50.0586% |
| 4 | 19,520 | -0.001682% | -0.008036 | 99.9089% |
| 5 through 12 | 24,320 through 58,256 | 0% | 0 | 100% |

PPO was not inert: measured clip fraction varied from 0 to 0.115519, with
terminal 0.022860; explained variance remained low (terminal 0.006221).
These are descriptive diagnostics, not a causal proof that cost alone caused
collapse. The matched staged-cost and combined-state arms are still required.

- Checkpoint SHA-256: `09307428fda4f83a7171a201c1eb7de156beee4c7dc233ff157a4e3fc82bfa9a`.
- Payload SHA-256: `45f6454dd793a2970b645a3c1c198d303c607499140ece839dcf8492a0145366`.
- Receipt directory: `C:/Users/Admin/ray_results/corrected-compat-r0-r4v1-20261005/study_attempt03_cell1_terminal_replay_attempt02`.
- Replay attempt01 mistakenly supplied the Tune-local trial directory instead
  of `_stable_metrics/<trial>` and failed before data/Ray load; logs are retained.
  Attempt02 used the verified stable directory and completed all parity gates.

The reporting repair now skips empty compatibility metric wrappers and leaves
absent economic observations as NaN, not fabricated zero. Real measured zero
remains zero. It does not change PPO, reward, action or evaluation logic.
The existing actual-PPO/checkpoint/replay integration test was extended through
CSV and flattened report persistence: red reported zero versus measured
`-2.6042683959009646e-05`; after repair both state modes passed. A second red/green
case verifies missing versus measured-zero economics. Focused training,
representation-integration and ablation tests: **132 passed**. The full suite
passed **412 tests**, healthcheck passed and serialized `pre_commit run --all-files`
passed. Independent GPT-6.1 Sol/MAX and Astra reviews both returned
`VERDICT: APPROVE` with no blockers or nits in this scoped repair. Sol reconstructed
19 retained raw rows / 114 economic fields exactly and executed 16 additional
base/current probes plus 10 focused tests. Astra executed 15 focused tests,
10 base/current and five persistence/selection probes. Both verified missing
economics cannot trigger eligible off-cadence retention and all five design
principles were addressed. Neither review claims to re-audit the entire dirty tree.

Attempt04 is cleared for preflight under the unchanged loadout and resource
limits. Its frozen 104-file Python source SHA-256 is
`944d7d308654b93b090fec83a8097728157405671ae8aff858d2d00658fcf233`.
Dataset, panel-file and loadout hashes were rechecked unchanged. Record actual
launch identity separately; clearance is not proof of launch or completion.
GitNexus remains stale at `1d5ba82` versus HEAD `d3eb4c5`; its Bash wrapper reports
missing `python3`, while the repository Python status summary succeeds.

A preceding launcher setup failed before training: PowerShell rejected
`New-Item -LiteralPath`; PID 45344 exited with no owned child, logs or study
outputs. Directory creation was corrected to `-Path`, absence was rechecked,
and the official launch above reused the previously unused attempt03 paths.
No training-source/configuration change or duplicate runner was introduced.

After reviewing the smoke, the human explicitly authorized the existing eight-cell
corrected-compatibility loadout: two representations (R0 / R0 + R4-v1), two cost
curricula (exact 1 bp / compressed causal staging), and two paired seeds. Each
cell has 12 updates / 57,600 nominal sampled steps, for 460,800 nominal training
steps total; record complete-episode overshoot. PBT remains disabled. Luna/xhigh
supervises official execution and sequential replay with the declared resource
and two-hour global stop gates. Record actual launch identities below when known;
authorization alone is not a running/completed result.

Renewed launch check found approximately 4.40 GiB available physical RAM, below
the unchanged 5 GiB startup gate. At that initial check the run had not launched.
No unrelated application was stopped; request approximately 1 GiB of headroom
and recheck all guards before launch. This is a resource hold, not cancellation
of the human's explicit longer-run authorization.

RAM subsequently recovered above 5 GiB; require a fresh passing read at launch.
The new 104-file Python fingerprint is
`5015ddc9fb84c9cfc71854f8070bb661aa1427132eab071ad174ff90e9a2181f`.
Initial test-line-only reversal did not match the smoke fingerprint. The exact
cause was the added browser-registry test ID plus three nearby line-ending
changes: removing that line and restoring only that test file's CRLF endings
in memory reproduces the original attempt-02 fingerprint
`b2739469a5a601aceb95b69e19d5f08ce44e8b8763b94300ebb7ea4f79234e95`.
No disk restoration was performed and no runtime Python drift remains in this
comparison. Freeze the actual new fingerprint for the longer study; do not
relabel it as the old bytes. Config, panel and dataset hashes are unchanged.

This is the separately declared January 32-task **development bridge**, not the
Q21 16-cell target screen. It does not authorize fresh 128-task terminal results,
an all-16 completion bypass, economic promotion or R1 implementation. The native
migration remains isolated and will be integrated later under the approved order.
Freeze runtime source and loadout during cells; docs-only updates are permitted.

## Completed smoke: mechanics, not promotion

Both official one-update cells completed under GPT-6 Luna/xhigh supervision.
Each collected 4,944 sampled environment steps against the nominal 4,800 target
(144 steps of recorded complete-episode overshoot), then evaluated the same
32-task January development panel at exact 1 bp. PBT was disabled throughout.

| Observed metric | R0 | R0 + R4-v1 |
|---|---:|---:|
| Mean gross return | +0.172819% | +0.461611% |
| Mean net return, 1 bp | -0.059056% | +0.441611% |
| Mean net Sharpe per sqrt-bar, not annualized | -0.04466533 | +0.02445458 |
| Short / flat / long action fractions | 82.43815% / 0% / 17.56185% | 100% / 0% / 0% |
| Mean economic turnover | 23.1875 | 2.0 |
| Mean cost return | 0.231875% | 0.020000% |
| Clipped / valid optimizer-sample visits | 0 / 145,920 | 804 / 145,920 |
| Clip fraction | 0 | 0.00550987 |
| Approximate KL | 0.00020183 | 0.00051533 |

Clip denominators count valid optimization sample visits, not unique market
bars. The native effective configuration readback confirms 30 epochs, minibatch
128, train batch 4,800, LR approximately `7.151674865214668e-6`, gamma `.99`,
lambda `.9`, clip `.1`, value coefficient `2`, value clip `20`, and the declared
entropy schedule. R4's economics equal the known short-only January control;
this is **not evidence of conditional trading edge or economic promotion**.
There is no update-zero comparison here, no independent-seed replication and
no fresh terminal-panel result. One update cannot establish learned uplift.

Fresh processes restored each indexed retained checkpoint and reproduced all
32 episodes. Both `payload_parity` (episode metrics and action/reward traces),
`checkpoint_parity`, and `selection_score_parity` are true. The replay contract
records training exposure as 4,944 steps; replay-local summary iteration/step
zero is not the checkpoint's training age. Canonical checkpoint SHA-256:

- R0: `c3939368bc2a89048710e75efbebf5132df8bc31cf8bfb4b82b17ea38731aee4`.
- R4: `a30ee697a029caece1515d29195fe31e1e9b8d40f70c51bc6efd85af6e83cd51`.

Training windows were approximately 45 seconds and 42 seconds respectively;
startup/replay costs and longer-run throughput cannot be inferred from this
single-update timing. Available physical RAM was 6.40 GiB before launch, with
5 GiB per-cell startup guards passing. The lowest sampled value was 3.35 GiB,
above the 2 GiB runtime floor; sampled monitoring is not a continuous peak-memory
measurement. After both replays it was approximately 6.30 GiB. All owned training
and replay processes exited; no eight-cell study was launched.

Ray warned about long Windows driver-artifact paths and custom policy durable
registration. Neither prevented these checkpoint writes or fresh restores.
Keep those warnings visible: two successful replays do not prove every future
path length/registration case safe. Attempt 01's pre-training failure is retained
below; no failed probe is relabelled as performance evidence.

### Durable outputs

- [Official ablation summary](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/smoke_attempt02_summary/summary.json).
- [Net-return plot](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/smoke_attempt02_summary/ablation_compare.png).
- [Sharpe plot](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/smoke_attempt02_summary/ablation_compare_validation_sharpe.png).
- [Action-fraction plot](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/smoke_attempt02_summary/ablation_compare_action_fractions.png).
- [PPO/behavior diagnostics](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/smoke_attempt02_summary/ablation_compare_diagnostics.png).
- [R0 fresh replay receipt](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/smoke_attempt02_replay/training_fn_ITraderEnv_11aad_00000_0_2026-10-05_10-45-08.json).
- [R4 fresh replay receipt](corrected-compatibility-r0-r4-controls-2026-10-05-evidence/smoke_attempt02_replay/training_fn_ITraderEnv_3333b_00000_0_2026-10-05_10-46-04.json).

The evidence directory retains both replay episode JSONL files alongside their
receipts. These are byte-preserving copies of official outputs; absolute source
paths remain in the original summaries. The experiment browser indexes the
official summary using its existing adapter; it does not invent missing fields.
After both replay processes exited, the browser registry test's expected ID list
was extended for this smoke. The recorded prelaunch Python-tree fingerprint
therefore identifies the executed snapshot, not an assertion that the current
dirty tree is byte-identical after this test-only reporting edit.

### Next gate

Q22 is human-approved: keep this corrected optimizer recipe identical across
all 16 future screen cells, with PBT off and no deliberate entropy-reset spikes.
The existing loadouts already declare it; no post-run configuration edit was
needed. Q21 initialization monitoring, fresh 128-task terminal-panel declaration
and all-16 completion enforcement remain required. Q23 metadata-only sampling
is now approved; eligibility audit and panel freeze remain pending. Q24's new
equal-symbol terminal-net summary is accepted but not implemented; it will not
rewrite this smoke's raw means. Q25 was subsequently accepted: secondary
net-bps-per-actual-bar diagnostics normalize per episode before hierarchical
aggregation; implementation remains pending and terminal net stays primary.
Do not launch that target screen or silently reuse the separate
eight-cell development bridge as qualifying evidence. This pass stops before
R1 implementation; native API work remains isolated.

## Resumed smoke and bounded correction

### Q23 metadata-only eligibility preflight (not a frozen panel)

Luna projected only regular-session timestamps and session identifiers from all
12 symbols: 5,515,061 rows, without loading price, volume, return or regime
values. Using the current 70/15/15 chronological partition and 200-row purge,
validation spans approximately 2025-04-24 through 2026-01-09 (symbol-dependent).
The audit checked 10:30 / 11:30 New York decision times, corresponding to 60 /
120 minutes after RTH open, with 60 prior and 200 continuous future minute bars
inside one session and eligible outcome split.

| Metadata result | Count |
|---|---:|
| Eligible offset slots per symbol before known-panel exclusions | 344–347 |
| Eligible offset slots per symbol after exclusions | 336–345 |
| Remaining offset slots across 12 symbols | 4,094 |
| Distinct symbol/session pairs | 2,052 |
| Distinct calendar dates | 176 |
| Dates with candidates for every symbol | 143 |

| Symbol | T+60 / T+120 slots | Distinct sessions | Retained date span |
|---|---:|---:|---|
| AAPL | 171 / 172 | 172 | 2025-04-24–2026-01-06 |
| AMD | 169 / 170 | 170 | 2025-04-24–2026-01-06 |
| AMZN | 170 / 171 | 171 | 2025-04-24–2026-01-06 |
| AVGO | 168 / 168 | 168 | 2025-04-30–2026-01-08 |
| GOOGL | 171 / 171 | 171 | 2025-04-28–2026-01-07 |
| IWM | 171 / 172 | 172 | 2025-04-24–2026-01-06 |
| META | 170 / 171 | 171 | 2025-04-24–2026-01-06 |
| MSFT | 170 / 171 | 171 | 2025-04-24–2026-01-06 |
| NVDA | 169 / 170 | 170 | 2025-04-24–2026-01-06 |
| QQQ | 170 / 171 | 171 | 2025-04-24–2026-01-05 |
| SPY | 171 / 172 | 172 | 2025-04-24–2026-01-05 |
| TLT | 172 / 173 | 173 | 2025-04-24–2026-01-06 |

Every symbol has eligible slots in all three chronological validation-row
thirds; this is metadata coverage, not a realized-regime classification.
The probe used the existing `DataPartition` on timestamp/session projections
and half-open history `[i-60,i)` / outcome `[i,i+200)` intervals. All four
horizons fit inside that validated 200-bar outcome window.

The exploratory probe initially left its per-panel attribution map unpopulated
while its separate union-exclusion accumulator was correct. A bounded metadata
recount reconciled this reporting defect: 4,158 candidates before exclusions;
16 overlaps for each of the four panels; 64 unique excluded candidates; 4,094
retained. No candidate overlapped more than one panel. The recount checked all
128 task windows by same symbol/session interval intersection. This fixes the
count interpretation, not the unresolved historical-exposure gate below. No
panel or eligibility receipt is frozen. The exploratory reporting defect did
not affect the completed smoke or its verified original-panel replay.

Exclusions cover overlapping observation/outcome windows from all four checked-in
panel definitions: original dataset-v2 R0, December R4, January actor-robustness,
and November G0. This demonstrates enough metadata coverage for the proposed
32-anchor panel, not a selected panel or balanced realized market direction.

**Open freshness gate:** historical randomized evaluation starts outside these
four panel files have not been enumerated. Non-overlap is currently proved only
against the checked-in panels, not universal prior non-exposure. Before freezing,
audit available historical episode/start manifests or explicitly resolve the
freshness limitation with GRILL; never label reused/unknown starts clean by
omission. No dates/anchors were selected, no panel hash was frozen and no new
terminal outcomes were inspected. Q24 aggregation and Q21 runner gates remain
implementation work, not completed by this metadata count.

A subsequent read-only historical inventory projected episode identities from
96 R0/R4 evidence JSONL files: 3,072 rows reduce to 128 unique
`(symbol,start_idx,end_idx,horizon)` identities / 32 starts, all already covered
by the four panels. However, earlier randomized evaluation is documented in
`dataset-v2-r0-audit-2026-09-24-evidence/gate_d_00000_resolved_config.json`
through `gate_d_00003_resolved_config.json`, with evaluation seed `30260943`,
32 tasks, null panel hash and null task start indices. Their originals are
under `C:/itrader-ray/v2p0-post-restore-gate35d-20260924/_stable_metrics`.
All four Gate-D `metrics_full.jsonl` files have 35 reports but no actual episode
start/end/symbol/session identities. Related `gate35`, `gate35b`, `gate35c` and
`v2p0-cost-sync60-20260924` roots also document randomized evaluation. This is a
bounded inventory, not proof that every historical run was found.

Recover original starts or reconstruct them using exact historical filtering,
RNG/reset/worker/runtime semantics before certifying non-exposure; a saved seed
alone is insufficient. Any exception must be explicit in the evaluation
contract. These missing historical receipts do not invalidate the current
development bridge, which deliberately uses the already-inspected January panel.

### Independent verification

A separate independent smoke-evidence review returned `VERDICT: APPROVE`, with
no open findings. It verified original checkpoint-tree and saved-configuration
hashes, actual native settings/optimizer visits, all 32 declared replay tasks
and their bound episode/action/reward hashes, distinct training/replay process
identities, and each episode's 1 bp accounting. In-memory progress, PID, row-count
and artifact-digest tampering was rejected. All five design principles were OK.
That reviewer did not reload/rehash the full bars dataset or reconstruct the
historical Python snapshot; their recorded identity chains agree. There is one
fresh replay per checkpoint, not multiple independent replications of each.

Post-smoke verification: repository healthcheck passed; all 408 tests passed
in 70.32 seconds; serialized all-files pre-commit passed. Optional local
`llama_cpp` and `transformers` packages are absent, which does not affect this
random/generated-task CPU probe. The dashboard-specific suite executed 45 tests.

A fresh independent dashboard review found and then verified fixes for three
issues: registry dates mislabeled as training dates; a fresh-panel label applied
to mixed source/fresh traces; and unknown run/member identities joined into a
trajectory. The final review approved all three fixes, checked 28 unknown-identity
cases, 25 ordered comparison selections, generated/template parity, and all 27
copied evidence files against their originals by SHA-256. All five REVIEW.md
design principles passed. No browser workaround was used; rendered layout
remains unverified because browser control rejected local file URLs.

GitNexus is stale (`1d5ba82` indexed versus current `d3eb4c5`). The status script
reported that drift but then failed because Git Bash lacks `python3`; running
its summary with the repo Python succeeded. Source/tests remain authoritative;
no graph refresh or commit was performed.

The human freed RAM and resumed the supervised run. Attempt 01 (launcher PID
34200) failed before training/evaluation/checkpoint creation: the builder checks
checkpoint/PBT cadence even with PBT off. Smoke specified checkpoint interval 1
but omitted PBT interval, whose CLI default was 3. The study similarly declared
checkpoint interval 4 without the matching PBT interval. Earlier schema/CLI-only
review did not exercise this complete builder path.

Both loadouts now declare equal cadence explicitly (smoke 1/1, study 4/4); PBT
remains disabled. A fresh focused Sol 6.1/MAX review executed actual ablation CLI
serialization, production argparse and full `build_base_config` for all ten
cells. Removing the fix reproduces rejection in all ten; fixed cells build with
the intended settings. New regression coverage passes both loadouts. Nine
focused tests passed; the earlier 406-test/pre-commit result predates this
test-only Python increment and must not be relabelled a new full-suite result.

Attempt 01 logs are preserved under the output root's `_supervision_smoke`.
Its process tree exited; no owned Ray workers remained. It produced no economic
evidence. Attempt 02 launched at 10:44:57 Europe/Madrid, PID 6600, with 6.40 GiB
available RAM. Logs are `_supervision_smoke_attempt02/{stdout,stderr}.log`; output
is isolated in `smoke_attempt02_summary`. Both smoke cells must pass before the
eight-cell study is authorized. Stop on the previously declared critical gates.

Revised prelaunch identities, superseding only the original preflight hashes below:

- Smoke JSON: `a6ee572995d610f445614be9aa58811ecb37118b90a4dd64d50003e7c3c32bb8`.
- Study JSON: `246ceb79af9d6e85482363ed0f4e3d3812590868031feb0482ac92402ea52264`.
- Python tree (104 files, new regression test): `b2739469a5a601aceb95b69e19d5f08ce44e8b8763b94300ebb7ea4f79234e95`.
- Dataset/panel hashes are unchanged and were canonically rechecked before launch.

The later human-approved Q21 protocol governs the future complete 16-cell
representation screen, not this separate development bridge. Fresh-panel
construction/enforcement remain pending. No R1 code or native API integration
is authorized by this smoke.

## Scope and interpretation

The human authorized fresh matched controls after the reviewed Ray 2.58
compatibility repairs. This supersedes the previous session's deferral. Stop
before R1 implementation and before native RLModule/Learner/EnvRunner migration.
Use the official ablation pipeline, not an isolated replacement trainer.

R0 and R0 + R4-v1 differ only in explicit position/remaining-horizon state.
The intended comparison keeps MLP-FiLM, Family 00, dataset-v2 RTH,
random/generated tasks, stream_worker and local-worker topology fixed.
PBT is disabled: this is representation-baseline measurement, **not a new
PBT search or proof that exploit/explore is economically effective**.

The final loadout, paired seeds, panel identity, effective optimizer fields,
source/dataset hashes, wall-time/resource limits and replay paths must be
recorded before launch. Inspect the existing development panel only; Q21 is now
accepted; Q23 sampling is approved but eligibility and panel freeze remain to
be completed. The final test remains sealed.

## Comparison declaration

The study loadout is
`itrader/scripts/experiments/configs/phase_a_corrected_ray258_r0_r4v1_fixed_2x2_cost_20261005.json`.
Smoke probes are separate from the eight study cells in
`phase_a_corrected_ray258_r0_r4v1_fixed_smoke_20261005.json` in the same directory.
The output root is
`C:/Users/Admin/ray_results/corrected-compat-r0-r4v1-20261005`; no old run is
overwritten or resumed as if it were a corrected-stack control.

| Factor | Declaration |
|---|---|
| Representation | `solver_mlp_film`, state `off` versus `both`, `zero_v1` residual, strict `r0_r1_v1` profile |
| Training seeds | Paired `20260924`, `20260925` |
| Development evaluation | January 2026 crossed 8-anchor x 4-horizon panel, seed `32260943`, 32 episodes every update |
| Training cost | Exact 1 bp versus compressed causal 0 / 0.25 / 0.5 / 1 bp at 0 / 14,400 / 28,800 / 43,200 sampled steps |
| Evaluation cost | Exact 1 bp throughout; not the current training-stage cost |
| Optimizer | Native 30 epochs, minibatch 128; LR `7.151674865214668e-6` |
| PPO/GAE | Gamma 0.99, lambda 0.9, clip 0.1, value loss coefficient 2, value clip 20 |
| Entropy | Native RLlib linear schedule `(0,.005), (96000,.002), (240000,.0005)`; no compressed entropy clock |
| Budget | 12 updates / 57,600 nominal sampled steps per cell; complete-episode overshoot recorded, not an exact exposure cap |
| Topology | Sequential single-member cells, CPU, local workers (`workers=0`), PBT off |
| Global bound | At most two hours under active supervision; this global cap is not encoded in the loadout. One-hour configured per-cell timeout remains a fallback |

This is a new matched-development screen, not reproduction of the historical
self-resume entropy spikes or a fresh-panel generalization test. The compressed
cost curriculum is an explicit changed factor. Cost staging is piecewise
constant on the target-owned step clock; entropy is linearly interpolated on
RLlib's native sampled-step clock. Do not conflate their scheduling semantics.

Preflight exposed a missing fixed-run entropy override. The narrow correction
adds `--entropy_coeff_schedule` to the existing builder/loadout seam, reuses
schedule parsing and preserves omitted-argument defaults. A failing builder
test reproduced the ignored recipe before the patch. Afterward 106 focused
configuration and real synthetic PPO/checkpoint/replay tests passed, followed
by 406 full-suite tests and a successful serialized all-files pre-commit run.
Independent review is required before launch; its scope is the schedule/loadout
increment, not a claim of completed economic validation.
No R1 implementation or training-algorithm change is part of that correction.

GPT-6 Astra/MAX independently approved the increment and prospective loadouts.
Its actual CLI/schema/serialization and installed-Ray interpolation probes
confirmed entropy `.005` at step 0, `.0032` at 57,600, `.002` at 96,000 and
`.0005` at/after 240,000. Base comparison used isolated production expressions
because the saved `d3eb4c5` trainer has a pre-existing indentation error. The
separate GPT-6.1 Sol/MAX review also returned `VERDICT: APPROVE`. It executed
actual CLI serialization/parsing for all ten declared cells, schema/type probes,
default-preservation checks and startup guard boundary checks: 4.999 GiB rejects,
5.0 and 5.001 GiB accept. Neither reviewer launched training. All five REVIEW.md
design principles were reported OK by both reviewers.

## Historical versus corrected evidence

| Claim | What remains supported | What must not be inferred |
|---|---|---|
| Historical economic curves | Observed outcomes of the actual saved policies, subject to their original replay/data qualifications | What the intended or repaired recipe would have achieved |
| Recorded PBT events | Config mutation requests and observed checkpoint transfers where their receipts validate | That all requested settings reached policy loss, GAE or Adam |
| Fixed-recipe runs | PPO activity and policy behavior under their actual effective settings | Intended epoch/minibatch settings from obsolete aliases alone |
| Corrected fresh controls | Prospective matched evidence once complete and replayed | Economic promotion, causal explanation of every prior basin, or full native-API parity |

The bounded [historical audit](ray258-historical-optimizer-audit-2026-10-05.md)
found effective 30 epochs / minibatch 128 in ten sampled Ray 2.58 terminals,
donor-setting divergence in five exploited targets, and six member-00002
self-resume entropy resets. The magnitude of lost search effectiveness is not
known without a matched experiment. It is not correct to retroactively erase
the returns or assert that correcting PBT necessarily improves profitability.

## Supervision and stop policy

- Startup guard: at least 5 GiB available physical RAM; stop the owned probe
  before availability falls below 2 GiB. These are conservative operational
  bounds, not a measured guarantee that this workload fits. Initial preflight
  samples ranged from 2.45 to 4.48 GiB; later checks reached 1.57 GiB and
  recovered to approximately 3.3 GiB, still below the startup gate. Launch
  was on resource hold at that initial preflight (superseded by completed
  attempt 02 above). The 2 GiB runtime floor requires active supervision;
  only the 5 GiB pre-variant check is enforced by the ablation runner. No other
  application has been stopped. Keep full dataset coverage unchanged.
  Final availability was about 4.88 GiB at 03:38 Europe/Madrid, still below
  the guard at that time. No automatic launch followed that historical check.
- The prior local-worker Ray-2.58 acceptance probe bottomed at 2.34 GiB free;
  the full RTH input contains 5,515,061 rows (about 1,210 MB loaded). Historical
  minimum free memory alone does not establish today's startup requirement.
  Disk headroom was approximately C: 67.5, D: 95.3 and E: 184.6 GiB.
- Freeze the declared source before training; do not patch a running cell.
- Check actual effective native epoch/minibatch/LR fields, clip counts and finite
  PPO metrics, cost-stage application, target-owned step progress, full panel
  coverage, non-empty checkpoint/provenance artifacts and deterministic replay.
- Run cells sequentially within measured memory and storage headroom.
- Stop this experiment on critical contract/provenance errors, non-finite
  training, failed replay, unexpected resource pressure or bounded-budget
  exhaustion. Preserve failed attempts; never overwrite them with a retry.
- Stop only processes owned by this experiment. Do not stop unrelated Ray,
  WSL, games, research agents or user applications.
- Supervisor escalates faults to the parent agent and returns a summary in this
  chat. Independent review of closure remains separate from economic judgment.

## Historical preflight record (before attempt 01)

This section preserves the initial handback and original hashes; it is not
current run status. Attempt-02 outcomes and revised identities are above.

Luna's initial preflight handback: both loadouts passed schema and CLI-name validation;
all eight study cells have matched declared factors, and the two smoke cells
were isolated. At that point no experiment process had launched, no unrelated
process was stopped, and no return/Sharpe/checkpoint result existed yet.

### Prelaunch identities

These are preflight identities, not producer receipts or completed-run evidence.
Recheck them before launch or later continuation; source includes the dirty
Python tree and untracked modules, not only the Git revision.

| Artifact | SHA-256 |
|---|---|
| Study loadout file | `574d13fea74a0752f871e4d01d10796653c1a94047d6c062fa22fb70d3fa2130` |
| Smoke loadout file | `d9e3651a5fbbae3c71d225697a24710910525c512abe2284d85702c3794739b9` |
| January development panel file | `0401993d22fef1bc69712f1b8022d0f363e80779af6bc58189ee979bdf2198b5` |
| Dataset bars tree, recomputed from bytes | `20aac96de6da77e20beeaff214d380ef06184faf301ff0dd53bed0ffd4c04ce9` |
| Dataset manifest file | `4c6202883a46cbe5bfbf4986003dcd1eb05b3738ac2d48eeb42d9a57fe974ca5` |
| Runtime Python source tree (104 files, includes tests) | `0716db3fa292c3a84a8a2029895402b5e3f84ba8444ccfe7224eabbc0ae390c2` |

The panel's expected dataset digest matches the recomputed bars-tree digest.
The output root resolves through the existing `ray_results` junction onto
`D:/iTRADER-offload/active/ray_results`; no new storage migration was performed.

Once resource and review gates pass, from the repository root:

```powershell
& .venv/Scripts/python.exe itrader/scripts/experiments/run_ablation.py itrader/scripts/experiments/configs/phase_a_corrected_ray258_r0_r4v1_fixed_smoke_20261005.json
```

Inspect both smoke outputs, actual optimizer settings, panel coverage, replay,
memory low-water and throughput before approving the eight-cell study command:

```powershell
& .venv/Scripts/python.exe itrader/scripts/experiments/run_ablation.py itrader/scripts/experiments/configs/phase_a_corrected_ray258_r0_r4v1_fixed_2x2_cost_20261005.json
```

At initial preflight these commands were recorded, **not executed**. The smoke
was subsequently executed through the official runner with unique attempt
output directories; the eight-cell study remains unexecuted. Do not rerun either unattended:
the resource floor and global wall-time stop need an active supervisor. The
January panel remains inspected development data; Q21 is now accepted but its
implementation remains pending, and the final test remains sealed. GitNexus is
stale (`1d5ba82` versus current `d3eb4c5`); direct
source/tests were used. No graph refresh was attempted under memory pressure.

Record terminal and retained-checkpoint return, Sharpe per square-root bar,
turnover, action fractions, critic/PPO diagnostics, actual sampled steps and
cost curriculum. Keep economic returns unscaled; do not call the PPO reward
scale a normalized return. Add official comparison plots/evidence to the
research browser only when actual artifacts exist.

After these controls, implement/test R1 separately, then run matched R1 and
R1 + R4-v1 on the same corrected compatibility stack. A later native-API
migration repeats the complete four-cell comparison as a platform factor.
