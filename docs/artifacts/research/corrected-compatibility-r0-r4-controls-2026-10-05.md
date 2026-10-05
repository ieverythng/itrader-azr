# Corrected-compatibility R0 / R0 + R4-v1 controls

Date: 2026-10-05 (Europe/Madrid). Status: terminal replay held after a supervision-launcher failure; no evaluation or study results; artifacts preserved.
Supervisor: GPT-6 Luna at xhigh, delegated within the current research chat.
Source branch remains dirty; no staging, commit, push or artifact deletion.

## Latest coordination update

The user permits a 1 GiB runtime RAM floor for a resumed supervised replay. This
supersedes the earlier 2 GiB runtime floor below for future attempts only; no new
startup threshold was specified, so do not infer a change to the previously
declared 5 GiB launch guard. The supervisor reports that the terminal replay is
held after a launcher failure, no evaluation ran, and its artifacts are
preserved. Resolve the supervision harness before resuming; this is not an
economic failure or a completed smoke.

At about 13:43 UTC a separate migration CLI E2E test launched Ray and completed
one synthetic Tune update. That timing plausibly matches the unowned Ray driver
process observed by the supervisor, but process IDs were not independently
mapped. The test shut Ray down; a later check found no remaining Python/Ray
processes. Do not attribute its run artifacts or evaluation metrics to these
compatibility controls.

## Resumed smoke and bounded correction

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
recorded before launch. Inspect the existing development panel only; no new
selection panel or sealed final-test access while Q21 remains pending.

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
  remains on resource hold. The 2 GiB runtime floor requires active supervision;
  only the 5 GiB pre-variant check is enforced by the ablation runner. No other
  application has been stopped. Keep full dataset coverage unchanged.
  Final availability was about 4.88 GiB at 03:38 Europe/Madrid, still below
  the guard; this is not a background-running or automatically queued experiment.
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

## Output and next boundary

Luna's preflight handback: both loadouts pass schema and CLI-name validation;
all eight study cells have matched declared factors, and the two smoke cells
are isolated. No experiment process was launched, no unrelated process was
stopped, and there is no return/Sharpe/checkpoint result from this round.

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

At the original preflight, these commands were recorded but not yet executed;
the supervised attempt-04 training and terminal-replay outcomes are summarized
in the closeout below. Do not run either unattended: the resource floor and
global wall-time stop need an active supervisor. The January panel remains
inspected development data; Q21 and final-test gates remain pending. GitNexus
was stale (`1d5ba82` versus then-current `d3eb4c5`); direct source/tests were
used at preflight.

Record terminal and retained-checkpoint return, Sharpe per square-root bar,
turnover, action fractions, critic/PPO diagnostics, actual sampled steps and
cost curriculum. Keep economic returns unscaled; do not call the PPO reward
scale a normalized return. Add official comparison plots/evidence to the
research browser only when actual artifacts exist.

After these controls, implement/test R1 separately, then run matched R1 and
R1 + R4-v1 on the same corrected compatibility stack. The native Ray 2.58
platform cutover is complete on the isolated migration branch; a future
four-cell scientific qualification must use that active native stack and keep
runtime identity explicit rather than treating the old-stack outcomes as native
parity evidence.

## Supervisor closeout update — 2026-10-05

All eight attempt-04 training cells completed their 12 updates by about 13:30
Madrid. In the first terminal-replay attempt, cells 4--7 passed, then cell 8
was proactively stopped at 1.198 GiB free RAM, below the 1.25 GiB stop threshold
but above the human-approved 1 GiB hard runtime floor. That cell-8 replay was
preserved and excluded; no evaluation result was attributed to it.

After the user-requested WSL shutdown, a separate fresh cell-8 replay retry
completed successfully, together with witnessed repeats of cells 1--3. The final
accepted set is eight retained checkpoints times 32 episodes, or 256 fresh
replay episodes, independently accepted for replay parity. No training was
resumed or rerun during closeout. Economic/fresh-panel promotion remains
unapproved. These compatibility-control results are separate from validation of
the isolated native migration; no owned Ray job processes remained after
closeout. The earlier launcher-hold and first cell-8 replay-stop notes are
preserved as intermediate events, not current status.
