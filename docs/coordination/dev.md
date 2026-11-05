# DEV Board

- `DEV-RECIPE39` (2026-10-13): prepare exactly the new eight-variant
  `pnl_linear` R0/R1 x state-off/both loadout and the paired inspected-progress
  January panel. Human/GRILL baseline decision is closed: training and eval
  TaskSpecs use unchanged `pnl_linear`; independent economics use 1 bp evidence
  scoring. Write set is only those two new JSON files, focused contracts in
  `test_run_ablation.py` and `test_task_contracts.py`, this board, and ignored
  `_local/dev-recipe39/`. Preserve the existing 16-cell net-return config and
  source panel byte-for-byte. Architecture route: ablation config/schema and
  `run_ablation.py` -> `train_solo_V2`/`TaskSampler00` -> `TaskSpec` and
  `RewardEngine`; evaluation panel enters the existing `build_base_config`
  loader and fixed-task sampler. No new wrapper or runtime/evaluator/source
  behavior. The predeclared public-contract probes are in
  [_local/dev-recipe39/scenarios.md](_local/dev-recipe39/scenarios.md).
  Branch `agent/Watson-dev`, HEAD `8f9fcf94e028b51d883a1a484c2c9466a3d6b547`;
  inherited dirty/untracked tree preserved. Validation lease is
  `LEASE-RECIPE39-FOCUSED`: synthetic config/task tests only, no Ray/GPU/model,
  market-row access, full suite, or workload. Refresh RAM before testing; start
  only at >=4 GiB, abort below 2 GiB, <=90s per child and <=240s cumulative.
  Deadline 2026-10-13 10:15 UTC. No staging/commit/push, install/sync/driver,
  WSL, data acquisition, outcomes, simulation, training, or PBT. Two fresh
  reviews must approve exact config/test bytes. `screen_completion.py` remains
  a known eight-vs-sixteen/two-curriculum adaptation prerequisite; do not
  bypass or modify it. Stop on unsupported sampler meaning, identity drift,
  memory gate, or review failure; configuration preparation is not run
  authorization.
- Bounded validation: `.venv\Scripts\python.exe -m pytest -q
  itrader/tests/test_run_ablation.py itrader/tests/test_task_contracts.py` —
  **38 passed in 2.64s**; the first panel-hash assertion exposed a transcription
  typo, then passed after correction. Scoped `git diff --check` passed. Exactly
  two tests were added: one matched-loadout contract and one panel/TaskSpec/
  sampler lifecycle. Independent review found no blocking issues and one
  optional NIT: lock exact per-variant override keys/effective optimizer
  settings so an accidental LR override fails the recipe test. The two tests
  cover distinct boundaries; nearby wraparound/reseed and semantic-embedding
  tests remain distinct. No test deletion or count-based consolidation was
  justified. Current hashes: loadout
  `7B1E5D7D17C25CD855DA4004D94FC65E60933C8DA947DFDABB497178A206FD03`, panel
  `A7CFF4F437EEC5938119234B483B63847A915308E3EEAA4C8E3B8FB1AC138B21`,
  `test_run_ablation.py`
  `C874D50B820ED355E23605CCC18F0DFD32DCD01455C1601E39A79456214069C9`,
  `test_task_contracts.py`
  `5F0D9785829DE432B6BB4B2C38848FBBE9994BB3C7EA1E7EB4F3C84760A1AB87`.
  Second independent review remains pending. Scope stopped at the lease
  deadline; no training, staging, commit, or push.
- Initial protected-byte receipts before edits: the existing 16-cell net
  loadout `phase_a_r1_corrected_compat_fixed_4x2x2_20261006.json` SHA256
  `D28B778B480391547D5F5DA4F8932AE461EA0ECFA5339EF2235BEF0B619DB38F` (6694
  bytes); inspected January progress panel
  `dataset_v2_r4_actor_jan2026_crossed_8x4_20260929.json` SHA256
  `0401993D22FEF1BC69712F1B8022D0F363E80779AF6BC58189EE979BDF2198B5` (8490
  bytes); `screen_completion.py` SHA256
  `E159818F3DB744016EE18D4C4582E23A847A20908CC8A823CE984410C3C0B0CE` (7972
  bytes). Both new target paths were absent. Initial free RAM was 4.73 GiB;
  no Python/Ray process was found. GitNexus status: stale index (2026-06-17),
  and Git Bash lacked `python3`; no WSL/reindex attempted.
- `DEV-DEFAULT38` closed 2026-10-13 03:12 UTC below; retained as prior work.

- `DEV-DEFAULT38` (2026-10-13): implement the narrow training-only CUDA
  interpreter default for ablation configs that omit `python_exe` and do not
  supply an explicit interpreter selector. Preserve explicit config paths,
  environment-selected CPU/custom interpreters, historical loadouts, collection,
  plotting, and general `run_repo_python.py` CPU-default behavior. Reuse its
  installed `cuda-tested` validation; never fall back to CPU or provision a
  profile. Review found that the runner's repository-root cwd changed relative
  path meaning, so add an opt-in preserve-cwd dispatch used only by ablation
  training. Write set: `itrader/scripts/experiments/run_ablation.py`,
  `itrader/tools/run_repo_python.py`, their nearest tests, focused
  `docs/environment_strategy.md`, this board, and ignored
  `docs/coordination/_local/dev-default38/`. Architecture map: ablation CLI
  owns train and collect dispatch; `run_repo_python.py` owns interpreter/profile
  resolution and the opt-in child cwd; plotter remains a non-training CPU-safe
  child. Nearest tests are
  `test_run_ablation.py`, `test_run_repo_python.py`, and the existing Windows
  launcher contract. No solver/PPO/reward/data/configured training recipe,
  environment install, WSL, CUDA query, Torch/Ray workload, training, PBT,
  native Ray, R1 seam, or Git operation is in scope.
- Predeclared behavior probes are recorded before edits in
  [_local/dev-default38/scenarios.md](_local/dev-default38/scenarios.md): no
  selector selects CUDA only for the training child; config and environment
  selectors remain authoritative; missing default CUDA fails before child
  training; empty `ITRADER_PYTHON_BIN` is not a valid explicit interpreter;
  collection and plotting do not probe CUDA; general runner default remains
  CPU-safe. Validation is limited to focused non-CUDA tests, each child
  <=45 seconds and total <=180 seconds; no individual run below 2 GiB available
  RAM. Stop on semantic ambiguity, resource floor, unrelated source drift, or
  validation budget exhaustion. Two fresh `REVIEW.md` reviewers and the
  Standards/Spec review are required after source freeze.
- Test ROI constraint: prefer a compact table-driven dispatch contract plus
  public fail-closed and caller-cwd launch scenarios; retain distinct profile-identity and
  argv contracts. Assess whether adjacent same-boundary unit cases can be
  parameterized, but do not delete useful invariants or create a monolithic
  integration fixture to reduce test count. Report each test's independent
  boundary and evidence value before closure.

- Owner chat: `01a092eb-9aa9-7571-b723-d91a84e78ea3`.
- Seeded by GRILL: 2026-10-09 08:42 UTC; DEV becomes sole writer at dispatch.
- Last updated: 2026-10-13 03:11 UTC.
- Current status: `DEV-DEFAULT38` implementation, focused validation, and
  bounded review are complete. The caller-cwd integration failed before repair
  as expected and passes after it. The final focused suite passes 57/57; the
  author run took 3.05s and independent `gpt-6.1-sol` and `gpt-6-astra` max
  reviews each reran it successfully (3.01s and 3.00s). Both fresh reviewers
  approved with zero blocking findings and zero nits. Both assessed Standards,
  Spec, and all five design principles. This is a bounded review, not an
  exhaustive pool or CUDA/trading qualification. No full suite or CUDA/Ray/
  Torch/WSL/training workload ran. GitNexus status was stale and its check could
  not find Git Bash `python3`; no reindex/WSL activation followed. No owned
  process remains. Branch `agent/Watson-dev`, HEAD
  `8f9fcf94e028b51d883a1a484c2c9466a3d6b547`; all inherited dirty work is
  preserved and no files were staged or committed.
  Test ROI: replacing a one-assertion path helper check with one synthetic
  real-wrapper/child integration oracle adds caller-cwd and child/collector
  result-root agreement without increasing the focused suite count (57).
  Distinct profile identity, selector rejection, argv fidelity, collect-only,
  and lifecycle contracts remain because they cover different boundaries; no
  count-based deletion was used. Approximate author-run cumulative pytest
  fixture time across red/green attempts was 11.3s; final author run 3.05s plus
  reviewer reruns 6.01s.
  SHA256 / byte receipts for implementation, tests, environment doc, and frozen
  local scenarios: `run_ablation.py` 38A019EFD128E56029C63D79F0C04DEB27C8DF910DA25533640523C9C1B91C77 / 29491;
  `run_repo_python.py` 5B01DF3D956EA949B0B77A9CA7608CD698B7EA6134B1CD8874BF683711F5351E / 9262;
  `test_run_ablation.py` 653EA83BCD319E19AF07E4932131C6E7094B2433DB1327693496BA7C6BF1FC70 / 32419;
  `test_run_repo_python.py` E7D3C1E8A611EBE96586B46AA64964F4BFAD84499A035DDF07B4F85BC1AC12F6 / 6065;
  `environment_strategy.md` 99EA6E9026387B74502FAF4502243DB903D835B76590DAF7DE7E9D0F89DAAA5C / 12059;
  `scenarios.md` 7F17B49025554228422DC365A2D1227C3A69F0F88D26D067C9D1CCE65A544D44 / 2997.
  The board hash is not self-recorded; its receipt edit necessarily changes it.
- Prior status: the repaired bytes were pending focused validation and final
  independent review. Earlier intermediate statuses and their hashes/results
  remain as historical evidence below and are superseded by this receipt.
- Prior status: the bounded `DEV-FIX36` repairs are implemented and the
  focused Windows contract set passes (37 passed in 4.20s); two fresh independent
  `REVIEW.md` lanes approve the exact frozen snapshot. They report zero blocking
  findings and one optional nit: direct calls to the identity validator do not
  canonicalize a base-prefix 8.3 alias, while the real reporter canonicalizes
  it and the runner rejects it. Relative/NUL path identities are rejected.
  Earlier `CHANGES` reviews and the superseded 01:08 approval remain preserved
  as history. Test ROI:
  keep the three distinct resolver contracts, fold CPU-default coverage into
  the existing direct-runner integration, table-drive input variants, and
  avoid a duplicate wrapper-overlap case or one-test-per-argument growth. No
  synthetic run is authorized. All inherited work remains dirty; no
  staging/commit. Exact current hashes and review outcome will be appended to
  the [repair receipt](../artifacts/research/dev-fix36-launcher-repair-2026-10-13.md).
  DEV's focused-test and reviewer processes are closed. A hash-bound handoff
  was sent to TRAINING for its separately GRILL-gated CUDA37 preparation; DEV
  launched no training and holds no active job.
- `DEV-FIX36` is under the GRILL-authorized bounded
  DEV--TRAINING repair scope. Branch `agent/Watson-dev`, HEAD
  `8f9fcf94e028b51d883a1a484c2c9466a3d6b547`; inherited dirty tree preserved.
  DEV writes are limited to `itrader/tools/run_repo_python.py`,
  `itrader/tools/cuda_identity_probe.py`,
  `itrader/tools/invoke_cuda_tested.ps1`, public launcher/runner contract tests,
  focused `docs/environment_strategy.md`, this board, ignored
  `_local/dev-fix36/`, and one compact research receipt. No R1, solver/PPO,
  reward, config/runtime pin, data, CUDA/device, WSL, or Git shipping changes.
  Architecture map: native public entrypoint and sole profile-validation owner
  `run_repo_python.py`; `cuda_identity_probe.py` is a standard-library identity
  reporter consumed by that validator; compatibility/argv boundary
  `invoke_cuda_tested.ps1`; existing native argv-list runner in
  `scripts/experiments/run_ablation.py`; nearest contracts are
  `test_run_repo_python.py`, `test_cuda_test_launcher.py`, and
  `test_cuda_provisioning_contract.py`. Predeclared external cases and test-ROI
  disposition are frozen in [_local/dev-fix36/repair-scenarios.md](_local/dev-fix36/repair-scenarios.md).
  Repair stop: 2026-10-13 01:10 UTC; two fresh review lanes by 01:30 UTC. Current validation reservation
  excludes full suite, environment sync/install/build, WSL, Torch/Ray/GPU import
  or query, training/PBT, and native Ray. Startup/runtime RAM floors are 4/2
  GiB; disk floors are 20/10 GiB; each child <=45s and focused group <=180s.
  Two fresh different-model `REVIEW.md` lanes remain required after the source
  is frozen; no files are staged or committed.
- Focused red/green result: before the initial repair, explicit CUDA-missing and
  unknown-profile subprocess cases launched the CPU fallback (exit 0 and
  sentinel created); a later explicit-empty-selector probe reproduced the same
  gap. After the bounded follow-up repair, the combined runner and Windows
  launcher suites pass 37/37. Coverage includes a fixed-profile identity
  matrix, 8.3 short-path overlap refusal, reparse/override refusal, direct
  profile and CPU-default runner integration, exact wrapper argv/streams/nonzero
  exit, child-only selector, and caller-state preservation using a
  standard-library fake module. This invokes neither
  actual pytest collection nor Torch/Ray, CUDA query/work, or training. Test ROI
  retains three distinct resolver invariants, uses table-driven input variants
  for the same selector/identity boundary, and reserves integration tests for
  public lifecycle behavior; no quota-driven test deletion occurred. Details:
  [_local/dev-fix36/repair-scenarios.md](_local/dev-fix36/repair-scenarios.md).
- Previous completed preparation: `DEV-PREP31` produced only its scenario
  packet; the packet and historical notes remain below.
- Previous scope: `DEV-REVIEW30` completed read-only with `CHANGES`; three
  blocking evidence/reporting findings and three nits remain. The exact
  snapshot, independent findings, test-ROI disposition, and execution limits
  are in the [compact review receipt](../artifacts/research/dev-review30-r1-evaluation-review-2026-10-12.md).
  Q23/Q26 qualification and R1 performance are not closed. Behavioral tests,
  healthcheck, pre-commit, and Torch/Ray/GPU execution were deferred while the
  TRAINING-CUDA30 preparation/execution scope was active; do not treat
  unrelated prior hook results as evidence for this snapshot. Changes remain
  dirty and unstaged.
- Last completed provisioning scope (`DEV-CUDA28`): two independent reviews approved the exact
  source and generated lock; the exact candidate `pyproject.toml`/`uv.lock` are
  applied, the one locked sync passed, and the CUDA tensor forward/backward
  smoke passed on the isolated Windows profile. CUDA27 remains closed and its
  evidence untouched. The run completed in 689.510s from 01:56:48 to 02:08:18
  UTC; installed pins include Ray 2.58.0, NumPy 1.26.4, Gymnasium 1.2.2,
  pandas 2.1.4, pyarrow 17.0.0, and Torch 2.9.0.dev20250801+cu128. The RTX
  5070 Ti (CC 12.0) forward/backward smoke passed with output `[4,4]`; minimum
  sampled RAM was 3.72 GiB, final verification 4.17 GiB, and no owned process
  remained. The lock review noted 76 shared non-Torch package version changes
  and removal of `decorator`; this is recorded for human review, not hidden.
  Test-ROI review retained the distinct parser, transport, and provenance
  contracts and did not add quota-driven tests. This does not qualify
  RLlib/PPO/PBT or economic performance. Changes remain dirty and unstaged; no
  commit or push. Full hashes and logs are in the compact CUDA28 receipt.
  Earlier scopes, failures, and hashes remain preserved below.
- Branch / last recorded HEAD: `agent/Watson-dev`; current HEAD is not certified
  by DEV in this scope. GRILL reports `8f9fcf94e028b51d883a1a484c2c9466a3d6b547`
  from its current pass.

## Closed Scope: DEV-REVIEW30 Read-Only R1/Evaluation Review

This review was performed against branch `agent/Watson-dev`, HEAD
`8f9fcf94e028b51d883a1a484c2c9466a3d6b547`, using the pre-read adversarial
probe matrix and frozen snapshot manifest recorded in
`docs/coordination/_local/dev-review30/snapshot.md` (ignored local evidence;
SHA256 `913de0c10c37ddf0f3a9306375bd671200e4a4726e7be4e7e1fcb3f590075ee6`).
All 48 source/test/runtime hash entries matched; the previous R1 snapshot
matched only 27/30 entries, so its approvals do not transfer beyond matching
bytes and original scope. Two independent review lanes returned `CHANGES`.
Three blocking findings: R4 analyzer acceptance lacks verified distinct-process
launch evidence; replay receipt verification accepts incomplete or malformed
producer timestamps/identity fields; ex-post regime summaries accept impossible
action fractions. Nits: cross-panel uncertainty checks compare cardinalities,
not identity sets (already disclosed as exploratory and not Q26-qualified);
early replay launch failures discard cleanup diagnostics; and the active
next-actions blocker summary conflates completed bounded CPU lifecycle evidence
with pending CUDA/performance qualification. Exact inputs/expected/actual,
principle assessments, and limitations are in the receipt.

Test-ROI disposition: do not impose a deletion quota or combine all tests into a
large brittle integration test. Retain distinct deterministic unit contracts
for canonical evidence/range/hash, exact arithmetic, selector/restore identity,
clip statistics, and process containment. Reduce duplication through shared
contract fixtures and table-driven variants where they exercise the same
boundary. Prefer a small number of public integration oracles for the complete
checkpoint lifecycle, accounting/evidence, causal complete-window handling,
matched aggregation, and receipt-to-report/browser disposition. Synthetic E2E
proves lifecycle wiring, not market edge. Current tests were not executed in
this review; no compaction or code change was performed.

## Completed Preparation: DEV-PREP31

Prepared a bounded red/green public-scenario and minimal repair packet for the
three DEV-REVIEW30 blockers. It reuses `compare_process_replays`,
`evidence_values.action_fractions`, and the existing analyzer/evidence tests;
it distinguishes structural receipt parsing from witnessed process
qualification so a caller-authored digest cannot count as process proof. The
packet is `docs/coordination/_local/dev-prep31/repair-scenarios.md` (local,
ignored; SHA256 `9767492de9cdcdb3a5b4733ea8b47a3634eae88cbe21a971a12d72a4dfc7ddd8`).
Only read-only source inspection was done; no tests/imports or CUDA30 operations
occurred. GRILL retains the activation and sequencing decision.

No-Mistakes discovery: native `Get-Command no-mistakes` returned not found.
Thus no help/doctor/AXI or installation/init/daemon/config/hook action ran.
Actual AXI was not run: the requested changes remain dirty/uncommitted, prior
initialization and pipeline-agent configuration are unverified, and an
isolated committed snapshot plus enforced aggregate budgets and explicitly
disabled shipping stages are prerequisites. Manual REVIEW is not an AXI pass.

## Closed Scope: DEV-CUDA28 Known-Error Repair

GRILL's separate dispatch at board lines 89-114 permits one repair, not a
continuation of spent CUDA27. Branch/HEAD at intake: `agent/Watson-dev` /
`8f9fcf94e028b51d883a1a484c2c9466a3d6b547`; the workspace is already dirty, so
only the following paths are in this write set: CUDA provisioner/profile/test
sources, `docs/coordination/_local/cuda-uv-bridge28/`, this board, and one
compact durable CUDA28 receipt. Root `pyproject.toml`/`uv.lock` may change only
after the complete generated lock receives its own independent approvals.
No staging, commit, push, runtime install before that gate, WSL, driver, native
Ray, market, or PPO work.

Known failure: uv 0.11.7 rejects a direct-wheel source URL containing a
`#sha256` fragment. The candidate now removes only that unsupported fragment;
the exact torch version, official wheel URL, and SHA256 remain separately
enforced by retained-proof validation and `_verify_lock`. The old downloader is
not reachable from the repaired stage. CUDA27's wheel, state, logs, and review
receipt remain immutable evidence. The reused proof is required to match its
old state, acquisition log, review receipt, exact size, and a fresh offline
SHA256 of the retained wheel.

Before reading the implementation, the frozen review probes are: fragmentless
official URL accepted; exact `#sha256` and unrelated fragments rejected;
changed host/path/query/credentials rejected; lock source and wheel hash must
match exact approved values; matching retained state/log/review/current bytes
accepted; altered state/log, review digest, file size, or current wheel bytes
rejected. The existing profile matrix is updated in-place; one compact
parameterized retained-proof contract was added because this is a separate
provenance boundary. The earlier 24-case parser and three-case byte-transport
matrices remain separate: they exercise content validation versus OS transport.
No count target or unrelated suite cleanup.

Preparation ceiling: 20 minutes from receipt and absolute stop `2026-10-11
02:05 UTC`. Both fresh focused source approvals are recorded in the ignored
candidate receipt. Review found zero blocking findings; both reviewers noted
the same non-blocking diagnostic nit: a malformed retained JSON object can
escape as `AttributeError` instead of the typed provisioning error, but still
fails closed before lock/sync. Per GRILL's instruction not to polish optional
nits in this one-shot scope, no change was made. Native offline uv syntax check
proved only that the fragmentless URL parses; the expected absent-cache error
occurred with `--locked --offline`, and the copied lock hash remained
unchanged. It is not a generated-lock result.

The CUDA27 measured attempt used 86.0122625 seconds through the final lock-log
timestamp. Conservatively charge 87 seconds, leaving 2,313 seconds of the
original 2,400-second owned budget; reserve 300 seconds for cleanup, so this
repair's active-stage ceiling is 2,013 seconds. The separate lease caps remain
45 minutes after activation and absolute stop `2026-10-11 02:45 UTC`.
RAM must be >=4 GiB at start / >=2 GiB at runtime; C: disk >=30 GiB / >=15 GiB.
The one normal uv full-wheel transfer was consumed by the successful lock
generation. Candidate lock SHA256 is
`0b12468efc1c2b376fa7818108e2e07016670fd01c71f1c67e41cd8375db09bb` (277,523
bytes). At this intermediate point the state was `awaiting_lock_review` and
root dependency files were unchanged. Both fresh lock reviewers subsequently
approved; exact candidate files were applied and the one sync/smoke sequence
completed as recorded below. Preserve the non-blocking dependency breadth
finding for later human review; no retry is authorized.

Test-ROI review: retain the 24-case URL/parser matrix and three-case raw process
transport matrix because they validate separate boundaries. Retain the
four-case retained-proof contract because its positive proof, changed payload,
state/log mismatch, and review-digest mismatch are independent provenance
oracles; they form one small helper-level evidence integration test, not full
provisioning E2E. Runtime-identity accept/reject checks could share a
parameterized table without deleting an oracle, but this is presentation-only
condensation. No quota-based reduction is justified. Reviews found all five
design principles satisfactory; the diagnostic nit above is the only finding.

The CUDA27 elapsed/remaining values are measured from its UTC start and log
timestamp; the 87-second budget charge is conservative because the exact final
process-exit instant was not persisted.

## Current Scope: DEV-CUDA27 Implementation And Qualification Preparation

GRILL's superseding dispatch authorizes the existing CUDA source set, root
`pyproject.toml`/`uv.lock`, this board, ignored candidate scratch, and one
compact durable receipt. Preserve the CPU default, optional `dev`/`etl`/
`research`/`llm-local` groups, Ray 2.58 compatibility pins, and exact tested
Torch `2.9.0.dev20250801+cu128` Windows AMD64 CPython 3.11 wheel. No native Ray
merge, toolkit/driver/WSL changes, copied packages, or Git shipping.

Preparation and both review gates stop within 120 minutes of dispatch and no
later than 2026-10-11 02:00 UTC. Build an exact-path scratch candidate and a
bounded, executable acquisition/lock/sync driver. It must validate ownership,
canonical paths/reparse points, CPython/platform identity, child-only uv
environment, live RAM/disk floors, per-stage/overall timeouts, and reuse
`itrader/rl_lib/evaluation/process_scope.py` for process-tree containment.
Freeze the candidate and full command/source identities for two fresh
independent `REVIEW.md` approvals. Once those pass and a fresh resource check
clears the thresholds, the authorized single provisioning job may acquire and
lock the candidate; do not wait for another generic install approval. Give the
complete generated portable lock its own fresh review before applying exact
candidate bytes to root `pyproject.toml`/`uv.lock` and running locked sync plus
the tiny target identity/GPU tensor gate. R1 training remains a separate later
TRAINING lease and is not part of this provisioning job.

The one authorized provisioning lease was used at 2026-10-11 01:34:26 UTC and
failed at lock generation; it is spent, with no active owned processes. The
original bounds remain part of the historical receipt: start RAM >=4 GiB / disk
>=30 GiB, abort below RAM 2 GiB / disk 15 GiB; provision <=55 minutes,
acquisition <=900 seconds, sync <=1200 seconds, and total owned runtime <=2400
seconds with cleanup reserve. At most two full reads of the exact wheel were
permitted; no retry or alternate version is authorized. Any further attempt
requires a new bounded dispatch and fresh native preflight.

GRILL's 01:04 UTC timing clarification keeps preparation/reviews at the 02:00
UTC stop; the one unused conditional provisioning job may finish by 02:30 UTC
only when its complete run plus >=300 seconds of cleanup fit. All activation,
resource, source, ownership and per-stage limits are unchanged. Bind actual
start/expiry before activation and do not launch near expiry.

GRILL clarified timing at 01:04 UTC: preparation and review still stop by
02:00 UTC, while the one unused conditional provisioning job may complete by
02:30 UTC if its full reviewed stages and at least 300 seconds of cleanup fit.
The 55-minute activation cap, 2,400-second owned-runtime cap, 900/1,200-second
acquisition/sync caps, and RAM/disk/source/ownership gates are unchanged. Record
the actual activation/expiry before starting; do not launch on a near-expiry
margin.

Test ROI is an explicit review axis, not a numerical target. Reviewers must
identify the independent contract each changed test protects and whether a
small public integration scenario can replace only truly duplicated unit/API/
CLI examples. Preserve focused tests for distinct invariants, keep independent
oracles, and do not broaden this scope into deleting or rewriting the inherited
suite. The existing audit's two candidate consolidations remain hypotheses, not
approved edits. No fourth old-fixture invocation, broad cleanup, full suite,
training, market/PBT work, staging, commit, or push.

### Review-prep checkpoint (2026-10-10 23:56 UTC)

The isolated profile candidate and one-shot provisioning driver are prepared
for independent review. Added an explicit CUDA 12.8 runtime assertion to the
GPU tensor smoke. The contract tests are parameterized by failure class rather
than duplicating test bodies; they cover the exact supported identity, five
unsupported runtime variants, the exact CUDA sync group selection, valid
independent approvals, and rejection of stale hashes, duplicate reviewer/model
identity, a CHANGES verdict, or a blocking finding. Author validation:
`black` completed without changes; focused pytest reported **13 passed in
0.06 s**; candidate `pyproject.toml` parsed with Python `tomllib`; scoped
`git diff --check` passed (Git emitted only the existing Windows LF/CRLF
normalization notice). An earlier red case exposed an aliased test oracle and
was fixed by copying the expected hash map; it did not expose a product defect.

The requested test-ROI review remains advisory and read-only for inherited
tests: compare the candidate's focused contract tests with the existing
API/CLI and model-state parity tests identified in the CUDA26 audit. Recommend
consolidation only when one public integration scenario preserves each
independent rejection path and oracle. No inherited tests were removed or
rewritten, and there is no count target.

### Corrected review snapshot (2026-10-11 00:01 UTC)

The first two review agents were stopped before returning verdicts after the
author found that `Path.exists()` could skip a dangling symlink/reparse
component. The guard now uses `lstat()` before deciding a component is absent,
and review-receipt/state/wheel target paths are checked before opening or
replacing them. A focused regression case simulates a dangling reparse entry
because this Windows account cannot create symlinks (`WinError 1314`); the
reviewers must assess this oracle and its limitation. The new exact snapshot
passes **14 focused tests in 0.07 s** (no skips), Black, candidate TOML parsing,
and scoped diff-check. The prior review hashes and any incomplete review are
invalid; a fresh independent pair is required for the new hashes.

The next review pass uncovered another blocking source-provenance gap before
either reviewer returned a verdict: `sync-smoke` checked the reviewed lock but
did not revalidate the source hashes approved before acquisition. It now
recomputes the complete candidate source hash set, requires byte-for-byte
equality with the acquisition state, and revalidates the source approval
receipt before proceeding. A focused regression test covers source drift
between stages. The latest source snapshot is **15 passed in 0.07 s** plus
Black, candidate TOML parse, and scoped diff-check. Both review attempts were
stopped without verdicts; only a new pair against this third frozen snapshot
can authorize acquisition.

### Contained acquisition and explicit test-ROI review (2026-10-11 00:07 UTC)

The third invalidated review also caught two source issues before any
provisioning: the scratch uv configuration parent could be created before its
reparse check, and the wheel download used blocking reads in the supervising
process. The driver now validates both uv config paths before creating either,
pins system uv configuration inspection to `C:\ProgramData`, and launches the
wheel-fetch worker through the same bounded `ReplayProcessScope` stage runner
used by lock/sync. The worker rechecks active acquisition state, owner, approved
candidate hashes, and the source-review receipt before network access; the
parent enforces the 900-second and global deadlines and validates the complete
wheel receipt. The runtime smoke also asserts CUDA 12.8 explicitly.

Post-fix author validation: Black check passed, the focused contract suite
reported **15 passed in 0.08 s**, Python compilation succeeded, candidate TOML
parsed, and scoped `git diff --check` passed (only the known Windows line-ending
notice). Root `pyproject.toml` and `uv.lock` remain at their frozen baseline
hashes. No provisioner stage, wheel download, package operation, GPU, or
training action was run. A fresh independent reviewer pair is now reviewing
the exact source hashes, with test ROI explicitly required: assess whether any
new or inherited tests have duplicated public oracles suitable for one named
integration scenario, while preserving focused independent rejection,
arithmetic, trust, and provenance boundaries. No tests were consolidated by
count; the existing two candidate consolidations remain review questions, not
assumed edits. No verdict has been received yet.

### Deadline/metadata repair snapshot (2026-10-11 00:30 UTC)

The renewed adversarial pass reproduced two more failures before provisioning:
malformed dependency metadata could bypass the type assumption in the exact
Torch-profile check, and a child that returned just after its monotonic stage
deadline could be accepted. The profile validator now rejects non-list or
non-string unconditional dependencies; the stage runner checks expiry again
when observing process completion and cleans its owned process scope before
returning the timeout. The stage tests share one fake process-scope helper while
keeping separate before-launch and late-completion expectations.

TDD evidence: both added cases were first observed red (one accepted malformed
dependency field; one accepted late child exit), then green after the narrow
patch. Final focused suite: **34 passed in 0.13 s**. Black reformatted the
provisioner; `py_compile`, candidate TOML parse, and scoped `git diff --check`
passed. Root project and lock hashes remain exactly at their frozen baselines.
Latest scoped hashes: driver `2a92a8bfc5e7eb1a1acf43fa3a5b4b130f5cf53048723a9810e2092297d9665d`,
contract tests `4e0aa84ec6ad7f57da6e928d14c8f1538481eca5ce700d46913e80a5a78deb85`;
other candidate/runtime hashes are preserved from the unchanged prior receipt.
The current source is frozen for a fresh independent review pair; the review
must specifically judge whether additional unit cases duplicate a useful
public integration oracle. No setup download, uv stage, install/sync, GPU, or
training has run; the conditional lease remains inactive.

### Fresh independent review result (2026-10-11 00:38 UTC)

Both fresh reviewers returned `VERDICT: CHANGES` on the exact snapshot above.
They agree on two blocking `_verify_lock` gaps: protected package versions are
collapsed by name in an order-dependent dictionary comprehension, allowing a
contradictory duplicate to pass when the expected version is last; and the CUDA
wheel check uses `any`, allowing a conflicting artifact record beside the
approved one. These are validator false-acceptances; neither reviewer ran uv
selection or installation. Their synthetic records demonstrate only that the
gate accepts conflicting lock bytes, not that the bytes install.

One reviewer also classified malformed receipt/lock table types escaping the
CLI's controlled `ProvisioningError` message as a nonblocking diagnostic nit.
Both reviewers found test-ROI opportunity: a bounded mocked public `main()`
lifecycle could absorb positive receipt, valid-lock, and sync-command checks
while asserting stage ordering, persisted identity, retry refusal, and
no-child-on-rejection. Keep focused negative trust, time, resource, path, and
provenance cases. Existing parameterized rejection matrices are not redundant;
no deletion by count. The repeated deadline harness setup is already shared by
one test helper. This broader lifecycle test is advisory and not a current
CUDA activation prerequisite; no optional test refactor is being added to this
bounded repair.

Both reviewers executed the 34-case focused suite once each (34 passed; 0.12 s)
and independently exercised lock multiplicity in memory. All five design
principles were reported; programming-by-intention was violated by the two lock
false-acceptances. A malformed-shape diagnostic nit was reported by one
reviewer. Source hashes remained stable during both reviews. The review pool is
bounded, not exhaustive; no provisioning/network/install/sync/GPU/Ray/training
work occurred. TDD regressions for both lock multiplicity cases are the next
implementation step, followed by a new independent review pair.

### Lock-cardinality repair snapshot (2026-10-11 00:40 UTC)

The lock regression matrix first failed on four adversarial inputs: a conflicting
protected version before the valid record, and a conflicting or foreign CUDA
wheel alongside the approved artifact (including both list orders). The
validator now requires exactly one record per protected package, exactly one
approved CUDA package source and wheel artifact, and well-formed package/source
tables. Review receipts with a non-object root also produce a controlled
`ProvisioningError` rather than escaping the CLI handler.

TDD evidence: the lock matrix was red before the patch (four failures); focused
lock and malformed-receipt rerun is green (**11 passed**), then the complete
focused contract suite passed (**42 passed in 0.15 s**). The repeated deadline
test setup now uses one shared process/monotonic-time patch helper, retaining
separate assertions for pre-launch expiration and late child completion.
`py_compile`, Black, candidate TOML parsing, and `git diff --check` on this board
passed. Root `pyproject.toml`/`uv.lock` hashes remain frozen at
`11ce4f40f9973549d27cf740b437fe767755ad0d12f7ade74c704a2b0c4e9c16` and
`4abdc1e100e8e86712babcb01ada0a23741c1b29abd896034017f2aad711126e`.

Test ROI disposition from the independent pair: keep each negative trust,
resource, path, time, and provenance oracle. A single mocked public `main()`
lifecycle is a useful future consolidation for the positive receipt/lock/command
path, but is non-blocking and would broaden this bounded repair; no unrelated
tests are removed or rewritten. This repairs the correctness blockers, not the
separate optional lifecycle-evaluation follow-up. Latest frozen code/test
hashes before re-review: `provision_cuda_tested.py`
`1023e2c0ee475ae7478efe7a3ae94112abb861b1703ce6802de27325734c86b3`;
`test_cuda_provisioning_contract.py`
`73947c70d22a93bc020ec92b71bdcf242e1a092a98314b8038e23c2b921a1c0d`.
Fresh two-model independent review is required for these bytes. No provisioning
stage, download/network, package install/sync, GPU, Ray, or training has run.

### Final source review and test-ROI disposition (2026-10-11 00:49 UTC)

Two fresh independent reviewers approved the exact lock-cardinality repair
snapshot: `cuda27_review_astra_repair` (assigned `gpt-6-astra`) and
`cuda27_review_sol_repair` (assigned `gpt-6.1-sol`). Both report zero blocking
findings; Astra notes one nonblocking opportunity to cover positive helper
checks through a future mocked public `main()` lifecycle. The full scoped suite
was executed once per reviewer: 42 passed (0.14 s and 0.15 s). Independent
in-memory probes also passed their stated assertions. Neither review exercised
real process containment, installation, CUDA, Ray, training, or market outcomes;
the candidate files are absent from the recorded base, so executable base
comparison is unavailable. All eleven dispatched hashes and root metadata
hashes stayed stable during review. The complete compact review/test-ROI receipt
is [here](../artifacts/research/cuda27-review-and-test-roi-2026-10-11.md).

Test ROI: preserve the negative trust, schema, path/reparse, runtime/platform,
resource, timestamp/deadline, source/hash, protected-pin, wheel-cardinality and
process-boundary oracles. Their parameterized variants distinguish failure
modes (including order dependence) and should not be removed to shrink the
count. Keep the shared fake-process-scope helper. A single mocked successful
public lifecycle could later replace some positive helper-only receipt, valid-
lock and command-construction checks if it verifies actual command submission,
stage order, persisted identities and fail-closed transitions. Both reviewers
defer that optional refactor outside this bounded source repair; it would
broaden scope without closing a current blocker. No unrelated test was removed
or rewritten.

Fresh native host snapshot at 2026-10-11 00:46 UTC: 4.74 GiB available RAM and
58.48 GiB free on C:, passing the 4 GiB/30 GiB activation floors; the post-
shutdown process inventory contained no Vmmem, Python, uv or Ray process. A
native `wsl.exe --shutdown` returned exit 0; no WSL guest was started. A later
provisioner preflight initially failed because the exact uv 0.11.7 binary
includes build metadata after the version string. No acquisition stage began.
See the follow-up below for the narrow fix and passing native preflight.

### Native uv version-output repair (2026-10-11 00:56 UTC)

The real native `--stage preflight` first exited 1 at the pinned-version check:
the installed binary reports `uv 0.11.7 (9d177269e 2026-04-15
x86_64-pc-windows-msvc)`, while the old comparison required the bare string
`uv 0.11.7`. No wheel fetch, lock generation, or profile creation occurred.
The TDD repair adds one version-output contract that accepts the exact pinned
release with optional parenthesized build metadata, and rejects 0.11.70, 0.11.6,
0.11.7rc1, and a nonzero command exit. The six-case regression initially failed
before the new validator; the focused matrix then passed **6/6**, and the full
focused suite passed **48/48 in 0.15 s**. Black reformatted only the touched
driver/test files. The previous pair's approval binds to the pre-fix bytes only;
a fresh independent pair must approve the final hashes before acquisition.
The first native `--stage preflight` after that parser repair passed at 00:51
UTC with Windows AMD64/CPython 3.11/x64, exact uv 0.11.7, 4.76 GiB available
RAM, 58.48 GiB free C:, and no target profile. It created only isolated uv
configuration directories; no wheel, lock, environment, or root metadata was
created or changed.

### Multiline uv-output boundary repair (2026-10-11 01:02 UTC)

The fresh review pair found that `\s+` accepts line breaks, including
`uv 0.11.7\n(uv 0.11.8)`. Added LF and CRLF rejection cases to the existing
parameterized version-output test; the pre-fix red run reproduced both (2
failed, 6 passed). Restricting the separator to horizontal space/tab now passes
the matrix (8/8) and full focused suite (50/50); `py_compile` passed for both
touched Python files. Native preflight passed again at 01:00 UTC with 4.76 GiB
RAM, 58.48 GiB free C:, Windows AMD64/CPython 3.11/x64, exact uv 0.11.7, and no
target profile. The updated source/test hashes need a fresh independent review;
the earlier approval and ignored review receipt are stale. The reviewers must
also report test ROI: no distinct negative oracle was removed, and the optional
positive public-lifecycle consolidation remains deferred. No wheel acquisition,
lock generation, install/sync, or GPU qualification has occurred.

### Version-metadata and Unicode-separator repair (2026-10-11 01:07 UTC)

The next independent probe showed that arbitrary parenthetical text could hide
`uv 0.11.8` (or bare `0.11.8`) and that `str.strip()` plus the broad metadata
class accepted VT, FF, NEL, U+2028 and U+2029. Added ten more boundary inputs to
the same parameterized CLI-output test: conflicting versions, five metadata
separators, and leading/trailing Unicode separators (plus the ordinary trailing
CLI newline positive control). The pre-fix run reproduced nine rejections that
were incorrectly accepted (9 failed, 9 passed). The validator now strips only
ASCII CLI whitespace, restricts metadata tokens to printable ASCII word/build
characters, and rejects semantic-version tokens inside metadata. Targeted matrix
passes 18/18; focused suite passes 60/60 in 0.16 s; `py_compile` passes and Black
formatted the driver. This remains one test function with table-driven
independent boundary cases; no distinct oracle was removed. Fresh review must
again assess that ROI against the public lifecycle, while retaining parser
boundary coverage. No acquisition or profile creation occurred.

The earlier test-ROI finding remains closed for this repair: preserve independent
negative trust/schema/provenance/resource/path/time/lock oracles and avoid
count-driven removal. A mocked positive `main()` lifecycle remains a potential
separate test redesign; it is not being added to this narrow fix.

### Exact uv build allowlist and receipt binding (2026-10-11 01:09 UTC)

To close the remaining open-ended metadata cases, GRILL recommended accepting
only the canonical `uv 0.11.7` line and the exact native build string observed
from the pinned executable, with at most one normal LF or CRLF line ending.
The helper now returns that normalized accepted line and `_check_base` records
it as `uv_version_output` in preflight evidence, which is then embedded in
provision state. The previous generic metadata grammar and semver scan are
superseded. The existing table-driven test now covers exact accepted forms,
line endings, the malformed/conflicting/Unicode variants found by review, and
preserves diagnostics; 24 cases pass. The focused suite passes 66/66 in 0.16 s,
`py_compile` passes, and Black completed. This remains one parameterized test
function; no tests were deleted. Fresh two-model review must assess whether any
boundary row is redundant and must bind the final exact hashes. No final-source
native preflight or acquisition has run; prior approval/preflight receipts do
not qualify this snapshot.

## Prior Scope: DEV-CUDA26 Implementation And Qualification Prep (superseded)

Received GRILL's scoped dispatch at 2026-10-10 20:45:19 UTC. Allowed write set:
the CUDA25 source set, this DEV board, ignored CUDA-local scratch, and one
compact durable receipt. Preparation/repair/review ends within 45 minutes of
receipt and absolutely by 22:45 UTC. No more than two bounded launcher red/green
rounds, each <=300 seconds and <=600 seconds total. The previous three uv
fixture invocations are consumed; no fourth invocation. No Torch/Ray/GPU/full
suite, WSL, driver changes, market/PBT work, or Git shipping during preparation.

Implement the accepted uv CPU-default / explicit mutually-exclusive CUDA
profile, no-sync launcher and stdlib identity probe, compact public launcher
tests, and explicit GPU/device opt-in in the existing four-cell checkpoint
integration. Resolve all CUDA25 review blockers. Preserve root CPU `.venv`,
Ray 2.58 behavior, optional groups, model/economic recipe, and ordinary CPU CI.
The complete real acquisition, hash, lock-binding and sync command set must
receive two fresh independent `REVIEW.md` approvals before any conditional
heavy lease can activate. If that gate or resource preflight is not complete,
handoff the precise blocker; do not acquire or sync.

### DEV-CUDA26 review outcome (2026-10-10)

The initial frozen snapshot received two independent `VERDICT: CHANGES` results:
both reviewers reproduced one blocker, where an initially absent
`ITRADER_PYTHON_BIN` remained present after a rejected launch. The cause was
unconditional `SetEnvironmentVariable(..., $null, 'Process')` in `finally`.
This was a launcher defect, not a test-oracle mismatch.

GRILL issued a superseding one-round repair scope at 21:04 UTC, within the
original preparation deadline: repair only exact caller-environment
restoration; allow one author test invocation and one per fresh reviewer; no
retries. The launcher now records entry presence and explicitly removes
`Env:ITRADER_PYTHON_BIN` when it was absent. The author's single invocation
passed **6 tests in 2.92 s**. Both fresh independent reviewers then returned
`VERDICT: APPROVE`, zero findings, and **6 passed** (2.90 s and 2.91 s); all six
source/metadata hashes matched before and after review. The five review
principles are OK for the bounded candidate. This approval does not authorize
package acquisition or sync. No CUDA provisioning, root `pyproject.toml`/
`uv.lock` change, GPU/Ray/Torch run, or Git shipping followed.

The bounded read-only test-ROI audit covered 28 changed test modules in the
inherited working tree (7 modified, 21 new; 276 added test-function
definitions). It found no basis for broad unit-to-integration conversion or
deletion by count: most cases cover distinct public, trust, arithmetic, or
provenance boundaries. Two plausible future consolidations are the duplicate
partial-terminal-screen rejection path (`test_terminal_screen.py` API and CLI
cases) and model-only state-neutrality parity cases in `test_no_nan_forward.py`.
Keep the lower-level completion validator and environment-to-model integration
oracles. Reviewers also identified one redundant final canonical-digest equality
assertion in the checkpoint integration; preserve its separate payload-extraction
and callback-payload digest checks. No test consolidation was made in CUDA26.

Final reviews marked all five principles OK, including Encapsulation after the
repair. See
[`cuda26-review-and-test-roi-2026-10-10.md`](../artifacts/research/cuda26-review-and-test-roi-2026-10-10.md)
for exact reproductions, review commands, current-snapshot identities, and
follow-up boundaries.

### CUDA26 packaging continuation gate (2026-10-10 21:23 UTC)

The user's follow-up explicitly retains test ROI as a review requirement. The
completed audit remains the bounded disposition: do not convert tests to
integration tests or remove them by count; preserve distinct trust, arithmetic,
restore, and provenance oracles, and consider only the two documented duplicate
scenario consolidations in a separately scoped change. No test source was
changed in this continuation.

Packaging preflight is still not qualified: no production routing/lock packet
has received the required two independent reviews, no wheel was acquired, and
no environment was synced. Read-only local checks observe uv 0.11.7, the exact
native base interpreter `C:\Users\Admin\itrader\python.exe` as CPython 3.11.13
64-bit, about 5.40 GiB free RAM, and 59.03 GiB free on C:. These resource values
clear the stated start thresholds but do not replace a fresh lease preflight.
The advertised wheel digest is not a locally measured payload digest. Do not
activate the conditional heavy lease or claim package/runtime qualification
without the complete reviewed lock/source/acquisition procedure. No CUDA,
Torch, Ray, WSL, PBT, test, or package operation was started in this
continuation.

The initial review packet is at
[`cuda26-production-routing-review.md`](_local/cuda-uv-bridge12/cuda26-production-routing-review.md).
Two fresh independent reviews of those bytes both returned `VERDICT: CHANGES`.
Their common blocking findings are: the lock target and sync project path are
not concretely bound to a scratch candidate; the sync has no enforceable
timeout, process-tree stop, or live RAM/disk floor monitor; and the fixed
environment target/`UV_PROJECT_ENVIRONMENT` caller state is not safely
validated/restored. One reviewer additionally found that the CUDA marker admits
PyPy 3.11 despite a CPython-only wheel; the other noted an explicit CUDA-group
selection can be empty on unsupported platforms unless the Windows-only
contract is made clear. Both reviewers attest the portable lock, local wheel
digest, and runtime remain unknown. No lock resolution, full payload read,
root metadata/lock edit, sync, or conditional lease activation occurred.

GRILL extended packet preparation to 21:45:19 UTC only. This extension is
closed early at 21:39 UTC because the exact-path guarded driver and required
fresh final review pair were not ready; the initial packet remains at the
CHANGES verdict. Do not treat the prose packet or existing replay process-scope
helper as an executable provisioning gate. The absolute 22:45 UTC stop,
>=55-minute activation margin, and all acquisition/sync/resource caps remain
unchanged. Root `pyproject.toml` and `uv.lock` are unchanged; no package
operation, test, download, sync, or runtime action occurred. Continue only from
a newly scoped, reviewable candidate that binds scratch project/lock/target
paths, verifies ownership/canonical paths/reparse points, enforces CPython ABI
and supported-platform rejection before selecting CUDA, and uses bounded
process-tree/resource supervision. The test-ROI disposition in the review
receipt remains unchanged and no test consolidation was made.

## CUDA-UV25 Review Outcome Carried Forward

## CUDA-UV25 Independent Review and Test ROI (2026-10-10)

Two fresh independent `REVIEW.md` reviews completed against the bounded CUDA25
candidate/evidence scope. Both returned `VERDICT: CHANGES`; no provisioning,
root metadata/lock edit, environment creation, or new oracle invocation
followed. Their findings are blockers, not approval: validate 64-bit process
identity rather than host architecture alone; reject protected `.venv` junction
or reparse-point aliases; define and verify the exact cached-wheel acquisition,
SHA-256, lock binding, and single-sync procedure; include the `dev` group when
the isolated profile must run pytest; and preserve native child exit codes under
PowerShell 7.4+ error preferences. Exact source-routing selection is supported
by retained raw subchecks, but the final redundant export-recheck receipt stays
`FAILED` because its extra HTTP-path assertion was overstrict. Do not relabel
that receipt green or claim TLS, payload, installation, or GPU qualification.

Test-ROI decision from both reviews: keep the existing resolver and checkpoint
integration cases because they cover distinct trust/lifecycle boundaries. For
the proposed launcher, prefer one public lifecycle integration (interpreter,
arguments including spaces, child working directory, exit propagation, and
caller environment/location restoration) plus a compact parameterized rejection
matrix. Keep distinct error and never-started signals; do not create one test per
field or replace solver/checkpoint/GPU coverage with launcher tests. No existing
unit test was found safe to delete or merge. Any corrective implementation and
its tests require a renewed bounded scope and fresh two-review gate; no code or
test edits were made in this review pass.

## Closed Scope: CONTINUATION22

GRILL's 40-minute DEV22 dispatch is closed at 18:45 UTC, before its hard stop
19:15 UTC. The current HEAD is externally recorded by GRILL as `8f9fcf9`; DEV
did not run Git, so neither HEAD nor the dirty/index state is newly certified.
All inherited work is preserved. Writes remained within the declared DEV22
board, accounting test, ignored local continuation22 files, and compact report.
No source implementation, runtime/lock promotion, test/import execution,
training, market data/outcome access, WSL, install/download, or Git shipping
occurred.

GRILL reports TRAINING rechecked the accepted 49-entry lifecycle20 identity
manifest with zero missing/mismatch. Its frozen lifecycle test and runtime
identities exclude both accounting target files; environment, evidence,
reward, R1 model and lifecycle identities remain untouched here. DEV owns no
process/job. Do not run tests or imports while TRAINING22 owns the only heavy
lease. The repository GitNexus status helper invokes Git; it is therefore not
run under this scope's no-Git rule.

The test-ROI pass is deliberately bounded to the accounting owner and nearby
public boundaries. The existing file has 11 test functions / 28 expanded cases
before this edit. Rather than adding a new green-test rung, the existing
environment/diagnostics integration scenario now uses the accepted literal
hold/reverse/liquidate path `[100, 100, 110, 121, 110]`, start 2, horizon 3,
actions `[2, 2, 0]`, and 10 bp. It asserts independent gross/cost/net,
turnover, reward, per-step transition, terminal evidence and reset values.
No unit boundary was removed. Static Standards and Spec reviews returned
APPROVE with zero findings; they explicitly did not execute code or compare a
base snapshot. No test was run due to TRAINING's lease.

Keep the nine-case transition matrix, finite-volatility rejection matrix,
trace-identity rejection, causal future-row probe, and partition fail-closed
cases: each tests a distinct arithmetic or trust boundary. Do not delete or
merge them merely to lower a count. After an authorized run, review whether the
basic step terminal-cost example is genuinely subsumed by the public lifecycle
oracle; retain its hypothetical-roundtrip distinction unless an independent
replacement is demonstrated. The reviewers found no unit case safe to collapse
into this lifecycle. Shared fixture-writing could be extracted, but keeping
each synthetic path/window/task/cost setup explicit yields more review value
than saving setup lines. No unit-test consolidation/removal is made in
CONTINUATION22; the existing public-path pilot was expanded in place.

Record the repricing API proposal as
`reprice_episode_cost_scenarios(*, steps, evidence, scenario_cost_bps)`: call
the existing episode reconciler once against the original evidence/cost,
derive turnover including terminal liquidation from that validated trace, and
compute `gross - turnover * scenario_cost_bps * 1e-4` for each declared cost.
Return ordered scenarios plus an algebraic break-even bp value only when
turnover is nonzero; zero turnover is explicitly unavailable, and negative
gross cannot become profitable at nonnegative costs. This is frozen-trace
accounting only—no reward mutation, retraining, or edge claim. Write its red
test after a separate execution-backed DEV lease is issued; no implementation
or result is claimed now.

In parallel, a full review-only uv candidate now reflects the accepted contract:
uv is supported, default groups are `dev` + pinned CPU, CUDA is an explicit
mutually exclusive group, and plain `pip install .` will require separate
PyTorch setup. It preserves the existing optional groups and uses the official
CPU wheel index for Windows/Linux; the actual installed CPU binary identity is
not certified by the candidate or prior lock. The old fragment is retained as
superseded history. Do not alter project metadata/lock, resolve, sync, download,
install or promote root runtime. Preserve the external cu128 benchmark's
unqualified public-stack/economic status. Hand off through this board by 19:15
UTC with final local hashes, review and execution status, and next lease
boundary.

Source-routing follow-up: the candidate now uses group-scoped CPU and CUDA
source declarations, but exact uv 0.11.7 behavior remains unresolved. Official
docs demonstrate source routing by optional dependency extras; open issue
#15730 reports similar group-source leakage in 0.8.15 and a later report on
0.9.26. This does not establish a 0.11.7 defect or correctness. Candidate
promotion is blocked until an exact-version, throwaway, local-wheel lock-only
smoke proves CPU and CUDA selections remain distinct. The independent Spec
review rated this `UNRESOLVED`; it was static only, not a full REVIEW.md
verdict. See the report for its exact oracle and URLs.

Evidence receipt: [CONTINUATION22 DEV report](../artifacts/research/continuation22-dev.md)
SHA-256 `BF765D6706C49F5CCA56483949A182DC2BD5828067AEE5478D53004C39B40643`.
Full uv candidate SHA-256
`F576DBC93B2F3901C76E9678FB290A053674A024DC25F2928763B37A6C4B7314`.
The final review note SHA-256 is recorded in the report. No test, import,
resolver/`uv`, Git, install, download, or runtime command was run in DEV22;
official documentation and upstream issue pages were read-only research.

## Conditional Scope Closed at First Gate: CUDA-UV23 / DEV23

GRILL dispatch recorded at 18:43 UTC. Light preparation and review completed
at 18:51 UTC, within its 15-minute cap (18:58 UTC). Architecture/package docs, current
root `pyproject.toml`/lock, existing launcher tests, review contract, and the
unexecuted local launcher/probe were inspected. GitNexus status through native
Git Bash reports the index stale (2026-06-17 commit `1d5ba82` versus current
`8f9fcf9`); its summary then stopped because `python3` is unavailable in Git
Bash. No WSL or index refresh was started. A compact execution plan is at
`docs/coordination/_local/cuda-uv23/bridge-preparation.md`, SHA-256
`2EBA49803388C3BF533AD38A2D9E9AF76F6E7933D212ABCE9B96F0F1EE7F9B08`.

A single conditional `LEASE-DEV-CUDA-UV23` may activate only after TRAINING22's board explicitly
closes with no owned process and fresh source/conflict/resource preflight.
Until that activation, do not invoke uv/resolution, acquire a wheel, sync or
install, edit root `pyproject.toml`/`uv.lock`, or start runtime checks.
Application/provisioning is capped at 55 minutes after activation, with an
absolute stop at 20:10 UTC; one acquisition, one sync, no automatic retries.

Allowed future writes: `pyproject.toml`, `uv.lock`, a no-sync CUDA launcher and
stdlib identity probe under `itrader/tools/`, compact launcher/profile tests
beside `test_run_repo_python.py`, this board, ignored
`docs/coordination/_local/cuda-uv23/`, and one compact research receipt. GRILL
retains active environment/plan docs. Preserve inherited dirty bytes. The
accepted Windows CPython 3.11 cu128 nightly URL is fixed; its HTTP 200 and
3,062,428,787-byte reported length are availability evidence only. Verify
actual size and SHA-256 against the advertised digest before locking/syncing.
Keep root CPU `.venv` untouched; use only the isolated CUDA profile. Preserve
Ray 2.58, Gymnasium 1.2.2, NumPy 1.26.4, and existing optional groups.

The mandatory first technical gate is an isolated uv 0.11.7 lock-only
source-routing smoke; the DEV22 group-scoped candidate remains unresolved until
that passes. Then review a full resolution copy for default CPU, CUDA-only,
group conflict behavior, cross-platform forks, package pins, and source URLs
before applying project lock changes. Review exact contained argv commands
before the single permitted download and sync. RAM start/abort is 4/2 GiB;
disk start/abort is 30/15 GiB; download <=900s, sync <=1200s, total owned
execution <=2400s. A failed attempt is preserved without retry. Package,
launcher, and GPU identity probes do not qualify Ray/PPO or economics; those
require a later TRAINING dispatch. Accounting TDD and DEV19 Spec closure stay
queued. No direct peer messages; routine progress, blockers, and final receipt
return on owned boards. Current HEAD is externally recorded as `8f9fcf9`. The
DEV23 scope permits read-only Git HEAD/status/diff checks for later base/source
review, but no stage/commit/push/rebase/reset or index-changing operation.
Provisioning must use native CPython 3.11, avoid source builds, and not download
an interpreter/toolkit or build unrelated optional packages.

### Activation and stop receipt — 2026-10-10 19:10 UTC

TRAINING/TESTING's board explicitly records R1-LIFECYCLE22 closed, four of four
synthetic cases passed, zero skips/failures/errors, no cleanup survivors, and
no owned process. The fresh DEV preflight at 19:06 UTC observed unchanged root
`pyproject.toml`/`uv.lock`, HEAD
`8f9fcf94e028b51d883a1a484c2c9466a3d6b547`, 6.13 GiB available RAM, 59.12 GiB
free C: disk, and no Python/Ray/uv process. The isolated destination did not
exist. Conditional activation followed that preflight; the 55-minute cap and
20:10 UTC hard stop were not approached.

The exact-uv 0.11.7, CPython 3.11.13, offline, local-wheel lock-only smoke was
the first technical gate. Its command was
`uv --no-config lock --offline --python C:\Users\Admin\itrader\python.exe`
from the ignored `source-routing-smoke/project` directory. Resolution failed
before producing a lock:
`Failed to download torch @ file:///.../torch-2.9.0.dev20250801%2Bcu128-py3-none-any.whl`;
`builder error for url`; `URL scheme is not allowed`. This is a failed oracle,
not proof that the production HTTPS candidate is broken or that its group
scoping passes. Per the stop rule, no alternate source design or retry was
attempted. The local fixture and failure output are retained in ignored
`docs/coordination/_local/cuda-uv23/source-routing-smoke/`.

Read-only official artifact metadata check clarified provenance: the exact
wheel's HTTP HEAD response advertises
`x-amz-meta-checksum-sha256: 8ff2cbe2a88bebfc14b71fea17f4937d77f0ff70aa9e4cebd6836b5bbe82765d`
with `Content-Length: 3062428787`; this digest came from the official wheel
object response header, not its PEP 658 `.whl.metadata` body. The sidecar
contains package metadata only. The value is an advertised checksum, not a
locally verified payload hash: no wheel bytes were downloaded. The official
HEAD response also reported an ETag that is a multipart identifier, not a
SHA-256.

No root project or lock bytes changed; no uv lock was produced in the smoke
project; no wheel acquisition, environment creation/sync, Torch import, GPU
probe, tests, Ray/PPO, training, or Git write occurred. The CUDA profile is
not promoted. A fresh GRILL authorization is required to adapt or replace the
local-only oracle and continue. The independent source-routing/application
and acquisition/containment reviews were not started because the mandatory
first oracle did not pass.

## Closed Scope: CUDA-UV24 local fixture repair

GRILL dispatched a fresh <=10-minute preparation window at approximately
19:14:50 UTC, followed by exactly one repaired source-routing oracle run of at
most 60 seconds. Preserve the failed UV23 `file://` fixture and result. This
scope may only write this board and ignored `docs/coordination/_local/cuda-uv24/`.
Use local wheel-path sources only with explicit source-kind limitations, or a
finite loopback-only HTTP fixture server with owned timeout/cleanup. No external
resolver or payload access is allowed in the tiny oracle. It must exercise
default CPU, explicit CUDA, and mutual-exclusion rejection using selected
versions and sources; fixture generation alone is not evidence.

The one repaired oracle ran at 19:20 UTC with uv 0.11.7 and CPython 3.11.13,
using one finite loopback-only server and tiny wheels. It exited 1 in 1.3 s
before producing `uv.lock`. uv fetched only from logged loopback paths, then
reported an unsatisfied `torch==2.3.1+cpu` requirement for the non-Windows,
non-Linux platform branch. The fixture CPU requirement lacked the same
Windows/Linux marker used by its source mapping, so this is a fixture-contract
failure; it does not establish a production HTTPS/index source-routing defect.
The server closed in `finally`; post-run inventory found no Python/Ray/uv
process, 6.21 GiB available RAM, and 59.12 GiB free C: disk. The predeclared
default CPU, CUDA-only, conflict, and default-without-CUDA assertions did not
all execute. Per the one-run stop rule, there was no retry or fallback.

The loopback transport exercised the accepted source kinds (CPU explicit index,
CUDA direct wheel URL) without external resolver or wheel payload access. This
attempt does not qualify selected lock sources, group exclusion, installation,
runtime identity, CUDA availability, Ray/PPO, or economics. No root package or
lock bytes changed; no real CUDA acquisition/sync, root tests, training, WSL,
or Git write occurred. The failed script, fixture, and predeclared cases are
preserved in ignored `docs/coordination/_local/cuda-uv24/`.

The light repair scope is closed. No independent reviews or conditional
`LEASE-DEV-CUDA-UV24` activation followed because the oracle did not pass. A
new GRILL decision is required before any further fixture, resolver, project,
or environment work. The accepted wheel, checksum, and package contract remain
unchanged; no UV upgrade, source-group-to-extra redesign, or fallback wheel was
selected.

## Current Scope: DEV-CUDA25

GRILL superseded the expired UV23/24 leases and dispatched a finite CUDA25
bridge. Preparation and independent review are capped at 30 minutes from the
20:25:58 UTC receipt; the absolute scope stop is 22:05 UTC. Up to three local-
only source-routing oracle invocations are permitted, at most 60 seconds each
and 180 seconds total. Preserve the UV23 and UV24 failures; any substantive
source, resolution, or integrity failure stops the scope.

The corrected oracle must model the accepted full platform branches: Windows/
Linux pinned CPU from the explicit CPU index; Darwin CPU from the default
package source; and Windows AMD64 CPython 3.11 CUDA from the exact direct wheel
URL. Assert default CPU selection, explicit CUDA selection, selected package
version/source, Darwin's default-source record, and CPU/CUDA mutual-exclusion
rejection. Tiny fixtures may use only loopback; no real wheel bytes or external
resolver access in these checks.

After a passing oracle, prepare two fresh independent REVIEW.md reviews of the
unchanged accepted root routing copy and contained provisioning commands. Only
after both approvals plus fresh source/conflict/resource checks may the sole
`LEASE-DEV-CUDA25` activate. Keep one real artifact acquisition and one isolated
sync, exact version/digest, root CPU protection, Ray 2.58/Gymnasium 1.2.2/
NumPy 1.26.4 and optional-group preservation. Follow the 4/2 GiB RAM and
30/15 GiB disk gates; <=55 minutes from activation, <=900 seconds download,
<=1200 seconds sync, <=2400 seconds owned total, and 22:05 UTC absolute stop.
Use only `%LOCALAPPDATA%\iTrader\envs\cuda-tested-cu128`; create no additional
full probe environment. No WSL, drivers, Git shipping, market access, training,
or native Ray API migration.

Allowed tracked writes are `pyproject.toml`, `uv.lock`, the no-sync CUDA
launcher and stdlib identity probe under `itrader/tools/`, compact launcher
tests beside `test_run_repo_python.py`, and (only if required) narrow explicit
GPU opt-in for the existing four-cell checkpoint integration. Ordinary CI
stays CPU-only with no silent GPU fallback. This board and ignored
`docs/coordination/_local/cuda-uv25/` hold routine receipts; GRILL owns active
environment/plan docs. No stage/commit/push or root `.venv` changes.

Human test-ROI requirement: the final source review must assess whether the
launcher/profile behavior can be covered by one public lifecycle integration
plus a compact rejection matrix, instead of adding a separate green unit test
for each option or field. Preserve focused tests for genuinely distinct
invariants; consolidate only where the independent oracle and failure signal
remain clear. Do not remove existing tests to meet a count or quota.

### CUDA25 oracle attempt 1 (20:27 UTC)

uv 0.11.7 resolved a throwaway full-branch lock in 0.294 s using only loopback
fixtures. Lock assertions passed: Windows/Linux Torch 2.3.1+cpu came from the
explicit CPU index; Darwin Torch 2.3.1 came from the default-source fixture;
CUDA Torch 2.9.0.dev20250801+cu128 came from the direct fixture URL. The
conflict request correctly exited 2 and named both groups. The attempt then
failed its own export oracle: uv's requirements export retained both mutually
exclusive, platform-marked CPU records, while the test incorrectly expected a
single current-platform line. A `uv tree` probe also showed only the project
root and is not accepted as Darwin selection evidence.

Scope incident: export/tree commands omitted an explicit interpreter and uv
automatically downloaded managed CPython 3.11.15 (24.4 MiB) despite the lock
command using native CPython 3.11.13. The download and cache are preserved;
nothing was deleted. This was outside the intended local-only fixture path.
No root config, root lock, CUDA environment, real Torch wheel, or `.venv`
changed. Post-attempt inventory: 5.49 GiB available RAM, 59.05 GiB free C:,
and no Python/Ray/uv process. One final bounded harness attempt will set
`UV_PYTHON_DOWNLOADS=never`, pass the exact interpreter to every uv command,
and assert universal marker-preserving export semantics. No real provisioning
can proceed unless that corrected oracle passes and the fresh reviews/resource
gates pass.

### CUDA25 oracle attempt 2 (20:29 UTC)

The corrected universal export assertions and the explicit CPython identity
passed: every uv command used native CPython 3.11.13 with
`UV_PYTHON_DOWNLOADS=never`; no interpreter download occurred. The lock-source
checks showed the expected Windows/Linux CPU index, Darwin default registry,
and CUDA direct URL. Default export preserved both CPU marker branches and no
CUDA record; CUDA-only export contained only the direct wheel URL; conflict
export exited 2 and named CPU and CUDA groups. The attempt failed only at the
final loopback-log assertion, which checked `request.startswith('/cpu-index')`
while records correctly begin `GET /cpu-index/...` or `HEAD /...`. All five
expected local source paths were present. No external endpoint was contacted;
the loopback server closed in `finally`. No root configuration/lock, `.venv`,
or CUDA profile changed. One permitted local oracle invocation remains; it will
correct the request-path assertion only, without altering the profile contract.

### CUDA25 oracle attempt 3 (20:31 UTC)

The final permitted local-only export recheck used native CPython 3.11.13 for
every uv command and disabled managed interpreter downloads. The decisive
selection checks passed: universal default export retained both platform-marked
CPU versions and no CUDA dependency; CUDA-only export selected the direct
cu128 wheel URL; the mixed CPU/CUDA request exited 2 and named both conflicting
groups. The lock records also matched the expected explicit Windows/Linux CPU
index, Darwin default source, and CUDA URL. The script nevertheless recorded
`failed` because it expected the *locked export* to request all five local
index/wheel paths. This command reused an already-resolved lock, so it made
three GETs only for the 750-byte synthetic CUDA fixture; it did not revisit CPU
indexes. This extra assertion does not negate the source-selection/export and
conflict evidence, but the final harness receipt is not all-green. All three
oracle invocations are consumed; no retry is permitted. The server stopped in
`finally`, no Python/Ray/uv process remained, and root `pyproject.toml`/`uv.lock`
hashes remain protected. No real wheel, environment, root config, or runtime
was changed. Do not begin provisioning unless the two independent reviews and
GRILL's source-oracle disposition explicitly accept the existing evidence.

## Current Scope: SALUKI-ADVISORY-ASSESSMENT21

Delivered 2026-10-10, before the 02:40 UTC stop. This was a static source/test/
configuration audit only, with allowed writes limited to this board and
`docs/artifacts/research/saluki-source-audit-2026-10-10.md`. No source/test
edits, test/runtime/Ray/training execution, market data/outcome access, or Git
operation occurred. The audit includes the corrected literal-vs-hypothetical
v2 concatenation behavior, source hashes, concrete public integration oracles,
and the requested test-ROI/consolidation review. GRILL owns synthesis; no code
change, test consolidation, method adoption, or experiment is authorized by
this receipt. Hash-integrity correction: the first recorded training-source
digest was only 62 hex characters. It was re-read from the source file and
corrected to its 64-character SHA-256; this amends the transcription, not a
retroactive certification of any changed source snapshot. Final audit SHA-256:
`54ad066d0e0af32304b53b1a1d417d85808ac6368e6e64d1575b2487171eaa4c`. See the
completion record near the end of this board.

## Previous Scope: EVAL-DISPOSITION-IMPLEMENT19

Closure handoff (2026-10-09): source is frozen with the lease withdrawn. No
further commands, tests, probes, child agents, or code edits are authorized in
this scope. The independent Standards run remains recorded; the static Spec
review's execution blocker is deferred to a fresh, separately scoped dynamic
review after TRAINING's sole Ray job ends. Reporting changes are outside that
frozen lifecycle identity. The earlier TDD-red and unexpected intermediate
failures remain disclosed; durations for those earlier invocations are
unknown, so cumulative runtime remains unknown. The corrected template digest
is an amended record based on the later independent hash report, not a
retroactive certification of the malformed first receipt.

GRILL dispatch: 2026-10-09 12:39 UTC; hard stop 13:15 UTC; <=360 seconds total
focused integration/test/reviewer execution, sequential native Windows root CPU
thread1. Fresh preflight required before every command: RAM >=4 GiB start / abort
below 2 GiB; disk >=20 GiB start / abort below 10 GiB. No other heavy job.
Ray/PPO/model/full-suite/pre-commit, installation, acquisition, real outcomes,
WSL and all Git operations are prohibited. Q23 clearance, Q26 calibration,
terminal execution, significance, all-16 run, method adoption and promotion
remain closed. Preserve all existing dirty work and failed evidence.

Before implementation, record the intended public interface:
`evaluation_disposition.build_evaluation_disposition(*, evidence_validity,
metric_availability, inference_qualification, promotion_eligibility) -> dict`.
It validates/version-tags the four explicit orthogonal status objects and must
not infer validity from metric values, infer Sharpe availability from scalar
zero, decide an unapproved method, or grant execution/promotion authority.
Production integration is limited to the existing terminal panel report,
read-only terminal preflight, and browser adapter/presentation.

Exact selected write set: this DEV board; new
`itrader/rl_lib/evaluation/evaluation_disposition.py`;
`itrader/rl_lib/evaluation/terminal_panel_report.py`;
`itrader/rl_lib/evaluation/run_terminal_screen.py`;
`itrader/scripts/experiments/render_experiment_browser.py`;
`itrader/scripts/experiments/schemas/experiment_registry.schema.json`;
`itrader/scripts/experiments/templates/experiment_browser.html`;
`itrader/tests/test_economic_diagnostics.py`;
`itrader/tests/test_terminal_panel_report.py`;
`itrader/tests/test_terminal_screen.py`;
`itrader/tests/test_experiment_browser.py`.
No other source, tests, registries, generated HTML, artifacts, architecture or
plan files may be changed without a returned scope request. `episode_evidence.py`,
`economic_diagnostics.py`, `browser_traces.py`, panel qualification and provenance
remain read-only owners/dependencies. No `CONTEXT.md` or ADR files were found.

Human test-ROI instruction: the independent reviews must decide whether any
new or existing unit checks can be consolidated into the public report/browser
integration without losing distinct rejection paths. Avoid one-test-per-field
growth; retain focused pure-contract checks only for behavior the public path
cannot isolate.

Implementation/test record (2026-10-09): added the pure status validator,
attached disposition to descriptive Q24 output and read-only preflight, and
exposed only explicitly recorded dispositions through existing browser
attributes. Legacy records remain unchanged rather than receiving inferred
statuses; their absent values stay absent. No registry, generated browser,
template, Q23 panel, Q26 method, or market outcome was changed. TDD first
failed collection because the new validator did not yet exist. An intermediate
focused check exposed two test-expectation mistakes; the broader run then caught
that defaulting every legacy arm would alter protected generated-browser bytes.
The adapter was narrowed to preserve exact legacy parity and display only
source-recorded statuses. A transient syntax error during that narrowing was
fixed before final validation.

Final focused command, after fresh resource/process/hash preflight, was
`.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider
itrader/tests/test_economic_diagnostics.py
itrader/tests/test_terminal_panel_report.py
itrader/tests/test_terminal_screen.py
itrader/tests/test_experiment_browser.py`: 109 passed in 3.55 s. This retains
the existing accounting and weighted-panel integrations, adds one compact
parameterized status-contract rejection matrix, one report-level disposition
assertion, one successful read-only preflight case, and extends an existing
source-to-rendered-browser integration with positive/zero/negative net plus a
legacy zero-Sharpe value. No new test file or per-status test ladder was added.
Each execution preflight observed >7.3 GiB available RAM, >62 GiB free disk,
and no conflicting Python/pytest/Ray processes; native Windows CPU thread count
was limited to one. No full suite, pre-commit, healthcheck, terminal evaluation,
training, all-16 run, or generated artifact refresh was run. Independent
reviewers must specifically report test ROI and any safe consolidation.

Execution-boundary reconciliation (2026-10-09): the GRILL stop rule did not
explicitly exempt the planned TDD-red invocation, so no further local test or
reviewer command may run under this lease. The planned initial collection
failure was expected because the new validator module did not exist yet; it is
not a product regression, but the lease wording was still ambiguous. Later
intermediate test failures were unexpected test-oracle failures (two incorrect
expectations) and a generated-browser byte-parity failure; these were repaired
before the final implementation run. A transient syntax error occurred while
editing the renderer and was fixed before final validation; it was not a test
result. The only recorded durations are the final focused suite (109 passed,
3.55 s) and the independent Standards rerun (109 passed, 3.57 s), totaling
7.12 s of measured test runtime. Earlier command durations were not recorded,
so total cumulative test execution time is unknown and is not reconstructed.
The pre-edit browser-template digest below was malformed in the first receipt;
it is corrected to the verified 64-character digest. This correction records
the identity reported by the later independent hash check and does not
retroactively certify the malformed receipt. One independent Standards review
is complete (0 blocking, 0 nit, APPROVE). The second independent Spec review is
complete as static-only: no source-level defect found and no test consolidation
recommended, but it records one BLOCKING review-process finding because the
required probes were not run after lease withdrawal (`VERDICT: CHANGES`). Its
test-ROI conclusion is to retain the compact parameterized malformed-contract
matrix and the distinct report, read-only preflight, and browser integration
checks; these protect different public boundaries. No tests, probes, Git
commands, or edits were performed by that reviewer. Thus the two-review gate is
not closed: only the Standards reviewer issued an execution-backed APPROVE. No
training, replay, or terminal evaluation process is owned or running by DEV.
Source remains frozen; further execution requires a new explicit lease and no
code edits are authorized unless that lease permits them.

Pre-edit SHA256 inventory (new module was absent):

| Path | SHA256 before implementation |
|---|---|
| `docs/coordination/dev.md` | `d0939c42f9bed53caedc0ab50dc0acda582fff107940b8da3d5702caaad5649b` |
| `itrader/rl_lib/evaluation/terminal_panel_report.py` | `66bc7fefbd725be8b907562965aac7863c052057fc9ec9c70ae6f4e0aa557dd8` |
| `itrader/rl_lib/evaluation/run_terminal_screen.py` | `4cbc034bd9f4942d5eab84601ec9e0b46b3228f9d12b75872b289fef928127a5` |
| `itrader/scripts/experiments/render_experiment_browser.py` | `aa0fcd2d18789bb495bf2ab9aaf31526de213482aabf57becf2d6524ccb4a6df` |
| `itrader/scripts/experiments/schemas/experiment_registry.schema.json` | `c0022b9948009910a31d27e1c2affbfc1e9eba008f25ac5daea589e7a160e829` |
| `itrader/scripts/experiments/templates/experiment_browser.html` | `ef6ca24fec129ee11e445ec7f1d31b2bc99f61a255f491703ab8ce60962a0b9d` |
| `itrader/tests/test_economic_diagnostics.py` | `0717f2b33bf771f4cc5ef0838696e25e1cc6e3a0570600417f3f0050f26ecaf6` |
| `itrader/tests/test_terminal_panel_report.py` | `445f9ac0f451922f6389b0bab8a3085022cbca2b11d56d8b661d8561b297e057` |
| `itrader/tests/test_terminal_screen.py` | `38b18c42681a1119198a8ace3867fb13df6580e3c854b2a56c2c84174b2a62b6` |
| `itrader/tests/test_experiment_browser.py` | `a1529f66d0926beab62c974f4142e9d97a0d451a0a3c76225471747a81dedab7` |

Branch/HEAD remain as documented `agent/Watson-dev` /
`8f9fcf94e028b51d883a1a484c2c9466a3d6b547`; no Git command is permitted here,
so current HEAD/index state is not reverified. The code-boundary baseline and
API above are recorded before source edits.

## Current Scope: CUDA-BROWSER-08

Render the completed CUDA-reuse07 report in the Experiment Browser through one
platform-qualification adapter at registry -> payload -> generated HTML. The
entry is platform evidence only: never include it in economic matched
comparisons or terminal-gate promotion. Do not infer or invent return, Sharpe,
action fractions, PPO update rate, checkpoints, or a training budget.

- Dispatch: 2026-10-09 11:12 UTC; hard stop 11:35 UTC; work budget <=20 minutes.
- Branch / HEAD: `agent/Watson-dev` /
  `8f9fcf94e028b51d883a1a484c2c9466a3d6b547`; inherited dirty tree preserved.
- Allowed writes only: `docs/coordination/dev.md`,
  `itrader/scripts/experiments/render_experiment_browser.py`,
  `itrader/scripts/experiments/schemas/experiment_registry.schema.json`,
  `itrader/scripts/experiments/templates/experiment_browser.html`,
  `itrader/tests/test_experiment_browser.py`,
  `docs/research/experiment_registry.json`, and generated
  `docs/research/itrader_experiment_browser.html`.
- No root runtime/interpreter/dependency changes, model/training source,
  unrelated UI redesign, Ray, solver/model jobs, installs/downloads, WSL, or
  Git staging/commit/push. Existing dirty bytes in allowed files must be
  inspected and preserved, not replaced from HEAD.
- Lease `LEASE-DEV-CUDA-BROWSER-08`: only focused browser tests and rendering,
  sequentially <=240 seconds cumulative; native CPU `.venv`, one thread. Before
  each invocation verify source identities, no conflicting job, available RAM
  >=4 GiB and disk >=20 GiB; abort below 2 GiB RAM or 10 GiB disk. A failed
  start closes the execution allowance: no retry and no threshold adjustment.
- Pre-edit SHA256 / byte snapshots: renderer
  `cf5f1e279670310a98672cf8647ac6eb235031723009ab8c62814145617885e0` (34,670);
  schema `f1b45ec3297ae74df34776f930d20ab594e107d799054352ab116aaf869c18ee`
  (4,016); template
  `b5a400d5cee3ca42427ceeee86aa8cae5b9895f359868851b0371878cce40027` (28,033);
  existing untracked browser test
  `5140dd29e4a00a542c1223ee72e785d3f701ee5b59c7a73652526543f40aaf1e`
  (29,042); registry
  `fd064673f7801fb2c052522d3b9e83b018e8ccf63573548ce7f26c9d3aa17347` (8,193);
  generated browser
  `5cca3254cf7fdbe98930e6251321e59b0db267c8043e380a28c69256f77efa72`
  (954,298). Report bytes currently available only at ignored
  `docs/coordination/_local/cuda-qualification/benchmark-reuse07-result.json`,
  SHA256 `c25efdfd7d6f63911e56002bbc501816bf5427bd1fa0d99ec4a1c386aa088f12`;
  the tracked copy was absent at initial preflight. TRAINING subsequently
  promoted it byte-identically to `docs/artifacts/research/cuda-reuse07/` at
  11:18 UTC; DEV does not own or modify that artifact.

Predeclared behavior inputs, recorded before opening the implementation:

1. The exact report digest `c25efdfd7d6f63911e56002bbc501816bf5427bd1fa0d99ec4a1c386aa088f12`
   with CPU/CUDA timings for B=1 and B=128, inference and forward/backward:
   render the authoritative measurements in milliseconds with batch/window,
   model/runtime/device identity, parity/cuDNN evidence, and explicit claim
   limitations.
2. Missing report, malformed JSON, digest mismatch, wrong-typed schema, and
   absolute or traversal source paths: fail closed with an actionable error;
   never substitute a cached or hand-entered metric.
3. A platform-qualification record presented to economic comparison or
   terminal-promotion selection: exclude/reject it as non-economic evidence.
   Missing economic metrics remain absent/unknown, never zero.
4. Existing experiment registry and entries in either order: preserve their
   current output and compare behavior while adding a discoverable platform
   evidence surface.

Test-ROI review requirement (human follow-up, 2026-10-09): do not grow a
test-per-field ladder. The intended high-value coverage is one registry ->
digest-bound source -> adapter -> payload -> rendered-tab integration, plus one
parameterized fail-closed matrix for missing, malformed, digest-mismatched,
wrong-typed schema, and unsafe-path inputs. During independent review, inspect
the existing browser tests for overlapping public-contract coverage and
recommend any safe consolidation with concrete test names and invariants
preserved. Keep narrow tests where they protect separate trust boundaries (for
example duplicate JSON
keys, source tampering, path containment, or comparison identity); do not delete
them merely to reduce counts. No test was consolidated or removed in this
static-only lease. Static inventory of the current browser-test module found
coverage is predominantly through the public `build_browser_payload` boundary,
not private-helper unit tests. A likely low-risk consolidation candidate is the
separate blank-factor and blank-experiment-metadata schema tests into one
parameterized schema-contract matrix; preserve all input cases and expected
errors. Review should also check that the multi-mutation
`test_individual_run_source_cannot_attach_unrelated_population` remains
diagnosable, rather than merging unrelated failure modes into a single opaque
scenario. These are review candidates, not approved edits.

Execution handoff: GRILL withdrew the unstarted `LEASE-DEV-CUDA-BROWSER-08`
before the next test/render/review invocation to prioritize TRAINING's Ray09
smoke. The only browser test invocation already completed before withdrawal:
the targeted new integration tests returned 2 expected failures in 0.44 s
because the schema rejected the new `platform_qualifications` property. Its
preflight showed 4.444 GiB free RAM, 62.22 GiB free disk and no other Python,
Ray, pytest or browser-renderer process. No test, renderer or reviewer is now
active; no further execution may start until GRILL releases a new lease. Static
source/registry work only may continue inside this scope and deadline. The
registry now contains the digest-bound CUDA entry, but the generated browser is
intentionally not refreshed yet and therefore does not show it. The promoted
report hash was re-read and matches the registry entry; focused
`git diff --check` passes. The schema, path-boundary and rendered-value edits
made after the expected-red invocation remain unexecuted pending lease release.

## Preflight Hold: CUDA-BROWSER-CLOSE-12 / LEASE-DEV-CUDA-BROWSER-CLOSE-12

- Dispatch: 2026-10-09 11:42 UTC; hard stop 12:00 UTC; cumulative execution
  ceiling 240 seconds.
- Ray09 was closed before this scope: its durable receipt says the PowerShell
  wrapper failed before pytest, zero cells completed, no descendants survived,
  and no Ray output directories were created. The raw log shows the quotes
  stripped from `python -c` and a subsequent `File.Replace` finalization error;
  this is not a solver test result.
- Preflight at 11:42 UTC: free RAM 0.225 GiB (a subsequent read-only sample
  during process inventory reported 0.142 GiB); free C: disk 62.158 GiB. No
  pytest, Python, Ray, or browser renderer was launched. The 4 GiB start gate
  therefore failed before invocation; no tests/render/reviewer or wrapper-child
  execution has consumed the 240-second lease.
- GRILL/TRAINING later reported 0.232 GiB available RAM at 11:50 UTC while
  coordinating a separate owned-process cleanup. DEV did not stop or modify
  that process. This remains below the 4 GiB start floor; the browser-close
  lease is closed and cannot be retried or relaxed without a fresh lease.
- Keep all generated/failed Ray09 artifacts unchanged. No process was stopped,
  no retry was attempted, and no RAM-floor change is authorized. Resume only on
  a new GRILL lease with a passing preflight; do not imply that browser output
  or R1 lifecycle behavior has been qualified.

## Closed Draft Scope: CUDA-UV-DRAFT-12

Completed source-only preparation at 2026-10-09 11:50 UTC, before the
12:25 UTC hard stop. No uv resolution, lock generation, download, install,
environment sync, Python import, test, or runtime command was executed. The
CUDA browser execution lease remains held; this independent static draft did
not consume it or change its thresholds.

Only this board and ignored files under
`docs/coordination/_local/cuda-uv-bridge12/` were written. Draft files:

- `README.md` SHA256 `0382124D60ED713FDC1D34F341584F729B9F47D8C3AE1BB9986BC7AFA71733F6`
- `pyproject.uv-profile.fragment.toml` SHA256 `FBB1544429F6A718D9ABB55AC66A994FC3F18FEA9F17A097195E31AE4AD2B925`
- `invoke-cuda-tested.ps1` SHA256 `6AC460E61F8FAF0E74415A7D5E766C4AA2BEF9094CD0B2F61703BCBF82C0068C`

Official uv documentation supports mutually exclusive dependency groups and
`default-groups`; extras are not selected by bare `uv sync`. The draft therefore
puts CPU torch `2.3.1` in a default `cpu-legacy` group and the exact tested
Windows CPython 3.11 wheel in an explicit `cuda-tested` group. It preserves the
current Ray/Gymnasium/NumPy pins and leaves existing optional groups untouched.
This entails a packaging decision: moving torch out of `[project].dependencies`
means non-uv installers would no longer receive torch. GRILL must decide
whether this research repository is uv-managed for runtime provisioning or
whether installable package metadata must remain complete before applying the
fragment.

The exact official wheel's metadata endpoint advertised SHA256
`8ff2cbe2a88bebfc14b71fea17f4937d77f0ff70aa9e4cebd6836b5bbe82765d`. This is
new server metadata, not a locally computed wheel digest; the wheel payload
was not downloaded. A future authorized acquisition must hash the bytes and
verify the advertised value before locking it. This also refines the earlier
research note that no trusted checksum was recovered: no locally verified
digest exists yet.

The launcher draft is Windows-only, checks the selected interpreter is x64
CPython 3.11, rejects a conflicting `ITRADER_PYTHON_BIN`, delegates through the
existing repository Python resolver, and never installs/syncs. It does not
qualify CUDA with Ray 2.58/RLlib/PPO or establish economic performance.
Static follow-up from GRILL found its embedded `python -c` identity probe may
recreate the PowerShell quote-marshalling failure. That CUDA launcher remains
unexecuted draft-only; replace the inline probe with a script/helper or witness
the exact native argument contract before promoting it.

Test-ROI review is explicitly in scope before any product edit: prefer one
public launcher-to-runtime integration path and a parameterized fail-closed
matrix; consolidate only redundant unit coverage where each invariant and
failure remains diagnosable. Preserve independent trust-boundary tests and
provenance checks. The browser-test consolidation candidates from the held
scope remain recommendations only; no tests were deleted or consolidated.

Decision for GRILL-IMPLEMENT: review the packaging tradeoff and profile groups;
bind only a locally verified wheel digest; then issue a separate, finite
provisioning lease if approved. Do not apply this draft or start sync based on
this preparation alone.

## Closed on Failure: BROWSER-HARNESS-CHECK-14 / LEASE-DEV-BROWSER-HARNESS-14

The fresh scope hard stop is 2026-10-09 12:12 UTC, with <=360 seconds of
sequential native-Windows focused tests/render/review oracles. At 11:58:39 UTC,
preflight passed at 8.71 GiB available RAM, 62.15 GiB free C: disk, and zero
Python/pytest/Ray conflicts. The first process scan had matched its own shell
command text and aborted before launching anything; the corrected check excluded
PowerShell and confirmed no conflicting compute process.

TDD red (before adding the harness implementation):
`python -B docs/coordination/_local/r1-harness-repair14/test_harness.py` failed
as expected because the new `invoke_case.ps1` and `run_case.py` did not yet
exist. No child process or production test ran in that red step.

GRILL clarified that this intentional red->patch->green sequence and the
explicit fake-child exit-7 case are authorized oracle coverage, not Ray-run
retries. The unexpected post-implementation assertion failure below is still
the stop condition; no automatic re-invocation followed it. The initial process
preflight false-positive was a checker defect: its own PowerShell command line
contained the literal `r1-harness-repair14` search string. It launched nothing;
the corrected process check examined Python/pytest/Ray executable names and
found zero conflicts. Independent review must examine this preflight rule
before it is reused.

The implemented stdlib-only fake-child integration then ran once, exit 1 in
1.616 seconds (3.566 seconds total across the red and green harness invocations).
`test_existing_inventory_is_never_overwritten_or_launched` passed. The
public-path subcase for exit 0 passed. The exit-7 child also returned its exact
argv and terminal `completed` inventory with exit code 7, but the test then
failed because both subcases reused one temporary Ray-results directory: the
second subcase correctly saw the first subcase's `new child output` as part of
its baseline, contradicting the test's expected single old directory. This is
test-fixture contamination, not evidence of a browser, solver, or Ray defect.

No automatic retry was attempted. A read-only post-run process check found no
surviving harness Python child; its only string match was the check's own
PowerShell command. No browser tests, generated-browser render, or independent
review ran after the failure. No root/product file, source test, browser output,
or environment was edited by the harness scope; failed09 and previous scratch
remain unchanged. The four new ignored harness files and their identities are:

- `test_harness.py` SHA256 `1F853AAF7C86D64F344C527C056DBD8200B7547AB8A569AB3ABFCC32BC65046E`
- `run_case.py` SHA256 `2A78C937BF1E48C4BEE6CBE25368DF37EE31A72E1BD99CC04B065E8A5947F353`
- `fake_child.py` SHA256 `12928FBAFFC2F3416CA378C261A518F00C0772C55715217B3FC1A8887ABFE644`
- `invoke_case.ps1` SHA256 `092279CC56D40460A0CFCAC4427184C467D8FB3E738C73BB5F9F8394DB960F4C`

Post-close integration review found an additional handoff limitation: this
`run_case.py` is explicitly fake-only (`scope == harness-oracle` and its
docstring forbids Ray/training). Its assertions cannot qualify the same
transport/inventory seam for a later R1 command. Do not hand this runner to
TRAINING as production-ready or claim real-launch parity. A renewed scope must
generalize the structured-argv runner to carry an explicit manifest scope while
continuing to execute only stdlib fake children; TRAINING must later place the
real command behind the existing Windows Job containment and fresh
source/resource authorization. Keep actual pytest driver/selector arguments as
manifest data, with no inline `python -c`. The current lease is closed, so no
runner edit or test was attempted after this finding.

Test-ROI remains open: the held browser suite's blank comparison-factor and
blank experiment-metadata parameter matrices are candidates for consolidation,
but no test was changed or removed. The independent Standards/Spec reviewers
were not dispatched because the execution lease closed on the failing oracle.
Next step requires a new lease: first generalize the runner and isolate each
exit-code subcase in its own temporary results root; then test the shared
structured-argv/inventory seam with fake children, run the browser tests and
render, and only then perform the two independent reviews. No R1 lifecycle run
is authorized by this closure.

## Current Scope: BROWSER-HARNESS-FIX-16 / LEASE-DEV-BROWSER-HARNESS-FIX-16

Fresh corrective lease, hard stop 2026-10-09 12:16 UTC; cumulative focused
execution ceiling 300 seconds. The prior attempt-14 scratch and its failed
receipts were preserved byte-for-byte. New harness files are isolated in
ignored `docs/coordination/_local/r1-harness-repair16/`.

The generic runner now takes an explicit non-empty scope identity and a bounded
structured `argv` list from JSON, launches with `shell=False` and a bounded
timeout, records a bounded top-level results-directory inventory, writes an
exclusive initial receipt and atomically replaces it on completion. It has no
fake-only scope guard. Its Windows PowerShell entrypoint passes script-file and
manifest paths as native arguments; the fake child receives spaces, both quote
types, and a backslash exactly. Exit 0 and exit 7 both produce terminal
inventories; an existing inventory is not overwritten; pre-existing result
directories remain intact. This qualifies only the transport/inventory seam
with fake children. Any real R1 command still requires TRAINING's Windows Job
containment and separate source/resource authority.

TDD red: the stdlib harness failed before implementation because the new runner
and PowerShell entrypoint did not exist (0.521 s, expected). Green: both
integration tests passed (1.203 s; 1.327 s measured by the outer command).
No Ray, pytest driver, model, or market data was executed by this harness.

The first browser render correctly rejected a registry/report SHA mismatch; the
report stayed untouched. The registry digest was corrected to the actual
report SHA256
`c25efdfd7d6f63911e56002bbc501816bf5427bd1fa0d99ec4a1c386aa088f12` and the
renderer passed, generating six experiments. The first focused browser suite
had 60 passes and one failure because a test expected client-rendered timing
text in server HTML. The client JS owns `toFixed(4)` formatting; the test now
checks the embedded authoritative timing values and formatter contract instead
of expecting JavaScript output before execution. Focused rerun passed: 61 tests
in 1.39 s. Renderer timings: initial mismatch 2.858 s; corrected render 0.951 s;
focused browser runs 2.671 s failed then 4.384 s passed. Total measured
test/render work remains far below the 300-second cap.

Final pre-review source identities:

- Renderer `render_experiment_browser.py`: `AA0FCD2D18789BB495BF2AB9AAF31526DE213482AABF57BECF2D6524CCB4A6DF`
- Registry schema: `C0022B9948009910A31D27E1C2AFFBFC1E9EBA008F25AC5DAEA589E7A160E829`
- Browser template: `EF6CA24FEC129EE11E445EC7F1D31B2BC99F61A255F491703AB8CE60962A0B9D`
- Browser tests: `A1529F66D0926BEAB62C974F4142E9D97A0D451A0A3C76225471747A81DEDAB7`
- Trace adapter: `8668940BDB632B1F2C0329B08239E2EF5FCE56C4ABD1D211315B58E52D24AAED`
- Trace tests: `CFAD47A08841D8B96E74FB36A03241D7E332429A2A09DCBCC28F2D6E3F9105DB`
- Experiment registry: `6BD4C0BEC56B6FEEEF7BBD66C13255E8CA23F21F4C1EFDB0DEC4EFC8F45552B0`
- Authoritative CUDA report: `C25EFDFD7D6F63911E56002BBC501816BF5427BD1FA0D99EC4A1C386AA088F12`
- Generated browser: `6CDBE2B5E966F499FA8250BF8A4D9976265624FC695825E37087B60E0B0C3962`

The exact fake-runner/test/PowerShell hashes are in the closure receipt below.

### Closure receipt: DEV16-RECEIPT-CLOSE

- Canonical execution lease: `LEASE-DEV-BROWSER-HARNESS16`; the earlier
  `LEASE-DEV-BROWSER-HARNESS-FIX-16` spelling in this board was an alias for
  this same lease, not a second lease. Execution hard stop was 2026-10-09
  12:16 UTC. Exact final report receipt time was not recorded; the last
  reviewer execution status arrived before the hard stop, but exact deadline
  compliance cannot be certified to the second.
- Reviewers: `/root/check16_standards` (requested `gpt-6.1-sol/max`) and
  `/root/check16_spec` (requested `gpt-6-luna/xhigh`). Both are independent
  reviewer IDs for this DEV scope. Exact serving identities were not observable.
  Their raw reports are in the DEV chat; no standalone review artifact was
  created. See [REVIEW.md](../../REVIEW.md) and the
  [TRAINING/TESTING handoff](training-testing.md). Neither approval authorizes
  Ray/PPO execution or R1 lifecycle acceptance.
- Both reviewers predeclared adversarial inputs before source inspection.
  Standards exercised 91 current/base registry cases across empty/single/multiple
  and reversed records, blanks, wrong/missing types, combined invalid fields,
  dates, Unicode and HTML labels; it also ran focused browser/trace tests,
  harness tests, renderer CLI/render and diff checks. Spec independently ran
  the focused browser/trace suite, scratch harness and renderer; its predeclared
  plan covered invalid identities/metadata, record count/order, malformed or
  unknown inputs, HTML rendering and trace edge cases. A simultaneous-blank
  probe failed in reviewer PowerShell quoting before reaching product code.
- Standards verdict: `APPROVE`, zero blockers and one optional nit at
  `itrader/tests/test_experiment_browser.py:549` and `:560`. Parameterize the
  nine blank-value/field cases as `(field_path, value, expected_error_path)`;
  assert the precise error path. Spec verdict: `APPROVE`, zero findings. Both
  recommend parameterizing input variants while retaining distinct trace/report
  unit contracts and end-to-end source-to-render and PowerShell-to-child
  provenance boundaries. All five REVIEW.md principles were reported OK.
- Base comparison limitation: HEAD equals the requested base commit, and the
  experiment-field schema diff showed no change. However, Standards' attempted
  schema overlay did not replace the file actually opened by the renderer, so
  the 91-case comparison is not claimed as an isolated base execution. No full
  base checkout/suite was run.
- Cumulative measured focused test/render runtime was approximately 31.4 s:
  this includes the expected TDD-red harness run, initial renderer digest
  rejection, first browser-test run (60 passed/1 failed on a client-rendered
  text expectation), corrected green runs, both reviewer runs, and the Standards
  matrix. Failures and corrections above remain preserved; no artifacts were
  deleted. Preflights reported no Python/pytest/Ray conflicts, and the Standards
  reviewer explicitly observed no command running at 12:13:08 UTC. No Ray,
  training, model, or market-data job was launched; a final OS process scan
  after the last Spec command was not captured.
- Source-bound identities: runner `73AD9D9EC810F73391F7E131BF0C2AA01C24B6496F5E3AF52477988241005F74`;
  fake child `CE1041B9074B1FE766EC7D9DD748D9BA23203269178EFD165E612BA097986DC1`;
  PowerShell entrypoint `092279CC56D40460A0CFCAC4427184C467D8FB3E738C73BB5F9F8394DB960F4C`;
  harness tests `3035A68751EB5FD35D2F839BC861EDF5F5CBF329128E481B7B081190F9504A7A`.
  Renderer `AA0FCD2D18789BB495BF2AB9AAF31526DE213482AABF57BECF2D6524CCB4A6DF`;
  schema `C0022B9948009910A31D27E1C2AFFBFC1E9EBA008F25AC5DAEA589E7A160E829`;
  template `EF6CA24FEC129EE11E445EC7F1D31B2BC99F61A255F491703AB8CE60962A0B9D`;
  browser tests `A1529F66D0926BEAB62C974F4142E9D97A0D451A0A3C76225471747A81DEDAB7`;
  trace adapter `8668940BDB632B1F2C0329B08239E2EF5FCE56C4ABD1D211315B58E52D24AAED`;
  trace tests `CFAD47A08841D8B96E74FB36A03241D7E332429A2A09DCBCC28F2D6E3F9105DB`;
  registry `6BD4C0BEC56B6FEEEF7BBD66C13255E8CA23F21F4C1EFDB0DEC4EFC8F45552B0`;
  report `C25EFDFD7D6F63911E56002BBC501816BF5427BD1FA0D99EC4A1C386AA088F12`;
  generated browser `6CDBE2B5E966F499FA8250BF8A4D9976265624FC695825E37087B60E0B0C3962`.
  The browser digest identifies generated working-tree output; it is not described
  as committed. These hashes were stable across reviewer preflights.

## Closed Preparation Scope: R1-LIFECYCLE-PREP-02

Prepare the existing synthetic PPO/checkpoint/replay integration for exactly
four production-loadout cells: R0 and R1, each with solver state `off` and
`both` (R4-v1). Keep the current tagged 600-row synthetic Parquet, fixed
12-sampled-step / one-SGD-epoch recipe, actual callbacks, strict model identity,
episode/reset evidence, checkpoint serialization, deterministic restored
logits/value/action, and replay/provenance comparisons. Reuse one parameterized
public integration; do not create near-duplicate test files or a test per field.

Write scope: `itrader/tests/test_representation_checkpoint_integration.py` and
this DEV board only. A readiness-doc change is deferred until execution
evidence exists. Do not edit models, training/runtime contracts, other parent
boards, data, checkpoints, or outputs. No tests, Ray/PPO/model jobs, full suite,
pre-commit, panel opening, collection, training, economic evaluation, stage,
commit, push, WSL, or CUDA/PBT. Existing dirty work outside this scope is
preserved.

### Frozen preparation snapshot

- App dispatch: 2026-10-09 09:37 UTC. The prior 09:20 UTC entry was an
  incorrect start-time estimate and is superseded. The 25-minute preparation
  budget therefore ends at 10:02 UTC; this is earlier than the original 10:05
  hard stop and governs this scope.
- Base/HEAD: `8f9fcf94e028b51d883a1a484c2c9466a3d6b547`; branch
  `agent/Watson-dev`; staged diff empty; workspace inherited dirty paths remain
  out of scope.
- Integration test before this scope:
  `itrader/tests/test_representation_checkpoint_integration.py`
  `a936b99e4adbc8ac13ace4ac907499059df86451a374fee594d7403d6e135b9a`.
- Frozen production dependencies: `registry.py`
  `4bfc0e84088f85acc47d20762e1766a019c2ef45b35f0843a88715438129ee3a`;
  `solver_window_lstm_film.py`
  `0f808c1db2908dc8a23a289ab23cd7353f53ee6f676c93ef4aa3daa99546b5e2`;
  `solver_mlp_film.py`
  `8473b004c3197ffccbddac8f587f3ba7c54b71dfeb32ae85cd20aee48903d145`;
  `train_solo_V2.py`
  `58a26a1308b38bff0e3dd503be7e622d54e59895db9d37a5a7bdb5160ad8beb0`.
- Additional declared hashes: `ppo_diagnostics.py`
  `b96d433ada705e89dfecd167a1d3791e91ac1c17441c66c3e15c0ca1e27296ac`;
  `policy_baselines.py` had malformed original receipt value
  `9a13e18a049d59edcfeecd0d3491659c8ae1ded5cdb89b340bbfe78150765ca52` (65
  characters; not a valid SHA256; see reconciliation below);
  `replay_evidence.py`
  `674d96a735331423de38e93d2e4839808b8d587453d86bc36d406d22fdc1b5f7`;
  `itrader_env.py`
  `cb262c630cbdefbf19b9a797bcc90e392234ae6cb771004b18dd4902c355a268`;
  `REVIEW.md` `b92c8e19dbf9f5ca5dbc5ec632afe4be4084d9d1acfe313e1021f5d1321ad167`.
  GitNexus is stale (indexed commit `1d5ba82`); advisory only; no index
  refresh in this bounded prep.

### Design and gate

- Four explicit IDs: `r0-state-off`, `r0-r4-state`, `r1-state-off`, and
  `r1-r4-state`. All use the same fixed synthetic market/task identities, seeds,
  horizon, cost, sampler/runner budget, and report path; only solver selector and
  the named state cross vary.
- All cells must be constructed by `build_base_config`, then use the existing
  actual `DiagnosticPPO` callbacks and save/restore/replay path. Keep expected
  model class/identity checks before learning and after restoration. The `both`
  state case must preserve reset evidence and the declared R4-v1 config.
- Required post-save equality remains exact: restored greedy logits, value,
  chosen action/probabilities, episode payload and canonical digest match the
  pre-save replay. The original callback rows and persisted metric parity remain
  checked; these are report/lifecycle evidence, not economic promotion.
- Static-only review can approve matrix construction and provenance wiring, but
  cannot approve serialized restore, callback delivery, or deterministic replay.
  The later execution gate requires a separate serialized resource lease; no
  test is executed in this preparation scope.
- Target hash after the four-cell parameterization and lifecycle-cleanup guard:
  `test_representation_checkpoint_integration.py`
  `db6cdb37c2c3f3486e0effcd88be600cce2e018870469673904c8d662f1b834c`.
  The other frozen dependency hashes above are unchanged. The original target
  bytes are represented by their pre-edit SHA only; because the test was already
  untracked, a byte-for-byte base diff is unavailable.
- Independent static Standards and Spec reviews were dispatched against this
  narrow file and frozen dependencies. Reviewers must predeclare probes, report
  every REVIEW.md principle, and separate design review from execution-required
  checkpoint/replay acceptance. An assertion originally outside the try/finally
  was moved inside it so any selected-model mismatch still stops the algorithm.
  A specific test-ROI review is requested: retain tests for distinct local
  invariants, and identify only genuinely duplicated end-to-end behavior for
  consolidation into the parameterized lifecycle path. No reviewer may run
  Ray/PPO/model tests here.
- `git diff --check` is permitted as a whitespace-only check; no test command,
  model construction, Ray initialization, healthcheck, or pre-commit is allowed.
- Final static Spec review: APPROVE for test preparation only, with no
  remaining findings on the final target hash. It explicitly does not qualify
  two-process producer/checkpoint-digest validation, changed weights, economics,
  or executed serialization/replay. All five REVIEW.md design principles were
  reported OK. Independent Standards review also APPROVES with no blocker or
  nit on the same hash and reports all five principles OK. The test file is
  untracked, so its recorded pre-edit digest does not provide a byte-level
  before/after diff.
- Test-ROI recommendation: after this lifecycle matrix executes successfully,
  consider removing only the duplicate positive `validate_restored_model` smoke
  assertion from `test_r1_loadout_to_environment_to_model_is_stateless_and_r4_neutral`;
  keep its wrong-class, policy-identity and model-config rejection cases. The
  two successful report-extraction cases may be parameterized as one compact
  reporting-contract matrix, while preserving wrapper, unknown-vs-zero, reward
  fallback and dynamic-header cases. Do not push distinct FiLM math/gradient/
  R4-neutrality, chronology/statelessness, PPO numerical/mask/nonfinite,
  optimizer-count or donor-restore/resumed-learning oracles into a slower
  integration: they localize different failure causes. No tests were deleted or
  moved in this prep scope.
- Static validation only: `git diff --check` exited 0, and the untracked target
  had no trailing whitespace. No pytest, healthcheck, pre-commit, Ray/PPO/model
  initialization or training was run.
- Receipt reconciliation (2026-10-09 10:00–10:03 UTC): the earlier board and both
  static-review summaries recorded `policy_baselines.py` as
  `9a13e18a049d59edcfeecd0d3491659c8ae1ded5cdb89b340bbfe78150765ca52` (65
  characters). A direct read-only `Get-FileHash -Algorithm SHA256` over the
  current file returned the valid 64-character digest
  `9a13e18a049d59edcfecd0d3491659c8ae1ded5cdb89b340bbfe78150765ca52`.
  A fresh independent read-only check returned the same current working-byte
  digest. The file is modified in the working tree; its HEAD and index blob IDs
  are both `8b52ad1f6ec8c2b80d4adffb0bb77ee05bf40dd0`, so neither binds the
  changed working bytes. The persisted preparation receipt has no valid original
  command capture or immutable snapshot tying the earlier review to this digest;
  both review summaries repeat the malformed value. Therefore transcription-only
  provenance and the exact earlier review input are unproven. The fresh check
  confirms current identity and API use only; it cannot retroactively certify the
  earlier review snapshot. The static reviews still approve test design, but the
  frozen runner-source gate remains open until the source is re-frozen/reviewed
  under a valid receipt. This is not proof of behavioral drift. No runtime or
  execution verdict is implied. The other eleven runner/lifecycle/research
  identities in the gate audit matched.

### Fresh source binding: R1-REVIEW-REBIND-05

- Started 2026-10-09 10:25 UTC; 15-minute budget ends 10:40 UTC; absolute stop
  10:45 UTC. Write scope is this board and ignored local receipts only.
- Machine-generated byte snapshots and manifest:
  `docs/coordination/_local/r1-lifecycle-review05/manifest.json`;
  manifest SHA256 `1dfa19783056790d3cc9e1f27d3dd912c1e7ab795f1a79b373893715cc05b0a4`.
  It binds 30 source/context files totaling 375,691 bytes to branch
  `agent/Watson-dev`, HEAD `8f9fcf94e028b51d883a1a484c2c9466a3d6b547`.
  For each entry, source-before, copied-snapshot and source-after SHA256 values
  matched during capture.
- Fresh valid identities: integration test
  `db6cdb37c2c3f3486e0effcd88be600cce2e018870469673904c8d662f1b834c`;
  current `policy_baselines.py`
  `9a13e18a049d59edcfecd0d3491659c8ae1ded5cdb89b340bbfe78150765ca52`.
  The latter is the current working-byte hash only; the earlier static review
  cannot be retroactively rebound to it.
- Two fresh independent static reviews were conducted against these exact
  snapshots: Standards and Spec. Their inputs cite the manifest hash, target
  hash, `policy_baselines.py` hash, and requested versus observable model/effort.
  No tests, imports, Ray/model jobs, product edits, or execution claims were
  authorized.
- Both fresh reviews completed against the manifest-bound snapshots with
  `VERDICT: APPROVE`, zero BLOCKING findings and zero NIT findings. Each
  independently verified all 30 snapshot hashes/lengths and the two required
  source identities. Parent re-verification also found zero mismatches.
  Standards reported all five `REVIEW.md` principles OK. Spec confirmed the
  four named lifecycle cells and production restore/replay evidence path, while
  limiting the claim: static lifecycle parity does not prove observation
  arithmetic, changed weights, checkpoint-byte provenance, or independent
  process replication. Both reviewers could not observe their serving model
  or reasoning effort, so requested model diversity is not independently
  attested.
- The earlier static reviews approved the test design, but their exact source
  identity remains unproven. This new receipt closes the static source-binding
  gap only for these manifested bytes; it does not retroactively certify the
  malformed earlier receipt. Runtime qualification remains a separate TRAINING
  gate.

### Required runtime qualification (not authorized in this scope)

- The frozen synthetic lifecycle test is not yet runtime-qualified. A later
  conditional `LEASE-TRAIN-R1-LIFECYCLE-05` belongs to TRAINING after CUDA closes;
  DEV must not start a Ray/test lease. Its required sequence is the R4 pair
  first, then the state-off pair, using the existing root CPU synthetic recipe.
- All four cells remain required: R0/R1 × state `off`/`both`. No failed cell
  may be dropped, and the test's parameter order/semantics stay unchanged; the
  runner may select node IDs to execute the agreed sequence.
- When the runtime comparison is later authorized, the primary fixed-1-bp
  terminal-net comparison is R1+R4 versus R0+R4. This does not reduce the four
  mechanical cells, authorize an interim market-panel opening, or close the
  still-open Q26 method/calibration decision.
- Static review is not execution acceptance. The later receipt must separately
  record each cell's command/node ID, environment and source identity, exit
  status, callback/checkpoint/replay evidence, and any failure; economic claims
  require their own declared gate. The current test does not assert changed
  weights or qualify two-process producer identity/economic performance.

## Closed Scope: R1-IMPLEMENT-01 — historical implementation plan

The following is retained as closed-scope context only; it is not current
authorization or an active instruction. R1 implementation completed under
`R1-IMPLEMENT-01`; current work is the bounded lifecycle preparation above.

Historical objective: implement the accepted stateless-per-decision R1 window
encoder and explicit model-selection/restore identity using TDD. The scope read
[readiness](../project-state/r1-development-readiness.md),
[R-series research](../research/recurrent-solver-ablation.md), nearby source/tests,
and [coordination](README.md), preserving corrected Ray-2.58 compatibility, R0
defaults and state-dict identity; no recurrent state across decisions or PPO
recipe change.

Historical write scope: R1 model/registry, selection/loadout/training/restore
integration, focused tests, and active R1 availability/next-action docs. It
excluded other parent boards, `AGENTS.md`, coordination index/grill/review docs,
the pinned skill port, native Ray migration, and new task families.

- Started: 2026-10-09 08:50 UTC. Implementation stop: 09:25 UTC; GRILL focused
  test lease expires 09:30 UTC. Maximum implementation window: 35 minutes;
  focused test runtime is cumulatively capped at 10 minutes by the lease.
- Base and HEAD: `3a3583d109af7a4c094124e73ad84008f371d8e3`; branch
  `agent/Watson-dev`; staged diff empty. Shared worktree is dirty: 35 tracked
  paths modified and 54 untracked path entries at dispatch; all pre-existing
  changes are preserved and remain outside this scope unless explicitly named.
- Scope intent: implement only the accepted stateless-per-decision R1 window
  LSTM/FiLM model, structural selection, train/loadout construction, and current
  R1 implementation-status docs. No native Ray migration, PPO recipe change,
  Q23/Q26 work, real data, training, replay, staging, commit, or push.
- Write set: `itrader/models/solver_window_lstm_film.py` (new),
  `itrader/models/task_film.py` (new), `itrader/models/solver_mlp_film.py`,
  `itrader/models/registry.py`, `itrader/rl_lib/train/train_solo_V2.py`,
  `itrader/tests/test_solver_selection.py`,
  `itrader/tests/test_semantic_observation_contract.py`,
  `docs/architecture.md`, `docs/architecture/itrader_system_architecture.md`,
  `docs/project-state/r1-development-readiness.md`,
  `docs/plans/itrader_masterplan.md`,
  `docs/research/itrader_research_dashboard.md`,
  `docs/project-state/current-state.md`, `docs/project-state/next-actions.md`,
  and `docs/project-state/handoff.md`, plus minimal manually synchronized HTML
  companions because no deterministic scoped renderer is available.
- Read dependencies: `AGENTS.md`, `REVIEW.md`,
  `docs/coordination/review-workflow.md`, `docs/coordination/grill.md`,
  `docs/coordination/README.md`, `docs/project-state/decisions.md`,
  `docs/plans/itrader_masterplan.md`, `itrader/models/observation_contract.py`,
  `itrader/rl_lib/evaluation/policy_baselines.py`,
  `itrader/rl_lib/evaluation/run_policy_baselines.py`,
  `itrader/rl_lib/evaluation/screen_completion.py`, and related tests.
- Excluded: every other pre-existing dirty/untracked path, parent coordination
  boards except this DEV board, `AGENTS.md`, review rules, Q23/Q26 artifacts,
  evidence/data/run outputs, and all historical checkpoints.
- Baseline SHA256 (current working bytes before this scope):
  `solver_mlp_film.py` `96bf09a5e51c5661e89d417bde6a60ec910f4c3aa940278352703ab3f260bd38`;
  `registry.py` `64a16d191cd9d6f6de9597729bba6a21b3c3baca91bab0563ec11eabd4ca24af`;
  `observation_contract.py` `e3deb74a2b1c82513383797e871d10784a61b398b5608af0649963d193a4d741`;
  `train_solo_V2.py` `8428d4b8737cae2fe1839c6776be0a5697e010ab5a8ec11705330fa88afda060`;
  `test_solver_selection.py` `a51bd3850f6c517dd03c0c33e1684c7f9f3e06ab17ea326b2f1994df525775a2`;
  `test_semantic_observation_contract.py` `b0b7225b24df4934a34a2616ecbb442a5c6a2261d8139ec7815c06d779d740d1`;
  `REVIEW.md` `b92c8e19dbf9f5ca5dbc5ec632afe4be4084d9d1acfe313e1021f5d1321ad167`.
- Test strategy: compact public-contract coverage, not a growing matrix of
  implementation-shaped tests. First preserve the existing fail-closed selector
  and R0 tests. Add one tagged synthetic-Parquet path from semantic observation
  through selected R1 model to logits/value, batch-order and prior-call
  invariance, and zero-state R1/R1+R4 initial parity; add one bounded adversarial
  config/restore-identity check for unknown/conflicting selectors and malformed
  schema. Keep focused architecture math/gradient assertions only where that
  public integration cannot establish the accepted Q15/Q18 contract. No Ray
  algorithm/checkpoint job in this lease; a synthetic lifecycle smoke requires
  a later review and separate lease.
- Allowed checks under `LEASE-DEV-R1-FOCUSED-01`: named synthetic R1/R0 contract
  and public-interface tests only, with each source hash captured before the
  command and no concurrent writes; cumulative test runtime <=10 minutes; stop
  on expiry, source drift, contention, or resource breach. No full suite,
  pre-commit, Ray jobs, server, real data/outcomes, or performance claims.
- Commands to run after edits are frozen: `./itrader/tools/run_repo_python.sh
  -m pytest -q itrader/tests/test_solver_selection.py
  itrader/tests/test_semantic_observation_contract.py`; then `git diff --check`.
  Record exact source hashes, command, exit code, wall time, and test count below.

## Run Record / Review Scope

- Pre-scope coordination advance at 2026-10-09 09:01 UTC set HEAD to
  `8f9fcf94e028b51d883a1a484c2c9466a3d6b547`, a GRILL board-system commit.
  It predates the 09:37 lifecycle-preparation dispatch and is outside that
  implementation write set. The shared tree still contains unrelated
  pre-existing dirty/untracked changes; they remain untouched.
- TDD red: the focused two-test command failed as expected before implementation
  (resolver not implemented / FiLM helper absent), 2 failed, 20 deselected,
  14.60 s. An intermediate check caught a test-import `NameError`, fixed before
  the final green run.
- Focused green: `bash ./itrader/tools/run_repo_python.sh -m pytest -q
  itrader/tests/test_solver_selection.py
  itrader/tests/test_semantic_observation_contract.py
  itrader/tests/test_no_nan_forward.py` — exit 0, 31 passed, 3.61 s.
  Source hashes immediately after that run matched at review preparation:
  `solver_window_lstm_film.py`
  `0f808c1db2908dc8a23a289ab23cd7353f53ee6f676c93ef4aa3daa99546b5e2`;
  `task_film.py` `391c5f902144f90d5fa35f29e77bb84a1d7e1d86635f73814faeba3776c4791f`;
  `registry.py` `4bfc0e84088f85acc47d20762e1766a019c2ef45b35f0843a88715438129ee3a`;
  `solver_mlp_film.py` `8473b004c3197ffccbddac8f587f3ba7c54b71dfeb32ae85cd20aee48903d145`;
  `train_solo_V2.py` `58a26a1308b38bff0e3dd503be7e622d54e59895db9d37a5a7bdb5160ad8beb0`;
  `test_solver_selection.py` `554991f9270e03bb1198e6a2e8c95f3ff6f80a909d70aa68fc09cbbceb4cd7c4`;
  `test_semantic_observation_contract.py` `7027e85031998370ec1adb583b0bc6a0784b0645000e5c6c03f670dbfa9842ea`;
  `test_no_nan_forward.py` `ee4825ef94ab5c5ae54ba4b3a4f4b9c48bdd9eaf431e9f8c5000e00c1437d27a`.
  No full suite,
  healthcheck, pre-commit, Ray lifecycle, checkpoint serialization or training
  was run under this lease.
- Independent review found one blocker at restore: checking only Algorithm
  config and class missed drift in Policy config and the instantiated model's
  config. The registry now checks each boundary and fails closed when the
  restored config is unavailable. The tagged integration now consumes the
  actual `build_base_config` environment config, with explicit synthetic task
  and start overrides. Two adversarial cases independently reject policy and
  model configuration drift.
- Test-ROI decision: keep 31 focused tests; this fix adds two adversarial
  assertions inside an existing integration rather than creating another test
  function/file. Independent reviewers recommend consolidating repeated
  public-path chronology/width/stateless checks only when diagnostics remain
  clear; retain distinct FiLM arithmetic, gradients, R4 neutrality, legacy
  compatibility, malformed-contract and restore invariants. No quota-driven
  deletions; not every field needs its own test.
- Independent Astra/Spec and Sol/Standards reviews approved the final scoped
  diff after re-review of the fix and final test delta. The focused three-file
  suite passed again after the final delta: 31 passed in 3.52 s. No Ray
  lifecycle, serialized checkpoint, full suite, healthcheck, training or market
  outcome was run. All other dirty/untracked paths remain excluded from review,
  not silently treated as authored here.

## Queue

1. `terminal_screen_only` Q23 role adapter and fail-closed consumer tests, after
   the current R1 scope and its prerequisite decision review.
2. Independent R1 gate reviews and exact snapshot receipts.

## Pending Evidence / Handoff

Q27 repair evidence is in
[the mechanical repair receipt](../artifacts/research/evaluation-mechanical-repairs-2026-10-07.md).
Prior green tests do not qualify new R1 bytes. Record new commands/results,
requested/observable reviewer identity, changed paths/hashes, and remaining gates.
Implementation may proceed; real screen execution is still closed on Q23/Q26.

## Completion Record: SALUKI-ADVISORY-ASSESSMENT21 — source audit handoff

Static audit completed 2026-10-10 under the bounded DEV21 dispatch. The durable
claim ledger, current source/test anchors, source hashes, architecture checks,
and test-ROI assessment are in
[`saluki-source-audit-2026-10-10.md`](../artifacts/research/saluki-source-audit-2026-10-10.md).
No implementation or test files were changed and no tests, runtime, Ray,
training, or market data/outcomes were accessed. Imported Saluki claims remain
advisory only. No current source defect was established; inaccurate cost,
gamma, transform-shape, and action-contract statements are flagged in the
claim ledger. The v2 concatenation finding distinguishes the literal dimension
error from the hypothetical 61×5 row-shaped repair. The first train-source hash
entry was 62 characters and has been re-read and corrected; see the top-level
receipt for the transcription correction and final audit digest. The accounting,
causal-observation, and model-identity public
integration oracles are proposed only; they were not executed.
The source review also records a prospective same-width action-ID/checkpoint
identity gap: current R1 width validation catches output-count changes, but its
model identity does not version the action vocabulary. The current mapping is
unchanged; add a restore oracle only before changing action meanings.

Test-ROI handoff: preserve the compact nine-case transition arithmetic matrix
as an independent oracle; review any overlap between the env/evidence and
economic-diagnostics integration paths before merging. A registry metadata test
consolidation is a low-risk future candidate, but do not delete tests by count.
Favor one literal-oracle public lifecycle scenario plus small rejection
matrices; retain distinct chronology, restore identity, schema, and action
arithmetic failure boundaries. No numerical test quota or consolidation is
approved by this static audit.

GRILL retains synthesis/decision ownership. No peer dispatch, Q23/Q26 approval,
training authorization, architecture adoption, test edit, Git operation, or
economic promotion follows from this handoff. Audit source identities are
working-tree hashes; Git state was intentionally not queried under the
read-only dispatch. The review pool is bounded, not exhaustive.

## CUDA27 final output-contract review (2026-10-11 01:32 UTC)

The final driver/test review pair approved the exact hashes recorded in the
ignored candidate receipt. A process-boundary test reproduced universal-newline
normalization of a lone CR; the driver now captures bytes and decodes after
completion. The original parser table remains parameterized; its nonzero-exit,
old-version, and near-version rows now use otherwise-valid banners so each
rejection oracle is independently meaningful. The focused file passes 69/69.

Test-ROI review recommends retaining the 24-case parser matrix plus the
three-case process-boundary matrix: they protect distinct contracts, so
combining them would reduce clarity without reducing meaningful test surface.
No unrelated unit tests were removed or consolidated. The source/test files
are untracked and absent at HEAD, so no executable base comparison is available.
Other candidate hashes match the earlier reviewed receipt exactly. This closes
source review only. No acquisition, lock generation, sync, install, or GPU
qualification has occurred.

## Final-source native preflight (2026-10-11 01:33 UTC)

After final pair review and exact eight-artifact receipt validation, the native
preflight passed with the expected Windows AMD64/CPython 3.11/x64 identity and
exact observed uv build output. It reported 4.51 GiB available RAM, 58.44 GiB
free on C:, and no existing target profile. A first attempt through the repo
`.venv` was refused by the interpreter identity guard before preflight work;
the approved external native interpreter then passed. Acquisition/lock had not
started at preflight time; its later outcome is recorded below. Recheck resource
headroom and the remaining global window before any new, separately authorized
attempt; keep the five-minute cleanup reserve.

## One-shot acquisition/lock outcome (2026-10-11 01:36 UTC)

The sole authorized job started at `2026-10-11T01:34:26.295177+00:00`. Wheel
acquisition completed once in 83.234 seconds: 3,062,428,787 bytes, expected
SHA256 `8ff2cbe2a88bebfc14b71fea17f4937d77f0ff70aa9e4cebd6836b5bbe82765d`,
final host `download.pytorch.org`. Preserve this file and the stage logs; no
second acquisition is allowed under this spent attempt.

`uv-lock.log` shows uv 0.11.7 failed while parsing the candidate `cuda-tested`
source because URL fragments are not permitted in that direct source entry.
State records `acquire=complete`, `lock=started`; no lock was accepted, and
`sync`/`smoke` stayed `not_started`. The driver exited 1 naturally; no owned
process remains. Post-run headroom was 4.55 GiB RAM and 55.59 GiB free C:, and
the target profile was still absent. Root `pyproject.toml` and `uv.lock` were
not changed. This is a candidate configuration/lock-generation blocker, not a
GPU/runtime result. Do not retry or edit the reviewed candidate without a new
bounded scope, corrected URL/hash representation, fresh source review, and a
newly authorized one-shot job.
