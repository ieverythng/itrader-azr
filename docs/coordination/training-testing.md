# TRAINING/TESTING Board

- Owner chat: `01a11fd7-d4c5-70d3-a1f3-80cd78ff4edd` (TRAINING/TESTING).
- Seeded by GRILL: 2026-10-09 08:42 UTC; parent becomes sole writer at dispatch.
- Status: `cuda_audit_active`; RUN-PREFLIGHT-01 closed, direct human follow-up active.
- Started: 2026-10-09 08:46:48 UTC (10:46:48 Europe/Madrid).
- Deadline: 2026-10-09 08:56:48 UTC; maximum 10 minutes.
- Branch / HEAD: `agent/Watson-dev` / `3a3583d109af7a4c094124e73ad84008f371d8e3`.
- Initial snapshot: dirty/unsealed; tracked source/tests/docs edits and untracked
  evaluation/registry/tests/coordination files already present. `git status --short`
  exits 0 with permission warnings for `.pytest-tmp-checkpoint/` and
  `.pytest-tmp-full/`; inaccessible contents remain unknown. Preserve all work.

## Current Scope: CUDA-PREFLIGHT-02

Direct human scope change at 2026-10-09 approximately 08:54 UTC: thoroughly
check CUDA availability/history and the working CUDA setup used by other
projects, then explicitly report findings/options to GRILL-IMPLEMENT. This is
authorization for that final message, superseding the routine board-only rule
for this requested report. Budget <=20 minutes, native Windows only. Board is
the sole write set; inspect environments/runtime metadata and official sources.
Permit bounded CUDA availability and tiny deterministic tensor checks, without
models, training, Ray, data, package installs, driver changes, or repo pin edits.
Stop on resource contention, <2 GiB available RAM, timeout, or unexpected service
startup. No ownership of DEV's lease; root experiment/runtime migration still
needs a separate decision. Use diagnosing-bugs for a differential probe.

## Closed Scope: RUN-PREFLIGHT-01

Read-only native Windows environment/resource and experiment-prerequisite audit.
Confirm interpreter, dependency and available CPU/GPU identities without starting
Ray, a server, collection, or training. Inspect accepted four cells, seeds,
cost stages, fixed recipes, and review/data gates; do not inspect new outcomes.
Report exactly what a later authorized run still needs.

Read dependencies: AGENTS, documentation index/architecture/masterplan,
GRILL board, R1 readiness, nearby selectors/loadouts/tests and native interpreter.
Stop on the deadline, ownership collision, unauthorized outcome access, or a
probe that starts a heavy service/job. Inspect process metadata only; do not
terminate or assume ownership of another chat's process. Final source/config
hashes and blockers will be recorded below; no qualified source snapshot exists.

Human refinement relayed by GRILL during this scope: future validation should
prefer named public-contract integration/E2E over expanding private-method/unit
counts. Preserve deterministic invariant coverage and existing tests; no test
deletion or token-saving claim. `LEASE-DEV-R1-FOCUSED-01` belongs to DEV only:
10-minute cumulative cap, expires 2026-10-09 09:30 UTC, no full suite/Ray jobs.
This parent has no lease and must not overlap that job with heavy validation.

## RUN-PREFLIGHT-01 Receipt

Observed 2026-10-09 08:46-08:51 UTC; all probes native Windows or Git Bash
(explicit `C:/Program Files/Git/bin/bash.exe`), never system Bash/WSL.

| Probe | Result / limit |
|---|---|
| Native repo launcher | `python itrader/tools/run_repo_python.py -B -c <stdlib/import probe>` exits 0; resolves `.venv/Scripts/python.exe`, Python 3.11.13, `os.name=nt` |
| Installed/imported dependencies | Ray 2.58.0, torch 2.3.1+cpu, Gymnasium 1.2.2, NumPy 1.26.4; all import successfully; `ray.is_initialized()` is false |
| Other installed metadata | PyArrow 17.0.0, DuckDB 0.10.2, pytest 9.0.3; metadata queried without tests |
| PATH interpreter | `C:/Users/Admin/itrader/python.exe`: Ray 2.9.0, torch 2.9.0.dev20250801+cu128, Gymnasium 1.0.0, NumPy 2.3.5; unsuitable for the root locked screen |
| GPU | NVIDIA RTX 5070 Ti, driver 617.42, 16,303 MiB VRAM, 861 MiB used, 15,137 MiB free, 2% utilization; WDDM per-process memory is N/A |
| Repo CUDA | `torch.version.cuda=None`, `cuda.is_available()=False`, device count 0; no tensors, inference, or CUDA computation launched |
| CPU/RAM | Ryzen 7 5800X, 8 cores / 16 logical processors, 6% sampled CPU load; 15.924 GiB physical RAM; free samples 1.131, 1.125, then 3.245 GiB, all below loadout's 5 GiB floor |
| Process context | At both process snapshots no Python, raylet, gcs_server, Ollama or vLLM process observed; many Codex/Cursor Node processes and desktop apps consume RAM. No process ownership inferred or process terminated; this chat owns no persistent job |
| Knowledge status | `& 'C:/Program Files/Git/bin/bash.exe' tools/knowledge/status.sh` exits 1: graph stale, indexed `1d5ba82` vs current `3a3583d`; then `line 23: python3: command not found` |
| Closest knowledge check | `python itrader/tools/run_repo_python.py -B tools/knowledge/status_summary.py --repo-root .` exits 0; index dated 2026-06-17, 189 files / 1,996 nodes / 3,202 edges. No index rebuild or package installation |
| Board format | `git diff --check -- docs/coordination/training-testing.md` exits 0; board is inside an untracked directory, so additionally reread its actual text; this is not an independent review |

Environment drift: [environment strategy](../environment_strategy.md) still
describes the repo `.venv` as Ray 2.9 and the shell launcher as Windows-broken.
Current native resolution/imports above establish Ray 2.58; the PATH environment
is Ray 2.9. GRILL's separately completed `LEASE-GRILL-HEALTH-01` healthcheck
(exit 0) agrees with the CPU-only finding; this parent did not rerun healthcheck.
Full pytest/pre-commit, Ray smoke, checkpoints, training/PBT, serving, acquisition,
and model APIs were not run. No market outcomes were parsed or displayed. An
initial overly broad recursive byte-hash probe included generated config outputs;
its digest is discarded and is not a source/run receipt. The final scope below
excludes generated outputs, datasets, run artifacts, and caches.

### Observed source/config identities (unsealed)

Branch/HEAD remain as declared above. A status sample had 35 modified tracked
entries and 50 untracked entries; later No-Mistakes/research additions changed
the overall dirty snapshot. Permission-denied temporary test directories remain
unknown. No staging, commits, pushes, source/config edits, or peer messages.

Preflight relevant manifest at approximately 08:51 UTC: **91 files**, SHA-256
`e8bd8ee09abc528cf5d3d9177993b5bc9bb265b47ac3ee52a66a6f22ea3cda60`.
Reproduction: collect `*.py` recursively from `itrader/models`, `itrader/envs`,
`itrader/rl_lib`, `itrader/tests` (exclude `__pycache__`), plus `pyproject.toml`,
`uv.lock`, the `.py`/`.sh` repo launchers, `run_ablation.py`, and the exact
16-cell config below. Sort relative POSIX paths, concatenate
`path + TAB + SHA256(file bytes) + LF` in UTF-8, then SHA-256 that stream.
This is a preflight identity, not a frozen run authority; rehash after DEV lands.
Final recheck at 08:53:53 UTC found **93 files**, digest
`122a048fb48c98d5a8d9413a8dbf1942b5d2ea5a6d1fabc3cbe5d685f53ab70b`;
DEV added R1 model/test files during this audit. The earlier source observations
below are timestamped findings, not current implementation/review claims. The
new R1 file is present; correctness and independent review remain unqualified
here. A stdlib config audit confirms all 16 unique combinations and PBT off.

| Relevant file (repo-relative) | SHA-256 |
|---|---|
| `pyproject.toml` | `11ce4f40f9973549d27cf740b437fe767755ad0d12f7ade74c704a2b0c4e9c16` |
| `uv.lock` | `4abdc1e100e8e86712babcb01ada0a23741c1b29abd896034017f2aad711126e` |
| `itrader/tools/run_repo_python.py` | `939c94a1d17f802d5d5e1ac8339fe44589b6b0497ed39eac1a695ecfdca53e7a` |
| `itrader/models/registry.py` | `64a16d191cd9d6f6de9597729bba6a21b3c3baca91bab0563ec11eabd4ca24af` |
| `itrader/models/solver_mlp_film.py` | `96bf09a5e51c5661e89d417bde6a60ec910f4c3aa940278352703ab3f260bd38` |
| `itrader/envs/itrader_env.py` | `cb262c630cbdefbf19b9a797bcc90e392234ae6cb771004b18dd4902c355a268` |
| `itrader/rl_lib/train/train_solo_V2.py` | `8428d4b8737cae2fe1839c6776be0a5697e010ab5a8ec11705330fa88afda060` |
| `itrader/rl_lib/evaluation/data_partition.py` | `0038e80b11840b7c1973a685f8caa4643f66bf3d1424fd1aa2bb692c2e605941` |
| `itrader/rl_lib/evaluation/run_terminal_screen.py` | `4cbc034bd9f4942d5eab84601ec9e0b46b3228f9d12b75872b289fef928127a5` |
| `itrader/rl_lib/evaluation/screen_completion.py` | `e159818f3db744016ee18d4c4582e23a847a20908cc8a823ce984410c3c0b0ce` |
| `itrader/scripts/experiments/run_ablation.py` | `07fa3dcfaf93ade6c49850186bd12945979e8f5d8e62421fb63ba526f217ca23` |
| `itrader/scripts/experiments/configs/phase_a_r1_corrected_compat_fixed_4x2x2_20261006.json` | `d28b778b480391547d5f5da4f8932ae461ea0ecfa5339ef2235bef0b619db38f` |

### Accepted experiment audit

The config contains all **16 distinct combinations**: R0, R0 + R4-v1,
R1, R1 + R4-v1; paired seeds **20260924 / 20260925**; exact 1 bp or causal
`0 -> 0.25 -> 0.5 -> 1 bp` at sampled steps **0 / 14,400 / 28,800 / 43,200**.
Evaluation stays 1 bp; PBT is false, population 1. Twelve iterations x batch
4,800 gives **57,600 nominal steps/cell**, **921,600 total**; actual batch
overshoot/exposure must be recorded, never called identical compute.

Q22 fields match readiness: 30 epochs, minibatch 128, LR
`7.151674865214668e-6`, gamma .99, lambda .9, clip .1, value coefficient 2,
value clip 20; original entropy clock `0:.005,96000:.002,240000:.0005`.
Window 60, `r0_r1_v1`, `market_relative_v1`, `NoFilter`, `zero_v1`, regular
session, net-return reward scale 1000; CPU-only, workers 0, Ray CPU cap 2,
per-cell timeout 1 hour. The timeout is config, not an approved whole-screen
wall-time budget. No volatility-normalization factor is enabled.

### Pending prerequisites / next decision

1. **DEV source and review:** at 08:51 registry raised `NotImplementedError`
   for R1 and `solver_window_lstm_film.py` was absent; this observation was
   superseded by concurrent file additions at 08:53. Require landed four-cell source,
   public selector/model/reset/restore evidence, and independent two-reviewer
   high-risk approval on exact dirty/untracked hashes. Historical approvals do
   not approve current/future R1. Existing checkpoint integration parameterizes
   R0 state `off/both`, not four encoder/state cells.
2. **Q23:** approved 2020-2021 candidate envelope is cross-period representation
   screening, not forward-time economic validation. Need manual/external-access
   attestation or conservative exclusion of unknown exposure, retained source-
   bound historical-start clearance, calendar/session metadata, complete eligible
   32 anchors / 12 symbols x four horizons, and a sealed 128-task panel/certificate.
   Current `DataPartition` accepts only all/train/validation/test; implement/review
   `terminal_screen_only`, certified intervals and dual training/terminal dataset
   identities. Config training path is the 2022-2026 dataset; existence alone was
   checked, not bytes/rows/outcomes. Current terminal preflight requires one equal
   dataset identity and is not the real executor; it cannot establish this new
   two-dataset contract. Human window approval does not certify freshness.
3. **Q26:** declare paired macro endpoint/contrasts, calendar dependence and block
   support, intervals, failure rules, training-seed treatment and multiplicity;
   calibrate with layout/synthetic outcomes before market results. Optional
   volatility normalization is off and is not a blocker. Do not treat two seeds
   or overlapping horizons as independent evidence.
4. **Execution sealing/resources:** reviewed actual terminal executor, matched
   controls and seeded update-zero monitoring, config/source/data/panel/runtime/
   process identities, predetermined terminal checkpoint, complete all-16 opening
   gate and sealed final test. Need a common runtime budget and explicit GRILL
   lease; the current 5 GiB start floor fails. No old eight-cell control receipt
   automatically substitutes for newly matched cells.

## Proposed next validation (not authorized or executed)

After DEV hands over a frozen source/review receipt and its lease completes,
request a separate GRILL lease for this existing public-interface command:

```powershell
python itrader/tools/run_repo_python.py -B -m pytest -q -p no:cacheprovider itrader/tests/test_env_core.py itrader/tests/test_semantic_observation_contract.py itrader/tests/test_solver_selection.py itrader/tests/test_terminal_screen.py
```

Proposed budget: one sequential process, CPU-only, native repo interpreter,
one PyTorch/BLAS thread, 5-minute hard wall cap, start only with >=5 GiB free RAM,
stop below 2 GiB or on hash drift/unauthorized outcome access/containment failure.
No Ray daemon, real data, API, or replay in this first command. Exact executed
node IDs and observable contracts matter more than case count. Retain distinct
deterministic invariants and current coverage. The command alone cannot certify
new R1 until DEV's named R1 public contracts join the frozen command manifest.

Then, on its own explicit Ray/replay lease and review, propose
`python itrader/tools/run_repo_python.py -B -m pytest -q -p no:cacheprovider itrader/tests/test_representation_checkpoint_integration.py`:
synthetic episode/reset -> selector/model -> PPO -> checkpoint restore -> replay.
Current file is R0-only; qualify its future four-cell extension before calling
it R1 E2E evidence. Proposed cap 10 minutes, one local Ray CPU, workers 0,
sequential cells, >=5 GiB start / 2 GiB stop reserve, supervised owned-process
cleanup. This is actual synthetic training/Ray/checkpoint activity and is excluded
from this preflight and the active DEV lease. Full suite/pre-commit need their
own serialized resource decision. Real 16-cell execution remains closed on all
prerequisites above; no learned/economic uplift claim follows from this receipt.

Write scope: this board only; GRILL/DEV own source/config implementation.
Declare a maximum 10-minute planning budget and read-only commands. No full
pytest/pre-commit, training, PBT, model calls, acquisition, checkpoint replay,
or source edits. Request any heavy-job lease on this board; do not self-authorize.

## Queue

1. Frozen-source focused and full validation, on an explicit GRILL lease.
2. Bounded synthetic R1 integration smoke after review and resource qualification.
3. Real matched R0/R1 x R4 screen only after Q23/Q26 and sealed run gates.

## Handoff Contract

Record job/source/config/data hashes, commands, elapsed time, exit codes,
hardware/resource availability, artifacts and stop reason. No learned/economic
uplift claim from a preflight or synthetic smoke. Keep verbose local logs under
`_local/` and retain the minimal durable reproduction receipt when authorized.
