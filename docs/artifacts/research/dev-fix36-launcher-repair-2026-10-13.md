# DEV-FIX36 launcher repair receipt

Status: bounded repairs implemented; focused contracts pass (37 passed in
4.20s); final independent reviews of the latest exact hashes are pending. The
two initial `REVIEW.md`
reviews returned `CHANGES` on an earlier snapshot; those findings and their
reproductions are preserved below as historical evidence and are addressed in
the follow-up disposition. This is not a CUDA/PPO/training qualification.
Branch `agent/Watson-dev`, HEAD `8f9fcf94e028b51d883a1a484c2c9466a3d6b547`;
all work remains dirty and unstaged.

## Finding and repair

The repository runner silently fell through to its CPU interpreter when an
explicit CUDA profile was unavailable or unknown. A second ownership gap meant
the PowerShell launcher carried the profile path/reparse/overlap/interpreter
identity guards, so direct native use of `run_repo_python.py` could bypass
them. An explicit empty selector also behaved like no selector.

`run_repo_python.py` is now the sole owner of profile selection and validation
for both direct and wrapped invocation. It rejects empty/unknown selectors;
requires Windows and the fixed `%LOCALAPPDATA%\iTrader\envs\cuda-tested-cu128`
target; rejects overlap with the protected project `.venv` and reparse-point
components; validates the selected runtime's exact CPython 3.11, AMD64, 64-bit,
executable, prefix and base-prefix identity; and fails closed before running the
requested child. The no-selector CPU resolution path is unchanged. The existing
PowerShell entrypoint is a compatibility/argv/state-restoration wrapper and
delegates the profile decision to the Python runner. No installer, runtime pin,
CUDA query, model, Ray, PPO, reward, or solver behavior changed.

Architecture map: public native entrypoint/validator `itrader/tools/run_repo_python.py`;
identity reporter `itrader/tools/cuda_identity_probe.py`; compatibility caller
`itrader/tools/invoke_cuda_tested.ps1`; nearest behavior contracts
`itrader/tests/test_run_repo_python.py` and
`itrader/tests/test_cuda_test_launcher.py`.

## Test ROI and evidence

Tests were defined before reading implementation in the ignored
[`repair-scenarios.md`](../../coordination/_local/dev-fix36/repair-scenarios.md).
The focused test strategy retains the three existing resolver invariants,
keeps invalid selector and identity alternatives table-driven, and covers the
two public routes with direct-profile and complete-wrapper integration oracles.
The wrapper oracle checks both separated/equals `--junitxml` spellings, both
node-ID orders, a single `--version` argument, empty-argument rejection,
stdout/stderr, nonzero child exit, profile selection, and restoration of caller
location, preferences, and environment. This avoids adding one test per flag or
saved variable while keeping direct-entrypoint and wrapper ownership
independently visible. Tests were not deleted to meet a count target.

## Initial independent review result (2026-10-13; superseded by follow-up below)

Both reviewers confirmed the same 11-entry frozen review snapshot before and
after their read-only passes; neither edited or staged files. The source/test
files remain frozen at the recorded hashes. Review outcomes were `CHANGES`.
One finding was independently reproduced by both reviewers, leaving five
unique blockers and one nit across both reports:

1. **Blocking — Windows short-path alias bypasses protected `.venv` overlap.**
   Input: put the selected profile beneath `<repo>/.venv`, set
   `LOCALAPPDATA=<repo>/.venv`, and invoke an unchanged runner copy using its
   `GetShortPathNameW` alias. Expected: exit 2 before the child. Actual: exit 0
   and `CHILD_RAN`; long-path invocation rejects the same profile.
2. **Blocking — malformed identity passes isolation validation.** Input: an
   otherwise valid CPython 3.11 AMD64/64-bit identity, with `base_prefix` set
   individually to `None`, `""`, `False`, `0`, `{}`, or `[]`. Expected: every
   malformed value raises `ValueError`. Actual: all six are stringified and
   accepted as paths different from the selected profile.
3. **Blocking — Windows PowerShell 5.1 mutates explicit argv.** Input:
   `['quoted "value"', '', 'argument with spaces']` in the documented
   `-PythonArguments` interface. Expected: exact child arguments. Actual:
   embedded quotes disappear and the empty token is dropped; pwsh 7 preserves
   the vector.
4. **Blocking — launcher test harness does not exercise its 5.1 fallback.**
   Input: run the existing lifecycle test under `powershell.exe` 5.1; its
   `@(ConvertFrom-Json ...)` nesting produces the wrong parameter array.
   Expected: the fake module receives the complete pytest-style vector and
   returns its markers/status 17. Actual: Python exits 2 with `Unknown option:
   - ` before the child markers. The fixture prefers pwsh when both engines
   exist, masking the fallback failure.
5. **Blocking — 5.1 restoration loses a present empty selector.** Input: caller
   process has `ITRADER_PYTHON_PROFILE` present as `""`, then runs the valid
   wrapper. Expected: it remains present and empty afterward. Actual: it is
   absent, changing subsequent default-selection behavior.
6. **Nit — rejection tests do not independently prove that no child started.**
   They name a nonexistent `unused.py`, while the asserted output marker belongs
   to the separate fake-module path. Replace that input with one runnable
   sentinel child in the existing parameterized scenario.

The path-alias and Windows PowerShell findings are compatibility-boundary
issues; no training, native solver, PPO, PBT, or CUDA workload was exercised.
The GRILL scope forbids another argv/JSON converter or helper layer, so the
PowerShell 5.1 repair must either fit the already approved interface or be
returned to GRILL for a decision; do not silently add a transport mechanism.

### Initial test ROI disposition

The initial reviewers did **not** recommend deleting the focused resolver
tests. Their specific follow-up suggestions were considered against the
repaired interfaces; final retained/consolidated coverage is recorded below.

The full reports contain supplementary probe limits. Astra executed the focused
suite (28 passed in 5.23s); Sol independently executed it (28 passed in 4.12s).
Both reported `git diff --check` passing with the existing DEV-board line-ending
warning. Neither compared executable HEAD behavior because the reviewed tool
and test files are absent from HEAD; recorded intake red evidence is not a base
execution. No installs, full suite, WSL, Torch/Ray/CUDA/device work, training,
or Git shipping occurred.

Observed red: before the repair, explicit CUDA-missing and unknown-profile
subprocess cases both ran the CPU fallback (exit 0 and a created sentinel). The
empty-selector probe later reproduced the same issue. After the repair:

```text
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider itrader/tests/test_run_repo_python.py itrader/tests/test_cuda_test_launcher.py
28 passed in 4.25s
git diff --check -- <scoped paths>
passed (existing DEV-board LF/CRLF warning only)
```

The Windows launcher integration uses a temporary standard-library fake
`pytest.py`, not real test collection. The direct-runner integration confirms
the selected profile executes the child without using PowerShell. No Torch/Ray
import, CUDA query/compute, GPU work, full repository suite, healthcheck,
pre-commit, PBT, or training ran. `ruff` is unavailable and was not installed.

The full per-file intake/final hashes and exact scope are in the ignored
[`snapshot.md`](../../coordination/_local/dev-fix36/snapshot.md). The initial
reviews apply only to the earlier source hashes. No files have been staged,
committed, or pushed.

## Bounded follow-up repair and test ROI (2026-10-13)

The follow-up resolved the initial five blockers and sentinel-oracle nit within
the existing public seams:

1. `_paths_overlap` now canonicalizes physical Windows paths after each path's
   reparse-point walk; a public subprocess invokes the copied runner through an
   actual 8.3 short path while the selected profile overlaps `.venv`. Expected
   exit 2/no child; observed exit 2 with the overlap diagnostic and no sentinel.
2. `validate_profile_identity` now rejects non-string/blank, relative, or
   NUL-containing interpreter path identity fields before path normalization.
   Six malformed `base_prefix` values are rows in the existing identity table,
   not six new test functions. A later independent review found relative and
   NUL path values were not covered by the first table; these are two added rows,
   not new test functions, and now reach the validator unchanged.
   The first fresh review caught that the table setup accidentally replaced
   those six values with the valid expected path. The setup is corrected so
   each malformed value reaches the validator; the focused test count did not
   increase.
3. The PowerShell 7+ wrapper uses the standard `.NET
   ProcessStartInfo.ArgumentList` and per-child `Environment` dictionary. The
   parent process, location, and preferences are not mutated; no manual quote
   encoder, JSON transport, or helper layer was introduced. The existing
   wrapper lifecycle oracle now includes an embedded quote, empty token, and
   spaced token while checking exact child argv, streams, nonzero exit, and
   caller state.
4. Windows PowerShell 5.1 is rejected before any caller-state mutation. The
   focused legacy-host test verifies the diagnostic, unchanged location and
   override, present-empty selector presence before/after, and no child start.
   The shared harness normalizes JSON parsing as a flat string array rather
   than relying on PowerShell's nested-array behavior.
5. The invalid-launch matrix now executes a real sentinel child when testing
   refusal, so “child did not start” is a valid oracle. The redundant normal
   wrapper-overlap row was removed: physical overlap is already covered by the
   new public short-path runner integration.

Final test ROI: preserve the three small resolver unit contracts because they
exercise different selection invariants; keep selector and runtime-identity
variants table-driven; combine CPU-default and explicit-profile behavior in
the existing direct-runner integration; retain the wrapper's single lifecycle
integration for exact argv/streams/status/state; and keep separate short-path
security and PowerShell-5.1 refusal boundaries. No unit tests were deleted by
quota, no one-test-per-token growth was added, and the duplicate overlap case
was consolidated rather than multiplied. This is the requested hybrid: focused
unit checks for local invariants plus a few public-contract integrations with
independent external oracles.

Focused command after the follow-up changes:

```text
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider itrader/tests/test_run_repo_python.py itrader/tests/test_cuda_test_launcher.py
37 passed in 4.20s
```

## Final independent review disposition (2026-10-13)

Two fresh independent `REVIEW.md` lanes (Astra and Sol) approved the exact
01:12 source snapshot; all eleven SHA256/byte-count pairs matched the manifest
before and after review. The review history found two issues: the first,
relative/NUL-containing `base_prefix`, was blocking and is fixed by the frozen
validator plus table-driven cases. The second is one optional nit: a direct
call to `validate_profile_identity` accepts a base-prefix written as the
selected profile's physical 8.3 alias because `_same_path` compares lexical
forms. This is not a demonstrated runner bypass: both reviewers executed the
unchanged reporter→JSON→validator path with that alias, observed the reporter
canonicalize it, and the validator reject it. No source change was made for
this optional internal-contract hardening within the validation-only window.

The separate PowerShell 5.1 present-empty concern also is not a product defect:
the actual wrapper refuses before mutation; both reviewers verified absent,
present-empty, and present-valued selectors remain unchanged, with no payload
launch. The table-driven legacy-host lifecycle test covers present-empty state.

Test-ROI disposition: retain three small resolver unit contracts because they
cover distinct path-selection invariants; retain table-driven selector and
identity variants; use direct-runner and full wrapper-lifecycle integrations
for their respective public entrypoints; keep physical 8.3-overlap security and
PowerShell 5.1 refusal as separate failure boundaries. One duplicate normal
wrapper-overlap row was removed because the short-path runner integration
already covers that invariant. No count-based unit-test deletion or new
one-test-per-argument growth was justified. Independent reviewers agree that
additional consolidation would weaken diagnostic specificity.

The focused command passed **37 tests** in the author rerun (4.20s); independent
reviewer reruns also passed 37 (4.14s and 4.58s). Scoped `git diff --check`
passed with only the existing DEV-board LF/CRLF warning. No full suite,
healthcheck, pre-commit, dependency operation, WSL, Torch/Ray/CUDA/device work,
training, PBT, staging, commit, or push occurred. The review hashes and exact
scope remain in the ignored [`snapshot.md`](../../coordination/_local/dev-fix36/snapshot.md).

All DEV test/reviewer jobs are closed. A hash-bound source/review handoff was
sent to TRAINING for its separately GRILL-gated CUDA37 preparation; this does
not constitute a run launch or authorize DEV to execute training.
