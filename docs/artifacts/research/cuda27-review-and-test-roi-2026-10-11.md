# CUDA27 Source Review and Test-ROI Receipt

Date: 2026-10-11 (UTC). Branch: `agent/Watson-dev`. Reviewed base: `8f9fcf94e028b51d883a1a484c2c9466a3d6b547`.

## Verdict

Two fresh independent `REVIEW.md` passes approved the same repaired source
snapshot. Astra (`gpt-6-astra`) reported zero blockers and one nonblocking test
ROI nit; Sol (`gpt-6.1-sol`) reported no blockers or nits. The model names are
the assigned reviewer configurations; no separate serving-model telemetry was
collected.

Each reviewer executed the focused command exactly once:

```powershell
.\.venv\Scripts\python.exe itrader\tools\run_repo_python.py -m pytest -q itrader\tests\test_cuda_provisioning_contract.py
```

Both runs passed: 42 tests (0.14 s and 0.15 s). Reviewers also ran isolated
in-memory adversarial probes; positive controls passed, while malformed,
ambiguous, duplicate, reordered, stale, spoofed-source, expired, and
conflicting-evidence inputs were rejected as expected. No review started a
provisioning stage, fetched a wheel, installed/synced packages, exercised real
process containment, imported CUDA on hardware, ran Ray/training, or opened
market outcomes. The candidate code/config paths are not present at the reviewed
base, so an executable base comparison is unavailable. The review pool is
bounded, not exhaustive.

## Test ROI decision

Keep focused negative contracts for independent trust, schema, path/reparse,
runtime/platform, resource, timestamp/deadline, source/hash, protected-pin,
CUDA artifact cardinality, and process-boundary failure modes. Parameterized
order variants are meaningful because they catch validator order dependence;
they are not redundant merely because they share a test function. The shared
fake-process-scope helper already removes repeated setup without merging the
separate pre-launch-expiry and late-child-completion assertions.

Potential future consolidation: a mocked successful public `main()` lifecycle
could cover approval, wheel receipt, generated-lock acceptance, exact submitted
sync arguments, identity/smoke stage order, persisted evidence and refusal of a
second attempt. It could absorb only the overlapping positive receipt, valid-
lock and command-builder checks, and only if it retains those independent
oracles. Both reviewers recommend deferring this optional test redesign: it is
not needed to close the repaired validator defect and would broaden this bounded
source-review scope. Do not reduce the suite to meet a count or delete distinct
rejection paths.

## Design principles

- Separation of concerns: OK; provisioning, interpreter identity, tensor
  smoke, launcher routing, and process ownership remain separate.
- Programming by intention: OK; explicit validation stages now enforce the
  protected package and artifact identities without order-dependent acceptance.
- Encapsulation: OK; process ownership remains with the existing
  `ReplayProcessScope` owner.
- High cohesion: OK; reviewed components stay focused on the isolated profile
  lifecycle and its guards.
- Low coupling: OK; no trading, reward, or execution-policy dependencies were
  introduced.

No Git stage, commit, push, root dependency/lock edit, installation, CUDA
qualification, Ray run, training, or economic promotion follows from these
approvals alone.

## Native-preflight follow-up

After the reviewed snapshot, the real native `--stage preflight` exposed a
concrete output-format mismatch: pinned uv reports
`uv 0.11.7 (9d177269e 2026-04-15 x86_64-pc-windows-msvc)`, not just `uv 0.11.7`.
The first invocation exited 1 before wheel acquisition, lock generation, or
profile creation. A narrow parser now accepts only the exact pinned release,
optionally followed by parenthesized build metadata; it rejects nearby
versions, prereleases, and command failure. The new parameterized test passed
6/6, and the full focused suite passed 48/48 in 0.15 s. The native preflight
then passed with 4.76 GiB available RAM, 58.48 GiB free C: space, the expected
Windows AMD64/CPython 3.11/64-bit identity, exact uv 0.11.7, and no existing
target profile. It created only isolated uv configuration directories.

This changed the driver and test hashes, so the approvals above do not qualify
the updated bytes. A fresh independent review pair is required before any
acquisition. The test-ROI result does not change: retain distinct negative
oracles; defer the optional mocked positive lifecycle consolidation to a
separate scope. No installation or runtime qualification has occurred.

## Multiline uv-output boundary follow-up (2026-10-11 01:02 UTC)

The fresh review pair found the first parser repair still used `\s+`, which
accepts LF/CRLF between the pinned version and optional build metadata. Two
cases were added to the existing parameterized version-output test (no new test
function): `uv 0.11.7\n(build)` and `uv 0.11.7\r\n(uv 0.11.8)` must reject. The
red run reproduced both failures (2 failed, 6 passed). Replacing the separator
with horizontal space/tab made the test matrix pass 8/8; the full focused suite
passed 50/50, and `py_compile` passed for both touched Python files. Native
preflight passed again at 01:00 UTC with 4.76 GiB available RAM, 58.48 GiB free
C:, expected Windows AMD64/CPython 3.11/x64, exact uv 0.11.7, and no target
profile.

These final driver/test bytes require a fresh independent review; the earlier
approval and ignored review receipt are stale. Review must include test ROI.
This fix added cases inside an existing parameterized external-output oracle;
no test was deleted or merged. The optional positive public-lifecycle
consolidation remains deferred because it would broaden this narrow repair
without closing its current defect. No acquisition, lock generation, install/sync,
or GPU qualification occurred on that review snapshot.

## Version-metadata and Unicode-separator follow-up (2026-10-11 01:07 UTC)

The next independent probe found the metadata grammar could accept a
contradictory `uv 0.11.8` or bare `0.11.8`, and could contain VT, FF, NEL,
U+2028, or U+2029; generic `str.strip()` also erased Unicode line separators at
the outer boundary. Added ten cases inside the same parameterized CLI-output
test: two conflicting-version inputs, five metadata separators, leading and
trailing Unicode separators, and the normal trailing CRLF positive control.
The final pre-fix run reproduced nine incorrect accepts (9 failed, 9 passed).
The validator now trims only ASCII CLI whitespace, accepts only printable ASCII
build-metadata tokens with horizontal separators, and rejects version tokens in
metadata. The matrix passes 18/18, full focused suite 60/60 in 0.16 s,
`py_compile` passes, and Black formatted the driver. This is still one
parameterized test function; its rows protect independent parser boundaries.
No cases were removed and the optional positive lifecycle consolidation remains
deferred. The final bytes require a fresh review pair; no acquisition, lock
generation, install/sync, or GPU qualification has occurred.

## Exact uv build allowlist and receipt binding (2026-10-11 01:09 UTC)

The next source direction tightens the CLI contract to the canonical `uv 0.11.7`
line and exactly the observed native build output
`uv 0.11.7 (9d177269e 2026-04-15 x86_64-pc-windows-msvc)`. Only one terminal LF
or CRLF is accepted; surrounding spaces, extra lines, arbitrary parenthetical
text, conflicting version records, and control/Unicode separators reject. The
helper returns the normalized accepted line, and `_check_base` stores it as
`uv_version_output` in preflight evidence embedded in provision state.

The existing parameterized test now contains 24 precise accepted/rejected CLI
forms; no new function was added and no distinct negative case was removed. The
matrix passes 24/24, focused suite 66/66 in 0.16 s, Python compilation passes,
and Black completed. This literal allowlist supersedes the intermediate generic
metadata parser described above. It remains under a fresh independent review
pair with test-ROI assessment; previous approval and preflight receipts do not
cover these final bytes. No final-source native preflight, payload acquisition,
lock generation, install/sync, or GPU qualification has occurred.

## Byte-preserving version gate and final review (2026-10-11 01:32 UTC)

Fresh process-boundary probes reproduced a gap in the strict literal allowlist:
with `text=True`, Python universal-newline conversion changed raw
`uv 0.11.7\r` to `uv 0.11.7\n`, which the gate accepted. The red test reproduced
that behavior. `_run_small` now collects bytes and decodes UTF-8 after the
process completes, preserving the raw terminator for the unchanged validator.
LF and CRLF remain accepted; lone CR rejects.

Review also improved the independence of existing rejection oracles without
growing the suite: the nonzero-exit case now uses a valid pinned banner, while
the old and near-version rows use bare wrong-version outputs. No parser rows
were added or removed. One 3-case integration matrix was added for the distinct
process-transport contract. Current results: **69 passed** in the full focused
file; the relevant version tests pass **27/27**; Black, `py_compile`, and
scoped `git diff --check` pass.

The final independent pair approved the exact driver and contract-test hashes
below with no findings. Each reviewer found no value in merging the parser and
transport matrices: the 24-case parser table proves the version/exit contract,
and the three integration cases prove byte-preserving transport. The prior
review's test-oracle nit was fixed by editing existing rows, not adding more
tests. This is the bounded hybrid requested: focused unit-level malformed-input
coverage plus one integration seam for the bug's real transport boundary; no
test-count quota and no broad test deletion.

- `itrader/tools/provision_cuda_tested.py`:
  `4c2e253b27fdb59b183ed01273bf635590cd325d31a53f0b6a7cda98952de626`
- `itrader/tests/test_cuda_provisioning_contract.py`:
  `7d9fd3be61d6a87a4116639200e2d4b182e0b87959f51c0b3458a309fd322f21`

The changed files are untracked and absent at HEAD, so executable base
comparison is unavailable. The six other candidate artifacts match their
previously approved hashes exactly; the ignored review receipt records this
carry-forward plus the current two-file approvals. Requested reviewer lanes
were GPT-6.1 Sol and GPT-6 Astra; serving-model telemetry was not available and
is not asserted. Review approval is source-only. A fresh native preflight and
resource check are still required before the one authorized provisioning job.
No acquisition, generated lock, sync, installation, WSL change, or GPU
qualification has occurred.

## Final-source native preflight (2026-10-11 01:33 UTC)

The eight-artifact receipt passed the provisioner's own hash-bound review check.
The exact final-source native preflight then passed: Windows AMD64, CPython
3.11 x64, uv `0.11.7 (9d177269e 2026-04-15 x86_64-pc-windows-msvc)`, 4.51 GiB
available RAM, 58.44 GiB free C: space, and target profile absent. A first
attempt using the repository `.venv` exited at the expected native-interpreter
guard before stage work; it did not satisfy preflight. The second attempt used
the approved external native Python and passed.

This preflight was current-source preparation evidence only. It did not acquire
the wheel or generate a lock, and did not qualify installed CUDA/GPU behavior.
The subsequent one-shot acquire/lock outcome is recorded below; the preflight
snapshot does not override that failure.

## One-shot acquire/lock outcome (2026-10-11 01:36 UTC)

The single authorized run activated at `2026-10-11T01:34:26.295177+00:00`.
Acquisition completed once in 83.234 seconds. The retained wheel receipt records
the exact expected byte count and SHA256, with final host `download.pytorch.org`;
the wheel and logs remain in their isolated paths and were not deleted.

The run stopped during `uv lock` with exit code 1. `uv-lock.log` reports
`Fragments are not allowed in URLs` for the exact wheel URL plus its `#sha256`
fragment in the candidate `cuda-tested` group. No lock was accepted, and sync,
installation, and GPU smoke did not run. State shows `acquire=complete`,
`lock=started`, with the other stages not started; the failed stage was not
persisted as a terminal failure status. Treat the process exit code and log as
the outcome; close this state-machine reporting gap before a future attempt.

The driver process and contained worker exited; no owned process remained.
Post-run RAM was 4.55 GiB and free C: space was 55.59 GiB; target profile stayed
absent. Root dependency manifests were unchanged. This consumed the authorized
one-shot attempt and must not be retried under this approval. A future attempt
needs a new bounded authorization, a uv-2.58-compatible way to pin/verify the
wheel hash without a forbidden URL fragment, fresh source reviews, and a new
native preflight. Preserve the current wheel, logs, and failed state for that
diagnosis.
