# CUDA26 Review and Test ROI — 2026-10-10

## Scope and disposition

This receipt records the bounded CUDA26 launcher repair/review and a read-only
audit of test growth in the inherited working tree. It does not authorize CUDA
package acquisition, environment creation/sync, Ray/Torch/GPU execution, market
experiments, or Git shipping. Root `pyproject.toml` and `uv.lock` remain
unchanged.

The initial independent review snapshot had one reproduced blocker: when
`ITRADER_PYTHON_BIN` was absent before a rejected launch, the launcher returned
the intended failure but left the variable present in the caller process. The
cause was unconditional `[Environment]::SetEnvironmentVariable(..., $null,
'Process')` in `finally`; the exercised PowerShell host did not restore
absence. Both independent reviewers classified this as a source contract
failure, not a test-oracle defect.

Under GRILL's narrowly renewed one-round scope, the launcher now records whether
the process environment entry existed and explicitly removes `Env:ITRADER_PYTHON_BIN`
when the caller had no entry. An originally present value still uses the normal
value-restoration path. The current author test passed. Two fresh independent
reviews then approved the same frozen source snapshot with zero findings; each
reviewer executed the authorized launcher suite exactly once. Final source and
package-metadata hashes were rechecked after both reviews and matched the frozen
identities below.

## Verification record

The author ran exactly once after the fix:

```text
.venv\Scripts\python.exe itrader\tools\run_repo_python.py -m pytest -q itrader\tests\test_cuda_test_launcher.py
```

Result: exit 0, **6 passed in 2.92 s**. Before the repair, both independent
reviews ran this same scoped suite once against the frozen bad snapshot; each
observed **4 failed, 2 passed** in about 3 seconds, all four failures at the
assertion that the initially absent environment variable remained absent. Those
failures are preserved here as evidence, not relabeled as passing.

The package metadata identities remain:

| File | SHA-256 |
| --- | --- |
| `pyproject.toml` | `11CE4F40F9973549D27CF740B437FE767755AD0D12F7ADE74C704A2B0C4E9C16` |
| `uv.lock` | `4ABDC1E100E8E86712BABCB01ADA0A23741C1B29ABD896034017F2AAD711126E` |

Current frozen source identities:

| File | SHA-256 |
| --- | --- |
| `itrader/tools/invoke_cuda_tested.ps1` | `13F2DB230F4E7FC21AB5875D0A7FF26885303C41612C7685F8E7283D711FF966` |
| `itrader/tools/cuda_identity_probe.py` | `4899BD74A3888F3293233AD0C669E1A45FD17EE23DCC95F661AF6D7E62CDCBD2` |
| `itrader/tests/test_cuda_test_launcher.py` | `55168F0FD58E791A572A84F7C7385630F8414C52BE31352C8890AC2CF8525AB7` |
| `itrader/tests/test_representation_checkpoint_integration.py` | `C6FA6671F58623F181A8A3294F4C52893B93BB86DD52F12D0AFA42C00C1CF518` |

Both independent reviewers returned `VERDICT: APPROVE` for this bounded source
candidate. Each reported 6 passed, with run times of 2.90 s and 2.91 s; neither
used a retry. Both found zero remaining BLOCKING or NIT findings, confirmed the
launcher fix restored absent and present caller values on the exercised paths,
and verified all six supplied hashes before and after review. The source files
are untracked and have no executable counterpart at base HEAD; `pyproject.toml`
and `uv.lock` are unchanged from base.

The five `REVIEW.md` principles are all OK for the scoped code: separation of
concerns (identity probe separate from routing), programming by intention
(named validation and clear failures), encapsulation (caller environment,
location and preferences restored), high cohesion (GPU checks stay within the
checkpoint lifecycle boundary), and low coupling (the launcher delegates to
the existing repository runner). The two reviewer executions and the author's
six-pass run are the only post-fix test executions. No CUDA runtime, Ray, Torch,
provisioning, download, install, or sync was executed here; those gates remain
unqualified.

## Test-efficiency review

The bounded read-only inventory compared the dirty test tree with HEAD
`8f9fcf94e028b51d883a1a484c2c9466a3d6b547`: **28 changed test modules** (7
modified, 21 new), containing 120 added test-function definitions in modified
modules and 156 in new modules. This is a large test delta, but the audit found
most additions protect distinct public contracts, failure boundaries, or
evidence provenance. A raw test-count target would discard useful checks and
does not demonstrate better value.

Two likely future consolidation candidates, each requiring a separate scoped
edit and review:

1. `itrader/tests/test_terminal_screen.py:13` and `:86` build the same
   eight-cell partial-screen rejection case through the API and CLI. These
   could share one named API-plus-CLI scenario while keeping both observable
   failure paths. Keep the lower-level validator cases in
   `itrader/tests/test_screen_completion.py:91` because they cover broader
   malformed-evidence inputs.
2. `itrader/tests/test_no_nan_forward.py:175` and `:268` separately compare
   model-level initial/state-neutral outputs against legacy behavior. These
   could become a parameterized parity scenario with explicit case IDs. Keep
   the environment-to-model integration in
   `itrader/tests/test_semantic_observation_contract.py:176`, which adds tagged
   Parquet preprocessing, environment-produced state, and gradient evidence.

The four checkpoint lifecycle cells in
`itrader/tests/test_representation_checkpoint_integration.py:248-258` are not
duplicates of the CUDA launcher test: they span the MLP/LSTM by state-off/state-
both design, including checkpoint/replay boundaries. Likewise retain separate
checkpoint selection, restore validation, and downstream lineage/digest
checks; combining them would hide which evidence boundary rejects an error.
The independent code reviewers identified a smaller exact redundancy in the
checkpoint integration: the final canonical-digest equality at line 355 is
implied by earlier row equality checks at lines 322 and 354. Preserve the
production payload-extraction check at line 323 and callback-payload digest
boundary. No test consolidation was made in this CUDA26 scope.

The practical policy is therefore: use one or a few public end-to-end
scenarios for lifecycle cohesion, retain small parameterized matrices for
genuinely distinct inputs, and keep focused tests where they protect
independent arithmetic, trust, or provenance invariants. Consolidate only
demonstrated duplicate oracles, not every unit test into a large integration.

## Review and next boundary

Initial frozen snapshot: both `REVIEW.md` passes returned `VERDICT: CHANGES`
with one blocker—the absent caller environment entry was not restored. The
repair was reviewed on fresh post-fix bytes, with test results, base comparison,
all five principles, test-ROI assessment, unknowns, and exact verdicts recorded
above. The scoped source approval does not imply package-resolution,
CUDA-runtime, GPU-placement, or economic qualification.

Even after the bounded source review passes, actual acquisition/sync remains
closed unless the separate complete package-source, wheel-hash, lock-binding,
command, and resource gates receive their required approvals and preflight.
