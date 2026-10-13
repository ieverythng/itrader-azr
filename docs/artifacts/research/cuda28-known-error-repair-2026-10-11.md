# CUDA28 known-error repair receipt — 2026-10-11

Status: completed 2026-10-11 02:08 UTC. The exact reviewed candidate lock was
applied, one locked sync completed, and the isolated CUDA tensor forward/backward
smoke passed. This qualifies this Windows CUDA profile for the tiny smoke only;
it does not qualify RLlib training, PBT, or economic performance.

## Scope and boundary

GRILL separately dispatched DEV-CUDA28 after CUDA27's only lock attempt failed
because uv 0.11.7 rejects `#sha256=...` URL fragments. CUDA27 remains closed;
its wheel, state, review receipt, and logs are preserved. This is not an
automatic retry. The new scope allows one corrected lock, a separate generated
lock review, one locked sync, and a tiny GPU tensor smoke, conditional on fresh
source/resource gates. No stage may start after the dispatch's absolute
02:45 UTC stop or exceed the 45-minute lease.

Working tree: `agent/Watson-dev`, HEAD
`8f9fcf94e028b51d883a1a484c2c9466a3d6b547`. Provisioner/test source paths are
untracked and absent at HEAD, so executable baseline comparison is unavailable.
Root `pyproject.toml` SHA256 remains
`11ce4f40f9973549d27cf740b437fe767755ad0d12f7ade74c704a2b0c4e9c16`; root
`uv.lock` remains
`4abdc1e100e8e86712babcb01ada0a23741c1b29abd896034017f2aad711126e`.

## Repair

The CUDA source entry now uses the exact official direct-wheel URL without a
fragment. The tested Torch version, encoded wheel path, and expected digest are
unchanged. Lock validation continues to require the exact official source URL
and `sha256:8ff2cbe2a88bebfc14b71fea17f4937d77f0ff70aa9e4cebd6836b5bbe82765d`.

The new stage first validates CUDA27's acquisition state, original log, two-
reviewer receipt, exact path/size/source, and a fresh offline digest of the
retained wheel. It then runs uv lock once. The prior standalone downloader and
internal fetch route were removed, so CUDA28 cannot acquire a replacement
proof archive; uv may make at most the one remaining normal full-wheel transfer
permitted by the dispatch.

## Checks so far

- TDD reproduction: the changed existing profile test went red against the
  old validator: exact `#sha256` fragment was accepted when it must be rejected.
- Focused suite after fix: **73 passed**.
- Black, `py_compile`, and scoped `git diff --check`: passed.
- The exact retained CUDA27 wheel revalidated offline: 3,062,428,787 bytes,
  SHA256 `8ff2cbe2a88bebfc14b71fea17f4937d77f0ff70aa9e4cebd6836b5bbe82765d`;
  state/log/review receipts agree.
- Native `uv lock --locked --offline` against a disposable copy of the
  fragmentless candidate reached the expected absent-cache error for that exact
  URL; it did not report a fragment parse error, access the network, or modify
  the copied lock (SHA256 unchanged at
  `4abdc1e100e8e86712babcb01ada0a23741c1b29abd896034017f2aad711126e`). This
  establishes parser acceptance only, not a generated lock.
- Earlier preflight: 4.55 GiB available RAM, 55.59 GiB free C:, no owned
  uv/Python/Ray provisioner process, and isolated target profile absent. The
  later, activation-adjacent preflight is recorded in the review section below.

## Independent source review and test return

Two fresh independent reviewers (GPT-6 Astra and GPT-6.1 Sol) reviewed the same
hash-bound candidate, ran the focused contract file once each, and both
returned `VERDICT: APPROVE` with zero blocking findings. At the final fresh
preflight (2026-10-11 01:55:48 UTC), available RAM was 4.45 GiB, free C: disk
55.59 GiB, no matching owned candidate process existed, and the target profile
was absent. Native preflight confirmed Windows AMD64, CPython 3.11 x64, and the
pinned `uv 0.11.7 (9d177269e 2026-04-15 x86_64-pc-windows-msvc)`.

Both reviewers identified one identical non-blocking diagnostic nit: malformed
top-level or nested retained-state JSON shapes (for example `[]`) escape as
`AttributeError` rather than `ProvisioningError`; the gate still stops before
lock/sync. No change was made because the current dispatch says not to polish
optional test nits or broaden the one-shot repair. Source approvals bind the
eight artifact hashes in `candidate-review.json`.

Test ROI: keep the 24-case URL/parser matrix and three-case process-byte
transport matrix separate because metadata validation and OS newline transport
are different public boundaries. Keep the four-case retained-proof table: the
positive proof, changed bytes, state/log disagreement, and review-digest
disagreement independently test artifact provenance and integrity. These are
not redundant test-count growth; the four cases share one public helper-level
evidence contract. Runtime-identity acceptance and rejection can be expressed
as one parameterized truth table without losing an oracle, but that only
condenses presentation. Both reviews advise against quota-based deletion. The
five design principles were assessed satisfactory; source files are absent at
the base commit, so no executable base comparison is available.

## Budget and gate closure

CUDA27 ran from `2026-10-11T01:34:26.295177Z` through the final `uv-lock.log`
write at `01:35:52.307439Z`: 86.0122625 seconds evidenced. Charge 87 seconds
conservatively; of the 2,400-second total, 2,313 remain. Preserve 300 seconds
for cleanup, limiting CUDA28 active stages to 2,013 seconds. Current live floors
must be rechecked immediately before activation: start RAM >=4 GiB / abort <2
GiB; disk >=30 GiB / abort <15 GiB.

The original sequence has now passed: exact-source reviews and preflight, one
retained-proof lock, two independent lock reviews, exact candidate application,
one locked sync, and the tiny GPU probe. No retry, PBT, market data, WSL,
driver/toolchain change, staging, commit, or push is authorized by this repair.

## Lock result (2026-10-11 01:58:57 UTC)

The one `reuse-proof-lock` invocation started at
`2026-10-11T01:56:48.645638Z` and ended naturally with exit code 0. State is
`awaiting_lock_review`, `lock=complete`, `acquire=reused_verified_proof`, and
`sync=smoke=not_started`. The generated candidate lock is 277,523 bytes with
SHA256 `0b12468efc1c2b376fa7818108e2e07016670fd01c71f1c67e41cd8375db09bb`.
Root `pyproject.toml` and `uv.lock` remain byte-identical to their frozen
baseline hashes at this point in the sequence. The lock reviews then approved
the candidate and the exact bytes were applied; final root hashes now equal the
candidate hashes below.

## Locked sync and CUDA smoke result (2026-10-11 02:08 UTC)

The two independent lock reviews both approved the exact candidate. One noted
as a NIT that 76 shared non-Torch package versions changed and `decorator` was
removed from the generated lock; this was recorded and accepted within the
dispatch's dependency-update scope. The other found no lock-content issues. No
extra test suite was added for lock bytes: both reviews judged the direct
artifact checks more useful, and tests duplicating a generated lock would add
little. This follows the test-ROI rule: retain tests for distinct behavior
contracts, consolidate only presentation-level duplicate cases, and do not
reduce coverage by quota.

The root files were guarded against the frozen pre-CUDA hashes, copied from the
approved candidate, and verified byte-for-byte afterward:

- `pyproject.toml`: `2fd34e7f17ebe5d0d9285d1dc5637e3c0d60174e52b688291b804695598b68cf`
- `uv.lock`: `0b12468efc1c2b376fa7818108e2e07016670fd01c71f1c67e41cd8375db09bb`

The one locked sync completed in the isolated profile using the same uv cache;
the sync log contains no second `Downloading torch` event. Target identity was
Windows AMD64, CPython 3.11 x64, at
`C:\Users\Admin\AppData\Local\iTrader\envs\cuda-tested-cu128`.
The tiny runtime smoke passed with Torch `2.9.0.dev20250801+cu128`, CUDA runtime
12.8, cuDNN 91002, NVIDIA GeForce RTX 5070 Ti (compute capability 12.0), output
shape `[4, 4]`, finite loss `0.15767672657966614`, and forward/backward gradient
shapes `[[4, 8], [4]]`. The provisioner records all stages complete and
`cuda_tensor_smoke_passed` at `2026-10-11T02:08:18.155214Z`.

The run started at `2026-10-11T01:56:48.645638Z` and reached final passed state
at `2026-10-11T02:08:18.155214Z` (689.510 seconds including review gates).
Lock generation itself took 2m08s; sync prepared 64 packages in 44.02s. The
sync log contains zero `Downloading torch` entries, so the retained 3.06 GB
wheel proof avoided a second Torch transfer. Installed pins include Ray
2.58.0, NumPy 1.26.4, Gymnasium 1.2.2, pandas 2.1.4, and pyarrow 17.0.0.

The minimum sampled runtime RAM was 3.72 GiB (above the 2 GiB abort floor); the
immediate post-run check was 4.22 GiB RAM and 49.71 GiB free C:. Later final
verification observed 4.17 GiB RAM. No matching owned uv, Python, Ray, Torch,
or provisioner process remained; no subprocess survivor was observed. The
activation preflight had 4.58 GiB available RAM and 55.59 GiB free C:.
This is not an RLlib/PPO/PBT test and does not establish compatibility of every
refreshed development dependency. The 76-package resolution change and missing
`decorator` remain visible for later code review. All changes remain dirty and
unstaged; no commit or push occurred.

Current relevant hashes:

| Path | SHA256 |
|---|---|
| `itrader/tools/provision_cuda_tested.py` | `04a011c6c9f4742bb89ce90c15e873ad546be761fe67c982c998772ddab0031a` |
| `itrader/tests/test_cuda_provisioning_contract.py` | `18816cc8b3263fc5d1ff4c721a6c949d662c0ef49891561b0cb1b40016854b53` |
| CUDA28 candidate `pyproject.toml` | `2fd34e7f17ebe5d0d9285d1dc5637e3c0d60174e52b688291b804695598b68cf` |
| CUDA28 candidate `README.md` | `70ba05b7742a31e64b2a3d54f6c8660ef77180938ba63e7eb6911264874daf48` |
| Applied root `pyproject.toml` | `2fd34e7f17ebe5d0d9285d1dc5637e3c0d60174e52b688291b804695598b68cf` |
| Applied root `uv.lock` | `0b12468efc1c2b376fa7818108e2e07016670fd01c71f1c67e41cd8375db09bb` |

Runtime and evidence artifact hashes:

| Artifact | Size | SHA256 |
|---|---:|---|
| Provisioner final state | 4,542 bytes | `458ecf04326bf98fb680db386e958ebe0c36a4e6953c48367d02906cee1b60dc` |
| Lock log | 149 bytes | `20e6c89f95ca2501645638dddc88e80e0c027ff3c2d983c9902c83e2dfe9a831` |
| Sync log | 2,888 bytes | `85e8946964b07b9b96a5e706d8b827a5278f754639f36f973b5a6dd595cd8fcb` |
| Target identity log | 320 bytes | `81b71e9c201ddba12cd74597c8ac30dc54b0e877149dd4849b65910b93132822` |
| CUDA smoke log | 397 bytes | `6f866ba803b94e0c2e127ed5dd29eda27d074446405af21cc19fddfe71200757` |
| Candidate review receipt (ignored local evidence) | — | `1c815822c707fee963092916627b171dc1eb3deac84aa9ac23fc17b3922cd8d4` |
| Lock review receipt (ignored local evidence) | — | `4148b4b75fd02d738cc92f6c9012922eb0024496b5b0d9ee644946153d2c951c` |
