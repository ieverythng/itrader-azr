# CUDA platform qualification: isolated compute plan

Date: 2026-10-09, Europe/Madrid. Scope: CUDA-PBT-PLAN-03-CLOSE.
Status: planning delivered; no installation, hardware probe, benchmark, PPO run,
or main-platform migration performed. Markdown is authoritative; HTML is its review companion.

Current authority: section 6 supersedes sections 1-5's provisioning recommendation.
The tested nightly-to-uv bridge is proposed; root CPU lifecycle remains protected.

## 1. Decision: one finite candidate

Recommend **torch 2.12.1+cu130**, native Windows x64 / CPython 3.11.13, in a
separate environment. Keep Ray 2.58.0, Gymnasium 1.2.2, NumPy 1.26.4 and the
existing ModelV2/TorchPolicy stack. This is a qualification candidate, not a
claim that iTrader is compatible or that it is the latest release.

The official Windows matrix documents CUDA 13.0 for 2.12.1 and its
[bugfix release](https://github.com/pytorch/pytorch/releases/tag/v2.12.1) is
published. The [2.12 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.12.0)
designate CUDA 13.0 the stable default and remove CUDA 12.8 from that build matrix.
The retrieved matrix also includes 2.13.0, while the start-local
selector displays 2.7.0 and the wheel index has additional newer entries:
do not infer “latest stable” from these inconsistent views. Use the fixed,
release-backed candidate above.
[Windows matrix](https://pytorch.org/get-started/previous-versions/),
[selector](https://pytorch.org/get-started/locally/).

| Item | Sealed candidate identity |
|---|---|
| Index | https://download.pytorch.org/whl/cu130 |
| Wheel | torch-2.12.1+cu130-cp311-cp311-win_amd64.whl |
| Official URL | https://download-r2.pytorch.org/whl/cu130/torch-2.12.1%2Bcu130-cp311-cp311-win_amd64.whl |
| Index-advertised SHA256 | 5ff38932260cb4d5a52170d955642f6ede17f565de64e62eaca12a875851471b |
| Python constraint | >=3.10; exact cp311 Windows wheel available |
| Historical minimum comparator | 2.7.1+cu128, first stable Blackwell-support generation; not selected for this finite run |

Wheel/hash were read from the [official index](https://download.pytorch.org/whl/cu130/torch/).
The advertised digest is **not** a locally verified downloaded-wheel digest.
[Official wheel metadata](https://download-r2.pytorch.org/whl/cu130/torch-2.12.1%2Bcu130-cp311-cp311-win_amd64.whl.metadata)
was fetched without downloading the wheel. CUDA 13.0 runtime choice does not
require installing local CUDA 13.0/13.1 or changing the driver. NVIDIA specifies
a CUDA 13.x driver floor of 580; TRAINING observed 617.42, but actual candidate
kernel execution remains untested.
[NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
Blackwell introduction is documented in [PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/).

## 2. Dependency identity and unresolved compatibility

The Windows wheel declares the following core dependencies; each current
uv.lock version satisfies its declared constraint. Preserve these exact versions,
and all other resolved packages, rather than allowing opportunistic upgrades.

| Dependency | Wheel requirement | Retained uv.lock version |
|---|---|---|
| filelock | unrestricted | 3.29.0 |
| typing-extensions | >=4.10.0 | 4.15.0 |
| setuptools | <82 | 69.5.1 |
| sympy | >=1.13.3 | 1.14.0 |
| networkx | >=2.5.1 | 3.6.1 |
| jinja2 | unrestricted | 3.1.6 |
| fsspec | >=0.8.5 | 2026.3.0 |

No torchvision, torchaudio, optional torch extras or LLM packages are required.
The Windows wheel metadata does not declare NumPy or Linux NVIDIA dependency
packages. Existing torch 2.3.1 alone changes; bundled runtime DLLs also change
inside the replacement wheel. This is not a full resolved-environment receipt.

Ray 2.58's tagged packaging supports Python 3.11 and pins RLlib Gymnasium 1.2.2;
it does not impose a torch bound in its RLlib extras. Its NumPy >=1.20
declaration does not contradict 1.26.4. None of this proves TorchPolicy API,
NumPy bridge, cuDNN, checkpoint or PPO compatibility.
[Ray 2.58 packaging](https://raw.githubusercontent.com/ray-project/ray/ray-2.58.0/python/setup.py).

The root project explicitly requires torch >=2.3,<2.4. Therefore ordinary root
installation/sync is intentionally incompatible with this candidate: do not
edit that pin, use the mixed PATH environment, or silently install the root
project with dependency resolution enabled. TRAINING must first retain an
isolated package manifest based on the Windows locked environment with exactly
one declared torch replacement, verify the acquired wheel digest, run package
consistency checking, and record Python/package/source hashes. Any additional
required dependency change stops this finite candidate and returns to GRILL.
No resolver/install operation was performed here.

## 3. Existing evidence, not repeated probes

Reuse [TRAINING's CUDA-PREFLIGHT-02](../../coordination/training-testing.md):
repo Python 3.11.13 imports torch 2.3.1+cpu, Ray 2.58.0, Gymnasium 1.2.2 and
NumPy 1.26.4. A different PATH interpreter executed tiny CUDA 12.8 FP32 matrix
forward/backward on RTX 5070 Ti (sm_120, about 16 GiB VRAM); its Ray/Gym/NumPy
versions differ. That proves hardware arithmetic, not candidate readiness.

Installed toolkit 13.2 and the human's earlier 13.1 recollection are distinct
from a wheel's runtime. Host RAM is about 16 GiB; observed availability was below
the existing 5 GiB start floor. VRAM does not remove host-RAM pressure.
Input SHA256: TRAINING board 3a727e1e2dff01c62c5fde176a517806e704529ed4b0e35cfe5a669a8b7225e2;
pyproject 11ce4f40f9973549d27cf740b437fe767755ad0d12f7ade74c704a2b0c4e9c16;
uv.lock 4abdc1e100e8e86712babcb01ada0a23741c1b29abd896034017f2aad711126e.
Branch/HEAD: agent/Watson-dev / 8f9fcf94e028b51d883a1a484c2c9466a3d6b547.
Dirty DEV work remains unsealed; freeze relevant model/helper/contract hashes before testing.

## 4. Bounded compute-only acceptance protocol

Proposed execution cap: 10 minutes of compute after isolated provisioning,
one process, one GPU, one CPU/BLAS thread, sequential cells, no Ray service,
PPO optimizer, market data, PBT or checkpoint run. Provisioning has its own
GRILL-controlled budget; timeouts stop rather than trigger a second candidate.
Only TRAINING's exact lease can authorize execution.

Test all four current structural cells: R0, R0+R4-v1, R1, R1+R4-v1.
Use the public selector and actual forward/value interfaces with fixed synthetic,
causal-format observations: 60x5 OHLCV, 32 task values and zero/two endogenous
state values. R1 is the actual 5-input/156-hidden LSTM plus 256-wide projection;
it resets hidden/cell state per decision. No R4-v2 modification.

**Correctness first:** fixed seed and identical serialized CPU initial weights/
inputs copied to GPU; float32 only; disable task dropout for comparison.
Compare logits, values and every parameter/input gradient using a fixed
synthetic scalar loss (sum of squared logits plus squared values, averaged).
Require finite outputs/gradients and elementwise
abs(GPU-CPU) <= 1e-5 + 1e-4*abs(CPU), including zero-reference entries.
These are proposed compute acceptance tolerances, not trading thresholds.
Compare B=1, B=128 and B=4800 forward; backward at B=128.
Do not relax tolerances after seeing failures. Near-tied argmax may differ within
tolerance; retain margins rather than declaring economic/action parity.

Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization, deterministic
algorithms enabled, cuDNN benchmark false, cuDNN deterministic true, TF32 disabled
for both matmul/cuDNN; AMP, BF16/FP16 and torch.compile disabled. Record effective
settings and cuDNN/runtime/architecture identities. CUDA LSTM can have
version-dependent nondeterminism; fail if the requested deterministic path is
unsupported instead of silently falling back.
[LSTM caveats](https://docs.pytorch.org/docs/2.11/generated/torch.nn.LSTM.html),
[reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html).
These 2.11 docs supply general caveats; candidate behavior still requires testing.

| Measurement | Predeclared method |
|---|---|
| B=1 inference | eval/no_grad; public forward plus value; resident-device compute and CPU-observation -> GPU -> CPU-logit/value round trip separately |
| Batched inference | same interfaces at B=128 and B=4800; identical weights/inputs |
| Batched forward/backward | B=128, training mode but dropout disabled on both devices; zero gradients each repetition, fixed loss, no optimizer step |
| Warmup | 10 untimed calls per cell/device/workload; record context/import startup separately |
| Repetitions | 5 groups x 20 calls after warmup; report all group times, median and min/max, latency and samples/sec |
| Synchronization | wall-clock perf_counter with CUDA synchronize immediately before and after each group; CUDA events additionally report resident kernel timing only |
| Transfer timing | round trip measured separately, not hidden in kernel-only throughput; host arrays and device buffers preallocated; no pinned-memory change in this first run |
| Memory | process RSS, free host RAM, GPU allocated/reserved peaks reset per workload; observed external GPU use and process lifetime retained |

No promised speedup: small B=1 policies can be transfer/launch-bound. Batched LSTM
may benefit even when rollout inference does not. CPU and GPU floating-point
results need not be bitwise identical.
[Numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html).

Start only with >=5 GiB free host RAM and >=4 GiB free VRAM; stop below 2 GiB
host reserve, below 2 GiB free VRAM, on OOM/kernel/DLL/import/dependency error,
NaN/Inf, tolerance failure, undeclared CPU fallback, source drift, lease timeout
or unexpected service/data access. Sample host memory between timing groups and
GPU memory after synchronization; these reserves are proposed safety limits.
Retain failure and partial results, then clean up only owned resources.

## 5. What a pass does not authorize

A pass establishes model compute feasibility and microbenchmark results, not
RLlib lifecycle, PPO throughput, training parity, edge or screen promotion.
Later, separately leased public synthetic four-cell PPO/reset/checkpoint/
restore/replay tests must retain ModelV2/TorchPolicy and evidence contracts.
Measure rollout, transfer, learner and evaluation times separately.

Use three lanes for eventual platform attribution: existing torch CPU,
candidate torch CPU, and the same candidate torch GPU. CPU-vs-GPU on the same
candidate separates device from torch-version drift; changing R0 CPU to R1 GPU
does not isolate representation. Keep market screen runtime uniform/refrozen.

PBT parallel trials add separate CUDA contexts, model/optimizer allocations,
Ray/object-store and host-memory costs. Fractional GPU resource reservations are
not VRAM quotas. Sequential compute comes first; any two-member GPU PBT test
needs a new resource/restore lease. Do not reopen the market screen from a
microbenchmark result.

Q23 external/pool/manual access remains unknown; legacy training has only its
bounded owner disposition. Historical status at the original note: Q26 endpoint/
method choices were unaccepted. Current canonical decisions now accept the
primary contrast, multiplier-first candidate and central equal-tailed95 interval
candidate; mechanics/numerical calibration remain pending.
[Existing Q23/Q26 decision note](q23-q26-decision-chain-2026-10-09.md).
CapabilityProfile remains online-facing, built from offline held-out evidence;
this platform test does not populate a calibrated capability profile.

## 6. Superseding bridge plan: preserve the tested nightly in uv

Scope CUDA-UV-BRIDGE-06, 2026-10-09. The human now prioritizes bringing a
qualified GPU torch into iTrader's reproducible uv runtime, **without another
CUDA build**. This section supersedes sections 1-5 as the current provisioning
recommendation: stable 2.12.1+cu130 is historical evidence, not adopted fallback.
Root CPU lifecycle remains frozen until CPU-LIFECYCLE05 closes; DEV lock/config
changes and TRAINING acquisition/install/run need separate reviewed scopes.

### 6.1 One candidate, available artifact, incomplete byte provenance

Recommend exactly **torch 2.9.0.dev20250801+cu128**, Windows x64 CPython 3.11,
the version used by TRAINING's existing tensor benchmark. Do not substitute the
PATH interpreter's Ray 2.9/Gym1.0/NumPy2.3.5 for iTrader's Ray2.58/Gym1.2.2/
NumPy1.26.4. Installed version.py reports CUDA12.8 and git commit
b4f42c92a0d44bf48321f1ba101f077d18ef57eb; INSTALLER says pip and direct_url.json
is absent. These observations do not prove the original installer URL or wheel hash.

| Metadata-only check | Observed result |
|---|---|
| Exact official nightly wheel URL | https://download.pytorch.org/whl/nightly/cu128/torch-2.9.0.dev20250801%2Bcu128-cp311-cp311-win_amd64.whl |
| HEAD | HTTP200; Content-Length 3,062,428,787 bytes; Last-Modified 2025-08-01 12:14:09 GMT |
| ETag | 5edc360313e9dc9017fc615b9ba181be-366; multipart identifier, not SHA256 |
| Exact .whl.metadata | HTTP200; torch 2.9.0.dev20250801+cu128, Python>=3.9; dependency fields match installed metadata |
| Current nightly index | Exact date/version not listed; index resolution alone cannot reproduce it |
| Same filename on stable cu128 path | HTTP403; not proof of a universal absence |
| Payload bytes/hash | Not downloaded or verified; no trusted wheel SHA256 recovered in this scope |

The [official nightly index](https://download.pytorch.org/whl/nightly/cu128/torch/)
and [exact metadata](https://download.pytorch.org/whl/nightly/cu128/torch-2.9.0.dev20250801%2Bcu128-cp311-cp311-win_amd64.whl.metadata)
are source receipts, not a lock. Later authorized acquisition must retain the
original URL, byte count, locally computed SHA256 and metadata, then verify that
the lock binds those bytes. A fresh hash pins the acquired artifact; it cannot
retroactively prove byte identity with the previously installed package.
Compare build/version metadata and repeat tensor qualification.

Core dependencies are filelock, typing-extensions>=4.10.0, sympy>=1.13.3,
networkx>=2.5.1, jinja2 and fsspec>=0.8.5; setuptools is conditional on Python>=3.12.
The prior note's retained versions satisfy these declarations. That does not
prove Ray2.58/TorchPolicy runtime compatibility. No torchvision/torchaudio,
optional torch extras or compiler are needed for the solver bridge.

If this exact archive becomes inaccessible or fails verified acquisition/
qualification, **stop**. Ask for an explicit fallback version/profile decision
or recovery of a retained original archive with provenance. Do not silently
choose stable2.12, latest nightly, installed site-packages or a reconstructed wheel.
A new download of a prebuilt wheel is not a CUDA/toolkit compilation; whether
that download is acceptable remains a separate bandwidth/storage authorization.

### 6.2 Reproducible uv profile design, for later DEV review

Observed uv is 0.11.7, Windows x64. Current root torch>=2.3,<2.4 forbids this
candidate. A GPU extra cannot override an unconditional constraint.
Propose one reviewed universal lock containing two **mutually exclusive**
runtime extras: cpu_legacy (torch==2.3.1, existing PyPI source/Windows artifact)
and cuda_tested (torch==2.9.0.dev20250801+cu128 from the exact URL above).
Move torch out of unconditional dependencies only under the later source scope;
preserve every other root pin and resolved package where constraints allow.

[uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
documents conflicting accelerator extras and source routing.
For this disappearing nightly, use an extra-qualified direct URL source rather
than a moving index; keep CPU on its original PyPI identity. Proposed structure,
not a patch or qualified configuration:

```toml
[project.optional-dependencies]
cpu_legacy = ["torch==2.3.1"]
cuda_tested = ["torch==2.9.0.dev20250801+cu128"]

[tool.uv]
conflicts = [[{ extra = "cpu_legacy" }, { extra = "cuda_tested" }]]

[tool.uv.sources]
torch = [{ url = "https://download.pytorch.org/whl/nightly/cu128/torch-2.9.0.dev20250801%2Bcu128-cp311-cp311-win_amd64.whl", extra = "cuda_tested" }]
```

Only native Windows CP311 GPU support is proposed. Cross-platform resolution and
markers must be reviewed against the existing repository's support policy; do
not silently drop non-Windows support to make the lock pass. If an index source
is later chosen, name it, use explicit=true and source-route only torch there;
generic dependencies stay on PyPI with safe first-index behavior.
Automatic torch-backend selection is not a sealed project profile.

Preserve existing dependency groups dev, etl, research and llm-local. uv resolves
groups/extras together unless declared conflicts, even when a group will not be
installed. Audit their locked transitive requirements and native-wheel
availability; do not upgrade NumPy, Ray, pandas or torch companions to repair
another wing invisibly. An auxiliary conflict requires an explicit disposition,
not dropping the group or source-building TA-Lib/llama-cpp in a CUDA lease.
[Dependency rules](https://docs.astral.sh/uv/concepts/projects/dependencies/).

Moving torch to extras makes a no-extra environment incomplete. Later launcher/
healthcheck review must enforce one runtime profile and preserve the historical
CPU default explicitly; do not rely on extras being automatically selected.
Use separate resolved absolute environment paths for CPU and CUDA, never toggle
the active .venv in place. Lock/source identity, selected extra/groups, Python,
uv version, wheel hashes and actual torch device belong in every run receipt.
Use --locked to check lock freshness; --frozen alone skips that check. uv run
can sync automatically and uv sync can remove extraneous packages, so neither
should be run casually on the protected root environment.
[Lock/sync rules](https://docs.astral.sh/uv/concepts/projects/sync/).
No configuration or lock command was executed here.

### 6.3 Cache, disk and rollback

Read-only metadata inspection found a uv cache at
C:/Users/Admin/AppData/Local/uv/cache, with archive-v0, wheels-v6 and simple-v21.
The bounded wheels-v6 scan found only a torch PyPI entry for a Linux 2.3.1
artifact; no matching Windows nightly cache entry was recovered. This is not a
complete cache inventory, and installed pip files are not evidence of uv cache
reuse. Cache byte completeness, extracted GPU size and total download requirement
remain unknown. Do not copy site-packages, inspect private cache blobs as a
substitute for an artifact, or clean/prune caches.

C: had 66,838,654,976 free bytes at the metadata snapshot. The known compressed
archive is about 3.06 GB; budget extracted cache, environment copies/linking,
temporary extraction and any missing dependencies in addition. No confident
zero-download or peak-disk claim follows. Freeze disk/start-stop reserves before
authorized provisioning and check the exact cache/environment volume layout.
[uv caching](https://docs.astral.sh/uv/concepts/cache/) permits reuse when source,
metadata and artifacts are available; linking versus copying depends on platform/
filesystem. An offline completeness check must fail closed rather than fall back
to a different version or network request.

Retain existing pyproject/uv.lock bytes and .venv; no environment deletion or
overwrite is needed. Rollback means selecting the preserved CPU environment/
locked profile and quarantining the candidate, not reinstalling over a failed
environment. Original pins/hashes: pyproject
11ce4f40f9973549d27cf740b437fe767755ad0d12f7ade74c704a2b0c4e9c16;
uv.lock 4abdc1e100e8e86712babcb01ada0a23741c1b29abd896034017f2aad711126e.

### 6.4 Bridge gates, separate from bare tensors and market claims

After CPU lifecycle closure and independent config/lock review, later scopes:

1. Authorized cache-aware artifact acquisition and isolated sync; verify hash,
   package consistency, all retained pins and explicit interpreter/profile.
2. Native sm120/cuDNN and NumPy bridge imports; repeat actual solver FP32
   CPU/GPU forward/backward under declared settings/tolerances and memory gates.
   A tensor benchmark or installed arch list alone is insufficient.
3. All four public cells R0/R0+R4/R1/R1+R4: synthetic env/reset -> ModelV2/
   TorchPolicy PPO -> save/restore -> deterministic replay, optimizer/evidence/
   process provenance and contained failures. No native RLModule migration.
4. Compare existing CPU torch, candidate torch on CPU and the same candidate on
   GPU to separate version/device effects. Test checkpoint portability explicitly;
   matching shapes or torch version strings do not prove it.
5. Only separately authorize uniform-runtime matched representation training
   after source/config/runtime seals and outstanding Q23/Q26 gates.

Record rollout, transfer, learner and evaluation time separately before any
speed claim. Small-batch inference can lose to transfer overhead; GPU PBT adds
CUDA contexts and host/VRAM allocations, and fractional reservations are not
VRAM quotas. No promised rollout/PBT speedup, economic uplift or resource cure.
Existing TRAINING tensor-only lease does not authorize these gates.

Installed-file metadata SHA256: version.py
a18e28b7c4849f8f58f04ee3b24d329f5f0892e07d89d93f1a3f7630464ef249;
METADATA 550a96f834ea68af3ce35ddc116a7074281d74801177727b89df16f41142e4c1.
The research board retains scope/review receipts. No imports, payload downloads,
installs, root changes, compiler work or benchmarks were performed in this pass.
