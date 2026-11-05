# Environment Strategy

This document describes environment authority and execution boundaries; it does
not change dependencies.

Latest runtime reconciliation: 2026-10-11. Protected root `.venv` is Python 3.11.13,
Ray 2.58.0, Torch 2.3.1+cpu, Gymnasium 1.2.2 and NumPy 1.26.4, as recorded by
the [metadata inventory](artifacts/research/saluki-runtime-inventory-2026-10-10.md).
The designated `%LOCALAPPDATA%\iTrader\envs\cuda-tested-cu128` profile is now
installed from the reviewed root lock. Its tiny GPU forward/backward passed
with Torch `2.9.0.dev20250801+cu128`, CUDA runtime12.8, cuDNN91002 and RTX5070Ti
(CC12.0). [CUDA28 receipt](artifacts/research/cuda28-known-error-repair-2026-10-11.md)
binds exact source/lock/wheel and cleanup evidence. CUDA is the preferred forward
training target after its separately reviewed public PPO/checkpoint bridge;
the tiny test alone does not close that gate. Preserve the CPU control.
The older external CUDA environment's Ray2.9 must not replace the project stack.
See [GRILL](coordination/grill.md) for finite execution authority.

## Dependency authority

The source-of-truth dependency declarations are `pyproject.toml` and `uv.lock`:

- Python `>=3.11,<3.12`
- Ray/RLlib `2.58.0`
- Gymnasium `1.2.2`
- DuckDB `0.10.2`
- PyArrow `17.0.0`
- setuptools `69.5.1`
- CPU uv group: PyTorch `2.3.1+cpu` on Windows/Linux; `2.3.1` on macOS
- CUDA uv group: PyTorch `2.9.0.dev20250801+cu128`, Windows AMD64 CPython3.11 only

The old `docs/phase_b_plan.md` Ray/RLlib `2.47.1` line is a historical proposal.
Ray 2.9 is retained only through `itrader/compat/ray29/` and the existing
reproduction environment. New R-series runs use the root Ray 2.58 lock.

uv is the supported installer. The applied mutually exclusive `cpu-default`/
`cuda-tested` groups replace the obsolete global Torch bound. Plain `uv sync`
selects `dev` plus the CPU control; an explicit CUDA selection excludes defaults.
`pip install .` no longer installs PyTorch: non-uv users must explicitly install
the platform-appropriate CPU pin or the tested CUDA wheel and all intended dev/
optional tools. uv dependency groups are not published package extras.
No launcher may install or sync implicitly, and no GPU profile may silently
fall back to CPU. CUDA learners, CPU environments and small-batch inference
have different throughput characteristics; measure the actual workload.

CUDA28 resolved, reviewed and provisioned the declared profile; its lock also
refreshes76 shared non-Torch versions and removes decorator. All protected
experiment pins remain, but this is not full-suite qualification of those
refreshed tools or an unchanged-transitive-runtime CPU/GPU comparison. Reuse one
designated GPU profile, not a second full probe venv. "Merge environments" means
resolve and synchronize a declared profile, never copy `site-packages` or binaries
between interpreters. Keep the CPU control until proper-stack device execution,
PPO update, save, strict restore and frozen replay qualify. Root-runtime promotion
then requires the resulting identity/review receipt. Accepted wheel/cache and
durable evidence remain; cleanup is limited to inventoried agent-owned temporary
paths, not the external working environment. Do not combine the native Ray API
migration with this CUDA bridge; it requires its own matched baseline.

CUDA26's earlier caller-state launcher repair is component evidence only.
CUDA28 separately qualifies root profile/lock application, locked sync and the
tiny device test; TRAINING-CUDA29 owns the separately leased four-cell synthetic
PPO/update/save/restore/frozen-replay qualification. No market training follows.

## CUDA Profile Operations

The designated profile is outside the checkout to preserve the CPU control,
but consumes this repository's `pyproject.toml` and `uv.lock`; it is not the old
external Conda environment. Provisioning requires its own authorized resource
scope. The declared uv selection is:

```powershell
powershell.exe -NoProfile -Command {
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA 'iTrader\envs\cuda-tested-cu128'
    uv sync --locked --python 3.11 --no-default-groups --group dev --group cuda-tested
}
```

The override lives only in that child provisioning process, not the caller
shell, so it cannot redirect a later CPU sync. Existing validated profiles
need no sync before execution. The no-sync launcher routes through the
repository Python runner, which is the single owner of CUDA profile selection.
It checks the native Windows platform, fixed profile path, reparse-point and
`.venv`-overlap boundaries, and the selected interpreter's isolated 64-bit
CPython 3.11 identity. Identity path fields must be non-empty absolute strings
without NUL characters before path checks. The same checks apply when `run_repo_python.py` is
invoked directly with `ITRADER_PYTHON_PROFILE=cuda-tested`; validation no longer
depends on going through PowerShell. The PowerShell 7+ wrapper forwards the
explicit argument array with `.NET ProcessStartInfo.ArgumentList` and sets the
CUDA selector only in the child environment; it does not mutate caller
environment or location. Windows PowerShell 5.1 is rejected before any caller
state changes; use a Python caller that constructs a list `argv` (for example
the existing `run_ablation.py`) for that host. With no selector, the Python
runner keeps its existing CPU resolution; an invalid explicit CUDA selection
fails closed rather than falling back to the CPU `.venv`.

## Ablation Training Default

`itrader/scripts/experiments/run_ablation.py` keeps a narrower default than the
general Python runner: when a loadout omits `python_exe` and the caller supplies
no `ITRADER_PYTHON_PROFILE` and no nonempty `ITRADER_PYTHON_BIN`, only the
training child is dispatched through `run_repo_python.py` with the installed
`cuda-tested` profile selected in that child's environment. The profile's
existing platform, path, and interpreter identity checks remain authoritative;
missing or invalid CUDA fails before the training module runs, with no CPU
fallback, installation, or sync. A loadout's explicit `python_exe` continues to
win unchanged, preserving historical CPU and other pinned interpreter runs.
An explicit environment selector is likewise preserved rather than replaced.
This ablation-only wrapper call preserves the caller's working directory, so
relative dataset, checkpoint, and artifact paths retain their loadout meaning;
the repository root is added to the child import path. Other calls to
`run_repo_python.py` keep their existing repository-root working directory.
Collection and plotting do not receive the default CUDA selection, and the
general `run_repo_python.py` default remains CPU-safe. This interpreter-routing
default does not qualify economic performance or make R1 the model default;
R0 remains the parity control.

Pass Python options as one explicit array to the wrapper, which requires
PowerShell 7 or newer. PowerShell otherwise interprets flag-leading tokens such
as `-B` as launcher parameters before they reach Python:

```powershell
$pythonArguments = @(
    '-B', '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider',
    '--junitxml=docs/coordination/_local/cuda pytest.xml',
    'itrader/tests/test_representation_checkpoint_integration.py::test_representation_survives_ppo_checkpoint_and_frozen_replay[r0-r4-state]',
    'itrader/tests/test_representation_checkpoint_integration.py::test_representation_survives_ppo_checkpoint_and_frozen_replay[r1-r4-state]'
)
& .\itrader\tools\invoke_cuda_tested.ps1 `
    -RepoRoot $PWD.Path `
    -PythonArguments $pythonArguments
```

For an independently authorized tiny device check:

```powershell
$pythonArguments = @('-B', 'itrader/tools/cuda_runtime_smoke.py')
& .\itrader\tools\invoke_cuda_tested.ps1 `
    -RepoRoot $PWD.Path `
    -PythonArguments $pythonArguments
```

No activation, Conda PATH change, toolkit build or driver upgrade is required
by this profile. The tested [Windows CP311 wheel](https://download.pytorch.org/whl/nightly/cu128/torch-2.9.0.dev20250801%2Bcu128-cp311-cp311-win_amd64.whl)
is3,062,428,787bytes, SHA256
`8ff2cbe2a88bebfc14b71fea17f4937d77f0ff70aa9e4cebd6836b5bbe82765d`.
Non-uv installers must verify and install this exact artifact explicitly for
the tested CUDA path; an arbitrary latest Torch/CUDA release is not equivalent.

## Supported execution surfaces

For a Windows-native checkout, use PowerShell with a Windows Python/uv
installation and the repo lock. For a POSIX checkout, use a POSIX `.venv` or
Conda interpreter. In an earlier WSL session, invoking `.venv/Scripts/python.exe`
failed with `Exec format error`; do not generalize that observation to every WSL
configuration or activate WSL for this Windows-native workflow.

The repository launcher checks `ITRADER_PYTHON_BIN`, POSIX `.venv`, Windows
`.venv/Scripts/python.exe`, legacy environments, Conda, and `python` on PATH.
That fallback order does not make a Windows interpreter executable inside WSL;
select the compatible shell/environment explicitly.

## Validation commands (existing qualified environment only)

Do not synchronize dependencies as part of validation. On native Windows:

```powershell
.venv\Scripts\python.exe itrader/tools/run_repo_python.py itrader/tools/healthcheck.py
.venv\Scripts\python.exe itrader/tools/run_repo_python.py -m pytest -q itrader/tests
.venv\Scripts\python.exe itrader/tools/run_repo_python.py -m pre_commit run --all-files
```

For uv-based validation against an already provisioned environment, use
`uv run --no-sync`; provisioning and lock updates require separate authority.
On POSIX, with a compatible interpreter selected by the repo launcher:

```text
./itrader/tools/run_repo_python.sh itrader/tools/healthcheck.py
./itrader/tools/run_repo_python.sh -m pytest -q itrader/tests
./itrader/tools/run_repo_python.sh -m pre_commit run --all-files
```

A command that resolves a Windows executable from WSL and fails before test
collection is a blocked validation, not a test pass. Record the exact command
and error in the dated audit.

## Recurrent implementation prerequisite

Before using RLlib recurrent APIs, verify them against the locked Ray 2.58.0
runtime in a compatible environment. Do not copy flags or state behavior from
the historical 2.47.1 plan or Ray-2.9 reproduction profile. The recurrent contract must cover sequence lengths,
padding masks, last valid h/c, terminal/truncated bootstrap, fragment detach,
vector-worker state ownership, training/inference parity, and checkpoint/replay
state.

## Current validation boundary

The earlier WSL `Exec format error` and separate-environment migration gate
remain historical observations. The root lock and installed `.venv` now use
Ray 2.58; the prior statement that root `.venv` was still the Ray-2.9
reproduction surface is superseded by the inventory above. The optional Ray
dashboard package is not required for training. WSL remains off in the current
Windows workflow; use the native repository Python launcher.

Ray 2.9 on this Windows installation can truncate Tune-owned manifests and PBT
text logs to zero bytes. Repository-owned run manifests/status are therefore
written atomically under `~/ray_results/_stable_runs`, and per-trial config
history/status/checkpoint indices/metrics live under `_stable_metrics`. Treat
those stable paths as provenance authority while keeping Tune directories for
checkpoints and diagnostic inspection.

The earlier Bash/WSL pre-commit invocation selected a Windows interpreter and
failed before healthcheck/test hooks. That is a historical integration failure,
not a current native-Windows execution receipt. Invoke pre-commit through the
selected native interpreter and report the actual result; do not start WSL or
infer a fresh hook pass from historical direct healthcheck/pytest results.
