# Environment Strategy

This document describes environment authority and execution boundaries; it does
not change dependencies.

## Dependency authority

The source-of-truth dependency declarations are `pyproject.toml` and `uv.lock`:

- Python `>=3.11,<3.12`
- Ray/RLlib `2.58.0`
- Gymnasium `1.2.2`
- DuckDB `0.10.2`
- PyArrow `17.0.0`
- setuptools `69.5.1`
- PyTorch `>=2.3,<2.4`

The old `docs/phase_b_plan.md` Ray/RLlib `2.47.1` line is a historical proposal.
Ray 2.9 is retained only through `itrader/compat/ray29/` and the existing
reproduction environment. New R-series runs use the root Ray 2.58 lock.

## Supported execution surfaces

For a Windows-native checkout, use PowerShell with a Windows Python/uv
installation and the repo lock. For a POSIX checkout, use a POSIX `.venv` or
Conda interpreter. In this session's WSL runtime, invoking `.venv/Scripts/python.exe`
failed with `Exec format error`; do not generalize that observation to every WSL
configuration.

The repository launcher checks `ITRADER_PYTHON_BIN`, POSIX `.venv`, Windows
`.venv/Scripts/python.exe`, legacy environments, Conda, and `python` on PATH.
That fallback order does not make a Windows interpreter executable inside WSL;
select the compatible shell/environment explicitly.

## Validation commands (compatible environment only)

```text
uv run python itrader/tools/healthcheck.py
uv run python -m pytest -q itrader/tests
uv run pre-commit run --all-files
```

Or, with a compatible interpreter selected by the repo launcher:

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

The earlier WSL `Exec format error` remains a historical observation. The root
lock now targets Ray 2.58. The existing repository `.venv` is intentionally left
as the Ray-2.9 reproduction surface until the user rebuilds it; the accepted
Ray-2.58 gate used a separate Python 3.11 environment. The optional Ray dashboard
package is not required for training.

Ray 2.9 on this Windows installation can truncate Tune-owned manifests and PBT
text logs to zero bytes. Repository-owned run manifests/status are therefore
written atomically under `~/ray_results/_stable_runs`, and per-trial config
history/status/checkpoint indices/metrics live under `_stable_metrics`. Treat
those stable paths as provenance authority while keeping Tune directories for
checkpoints and diagnostic inspection.

The current pre-commit command is still launched through Bash/WSL and selects the
Windows `.venv`, producing `Exec format error` before its healthcheck/test hooks.
Direct Windows healthcheck and pytest pass. Until the launcher becomes shell-
aware, record pre-commit as an environment-integration failure rather than
claiming the code checks ran inside that hook.
