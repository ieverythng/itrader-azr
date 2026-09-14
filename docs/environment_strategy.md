# Environment Strategy

This document describes environment authority and execution boundaries; it does
not change dependencies.

## Dependency authority

The source-of-truth dependency declarations are `pyproject.toml` and `uv.lock`:

- Python `>=3.11,<3.12`
- Ray/RLlib `2.9.0`
- Gymnasium `0.28.1`
- DuckDB `0.10.2`
- PyArrow `13.0.0`
- setuptools `69.5.1`
- PyTorch `>=2.3,<2.4`

The old `docs/phase_b_plan.md` Ray/RLlib `2.47.1` line is retained only as a
historical proposal and is not an API authority. No lock or environment change
was made in the 2026-09-12 docs pass.

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

Before using RLlib recurrent APIs, verify them against the locked Ray 2.9.0
runtime in a compatible environment. Do not copy flags or state behavior from the
historical 2.47.1 plan. The recurrent contract must cover sequence lengths,
padding masks, last valid h/c, terminal/truncated bootstrap, fragment detach,
vector-worker state ownership, training/inference parity, and checkpoint/replay
state.

## Current validation boundary

The earlier WSL `Exec format error` remains a historical observation. The current
Windows-native `.venv\Scripts\python.exe` executes the repository suite and Ray
2.9 training stack. The optional Ray dashboard package is absent, so the trainer
retries without the dashboard; this does not affect training.

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
