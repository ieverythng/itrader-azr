# Environment Strategy

This document describes environment authority and execution boundaries after the
Ray 2.58 migration.

## Dependency authority

The source-of-truth dependency declarations are `pyproject.toml` and `uv.lock`:

- Python `>=3.11,<3.12`
- Ray/RLlib `2.58.0`
- Gymnasium `1.2.2`
- DuckDB `0.10.2`
- PyArrow `17.0.0`
- setuptools `69.5.1`
- PyTorch `>=2.3,<2.4`

The old Ray/RLlib `2.9.0` and `2.47.1` lines in dated/historical documents are
reproduction history, not API authority. `itrader/compat/ray258/` is the isolated
acceptance fixture used before promoting these versions into the root lock.

## Supported execution surfaces

For a Windows-native checkout, use PowerShell with a Windows Python/uv
installation and the repo lock. The Ray 2.58 local control, PBT restore, and
fresh-process replay pass on that surface. Ray officially labels Windows support
beta and does not test multi-node Windows clusters, so use Linux/WSL2 or the
pinned Linux container for cluster and unattended long-run validation. WSL2 was
not tested in the migration workstream.

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

## RLlib architecture migration prerequisite

The dependency migration does not silently promote the new RLlib architecture.
R0 explicitly selects the legacy ModelV2/RolloutWorker stack on Ray 2.58 until an
RLModule/EnvRunner/Learner/ConnectorV2 port passes tensor/update parity, metrics
parity, deterministic checkpoint replay, and a real PBT exploit/restore. Any
future recurrent contract must additionally cover sequence lengths, padding
masks, last valid h/c, terminal/truncated bootstrap, fragment detach,
vector-worker state ownership, training/inference parity, and checkpoint/replay
state.

## Current validation boundary

The earlier WSL `Exec format error` remains a historical observation. The current
Windows-native `.venv\Scripts\python.exe` executes the repository suite and Ray
2.58 training stack. The trainer runs without the optional dashboard.

Repository-owned run manifests/status are
written atomically under `~/ray_results/_stable_runs`, and per-trial config
history/status/checkpoint indices/metrics live under `_stable_metrics`. Treat
those stable paths as provenance authority while keeping Tune directories for
checkpoints and diagnostic inspection.

Ray 2.58 emitted child-process access-violation messages during Windows teardown
even though trials and replay completed. Treat them as a reason to validate long
runs on Linux/WSL2, not as a failed metric/checkpoint acceptance or as permission
to ignore a future nonzero driver exit.

The detailed migration matrix and acceptance receipts are recorded in
`artifacts/research/ray-2.58-migration-2026-09-14.md` and
`itrader/compat/ray258/evidence/`.
