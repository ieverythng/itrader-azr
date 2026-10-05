# Ray 2.58 compatibility environment

This directory is intentionally independent of the repository root lock. It was
the migration acceptance environment used before Ray 2.58 was promoted into the
authoritative root `pyproject.toml` and `uv.lock`; it remains a reproducible
fixture for reviewing that resolution. It is an environment profile, not the
source namespace for ModelV2/TorchPolicy adapters; those live under
`../rllib_legacy/`.

Create or refresh the compatibility environment from the repository root:

```powershell
uv sync --project itrader/compat/ray258 --frozen
uv run --project itrader/compat/ray258 python -c "import ray; print(ray.__version__)"
```

The compatibility environment imports the working tree directly when commands
run from the repository root. Historical Ray 2.9 checkpoints remain immutable
evidence and must not be rewritten in place.
