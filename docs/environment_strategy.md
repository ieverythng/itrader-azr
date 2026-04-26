# Environment Strategy

This repo now has two supported environment surfaces:

1. `pyproject.toml` for uv/pip style reproducibility.
2. `itrader/environment.yml` for Conda-first workstation setup, especially when native packages such as TA-Lib or CUDA-sensitive PyTorch installs are easier through Conda.

## Preferred New Workflow

Install uv, then lock and sync from the repo root:

```powershell
python -m pip install uv
uv lock
uv sync --group dev
```

Validate:

```powershell
uv run python itrader/tools/healthcheck.py
uv run python -m pytest -q itrader/tests
```

Run pre-commit without relying on Conda being available on `PATH`:

```powershell
uv run pre-commit run --all-files
```

## Current Main-PC Reality

The current validated workstation environment reports:

- Python `3.11.13`
- NumPy `1.26.4`
- pandas `2.1.4`
- Ray `2.9.0`
- Gymnasium `1.0.0`
- DuckDB `0.10.2`
- llama-cpp-python `0.3.9`
- CUDA available on `NVIDIA GeForce RTX 5070 Ti`

The root `pyproject.toml` is aligned to the resolver-consistent training stack where practical. One important mismatch exists in the current Conda environment: `ray[rllib]==2.9.0` resolves with `gymnasium==0.28.1`, but the live environment currently contains `gymnasium==1.0.0`. The uv lock follows the Ray-compatible pin rather than reproducing the inconsistent environment.

Ray `2.9.0` also imports `pkg_resources._vendor.packaging`, so `setuptools` is pinned to `69.5.1`. Newer setuptools versions expose different compatibility behavior and break Ray during test collection.

PyTorch is pinned as a stable pip range in `pyproject.toml`. The current workstation is using a CUDA/nightly-style build, so GPU-specific PyTorch installation may still need a machine-local install command or a dedicated extra/index strategy later.

Optional dependency groups:

- `dev`: tests, formatters, pre-commit.
- `etl`: TA/TA-Lib data feature tooling.
- `llm-local`: in-process local LLM bindings and Hugging Face tooling.
- `research`: heavier analysis/backtesting/data validation packages not required for core training smoke tests.

The current proposer path for Qwen via llama.cpp uses HTTP through `itrader.proposers.openai_compat`, so `llama-cpp-python` is not required for normal task-pool generation. Keep it optional unless you need in-process GGUF inference.

## Conda Fallback

For machines where native dependencies are easier through Conda:

```powershell
conda env create -f itrader/environment.yml
conda activate itrader
python itrader/tools/healthcheck.py
python -m pytest -q itrader/tests
```

After a validated Conda install, refresh:

```powershell
conda env export --from-history > env.lock.yml
```

## Before Task-Family Expansion

Before family `01` or `02` implementation starts:

1. Generate and commit `uv.lock`.
2. Reconcile `itrader/environment.yml` with the locked Python dependency set.
3. Run healthcheck, full tests, pre-commit, and one `train_solo_V2` smoke from a PBT checkpoint.
