# Other — env.lock.yml

# Environment Lock File (env.lock.yml)

## Overview

`env.lock.yml` is a **conda environment lock file** that defines the complete, reproducible dependency specification for the `itrader` project. This file ensures consistent environment reproduction across different machines and deployments by pinning exact package versions.

## Purpose

- **Reproducibility**: Guarantees identical package versions across all installations
- **Dependency Resolution**: Pre-resolved dependency tree to avoid conflicts during environment creation
- **Version Control**: Serves as a single source of truth for the project's Python environment

## Usage

### Creating the Environment

```bash
# Create environment from lock file
conda env create -f env.lock.yml

# Activate the environment
conda activate itrader
```

### Updating the Environment

```bash
# Apply changes after modifying the lock file
conda env update -f env.lock.yml --prune
```

### Exporting from Existing Environment

```bash
# Generate a new lock file from current environment
conda env export > env.lock.yml
```

## Dependency Categories

### Core Runtime

| Package | Version | Purpose |
|---------|---------|---------|
| `python` | 3.11 | Language runtime |
| `numpy` | 1.26 | Numerical computing foundation |
| `pandas` | 2.2 | Data manipulation and analysis |
| `pyarrow` | 13.0.0 | Arrow columnar format, Parquet support |
| `duckdb` | 0.10.2 | In-process analytical database |

### Machine Learning & Deep Learning

| Package | Version | Purpose |
|---------|---------|---------|
| `pytorch` | 2.3 | Deep learning framework |
| `torchvision` | 0.18 | Computer vision utilities |
| `transformers` | 4.41.1 | Hugging Face model hub |
| `accelerate` | 0.27.2 | Distributed training utilities |
| `llama-cpp-python` | 0.3.9 | LLM inference bindings |

### Reinforcement Learning

| Package | Version | Purpose |
|---------|---------|---------|
| `ray[rllib]` | 2.47.1 | Distributed RL framework |
| `gymnasium` | ≥0.29,<1.0 | RL environment interface |
| `pettingzoo` | 1.24.1 | Multi-agent RL environments |

### Trading & Finance

| Package | Version | Purpose |
|---------|---------|---------|
| `ta-lib` | (latest) | Technical analysis indicators |
| `vectorbt` | 0.26.0 | Vectorized backtesting |

### Networking & APIs

| Package | Version | Purpose |
|---------|---------|---------|
| `requests` | (latest) | HTTP client |
| `websockets` | (latest) | WebSocket protocol support |

### Data Quality

| Package | Version | Purpose |
|---------|---------|---------|
| `great_expectations` | ≥0.19 | Data validation and profiling |

### Development Tools

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | (latest) | Testing framework |
| `black` | (latest) | Code formatter |
| `isort` | (latest) | Import sorter |

## Version Pinning Strategy

The file uses three pinning approaches:

1. **Exact pin** (`==2.47.1`): Strict version lock for critical dependencies
2. **Minimum version** (`>=0.19`): Allows compatible upgrades
3. **Range constraint** (`>=0.29,<1.0`): Accepts minor/patch updates within major version

## Channels

Dependencies are resolved from two channels in order:

1. **conda-forge**: Community-maintained packages (primary source)
2. **pytorch**: Official PyTorch distributions

Channel order matters for package resolution. Packages are pulled from the first channel that contains them.

## Mixed Dependency Sources

This configuration uses both conda and pip:

```yaml
dependencies:
  - pip  # Required to enable pip installs
  - pip:
      - ray[rllib]==2.47.1  # pip-only packages
```

Pip dependencies are installed after conda packages resolve. Use pip for packages unavailable in conda channels or when specific extras (like `[rllib]`) are required.

## Maintenance Notes

### When to Update

- Security vulnerabilities in dependencies
- Required features in newer versions
- Compatibility with new Python versions

### Regenerating the Lock File

```bash
# After modifying env.yml (flexible spec)
conda env create -f env.yml
conda activate itrader
conda env export > env.lock.yml
```

### Common Issues

| Issue | Resolution |
|-------|------------|
| Platform mismatch | Regenerate lock file on target platform |
| Package not found | Check channel availability or use pip |
| Version conflict | Use `conda config --set allow_conda_downgrades true` |
