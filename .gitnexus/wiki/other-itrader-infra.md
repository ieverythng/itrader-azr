# Other — itrader-infra

# itrader-infra

Infrastructure configuration module for deploying the iTrader system using Ray cluster orchestration.

## Overview

This module contains configuration files for provisioning and managing Ray clusters. Currently, it provides a local development configuration optimized for single-machine development with GPU support for reinforcement learning workloads.

## Configuration Files

### cluster.yaml

Ray cluster configuration for local development environments.

#### Key Settings

| Setting | Value | Purpose |
|---------|-------|---------|
| `cluster_name` | `itrader-local` | Identifier shown in Ray dashboard |
| `head_resources.CPU` | 4 | CPU cores reserved for head node |
| `head_resources.GPU` | 1 | GPU reserved for learner/training |
| `min_workers` | 0 | No worker nodes in local dev |
| `max_workers` | 0 | Head-only deployment |

#### Docker Configuration

```yaml
docker:
  image: pytorch/pytorch:2.3.0-cuda12.2-cudnn8-runtime
```

Uses the official PyTorch runtime image with:
- PyTorch 2.3.0
- CUDA 12.2
- cuDNN 8

This image provides GPU-accelerated PyTorch without development tools, keeping container size minimal.

#### File Mounts

```yaml
file_mounts:
  /workspace/iTrader: .
```

Syncs the repository root to `/workspace/iTrader` inside the container. This enables live code changes without rebuilding the image.

#### Setup Commands

Installs required Python packages on cluster startup:

| Package | Purpose |
|---------|---------|
| `ray[air]==2.47` | Ray with AIR (RL/training APIs) |
| `gymnasium` | RL environment interface |
| `pandas` | Data manipulation |
| `duckdb` | Embedded analytical database |
| `ta` | Technical analysis indicators |

## Usage

### Starting the Cluster

```bash
ray up itrader/infra/cluster.yaml
```

### Accessing the Dashboard

After startup, the Ray dashboard is available at `http://localhost:8265` showing the `itrader-local` cluster.

### Stopping the Cluster

```bash
ray down itrader/infra/cluster.yaml
```

## Design Decisions

### Head-Only Architecture

The local configuration runs all workloads on the head node (`max_workers: 0`). This simplifies development by:

1. Eliminating worker node provisioning overhead
2. Providing direct access to logs and debugging
3. Reducing resource requirements for development machines

### GPU Reservation

The head node reserves 1 GPU specifically for the learner component. This ensures training workloads have guaranteed GPU access without competing with other Ray actors.

## Integration Points

This configuration supports the broader iTrader system:

- **Ray AIR**: Provides distributed training infrastructure for RL agents
- **Gymnasium environments**: Trading environments run as Ray actors
- **Learner component**: Consumes GPU resources for policy optimization
- **Data pipeline**: DuckDB and pandas support market data processing

## Extending for Production

For production deployments, create additional cluster configurations with:

```yaml
min_workers: 2
max_workers: 10
worker_default_resources:
  CPU: 4
```

Production configs should be added as separate files (e.g., `cluster-aws.yaml`, `cluster-gcp.yaml`) in this directory.
