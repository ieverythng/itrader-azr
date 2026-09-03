# Other — itrader-rl_lib

# Other — itrader-rl_lib

## Overview

This module contains a Prometheus configuration file for monitoring Ray-based applications. It provides the scrape configuration needed to collect metrics from Ray drivers and workers.

## Purpose

The `prometheus_ray.yml` file configures Prometheus to scrape metrics from Ray processes. This enables observability for reinforcement learning workloads distributed across Ray clusters, allowing developers to monitor:

- Task execution metrics
- Memory usage across workers
- Actor lifecycle events
- Custom application metrics exposed via Ray's metrics API

## Configuration Details

### Active Configuration

```yaml
global:
scrape_configs:
  - job_name: ray
    metrics_path: /metrics
    static_configs:
      - targets:
          - "127.0.0.1:8888"
```

**Parameters:**

| Field | Value | Description |
|-------|-------|-------------|
| `job_name` | `ray` | Identifier for this scrape job in Prometheus |
| `metrics_path` | `/metrics` | HTTP endpoint path where Ray exposes metrics |
| `targets` | `127.0.0.1:8888` | Default address for Ray driver metrics |

### Commented Configuration (Workers)

The file includes a commented template for scraping Ray workers:

```yaml
# - job_name: ray_workers
#   metrics_path: /metrics
#   static_configs:
#     - targets:
#         # must match the ports you configured above:
```

This section is disabled by default because worker ports are typically dynamic and environment-specific. Enable this when you need to collect metrics from individual Ray worker processes.

## Usage

### Starting Prometheus with This Configuration

```bash
prometheus --config.file=itrader/rl_lib/prometheus_ray.yml
```

### Integration with Ray

Ray must be configured to expose metrics on the expected port. When initializing Ray in your application:

```python
import ray

ray.init(
    metrics_export_port=8888,  # Must match Prometheus target port
    # ... other init parameters
)
```

### Scaling to Multiple Workers

For distributed workloads with multiple workers, uncomment and configure the `ray_workers` job with the appropriate worker ports. Worker ports are typically assigned dynamically, so you may need to:

1. Configure Ray workers to use fixed ports via `object_store_memory` and related settings
2. Use a service discovery mechanism (e.g., file-based SD or DNS SD) instead of static targets
3. Deploy Prometheus alongside your Ray cluster with dynamic configuration

## Integration Points

This configuration file integrates with:

- **Ray Runtime**: Expects Ray to expose metrics at the configured endpoint
- **Prometheus Server**: Loaded at startup to define scrape behavior
- **Grafana/Alertmanager**: Downstream consumers of the collected metrics

## Related Files

This configuration is typically used alongside:
- Ray application code that exposes custom metrics
- Grafana dashboards for visualizing Ray performance
- Alerting rules for cluster health monitoring
