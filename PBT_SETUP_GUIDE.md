 # PBT Setup Guide for Long-Running Sweeps

## Quick Start Checklist

### Before Starting PBT:
- [ ] **CPU PBT Mode**: Use `--cpu_pbt` flag for multi-trial exploration
- [ ] **GPU Retraining**: After CPU PBT completes, retrain best checkpoint on GPU
- [ ] **Timeout**: Set `--timeout_hours` appropriately (8-24 hours for full sweeps)
- [ ] **Population Size**: Start with `--pop 10` for initial exploration, scale up if needed

### Recommended Command:
```bash
# Phase 1: CPU PBT exploration (many trials)
python -m itrader.rl_lib.train.train_solo_V2 \
    --cpu_pbt \
    --pbt \
    --pop 10 \
    --workers 4 \
    --iters 200 \
    --timeout_hours 12 \
    --window 30

# Phase 2: GPU retraining of best checkpoint
# (extract best checkpoint path from ray_results, then retrain without --pbt)
```

---

## Checkpoint Management

### Where Checkpoints Are Saved:
- **Trial checkpoints**: `~/ray_results/Phase_A/<trial_id>/rllib_checkpoints/`
- **Best trial**: Check `analysis.best_trial` after completion
- **Checkpoint interval**: Every 3 iterations (configurable via `checkpoint_interval`)

### Extracting Best Checkpoint:

```python
import ray
from ray import tune
from pathlib import Path

# Load results (if Ray session still active)
analysis = tune.ExperimentAnalysis("~/ray_results/Phase_A")

# Get best trial
best_trial = analysis.best_trial
print(f"Best trial: {best_trial.trial_id}")
print(f"Best checkpoint: {best_trial.checkpoint}")

# Or manually find best:
import pandas as pd
df = analysis.results_df
best_row = df.loc[df['custom_metrics/episode_return_mean'].idxmax()]
print(f"Best trial ID: {best_row['trial_id']}")
print(f"Best return: {best_row['custom_metrics/episode_return_mean']}")

# Checkpoint path structure:
# ~/ray_results/Phase_A/<trial_id>/rllib_checkpoints/checkpoint_<iter>
```

### Retraining Best Checkpoint:

```python
from ray.rllib.algorithms.ppo import PPO
import ray

ray.init()

config = {...}  # Your base config (without PBT)
algo = PPO(config=config)

# Restore from best checkpoint
checkpoint_path = "~/ray_results/Phase_A/<trial_id>/rllib_checkpoints/checkpoint_<iter>"
algo.restore(checkpoint_path)

# Continue training
for i in range(additional_iters):
    result = algo.train()
    print(f"Iter {i}: {result['custom_metrics/episode_return_mean_mean']}")

algo.stop()
```

---

## Monitoring Long Runs

### 1. **Ray Dashboard**
- Access at `http://localhost:8265` (default)
- Monitor: trial progress, resource usage, failures
- **Key tabs**: Experiments, Actors, Logs

### 2. **Progress Files**
- Each trial writes: `~/ray_results/Phase_A/<trial_id>/progress.csv`
- Monitor with: `tail -f ~/ray_results/Phase_A/*/progress.csv`
- Or use: `ray.tune.ExperimentAnalysis` to load mid-run

### 3. **Log Files**
- Trial logs: `~/ray_results/Phase_A/<trial_id>/` (stdout/stderr)
- Look for: checkpoint saves, restore messages, errors

### 4. **Resource Monitoring**
```bash
# CPU usage across trials
top -p $(pgrep -f "python.*train_solo_V2")

# Memory usage
watch -n 5 'free -h'

# Disk space (checkpoints can be large)
df -h ~/ray_results
```

---

## Troubleshooting

### Common Issues:

#### 1. **Port Conflicts (Prometheus)**
- **Symptom**: `Address already in use` errors
- **Fix**: Prometheus is automatically disabled for PBT (multiple trials would conflict)
- **Verify**: Check `metrics_sinks` is empty when `--pbt` is set

#### 2. **Checkpoint Save Failures**
- **Symptom**: `⚠️ Checkpoint save failed` messages
- **Cause**: Disk space, permissions, or path issues
- **Fix**: Check disk space, ensure `~/ray_results` is writable
- **Impact**: Training continues, but PBT weight copying may fail

#### 3. **Trial Failures**
- **Symptom**: Trials crash after a few iterations
- **Monitor**: Check `max_failures=3` (allows 3 retries)
- **Debug**: Check trial logs in `~/ray_results/Phase_A/<trial_id>/`
- **Common causes**: OOM, NaN gradients, environment errors

#### 4. **Plot Directory Flooding**
- **Symptom**: Too many plot directories created
- **Fix**: `MetricsOnlyCallback` used for PBT (no plots), `CombinedCallback` only for single-trial
- **Verify**: No `plots/` directories in PBT trial dirs

#### 5. **Memory Issues with Many Trials**
- **Symptom**: System becomes slow, OOM errors
- **Fix**:
  - Reduce `--pop` size
  - Reduce `--workers` per trial
  - Use `reuse_actors=False` (already set) to avoid memory leaks

#### 6. **Checkpoint Restoration Issues**
- **Symptom**: `ℹ️ No checkpoint to restore` on every iteration
- **Expected**: First iteration of each trial has no checkpoint
- **Problem**: If later iterations show this, checkpointing may be failing
- **Check**: Verify `checkpoint_interval=3` and checkpoint directories exist

---

## Best Practices for Long Runs

### 1. **Pre-Run Validation**
```bash
# Quick test run (3 iterations, 1 trial)
python -m itrader.rl_lib.train.train_solo_V2 \
    --pbt \
    --pop 1 \
    --iters 3 \
    --workers 2 \
    --timeout_hours 1

# Verify:
# - Checkpoints are created
# - Metrics are logged
# - No errors
```

### 2. **Resource Planning**
- **CPU PBT**: 10 trials × 4 workers = 40+ CPU cores needed
- **Memory**: ~2-4GB per trial (workers + model)
- **Disk**: ~100MB per checkpoint, ~1GB per trial (full run)

### 3. **Early Stopping Strategy**
- Use `TimeoutStopper` (already configured)
- Monitor best trial performance; can manually stop if plateaus
- Consider `tune.stopper.MaxIterStopper` for iteration limits

### 4. **Checkpoint Frequency**
- **Current**: Every 3 iterations (good for PBT)
- **Too frequent**: Wastes disk I/O
- **Too rare**: PBT can't exploit good weights often enough

### 5. **Recovery After Interruption**
- Ray saves progress automatically
- Resume by re-running same command (will continue from last checkpoint)
- Or extract best checkpoint and retrain manually

---

## Extracting Results

### After PBT Completes:

```python
from ray import tune
from pathlib import Path

# Load analysis
analysis = tune.ExperimentAnalysis("~/ray_results/Phase_A")

# Get results dataframe
df = analysis.results_df

# Sort by best return
df_sorted = df.sort_values('custom_metrics/episode_return_mean', ascending=False)

# Top 5 trials
print(df_sorted[['trial_id', 'custom_metrics/episode_return_mean', 'config']].head(5))

# Best hyperparameters
best_config = analysis.best_config
print("Best hyperparameters:", best_config)

# Plot comparison (if matplotlib available)
analysis.get_best_trial('custom_metrics/episode_return_mean', 'max').logdir
```

### Exporting Best Checkpoint:

```bash
# Find best trial
BEST_TRIAL_ID=$(python -c "
from ray import tune
analysis = tune.ExperimentAnalysis('~/ray_results/Phase_A')
best = analysis.best_trial
print(best.trial_id)
")

# Copy best checkpoint
cp -r ~/ray_results/Phase_A/$BEST_TRIAL_ID/rllib_checkpoints/checkpoint_* ./best_checkpoint/
```

---

## Performance Optimization Tips

### For CPU PBT:
1. **Reduce workers per trial**: `--workers 2-4` (instead of 6)
2. **Smaller batch sizes**: Reduces memory per trial
3. **Fewer trials**: Start with `--pop 5-8`, scale up if needed

### For GPU Retraining:
1. **Larger batch sizes**: Use GPU memory efficiently
2. **More workers**: `--workers 6-8` for better throughput
3. **Longer training**: Let it run 500-1000 iterations

### Monitoring Resource Usage:
- **CPU**: Keep below 90% to avoid thrashing
- **Memory**: Monitor swap usage (should be ~0)
- **Disk I/O**: Checkpoint saves can be slow on HDD

---

## Expected Timeline

### CPU PBT (10 trials, 200 iters each):
- **Per iteration**: ~10-30 seconds (CPU, 4 workers)
- **Full run**: ~6-12 hours (depending on CPU cores)
- **Checkpoint saves**: ~100MB each, ~33 checkpoints per trial

### GPU Retraining (single trial, 500 iters):
- **Per iteration**: ~5-10 seconds (GPU, 6 workers)
- **Full run**: ~1-2 hours
- **Final checkpoint**: Ready for Phase B proposer integration

---

## Next Steps After PBT

1. **Extract best checkpoint**
2. **Validate on held-out data** (if available)
3. **Analyze hyperparameter correlations** (which params matter?)
4. **Start Phase B proposer integration** with optimized baseline

Good luck! 🚀
