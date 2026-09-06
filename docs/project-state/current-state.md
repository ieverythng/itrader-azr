# Current State

Last updated: 2026-09-03

## Branch Position

The active branch is `feat/codex-adversarialproposer` at `6714b22`. Before this
working-tree pass it was five commits ahead of `origin/Phase-B-Training` and had
no divergence. The latest commits add Sharpe/recurrent research and ablation
scaffolding; they do not add a recurrent solver implementation.

The GitNexus index is stale at `1d5ba82` and the local query wrapper reports no
indexed repository. A refresh was attempted but exited with only upstream
tree-sitter deprecation output and no completed index. Source, tests, configs,
and active docs were used directly.

## Implemented Baseline

- Main PPO entrypoint: `itrader/rl_lib/train/train_solo_V2.py`.
- Default solver: `SolverMLPFiLM`.
- Main environment: `ITraderEnv`, emitting a chronological flattened market
  window plus task embedding.
- Active family: `00`, `directional_predictor`, with three actions.
- Default reward: `pnl_linear`.
- Candidate proposer endpoints: Ollama and OpenAI-compatible local servers,
  including llama.cpp.
- Future contracts: `CandidateTask`, `ExecutionIntent`, `CapabilityProfile`, and
  `DecisionRecord`.

## This Working-Tree Pass

- Kept proven `pnl_linear` as the sole default, replaced placeholder
  `pnl_sharpe` with an explicit PnL+incremental-Sharpe blend, and added a
  separate experimental `sharpe_only` reward.
- Added episode strategy-return history to the environment reward context.
- Added reward tests for warmup, boundedness, and component accounting plus an
  environment plumbing test.
- Added `--task_horizons` to `train_solo_V2` for controlled ablations.
- Added `--task_reward_ids` and a fixed-horizon PnL-versus-Sharpe experiment
  config.
- Split task/proposer reward-parameter RNG from task-identity RNG so changing a
  reward does not silently change sampled horizons or market-window seeds.
- Repaired the Lazarus runner invocation.
- Changed the alignment design from four cells with a duplicate baseline to
  five cells, including a matched random-128 versus stream-128 comparison.
- Corrected the Sharpe and recurrent research docs and their HTML companions.

## Important Non-Capabilities

- `SolverLSTMModel` is a reset-every-step window encoder and incorrectly infers
  market feature width from an observation that also contains the task embed.
- `SolverWindowLSTMFiLM` and a native RLlib stateful recurrent FiLM head remain
  unimplemented.
- The current solver observation does not expose position, previous action,
  reward, unrealized PnL, or episode progress. Recurrence alone cannot be
  described as a complete trade-state tracker.
- Families `01` (`breakout_expansion`) and `02` (`mean_reversion`) are disabled
  bootstrap contracts. Family `03` is trade management, not trend following.
- Capability reports are not yet family/regime calibrated unless task metadata
  is present in the metrics records.
- No Sharpe or recurrent training result has been produced by this pass.

## Validation State

The recreated lock-synchronized `.venv` passes healthcheck and all 67 tests.
Pre-commit passes all hooks through the shared launcher, which now prefers the
repo `.venv` over legacy or ambient interpreters.
About 52 GB is free on C:. The ambient non-project interpreter remains
ABI-incompatible (NumPy 2.3.5 versus its installed pandas/pyarrow binaries), so
repository commands should continue to use `.venv` or the runner script.

## Highest-ROI Sequence

1. Run the five-cell alignment ablation.
2. Run matched `pnl_linear`, `sharpe_only`, and `pnl_sharpe` cells on MLP-FiLM.
3. Repair metrics metadata for capability slices.
4. Implement and measure R1 window-LSTM-FiLM.
5. Implement R2a recurrence, then isolate per-step FiLM as R2b.
6. Lock families `01` and `02` only after those measurements are credible.
