# Handoff

Last updated: 2026-09-03

## Resume Point

The branch audit and first Sharpe/alignment correction are in the working tree.
No recurrent model was implemented. The pinned `.venv` and repository hooks are
validated; review the diff before launching training.

## Working-Tree Changes

- Reward candidates: proven/default `pnl_linear`, experimental `sharpe_only`,
  and experimental `pnl_sharpe` blend.
- `ITraderEnv`: episode strategy-return history in reward context.
- `train_solo_V2`: optional comma-separated `--task_horizons` override.
- `train_solo_V2`: optional comma-separated `--task_reward_ids` override.
- Task and rule-based proposer reward RNGs are separated from task-identity RNGs
  so reward comparisons preserve sampled horizons and task seeds.
- Sharpe reward config:
  `itrader/scripts/experiments/configs/sharpe_reward_ablation.json`.
- Alignment config: five cells, including random-128 versus stream-128.
- Lazarus launcher: corrected positional config invocation.
- Tests: reward behavior, environment plumbing, horizon override, and matched
  alignment pair.
- Docs: corrected Sharpe/recurrent research, masterplan, decisions, current
  state, next actions, and HTML companions.
- Environment tooling: repository hooks and the shared Python launcher now
  resolve the repo `.venv` instead of relying on ambient `python`.

## Validation So Far

- Direct reward seam execution: pass.
- `bash -n` on Lazarus launcher: pass.
- `git diff --check`: pass.
- The recreated lock-synchronized `.venv` passes healthcheck and all 67 tests.
- `pre-commit run --all-files`: pass, including repository healthcheck and core
  tests through `.venv`.
- C: has about 52 GB free. The ambient interpreter still has NumPy 2.3.5 with
  incompatible pandas/pyarrow binaries; use the project environment.
- GitNexus: stale index and no local indexed repository available. Refresh was
  attempted but exited nonzero after upstream tree-sitter deprecation warnings.

## Architectural Conclusions

- `SolverMLPFiLM` remains the default and control.
- The legacy LSTM is a broken-for-current-observation window encoder, not a
  stateful policy.
- Recurrence should be measured as R0 -> R1 -> R2a -> R2b, separating window
  encoding, state persistence, and FiLM.
- The current observation lacks explicit position and episode progress; test
  those later as matched features.
- `pnl_linear` is the sole proven/default reward. Both Sharpe candidates are
  experimental and have no family default assignment.
- Canonical families are 00 directional, 01 breakout, 02 mean reversion, 03
  trade management, and 04 abstention.

## Commands

```bash
uv sync --frozen --group dev
./itrader/tools/run_repo_python.sh -m pytest -q itrader/tests/test_rewards.py itrader/tests/test_env_core.py itrader/tests/test_train_config.py
./itrader/tools/run_repo_python.sh itrader/tools/healthcheck.py
./itrader/tools/run_repo_python.sh -m pytest -q itrader/tests
./itrader/tools/run_repo_python.sh -m pre_commit run --all-files
bash itrader/scripts/experiments/run_ablation_lazarus.sh
```

Run llama.cpp and PPO training separately on the 16 GB target when the model
cannot coexist with training memory.

## Next Experiment

Complete the five-cell alignment ablation on MLP-FiLM. Then run the matched
three-way reward experiment from `sharpe_reward_ablation.json` before any
recurrent implementation or family default changes.
