# iTrader-AZR

Autonomous trading research platform inspired by the Absolute Zero Reasoner (AZR) paradigm, adapted to a proposer-solver setting for Reinforcement Learning with Verifiable Rewards (RLVR).

This repository is currently focused on:
- Phase A: stable solver learning with verifiable reward shaping in a controllable trading environment.
- Phase B: integrating an LLM-based proposer and validating AZR-style loop behavior at scale.

## Core Idea

iTrader separates two roles:
- **Solver**: learns trading behavior (currently PPO + custom FiLM-conditioned network).
- **Proposer**: generates tasks/curriculum (currently scaffolded, moving toward LLM-native task generation).

Training follows RLVR principles:
- reward is explicit, inspectable, and decomposed into components,
- tasks are sampled from parameterized specs,
- behavior is optimized under measurable constraints (PnL, drawdown, turnover, etc.).

## Repository Map

- `itrader/envs/itrader_env.py`
  - Main Gymnasium trading environment (`ITraderEnv`).
  - Supports `start_idx_mode` variants (`random_episode`, `stream_worker`, `task_spec`).
  - Includes task-conditioned observations and reward context plumbing.

- `itrader/rl_lib/tasks/rewards.py`
  - Reward registry + reward definitions (`pnl_linear`, `pnl_sharpe`, `composed_linear`).
  - Enforces finite outputs and decomposed reward components.

- `itrader/rl_lib/tasks/spec.py`
  - `TaskSpec` contract (family, horizon, reward config, embedding-related metadata, start index).

- `itrader/rl_lib/tasks/samplers.py`
  - `TaskSampler00` for task generation and optional task pool sampling.

- `itrader/models/solver_mlp_film.py`
  - Current primary solver model (`SolverMLPFiLM`), with FiLM conditioning on task embedding.

- `itrader/rl_lib/train/train_solo_V2.py`
  - Current main training entrypoint (Ray RLlib + PPO, optional PBT/CPU-PBT mode, checkpointing, metrics).

- `itrader/rl_lib/train/train_AZR.py`
  - AZR bootstrap runner (proposer task-pool generation + solver training handoff).

- `itrader/llm/proposer_llm.py` and `itrader/llm/engine.py`
  - Early LLM proposer integration paths.

- `itrader/proposers/`
  - Proposer abstraction layer (`BaseProposer`), backend factory, and deterministic rule-based fallback.

- `itrader/rl_lib/tasks/validation.py`
  - Shared task validation/clamping utility for proposer outputs.

- `PBT_SETUP_GUIDE.md`
  - Operational guide for long-running PBT sweeps and checkpoint workflows.

## Planning and Operations Docs

- `docs/architecture.md`
  - Current baseline architecture and Phase B target architecture.
- `docs/phase_b_plan.md`
  - Concrete milestone plan, acceptance criteria, and risk controls.
- `docs/ops_runbook.md`
  - Main-PC setup, lock workflow, and run/recovery procedures.

## Current Phase Status

### Phase A (mostly complete)
- Learnable reward setup is in place.
- Solver training loop has practical stability guards (sanitization, finite reward handling, callback metrics).
- PBT flow exists with documented operational practices.

### Phase B (active)
- Integrate proposer LLM into the actual training/runtime loop.
- Validate and benchmark `stream_worker` stepping/reset strategy.
- Expand `train_AZR.py` from bootstrap handoff into full proposer-solver co-adaptation loop.

## How To Run (Reference)

Main training script:
- `python -m itrader.rl_lib.train.train_solo_V2 --iters 200 --workers 6 --window 30 --seed 1234 --experiment_name Phase_A`

CPU-safe smoke run (no GPU required):
- `python -m itrader.rl_lib.train.train_solo_V2 --cpu_pbt --iters 3 --workers 2 --window 30 --seed 2026 --experiment_name Phase_A_Smoke --data_path itrader/data/ohlcv.parquet`

Useful start-index/repro flags:
- `--start_idx_mode` (`random_episode`, `stream_worker`, `task_spec`)
- `--ret_norm_reset` (`episode`, `stream`)
- `--stream_jump_every_episodes`, `--stream_stride`, `--stream_jump_on_wrap/--no-stream_jump_on_wrap`

Training outputs (Phase A):
- Ray trial artifacts: `~/ray_results/<experiment_name>/training_fn_*/`
- Stable metrics mirror: `~/ray_results/_stable_metrics/<trial_dir_name>/`
- Stable files:
  - `metrics_full.jsonl` (flattened scalar metrics for robust plotting)
  - `report_rows.csv` (session.report rows for quick comparison scripts)
  - `plots/` (auto-generated plot outputs)

Auto-plotting:
- End-of-run plotting is enabled by default (`--auto_plot_on_end`).
- Manual plotting options:
  - `python itrader/tools/plot_ray_results.py <trial_dir>`
  - `python itrader/tools/plot_ray_results.py --metrics-jsonl ~/ray_results/_stable_metrics/<trial_dir_name>/metrics_full.jsonl`

Experiment scripts:
- Start-index ablation runner:
  - `python -m itrader.scripts.experiments.run_ablation itrader/scripts/experiments/configs/ablation_start_idx.example.json`
- Script outputs:
  - `itrader/scripts/experiments/configs/outputs/<run_tag>/summary.json`
  - `ablation_compare.csv`, `ablation_compare.png`, `ablation_compare_diagnostics.png`

PBT workflow:
- See `PBT_SETUP_GUIDE.md` (CPU exploration -> best-checkpoint extraction -> GPU retraining).

Quality checks (CPU-safe):
- `./itrader/tools/run_repo_python.sh -m pytest -q itrader/tests`
- `./itrader/tools/run_repo_python.sh itrader/tools/healthcheck.py`
- Optional pre-commit:
  - `./itrader/tools/run_repo_python.sh -m pre_commit install`
  - `./itrader/tools/run_repo_python.sh -m pre_commit run --all-files`
  - Hooks also resolve through `./itrader/tools/run_repo_python.sh`, so they do not depend on whichever `python3` is in `PATH`.

Task generation smoke commands (Phase B):
- Rule-based:
  - `./itrader/tools/run_repo_python.sh -m itrader.scripts.gen_task_pool --backend rule_based --out /tmp/itrader_task_pool_rule.jsonl --n_tasks 8 --batch_size 4`
- Ollama:
  - `./itrader/tools/run_repo_python.sh -m itrader.scripts.gen_task_pool --backend ollama --ollama_model qwen2.5:3b --out /tmp/itrader_task_pool_ollama.jsonl --n_tasks 8 --batch_size 4`
- vLLM or SGLang (OpenAI-compatible server on `:8000`):
  - `./itrader/tools/run_repo_python.sh -m itrader.scripts.gen_task_pool --backend vllm --api_base_url http://localhost:8000/v1 --api_model Qwen/Qwen3.5-9B --out /tmp/itrader_task_pool_vllm.jsonl --n_tasks 8 --batch_size 4`
  - `./itrader/tools/run_repo_python.sh -m itrader.scripts.gen_task_pool --backend sglang --api_base_url http://localhost:8000/v1 --api_model Qwen/Qwen3.5-9B --out /tmp/itrader_task_pool_sglang.jsonl --n_tasks 8 --batch_size 4`

Run metadata:
- Each run writes `run_manifest_*.json` under `~/ray_results/<experiment_name>/`.
- Manifest includes git commit, config hash, dataset fingerprint, seed, and runtime mode.

Daily session logging:
- Use `docs/daily_logs/` for ongoing engineering logs.
- Prompts and workflow are documented in `docs/daily_logs/COMMANDS.md`.

Secrets:
- Do not commit API keys in source files.
- Copy `.env.example` and set `POLYGON_API_KEY` in your shell/session before using ingestion scripts.

## Engineering Gaps To Close For Scale

To move from research iteration toward robust long-term development, prioritize:

1) **Configuration discipline**
- Centralize run configs (YAML/TOML + schema validation).
- Minimize hardcoded defaults across train/env/model files.

2) **Clear interface contracts**
- Define strict typed interfaces between proposer, sampler, solver, and environment.
- Keep `TaskSpec` as the canonical cross-module contract.

3) **Testing layers**
- Fast unit tests for reward math and env transitions.
- Integration tests for training config assembly.
- Regression tests for known stability failure modes (NaN/Inf, checkpoint restore).

4) **Observability**
- Standardize metric namespaces and run metadata.
- Track experiment lineage (git commit, config hash, dataset fingerprint, seed strategy).

5) **Reproducibility**
- Explicit seed policies per worker/component.
- Versioned dataset snapshots and deterministic task sampling modes where needed.

6) **Failure isolation**
- Harden retry/recovery around checkpoint I/O and LLM task generation.
- Add clear health checks and run diagnostics scripts for common failure classes.

## Suggested Refactor Track (Branch-Friendly)

Short-term:
- Create a `docs/` directory with:
  - `docs/architecture.md` (AZR adaptation, data/control flow),
  - `docs/phase_b_plan.md` (deliverables, acceptance criteria),
  - `docs/ops_runbook.md` (training + monitoring + recovery).
- Add config module (`itrader/config/`) for typed run specs.
- Add a small test matrix that can run CPU-only in CI.

Mid-term:
- Evolve `train_AZR.py` bootstrap into a full executable Phase B loop wired to current env/task/reward stack.
- Introduce proposer abstraction (`BaseProposer`) with interchangeable backends (rule-based, HF Transformers, llama.cpp).
- Add experiment manifest generation (JSON) per run for traceability.

Long-term:
- Multi-agent curriculum loop where proposer and solver co-adapt under explicit constraints.
- Structured evaluation suite (generalization, robustness, risk profile, behavior consistency).

## Notes

- This environment may not support full GPU training in all development machines; use logic-level checks and surface-level validation locally, then run heavy training on the target GPU workstation.
- Branch (`feat-refactor-codebase`) is focused on maintainability, debuggability, and architecture hardening before scaling experiment volume.
