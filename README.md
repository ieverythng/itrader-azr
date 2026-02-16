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
  - AZR loop skeleton (conceptual structure, not yet updated to latest architecture/runtime).

- `itrader/llm/proposer_llm.py` and `itrader/llm/engine.py`
  - Early LLM proposer integration paths.

- `PBT_SETUP_GUIDE.md`
  - Operational guide for long-running PBT sweeps and checkpoint workflows.

## Current Phase Status

### Phase A (mostly complete)
- Learnable reward setup is in place.
- Solver training loop has practical stability guards (sanitization, finite reward handling, callback metrics).
- PBT flow exists with documented operational practices.

### Phase B (active)
- Integrate proposer LLM into the actual training/runtime loop.
- Validate and benchmark `stream_worker` stepping/reset strategy.
- Update and operationalize `train_AZR.py` to reflect real proposer-solver interaction.

## How To Run (Reference)

Main training script:
- `python -m itrader.rl_lib.train.train_solo_V2 --iters 200 --workers 6 --window 30 --seed 1234`

PBT workflow:
- see `PBT_SETUP_GUIDE.md` (CPU exploration -> best-checkpoint extraction -> GPU retraining).

Quality checks (CPU-safe):
- `python -m pytest -q itrader/tests`
- `python itrader/tools/healthcheck.py`
- optional pre-commit: `pre-commit install && pre-commit run --all-files`

Run metadata:
- each run now writes a manifest under `~/ray_results/Phase_A/` with git commit, config hash, dataset fingerprint, and seed.

Daily session logging:
- use `docs/daily_logs/` for ongoing engineering logs.
- prompts and workflow are documented in `docs/daily_logs/COMMANDS.md`.

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
- Replace `train_AZR.py` skeleton with an executable Phase B loop wired to current env/task/reward stack.
- Introduce proposer abstraction (`BaseProposer`) with interchangeable backends (rule-based, HF Transformers, llama.cpp).
- Add experiment manifest generation (JSON) per run for traceability.

Long-term:
- Multi-agent curriculum loop where proposer and solver co-adapt under explicit constraints.
- Structured evaluation suite (generalization, robustness, risk profile, behavior consistency).

## Notes

- This environment may not support full GPU training in all development machines; use logic-level checks and surface-level validation locally, then run heavy training on the target GPU workstation.
- Branch (`feat-refactor-codebase`) is focused on maintainability, debuggability, and architecture hardening before scaling experiment volume.
