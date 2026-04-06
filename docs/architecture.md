# iTrader Architecture (Phase A Baseline -> Phase B Target)

## Current Baseline (February 24, 2026)

Primary executable training path:

- `itrader/rl_lib/train/train_solo_V2.py`
- `itrader/envs/itrader_env.py`
- `itrader/rl_lib/tasks/rewards.py`
- `itrader/rl_lib/tasks/samplers.py`
- `itrader/models/solver_mlp_film.py`
- `itrader/rl_lib/callbacks/ps_metrics.py`
- `itrader/rl_lib/callbacks/auto_plotter.py`

This path is the effective source of truth for Phase A and experiment operations.

## Runtime Components

1. Data

- Input parquet: `itrader/data/ohlcv.parquet`
- Environment loads OHLCV window features and close prices for reward context.

1. Environment

- `ITraderEnv` emits observation = `flattened_price_window + task_embedding`.
- Start-index modes:
  - `random_episode`
  - `stream_worker`
  - `task_spec`
- Reward computed via registry (`RewardEngine`) using inspectable per-step context.

1. Task Layer

- `TaskSpec` is the contract between sampler/environment/proposer.
- `TaskSampler00` currently generates task specs stochastically and can optionally sample from a persistent `TaskPool`.

1. Solver

- PPO (RLlib) with custom FiLM-conditioned model (`SolverMLPFiLM`).
- Training loop supports PBT, checkpointing, stable metrics mirror, and final plotting.

1. Observability and Experimentation

- Stable metrics mirror:
  - `~/ray_results/_stable_metrics/<trial>/metrics_full.jsonl`
  - `~/ray_results/_stable_metrics/<trial>/report_rows.csv`
- Ablation harness:
  - `itrader/scripts/experiments/run_ablation.py`

## Architectural Gaps Blocking Phase B

1. AZR loop executable gap

- `itrader/rl_lib/train/train_AZR.py` is conceptual pseudocode, not runnable integration.

1. Proposer integration gap

- LLM proposer exists (`itrader/llm/proposer_llm.py`) but is not wired into training/runtime loop.

1. Contract and ownership clarity

- `TaskSpec` exists but module boundaries for proposer/sampler/solver runtime ownership are not fully explicit.

1. Dependency and environment drift

- Multiple manifests with divergent version intent make reproducibility and onboarding unstable.

1. Legacy surface area

- Legacy/stale entrypoints and cached artifacts increase confusion and accidental usage risk.

## Target Phase B Architecture

Control loop target (single process first, distributed second):

1. Proposer generates `TaskSpec` candidates from market state summary + curriculum state.
2. Validator clamps/spec-checks candidate tasks.
3. Sampler selects task(s) from proposer stream + optional pool mixing policy.
4. Solver trains/evaluates on selected tasks.
5. Outcome feedback (`return`, `success`, risk stats) updates proposer memory and task pool stats.
6. All phases emit versioned run metadata and comparable metrics.

## Immediate Refactor Principles

1. Single executable truth for each workflow.
2. Typed contracts before additional model complexity.
3. Deterministic and inspectable control paths.
4. Dependency lock process documented and reproducible on target machine.
