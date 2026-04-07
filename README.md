# iTrader-AZR

iTrader is a trading research stack following an Absolute Zero Reasoner (AZR)-style proposer-solver design under Reinforcement Learning with Verifiable Rewards (RLVR).

## Current Status (March 4, 2026)

- Phase A is operational and is the known-good training baseline.
- Phase B proposer infrastructure is active (rule-based, Ollama, OpenAI-compatible backends).
- Task generation is intentionally constrained right now:
  - `family_id` must be `00`
  - `reward_id` must be `pnl_linear`
- Additional task families and reward functions are defined as roadmap specs, but are not yet enabled for runtime generation.

## Known-Good Baseline Loop (Pre-Proposer-In-Loop)

This is the stable reference loop before full proposer-solver co-adaptation:

- Entry point: `itrader/rl_lib/train/train_solo_V2.py`
- Environment: `itrader/envs/itrader_env.py`
- Sampler: `TaskSampler00` (`family_id=00`)
- Reward path in use: `reward_pnl_linear` (`reward_id="pnl_linear"`)
- Solver: PPO + FiLM-conditioned MLP (`SolverMLPFiLM`)

Reference smoke run:

```bash
python -m itrader.rl_lib.train.train_solo_V2 --cpu_pbt --iters 3 --workers 2 --window 30 --seed 2026 --experiment_name Phase_A_Smoke --data_path itrader/data/ohlcv.parquet
```

Use this as the control run when debugging proposer integration.

## Task Family Plan

Task families are now formalized in `itrader/rl_lib/tasks/families.py` for modular expansion.

| Fam ID | Scope | Extra features (planned) | First target learner | Runtime enabled now |
|---|---|---|---|---|
| 00 | Single-instrument intraday directional | `rsi14`, `sma20`, `sin_tod`, `cos_tod` | solver_mlp_film | Yes |
| 01 | High-vol breakout | `atr14`, `bb_width` | solver_mlp_film | No |
| 02 | News-driven jump | `news_sentiment` merge via NewsAPI | LLM proposer | No |
| 03 | Mean-revert pairs | `spread_zscore(sym_a, sym_b)` | solver_mlp_film | No |
| 04 | Portfolio risk | per-asset beta, correlation matrix slice | Both | No |

Current gating behavior:

- Validation clamps non-enabled families to `00`.
- Validation clamps non-approved rewards to `pnl_linear`.
- Proposer prompts explicitly instruct models to generate only `family_id="00"` and `reward_id="pnl_linear"`.

## Reward Function Status

Reward registry exists in `itrader/rl_lib/tasks/rewards.py`.

- Active runtime reward:
  - `pnl_linear` (`reward_pnl_linear`)
- Implemented but not yet production-validated in your current workflow:
  - `pnl_sharpe`
  - `composed_linear`

The training and task-generation path is currently constrained to `pnl_linear` until the proposer is fully integrated in-loop and reward expansion is explicitly tested.

## Phase B Scope (Current)

Phase B objective is to move from bootstrap proposer usage to full AZR proposer-solver adaptation:

1. Generate validated task pools from proposer backends.
2. Feed tasks safely into solver training without destabilizing Phase A guarantees.
3. Extend from bootstrap handoff into the full AZR loop where proposer updates are trained from verifiable signals, including `r_prop`.
4. Keep strict validation, reproducibility, and rollback path to solver-only baseline.

AZR-aligned intent for this phase:

- task proposal is optimized for learnability (not static human-authored curricula),
- rewards remain environment-verifiable,
- solver and proposer are optimized jointly as the loop matures.

## AZR Theory Alignment

Absolute Zero Reasoner (AZR) reframes RLVR as self-play over verifiable environments:

1. The model proposes tasks instead of consuming fixed human-curated QA distributions.
2. The model solves proposed tasks and is trained from verifiable outcome signals.
3. Curriculum quality is measured by learning progress, enabling iterative self-improvement.

iTrader mapping:

- `proposer`: emits `TaskSpec` candidates conditioned on market context.
- `solver`: executes policy under the sampled task.
- `verifier`: environment + reward engine provide inspectable, decomposed reward signals.
- Phase B extension: add proposer-side signal (`r_prop`) and optimize the joint loop.

Primary docs:

- `docs/phase_b_plan.md`
- `docs/architecture.md`
- `docs/ops_runbook.md`
- `docs/phase_b_handoff_2026-03-04.md`

## Phase C Direction (Planned)

Planned Phase C expands architecture to dual-LLM proposer-solver research:

1. Replace or augment current solver with an LLM-based solver path.
2. Scale the already introduced Phase-B AZR loop (`r_prop` + solver reward) into a stronger dual-LLM regime.
3. Optimize joint RLVR objective where solver and proposer co-adapt under verifiable constraints.
4. Evaluate scalability toward broader iTrader orchestration (potential future agentic workflows).

This is roadmap intent, not current production behavior.

## Repository Map

- `itrader/envs/itrader_env.py`: main Gymnasium trading environment.
- `itrader/rl_lib/tasks/spec.py`: `TaskSpec` contract.
- `itrader/rl_lib/tasks/families.py`: task family registry and active constraints.
- `itrader/rl_lib/tasks/validation.py`: proposer output clamping and safety normalization.
- `itrader/rl_lib/tasks/rewards.py`: reward registry and reward implementations.
- `itrader/rl_lib/tasks/samplers.py`: `TaskSampler00` and task-pool sampling.
- `itrader/proposers/`: proposer abstraction and backends (`rule_based`, `ollama`, `openai_compat`, `llm`).
- `itrader/scripts/gen_task_pool.py`: task-pool generation CLI.
- `itrader/rl_lib/train/train_AZR.py`: Phase B bootstrap (task generation + solver handoff).
- `itrader/rl_lib/train/train_solo_V2.py`: stable Phase A solver training entrypoint.

## Running

### 1) Quality Checks

```bash
python -m pytest -q itrader/tests
python itrader/tools/healthcheck.py
python -m pre_commit run --all-files
```

### 2) Task Pool Generation (Phase B)

Rule-based:

```bash
python -m itrader.scripts.gen_task_pool --backend rule_based --out data/task_pool_rule.jsonl --n_tasks 32 --batch_size 4
```

Ollama (example):

```bash
python -m itrader.scripts.gen_task_pool --backend ollama --ollama_model qwen3.5:9b --ollama_base_url http://localhost:11434 --ollama_timeout_s 180 --batch_size 4 --n_tasks 32 --max_backend_errors 5 --out data/task_pool_ollama_qwen35_9b.jsonl
```

Ollama with generated market summary scaffold (from current OHLCV window):

```bash
python -m itrader.scripts.gen_task_pool --backend ollama --ollama_model qwen3.5:9b --market_summary_mode ohlcv_window --summary_df_path itrader/data/ohlcv.parquet --summary_window 256 --summary_end_idx -1 --batch_size 4 --n_tasks 32 --max_backend_errors 5 --out data/task_pool_ollama_qwen35_9b_summary.jsonl
```

vLLM or SGLang (OpenAI-compatible endpoint):

```bash
python -m itrader.scripts.gen_task_pool --backend vllm --api_base_url http://localhost:8000/v1 --api_model Qwen/Qwen3.5-9B --batch_size 4 --n_tasks 32 --max_backend_errors 5 --out data/task_pool_vllm_qwen35_9b.jsonl
python -m itrader.scripts.gen_task_pool --backend sglang --api_base_url http://localhost:8000/v1 --api_model Qwen/Qwen3.5-9B --batch_size 4 --n_tasks 32 --max_backend_errors 5 --out data/task_pool_sglang_qwen35_9b.jsonl
```

### 3) AZR Bootstrap Handoff Run

```bash
python -m itrader.rl_lib.train.train_AZR --backend rule_based --n_tasks 64 --gen_batch_size 8 --experiment_name Phase_B_AZR_Smoke --iters 3 --workers 2 --window 30
```

## Outputs and Tracking

- Ray artifacts: `~/ray_results/<experiment_name>/...`
- Stable metrics mirror: `~/ray_results/_stable_metrics/<trial_dir>/...`
- Run manifests: `run_manifest_*.json` under experiment directories.

## Safety and Secrets

- Never commit credentials.
- Use `.env.example` as template.
- Keep API keys in environment variables only.
