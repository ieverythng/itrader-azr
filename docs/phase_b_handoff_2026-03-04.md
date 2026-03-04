# Phase-B Handoff (2026-03-04)

## Goal of this handoff

Provide a no-context-loss continuation point for Codex sessions on main PC, focused on:

- deterministic local checks (`pre-commit`, tests, healthcheck),
- proposer task-pool generation via local Ollama,
- immediate migration path to high-throughput OpenAI-compatible serving (`vLLM`/`SGLang`),
- readiness for Qwen3.5-based proposer experiments.

## Branch and state

- Branch: `Phase-B-Training`
- Base commit before this work: `405f86c5`
- This handoff includes:
  - deterministic interpreter routing for checks/hooks,
  - improved Ollama error surfacing,
  - new OpenAI-compatible proposer backend for `vLLM`/`SGLang`,
  - safer backend failure handling in task generation.

## What changed

### 1) Deterministic Python path for checks/hooks

Files:

- `itrader/tools/run_repo_python.sh`
- `.pre-commit-config.yaml`
- `README.md`

What it fixes:

- `pre-commit` no longer depends on whichever `python3` is in shell `PATH`.
- Hooks always run using repo-local `.itrader_env/bin/python`.
- Eliminates false missing-package errors caused by system Python.

### 2) Ollama proposer diagnostics

Files:

- `itrader/proposers/ollama.py`

What it adds:

- HTTP/URL error handling with detailed backend message snippets.
- Faster root-cause detection (model load failures, daemon/network issues).

### 3) OpenAI-compatible proposer path (for vLLM/SGLang)

Files:

- `itrader/proposers/openai_compat.py` (new)
- `itrader/proposers/factory.py`
- `itrader/proposers/__init__.py`

What it adds:

- `OpenAICompatProposer` calling `POST /v1/chat/completions`.
- Factory aliases:
  - `vllm`
  - `sglang`
  - `openai` / `openai_compat`
- Reuses same JSON repair strategy as Ollama path.

### 4) Safer task generation CLI behavior

Files:

- `itrader/scripts/gen_task_pool.py`
- `itrader/tests/test_rule_based_proposer.py`
- `README.md`

What it adds:

- New backends in CLI:
  - `--backend vllm`
  - `--backend sglang`
- New API args:
  - `--api_model`
  - `--api_base_url`
  - `--api_key_env`
  - `--api_timeout_s`
  - `--api_temperature`
  - `--api_max_tokens`
- New backend-failure guard:
  - `--max_backend_errors` (prevents endless retry loops).
- Unit tests for factory routing + JSON repair on openai-compatible backend.

## Validation snapshot

Validated on repo-local env:

- `./itrader/tools/run_repo_python.sh -m pytest -q itrader/tests` -> pass
- `./itrader/tools/run_repo_python.sh -m pre_commit run --all-files` -> pass

Validated failure handling:

- Ollama + `hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M` now aborts cleanly after configured backend error threshold, with explicit `unable to load model` details.
- vLLM path exits cleanly when endpoint is unavailable.

## Current known blocker

Local Ollama compatibility/runtime issue for pulled Qwen3.5 4B GGUF on this workstation:

- error: `500 ... unable to load model ...`

Implication:

- keep Ollama path for models confirmed to load on target machine,
- use `vLLM`/`SGLang` for Qwen3.5 serving when throughput and compatibility are priority.

## Next commands (main PC)

### A) Quick sanity checks

```bash
./itrader/tools/run_repo_python.sh -m pytest -q itrader/tests
./itrader/tools/run_repo_python.sh itrader/tools/healthcheck.py
./itrader/tools/run_repo_python.sh -m pre_commit run --all-files
```

### B) Local Ollama proposer task-pool run

Use this once your target model is confirmed to run with `ollama run`:

```bash
./itrader/tools/run_repo_python.sh -m itrader.scripts.gen_task_pool \
  --backend ollama \
  --ollama_model qwen3.5:9b \
  --ollama_base_url http://localhost:11434 \
  --ollama_timeout_s 180 \
  --batch_size 4 \
  --n_tasks 32 \
  --max_backend_errors 5 \
  --out /tmp/itrader_task_pool_ollama_qwen35_9b.jsonl
```

If your local model tag differs, replace `qwen3.5:9b` with the exact `ollama list` name.

### C) vLLM-backed proposer run

Start vLLM server separately (main PC), then:

```bash
./itrader/tools/run_repo_python.sh -m itrader.scripts.gen_task_pool \
  --backend vllm \
  --api_base_url http://localhost:8000/v1 \
  --api_model Qwen/Qwen3.5-9B \
  --api_timeout_s 120 \
  --api_temperature 0.2 \
  --api_max_tokens 768 \
  --batch_size 4 \
  --n_tasks 32 \
  --max_backend_errors 5 \
  --out /tmp/itrader_task_pool_vllm_qwen35_9b.jsonl
```

### D) SGLang-backed proposer run

Start SGLang server separately (main PC), then:

```bash
./itrader/tools/run_repo_python.sh -m itrader.scripts.gen_task_pool \
  --backend sglang \
  --api_base_url http://localhost:8000/v1 \
  --api_model Qwen/Qwen3.5-9B \
  --api_timeout_s 120 \
  --api_temperature 0.2 \
  --api_max_tokens 768 \
  --batch_size 4 \
  --n_tasks 32 \
  --max_backend_errors 5 \
  --out /tmp/itrader_task_pool_sglang_qwen35_9b.jsonl
```

## Backend options at this point

### `rule_based`

- Fast, deterministic fallback.
- Best for CI/smoke/repro checks.

### `ollama`

- Easiest local path.
- Good for low-ops experimentation.
- Throughput and new-model support can lag.

### `vllm`

- Preferred for high-throughput serving on main PC.
- Strong OpenAI-compatible API path.
- Better production-like behavior for proposer scaling.

### `sglang`

- Also high-performance serving.
- Strong advanced inference/tooling support.
- Useful alternative when tuning latency/throughput behavior.

## Unsloth clarification

Unsloth is primarily for efficient fine-tuning/training workflows (LoRA/QLoRA) and can also accelerate inference in Hugging Face-style pipelines.

Important practical distinction:

- If you are serving via Ollama with GGUF models, that is a separate runtime path from Unsloth.
- Unsloth generally expects HF model workflow for training/inference optimization, not Ollama GGUF serving as-is.

Recommended split:

- Serving proposer now: `vLLM` or `SGLang` (or Ollama if model compatibility is stable).
- Fine-tuning adaptation: Unsloth + QLoRA on the training-capable stack, then export/deploy by your serving strategy.

## Codex continuation prompt seed

Use this directly in the next Codex session:

1. Read `docs/phase_b_handoff_2026-03-04.md`.
2. Verify `./itrader/tools/run_repo_python.sh -m pytest -q itrader/tests` passes.
3. Verify local model runtime (`ollama run <model> "ping"` or API health for vLLM/SGLang).
4. Execute `itrader.scripts.gen_task_pool` with backend `ollama` first (or `vllm` if available) for `n_tasks=32`, `batch_size=4`.
5. Print summary counts and first 2 generated task entries from output JSONL.
6. If backend errors exceed threshold, stop and report first backend error payload exactly.
