"""
CLI script to generate tasks via proposer backends and populate TaskPool.

Usage:
    python -m itrader.scripts.gen_task_pool --backend rule_based --out task_pool.jsonl --n_tasks 100 --seed 42
"""
from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict

from itrader.proposers import build_proposer
from itrader.rl_lib.tasks.families import DEFAULT_BAR_RESOLUTION
from itrader.rl_lib.tasks.market_summary import build_market_summary_from_ohlcv_path
from itrader.rl_lib.tasks.spec import TaskSpec
from itrader.rl_lib.tasks.task_pool import TaskPool
from itrader.rl_lib.tasks.validation import validate_and_clamp_task


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tasks and populate TaskPool")
    parser.add_argument(
        "--backend",
        type=str,
        default="rule_based",
        choices=["rule_based", "llm", "ollama", "openai_compat", "llama_cpp", "vllm", "sglang"],
        help="Proposer backend",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model identifier (used when --backend llm)",
    )
    parser.add_argument(
        "--ollama_model",
        type=str,
        default="qwen2.5:7b-instruct",
        help="Ollama model name (used when --backend ollama)",
    )
    parser.add_argument(
        "--ollama_base_url",
        type=str,
        default="http://localhost:11434",
        help="Ollama base URL (used when --backend ollama)",
    )
    parser.add_argument(
        "--ollama_timeout_s",
        type=float,
        default=60.0,
        help="Ollama request timeout seconds",
    )
    parser.add_argument(
        "--ollama_temperature",
        type=float,
        default=0.2,
        help="Ollama generation temperature",
    )
    parser.add_argument(
        "--ollama_num_predict",
        type=int,
        default=768,
        help="Ollama max generated tokens per request",
    )
    parser.add_argument(
        "--api_model",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="Model id for OpenAI-compatible APIs (used when --backend vllm|sglang)",
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI-compatible base URL (used when --backend vllm|sglang)",
    )
    parser.add_argument(
        "--api_key_env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable with bearer token for OpenAI-compatible API",
    )
    parser.add_argument(
        "--api_timeout_s",
        type=float,
        default=60.0,
        help="OpenAI-compatible request timeout seconds",
    )
    parser.add_argument(
        "--api_temperature",
        type=float,
        default=0.2,
        help="OpenAI-compatible generation temperature",
    )
    parser.add_argument(
        "--api_max_tokens",
        type=int,
        default=768,
        help="OpenAI-compatible max_tokens per request",
    )
    parser.add_argument("--out", type=str, default="task_pool.jsonl", help="Output JSONL file path")
    parser.add_argument("--n_tasks", type=int, default=100, help="Number of tasks to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--window_size", type=int, default=30, help="Window size for obs_dim calculation")
    parser.add_argument("--n_features", type=int, default=5, help="Number of OHLCV features")
    parser.add_argument("--t_embed_dim", type=int, default=32, help="Task embedding dimension")
    parser.add_argument("--action_dim", type=int, default=3, help="Action space dimension")
    parser.add_argument("--bar_resolution", type=str, default=DEFAULT_BAR_RESOLUTION, help="Bar resolution label recorded on generated tasks")
    parser.add_argument("--batch_size", type=int, default=5, help="Tasks requested per proposer call")
    parser.add_argument("--max_empty_batches", type=int, default=10, help="Abort after repeated empty proposer outputs")
    parser.add_argument(
        "--max_backend_errors",
        type=int,
        default=25,
        help="Abort after repeated backend call failures",
    )
    parser.add_argument(
        "--market_summary",
        type=str,
        default="Generate diverse trading tasks with varying horizons and reward parameters.",
        help="Market summary/context for task generation",
    )
    parser.add_argument(
        "--market_summary_mode",
        type=str,
        choices=["static", "ohlcv_window"],
        default="static",
        help="How to build market_summary text for proposer calls",
    )
    parser.add_argument(
        "--summary_df_path",
        type=str,
        default="itrader/data/ohlcv.parquet",
        help="Parquet path used when --market_summary_mode ohlcv_window",
    )
    parser.add_argument(
        "--summary_window",
        type=int,
        default=256,
        help="Rows for market summary window (ohlcv_window mode)",
    )
    parser.add_argument(
        "--summary_end_idx",
        type=int,
        default=-1,
        help="Inclusive end index for summary window, -1 means latest row",
    )
    return parser.parse_args()


def _build_proposer_from_args(args: argparse.Namespace):
    if args.backend == "ollama":
        print(f"Initializing Ollama proposer with model: {args.ollama_model} @ {args.ollama_base_url}")
        return build_proposer(
            "ollama",
            model=args.ollama_model,
            base_url=args.ollama_base_url,
            timeout_s=args.ollama_timeout_s,
            temperature=args.ollama_temperature,
            num_predict=args.ollama_num_predict,
        )
    if args.backend in {"openai_compat", "llama_cpp", "vllm", "sglang"}:
        print(f"Initializing {args.backend} proposer with model: {args.api_model} @ {args.api_base_url}")
        token = os.getenv(args.api_key_env, "").strip()
        return build_proposer(
            args.backend,
            model=args.api_model,
            base_url=args.api_base_url,
            api_key=token if token else None,
            timeout_s=args.api_timeout_s,
            temperature=args.api_temperature,
            max_tokens=args.api_max_tokens,
        )
    if args.backend == "llm":
        print(f"Initializing LLM proposer with model: {args.model_id}")
        return build_proposer("llm", model_id=args.model_id)
    print(f"Initializing rule-based proposer with seed: {args.seed}")
    return build_proposer(
        "rule_based",
        seed=args.seed,
        window_size=args.window_size,
        n_features=args.n_features,
        t_embed_dim=args.t_embed_dim,
        action_dim=args.action_dim,
        bar_resolution=args.bar_resolution,
    )


def _metadata(args: argparse.Namespace) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "generation_method": f"{args.backend}_proposer",
        "backend": args.backend,
        "seed": int(args.seed),
    }
    if args.backend == "llm":
        meta["model_id"] = args.model_id
    elif args.backend == "ollama":
        meta["model_id"] = args.ollama_model
        meta["base_url"] = args.ollama_base_url
    elif args.backend in {"openai_compat", "llama_cpp", "vllm", "sglang"}:
        meta["model_id"] = args.api_model
        meta["base_url"] = args.api_base_url
    return meta


def _market_summary(args: argparse.Namespace) -> str:
    if args.market_summary_mode == "static":
        return str(args.market_summary)

    end_idx = int(args.summary_end_idx)
    if end_idx < 0:
        end_idx = None

    try:
        summary = build_market_summary_from_ohlcv_path(
            args.summary_df_path,
            window_size=int(args.summary_window),
            end_idx=end_idx,
        )
        print("Using generated market summary from OHLCV window:")
        print(summary)
        return summary
    except Exception as e:
        print(f"Warning: failed to generate market summary from data: {e}")
        print("Falling back to static --market_summary text.")
        return str(args.market_summary)


def main() -> int:
    args = _parse_args()
    random.seed(args.seed)
    market_summary = _market_summary(args)

    proposer = _build_proposer_from_args(args)
    pool = TaskPool(args.out, read_only=False)
    print(f"Task pool initialized. Current size: {pool.size()}")

    env_config = {
        "window_size": args.window_size,
        "n_features": args.n_features,
        "t_embed_dim": args.t_embed_dim,
        "action_dim": args.action_dim,
        "bar_resolution": args.bar_resolution,
    }
    meta = _metadata(args)

    n_generated = 0
    n_valid = 0
    n_invalid = 0
    n_duplicate = 0
    empty_batches = 0
    backend_errors = 0
    max_attempts = max(int(args.n_tasks), int(args.n_tasks) * 10)

    print(f"Generating {args.n_tasks} tasks...")
    while n_valid < args.n_tasks and n_generated < max_attempts:
        remaining = args.n_tasks - n_valid
        current_batch = min(max(1, args.batch_size), remaining)
        batch_context = (
            f"{market_summary}\n\n"
            f"Generation batch index: {n_generated // max(1, args.batch_size)}. "
            "Avoid exact duplicates of previously generated tasks by varying seed, horizon, "
            "start_idx, reward parameters, regime tags, or metadata where valid."
        )

        try:
            raw_tasks = proposer.propose_tasks(
                market_summary=batch_context,
                n=current_batch,
            )
        except Exception as e:
            backend_errors += 1
            print(f"  Error during proposer call (#{backend_errors}): {e}")
            if backend_errors >= max(1, int(args.max_backend_errors)):
                print("  Error: too many backend failures, aborting.")
                break
            continue
        backend_errors = 0

        if not raw_tasks:
            empty_batches += 1
            print(f"  Warning: proposer returned no tasks (empty batch #{empty_batches})")
            if empty_batches >= max(1, int(args.max_empty_batches)):
                print("  Error: too many empty batches, aborting.")
                break
            continue
        empty_batches = 0

        for raw_task in raw_tasks[:current_batch]:
            n_generated += 1
            try:
                validated_task = validate_and_clamp_task(
                    raw_task,
                    env_config,
                    fallback_seed=random.randint(0, 1_000_000),
                )
                TaskSpec.from_dict(validated_task)
                size_before = pool.size()
                pool.add(validated_task, meta=meta)
                if pool.size() > size_before:
                    n_valid += 1
                else:
                    n_duplicate += 1
                if n_valid % 10 == 0:
                    print(f"  Generated {n_valid} valid tasks (attempts: {n_generated})")
            except Exception as e:
                n_invalid += 1
                if n_invalid <= 5:
                    print(f"  Warning: Invalid task dropped: {e}")

    print("\nGeneration complete")
    print(f"  Valid tasks added: {n_valid}")
    print(f"  Duplicate tasks skipped: {n_duplicate}")
    print(f"  Invalid tasks dropped: {n_invalid}")
    print(f"  Total proposer attempts: {n_generated}")
    print(f"  Final pool size: {pool.size()}")
    print(f"  Output file: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
