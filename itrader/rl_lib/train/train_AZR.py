from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from itrader.proposers import build_proposer
from itrader.rl_lib.tasks.families import DEFAULT_BAR_RESOLUTION
from itrader.rl_lib.tasks.spec import TaskSpec
from itrader.rl_lib.tasks.task_pool import TaskPool
from itrader.rl_lib.tasks.validation import validate_and_clamp_task


def _has_flag(tokens: List[str], flag: str) -> bool:
    return any(t == flag or t.startswith(f"{flag}=") for t in tokens)


def _parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Phase B AZR bootstrap: proposer task generation + solver training handoff",
    )
    parser.add_argument("--backend", choices=["rule_based", "llm"], default="rule_based")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--task_pool_path", default="itrader/data/task_pool_azr.jsonl")
    parser.add_argument("--n_tasks", type=int, default=200)
    parser.add_argument("--gen_batch_size", type=int, default=8)
    parser.add_argument("--max_empty_batches", type=int, default=10)
    parser.add_argument("--market_summary", default="Generate diverse curriculum tasks for trading policy improvement.")
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--n_features", type=int, default=5)
    parser.add_argument("--t_embed_dim", type=int, default=32)
    parser.add_argument("--action_dim", type=int, default=3)
    parser.add_argument("--bar_resolution", default=DEFAULT_BAR_RESOLUTION)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--p_pool", type=float, default=0.7)
    parser.add_argument("--experiment_name", default="Phase_B_AZR")
    parser.add_argument("--no_train", action="store_true", help="Generate task pool only; skip solver training handoff")
    return parser.parse_known_args()


def _build_proposer_from_args(args: argparse.Namespace):
    if args.backend == "llm":
        print(f"[AZR] Initializing LLM proposer: {args.model_id}")
        return build_proposer("llm", model_id=args.model_id)
    print(f"[AZR] Initializing rule-based proposer (seed={args.seed})")
    return build_proposer(
        "rule_based",
        seed=args.seed,
        window_size=args.window_size,
        n_features=args.n_features,
        t_embed_dim=args.t_embed_dim,
        action_dim=args.action_dim,
        bar_resolution=args.bar_resolution,
    )


def _generate_task_pool(args: argparse.Namespace) -> Path:
    random.seed(int(args.seed))
    proposer = _build_proposer_from_args(args)

    pool_path = Path(args.task_pool_path).expanduser()
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool = TaskPool(pool_path, read_only=False)

    env_cfg: Dict[str, Any] = {
        "window_size": int(args.window_size),
        "n_features": int(args.n_features),
        "t_embed_dim": int(args.t_embed_dim),
        "action_dim": int(args.action_dim),
        "bar_resolution": str(args.bar_resolution),
    }
    meta = {
        "generation_method": f"{args.backend}_proposer",
        "backend": args.backend,
        "seed": int(args.seed),
    }
    if args.backend == "llm":
        meta["model_id"] = args.model_id

    n_generated = 0
    n_valid = 0
    n_invalid = 0
    empty_batches = 0
    target_n = max(1, int(args.n_tasks))
    batch_size = max(1, int(args.gen_batch_size))
    max_empty = max(1, int(args.max_empty_batches))

    print(f"[AZR] Generating {target_n} tasks into pool: {pool_path}")
    while n_generated < target_n:
        current_batch = min(batch_size, target_n - n_generated)
        try:
            raw_tasks = proposer.propose_tasks(
                market_summary=args.market_summary,
                n=current_batch,
            )
        except Exception as e:
            print(f"[AZR] Proposer call failed (continuing): {e}")
            continue

        if not raw_tasks:
            empty_batches += 1
            print(f"[AZR] Empty proposer batch #{empty_batches}")
            if empty_batches >= max_empty:
                print("[AZR] Aborting generation after repeated empty batches.")
                break
            continue
        empty_batches = 0

        for raw_task in raw_tasks[:current_batch]:
            n_generated += 1
            try:
                validated = validate_and_clamp_task(
                    raw_task,
                    env_cfg,
                    fallback_seed=random.randint(0, 1_000_000),
                )
                TaskSpec.from_dict(validated)
                pool.add(validated, meta=meta)
                n_valid += 1
            except Exception as e:
                n_invalid += 1
                if n_invalid <= 5:
                    print(f"[AZR] Dropped invalid task: {e}")

    print(
        f"[AZR] Generation complete. valid={n_valid} invalid={n_invalid} "
        f"pool_size={pool.size()} path={pool_path}"
    )
    return pool_path


def _build_train_cmd(args: argparse.Namespace, passthrough: List[str], pool_path: Path) -> List[str]:
    cmd = [sys.executable, "-m", "itrader.rl_lib.train.train_solo_V2"]
    final_args = list(passthrough)

    defaults = [
        ("--task_pool_path", str(pool_path)),
        ("--p_pool", str(float(args.p_pool))),
        ("--seed", str(int(args.seed))),
        ("--experiment_name", str(args.experiment_name)),
    ]
    for flag, value in defaults:
        if not _has_flag(final_args, flag):
            final_args.extend([flag, value])

    cmd.extend(final_args)
    return cmd


def main() -> int:
    args, passthrough = _parse_args()
    pool_path = _generate_task_pool(args)

    if args.no_train:
        print("[AZR] --no_train set; skipping solver handoff.")
        return 0

    cmd = _build_train_cmd(args, passthrough, pool_path)
    print("[AZR] Launching solver training handoff:")
    print("[AZR] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
