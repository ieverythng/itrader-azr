"""
CLI script to generate tasks using LLM proposer and populate TaskPool.

Usage:
    python -m scripts.gen_task_pool --model_id Qwen/Qwen2.5-7B-Instruct --out task_pool.jsonl --n_tasks 100 --seed 42
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from typing import Dict, Any

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from itrader.llm.proposer_llm import ProposerLLM
from itrader.rl_lib.tasks.task_pool import TaskPool
from itrader.rl_lib.tasks.spec import TaskSpec


def validate_and_clamp_task(task_dict: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clamp TaskSpec fields to safe bounds.

    Args:
        task_dict: Raw task dict from LLM
        env_config: Environment config with defaults (window_size, n_features, etc.)

    Returns:
        Validated and clamped task dict
    """
    # Extract defaults from env_config
    window_size = int(env_config.get("window_size", 30))
    n_features = int(env_config.get("n_features", 5))
    t_embed_dim = int(env_config.get("t_embed_dim", 32))
    action_dim = int(env_config.get("action_dim", 3))
    
    # Required fields with defaults
    task_dict.setdefault("family_id", "00")
    task_dict.setdefault("price_id", "SIM")
    task_dict.setdefault("reward_id", "pnl_linear")
    task_dict.setdefault("seed", int(np.random.randint(0, 1_000_000)))
    
    # Clamp horizon (episode length)
    horizon = int(task_dict.get("horizon", 128))
    horizon = max(64, min(512, horizon))  # Clamp to [64, 512]
    task_dict["horizon"] = horizon
    
    # Handle start_idx: accept if present, but don't clamp (env will clamp based on df length)
    # If not present, leave as None (env will sample uniformly)
    if "start_idx" in task_dict:
        # Ensure it's an int, but don't clamp range (env will handle that)
        task_dict["start_idx"] = int(task_dict["start_idx"])
        # Basic sanity check: should be >= 0
        if task_dict["start_idx"] < 0:
            task_dict["start_idx"] = None
    else:
        task_dict["start_idx"] = None
    
    # Validate and clamp reward_params
    reward_params = task_dict.get("reward_params", {})
    
    # Clamp numeric reward parameters
    if "pnl_w" in reward_params:
        reward_params["pnl_w"] = float(np.clip(float(reward_params["pnl_w"]), 0.0, 10.0))
    else:
        reward_params["pnl_w"] = 1.0
    
    if "turnover_w" in reward_params:
        reward_params["turnover_w"] = float(np.clip(float(reward_params["turnover_w"]), 0.0, 0.01))
    else:
        reward_params["turnover_w"] = 0.0
    
    if "dd_w" in reward_params:
        reward_params["dd_w"] = float(np.clip(float(reward_params["dd_w"]), 0.0, 0.2))
    else:
        reward_params["dd_w"] = 0.04
    
    if "dd_T_weight" in reward_params:
        reward_params["dd_T_weight"] = float(np.clip(float(reward_params["dd_T_weight"]), 0.0, 1.0))
    else:
        reward_params["dd_T_weight"] = 0.0
    
    if "turnover_cap" in reward_params:
        reward_params["turnover_cap"] = float(np.clip(float(reward_params["turnover_cap"]), 0.0, 0.1))
    else:
        reward_params["turnover_cap"] = 0.02
    
    if "dd_cap" in reward_params:
        reward_params["dd_cap"] = float(np.clip(float(reward_params["dd_cap"]), 0.0, 0.2))
    else:
        reward_params["dd_cap"] = 0.05
    
    if "reward_scale" in reward_params:
        reward_params["reward_scale"] = float(np.clip(float(reward_params["reward_scale"]), 0.1, 10.0))
    else:
        reward_params["reward_scale"] = 1.0
    
    if "norm_clip" in reward_params:
        reward_params["norm_clip"] = float(np.clip(float(reward_params["norm_clip"]), 1.0, 20.0))
    else:
        reward_params["norm_clip"] = 5.0
    
    if "inact_w" in reward_params:
        reward_params["inact_w"] = float(np.clip(float(reward_params["inact_w"]), 0.0, 0.01))
    else:
        reward_params["inact_w"] = 0.0
    
    if "vol_floor_mult" in reward_params:
        reward_params["vol_floor_mult"] = float(np.clip(float(reward_params["vol_floor_mult"]), 0.1, 5.0))
    else:
        reward_params["vol_floor_mult"] = 1.0
    
    if "clip_abs" in reward_params:
        reward_params["clip_abs"] = float(np.clip(float(reward_params["clip_abs"]), 0.0, 1.0))
    else:
        reward_params["clip_abs"] = 0.0
    
    # Validate string/enum parameters
    reward_params.setdefault("dd_mode", "delta")
    if reward_params["dd_mode"] not in ["none", "delta", "terminal"]:
        reward_params["dd_mode"] = "delta"
    
    reward_params.setdefault("normalize", True)
    reward_params["normalize"] = bool(reward_params["normalize"])
    
    reward_params.setdefault("norm_mode", "all")
    if reward_params["norm_mode"] not in ["all", "penalties", "off"]:
        reward_params["norm_mode"] = "all"
    
    reward_params.setdefault("normalize_dd", True)
    reward_params["normalize_dd"] = bool(reward_params["normalize_dd"])
    
    task_dict["reward_params"] = reward_params
    
    # Compute obs_dim and action_dim
    obs_dim = window_size * n_features + t_embed_dim
    task_dict["obs_dim"] = obs_dim
    task_dict["action_dim"] = action_dim
    
    return task_dict


def main():
    parser = argparse.ArgumentParser(description="Generate tasks using LLM proposer")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model identifier")
    parser.add_argument("--out", type=str, default="task_pool.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--n_tasks", type=int, default=100,
                        help="Number of tasks to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--window_size", type=int, default=30,
                        help="Window size for obs_dim calculation")
    parser.add_argument("--n_features", type=int, default=5,
                        help="Number of features (OHLCV)")
    parser.add_argument("--t_embed_dim", type=int, default=32,
                        help="Task embedding dimension")
    parser.add_argument("--action_dim", type=int, default=3,
                        help="Action space dimension")
    parser.add_argument("--market_summary", type=str,
                        default="Generate diverse trading tasks with varying horizons and reward parameters.",
                        help="Market summary/context for task generation")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Initialize proposer
    print(f"Initializing proposer with model: {args.model_id}")
    proposer = ProposerLLM(model_id=args.model_id)
    
    # Initialize task pool
    pool = TaskPool(args.out, read_only=False)
    existing_size = pool.size()
    print(f"Task pool initialized. Current size: {existing_size}")
    
    # Prepare env_config for validation
    env_config = {
        "window_size": args.window_size,
        "n_features": args.n_features,
        "t_embed_dim": args.t_embed_dim,
        "action_dim": args.action_dim,
    }
    
    # Generate tasks in batches (LLM may return multiple per call)
    batch_size = 5  # Generate 5 at a time
    n_generated = 0
    n_valid = 0
    n_invalid = 0
    
    print(f"Generating {args.n_tasks} tasks...")
    while n_generated < args.n_tasks:
        remaining = args.n_tasks - n_generated
        current_batch = min(batch_size, remaining)
        
        try:
            # Generate tasks
            raw_tasks = proposer.generate_tasks(
                market_summary=args.market_summary,
                n=current_batch,
            )
            
            for raw_task in raw_tasks:
                n_generated += 1
                try:
                    # Validate and clamp
                    validated_task = validate_and_clamp_task(raw_task, env_config)
                    
                    # Try to create TaskSpec to ensure it's valid
                    task_spec = TaskSpec.from_dict(validated_task)
                    
                    # Add to pool
                    meta = {
                        "generation_method": "llm_proposer",
                        "model_id": args.model_id,
                        "seed": args.seed,
                    }
                    pool.add(validated_task, meta=meta)
                    n_valid += 1
                    
                    if n_valid % 10 == 0:
                        print(f"  Generated {n_valid} valid tasks (total attempts: {n_generated})")
                
                except Exception as e:
                    n_invalid += 1
                    if n_invalid <= 5:  # Only print first few errors
                        print(f"  Warning: Invalid task (dropped): {e}")
        
        except Exception as e:
            print(f"  Error during generation: {e}")
            continue
    
    final_size = pool.size()
    print(f"\nGeneration complete!")
    print(f"  Valid tasks added: {n_valid}")
    print(f"  Invalid tasks dropped: {n_invalid}")
    print(f"  Final pool size: {final_size}")
    print(f"  Output file: {args.out}")


if __name__ == "__main__":
    main()

