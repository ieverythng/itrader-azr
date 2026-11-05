"""Synthetic public-interface audit; no historical or real market data."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from itrader.envs.itrader_env import ITraderEnv


def run():
    with TemporaryDirectory(prefix="itrader-eval-audit-") as directory:
        path = Path(directory) / "synthetic.parquet"
        pd.DataFrame({name: np.full(100, 100.0) for name in
                      ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME")}).to_parquet(path)
        task = {"family_id": "00", "price_id": "SIM", "horizon": 4,
                "reward_id": "net_return", "reward_params": {"cost_bps": 1, "reward_scale": 1000},
                "seed": 17, "obs_dim": 57, "action_dim": 3, "bar_resolution": "1m"}

        def config(mode, tasks, worker=0):
            return {"df_path": str(path), "window_size": 5, "start_idx_mode": mode,
                    "solver_state_mode": "both", "worker_index": worker,
                    "task_sampler": {"seed": 42, "window_size": 5, "n_features": 5,
                                     "t_embed_dim": 32, "action_dim": 3, "fixed_tasks": tasks}}

        def episode(env):
            obs, reset = env.reset()
            initial_state = obs[-2:].tolist()
            done, count = False, 0
            while not done:
                obs, reward, terminated, truncated, info = env.step(2)
                count += 1
                done = terminated or truncated
            evidence = info["episode_evidence"]
            return {"requested_start": env.current_task.start_idx, "reset_reported_start": reset["task"]["start_idx"],
                    "evidence_start": evidence["start_idx"],
                    "steps": count, "horizon": reset["task"]["horizon"], "end": evidence["end_idx"],
                    "initial_state": initial_state, "terminal_state": obs[-2:].tolist(),
                    "net": evidence["net_return"], "turnover": evidence["turnover"],
                    "terminated": terminated, "truncated": truncated}

        end_env = ITraderEnv(config("task_spec", [{**task, "start_idx": 96}]))
        end_case = episode(end_env)
        short_horizons = [episode(ITraderEnv(config("task_spec", [
            {**task, "horizon": horizon, "start_idx": 100 - horizon}]))) for horizon in (1, 2)]
        interior = episode(ITraderEnv(config("task_spec", [{**task, "start_idx": 95}])))
        clamped = episode(ITraderEnv(config("task_spec", [{**task, "start_idx": 1000}])))
        fixed = [dict(task), {**task, "seed": 29}]
        random_env = ITraderEnv(config("random_episode", fixed))
        random_starts = [random_env.reset()[1]["task"]["start_idx"] for _ in range(6)]
        worker_env = ITraderEnv(config("random_episode", fixed, worker=1))
        worker_starts = [worker_env.reset()[1]["task"]["start_idx"] for _ in range(6)]
        stream_env = ITraderEnv(config("stream_worker", [task]))
        first_stream = stream_env.reset(seed=7)[1]["task"]["start_idx"]
        second_stream = stream_env.reset()[1]["task"]["start_idx"]
        reseeded_stream = stream_env.reset(seed=7)[1]["task"]["start_idx"]
        result = {"dataset_end_expected_steps": 4, "dataset_end_actual": end_case,
                  "last_eligible_horizons_1_2": short_horizons,
                  "interior": interior, "out_of_range_task_spec": clamped,
                  "fixed_random_starts": random_starts, "worker1_fixed_random_starts": worker_starts,
                  "fixed_random_two_task_cycle_repeats": random_starts[:2] == random_starts[2:4] == random_starts[4:6],
                  "fixed_random_worker_identity_does_not_change_task_owned_seeds": random_starts == worker_starts,
                  "stream_seed7": [first_stream, second_stream, reseeded_stream],
                  "stream_reseed_restarts": first_stream == reseeded_stream,
                  "scope": "synthetic_mechanics_not_historical_start_recovery"}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
