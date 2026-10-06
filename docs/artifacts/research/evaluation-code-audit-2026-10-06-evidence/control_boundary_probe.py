"""Bounded footer/identity-only EOF check of the eight accepted controls."""

import json
from pathlib import Path

import pyarrow.parquet as pq

from itrader.rl_lib.evaluation.provenance import sha256_file


def run():
    repo = Path(__file__).resolve().parents[4]
    root = Path("C:/Users/Admin/ray_results/corrected-compat-r0-r4v1-20261005")
    portable = repo / "docs/artifacts/research/corrected-compatibility-r0-r4-controls-2026-10-05-evidence"
    rows = []
    for number in range(1, 9):
        batch = "replay_jobscope_20261005" if number in (4, 5, 6, 7) else "replay_jobscope_retry01_20261005"
        accepted = json.loads((portable / batch / f"cell{number:02}" / "acceptance.json").read_text())
        trial = accepted["trial"]
        config_file = root / "_stable_metrics" / trial / "resolved_config.json"
        config = json.loads(config_file.read_text())
        env = config["config"]["evaluation_config"]["env_config"]
        symbols, sessions = sorted(env["data_symbols"]), env["data_sessions"]
        assert sessions == ["regular"] and env["data_split"] == "validation"
        data = Path(env["df_path"])
        footers = [file for symbol in symbols for file in
                   (data / "session=regular" / f"symbol={symbol}").rglob("*.parquet")]
        length = sum(pq.read_metadata(file).num_rows for file in footers)
        assert length > 0
        original_batch = ("study_attempt04_terminal_replay_jobscope20261005" if number in (4, 5, 6, 7)
                          else "study_attempt04_terminal_replay_jobscope_retry01_20261005")
        episode_file = root / original_batch / f"cell{number:02}" / "out" / f"{trial}.episodes.jsonl"
        # Deliberately project only location/count identity; no outcome metrics.
        identities = [{key: record[key] for key in ("start_idx", "end_idx", "steps", "horizon", "task_seed")}
                      for record in (json.loads(line) for line in episode_file.read_text().splitlines())]
        tasks = env["task_sampler"]["fixed_tasks"]
        expected = {(task["start_idx"], task["horizon"], task["seed"]) for task in tasks}
        assert len(identities) == 32 and len(expected) == 32
        assert {(item["start_idx"], item["horizon"], item["task_seed"]) for item in identities} == expected
        assert all(item["steps"] == item["horizon"] == item["end_idx"] - item["start_idx"] for item in identities)
        assert all(item["end_idx"] < length - 1 for item in identities)
        index_file = config_file.with_name("checkpoint_index.jsonl")
        index = [json.loads(line) for line in index_file.read_text().splitlines() if line.strip()]
        terminal_update = max(item["pbt_progress_updates"] for item in index)
        terminal_count = sum(item["pbt_progress_updates"] == terminal_update for item in index)
        assert terminal_count == 1
        rows.append({"cell": number, "trial": trial, "projected_frame_rows_from_current_footers": length,
                     "parquet_footer_count": len(footers), "episode_count": 32,
                     "min_start": min(item["start_idx"] for item in identities),
                     "max_end": max(item["end_idx"] for item in identities),
                     "all_full_horizon": True, "all_starts_match_fixed_tasks": True,
                     "physical_eof_hit": False, "config_file_sha256": sha256_file(config_file),
                     "highest_indexed_update": terminal_update, "highest_update_entry_count": terminal_count,
                     "checkpoint_index_sha256": sha256_file(index_file),
                     "episode_file_sha256": sha256_file(episode_file)})
    print(json.dumps({"scope": "current_footer_counts_and_retained_episode_identity_only_no_outcomes",
                      "dataset_bytes_rehashed": False, "cells": rows}, indent=2))


if __name__ == "__main__":
    run()
