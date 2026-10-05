"""Run two sequential fresh-process checkpoint replays with launch-bound receipts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import psutil

from .replay_evidence import compare_process_replays


def launch_replay(command: list[str], *, log_path: Path, timeout_seconds: float) -> dict:
    launch_id = str(uuid4())
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "ITRADER_REPLAY_LAUNCH_ID": launch_id})
        launch = {"pid": process.pid, "process_started_unix": psutil.Process(process.pid).create_time(),
                  "command": command, "launch_id": launch_id}
        observed = {(launch["pid"], launch["process_started_unix"])}
        deadline = time.monotonic() + timeout_seconds
        try:
            while True:
                # Windows venv launchers create a child Python interpreter.
                try:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        observed.add((child.pid, child.create_time()))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                try:
                    launch["returncode"] = process.wait(timeout=min(0.05, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            launch.update({"returncode": process.returncode, "timed_out": True})
        launch["observed_processes"] = [
            {"pid": pid, "process_started_unix": started} for pid, started in sorted(observed)
        ]
    return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-trial-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-selection", choices=("best", "terminal"), default="best")
    parser.add_argument("--checkpoint-progress-steps", type=int)
    parser.add_argument("--evaluation-task-panel-path", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("timeout-seconds must be positive")
    root = args.out_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    paths, launches = [], []
    for index in range(2):
        output = root / f"process_{index + 1}"
        command = [sys.executable, "-m", "itrader.rl_lib.evaluation.run_checkpoint_replay",
                   "--stable-trial-dir", str(args.stable_trial_dir.resolve()),
                   "--out-dir", str(output), "--checkpoint-selection", args.checkpoint_selection]
        for name in ("checkpoint_progress_steps", "evaluation_task_panel_path"):
            value = getattr(args, name)
            if value is not None:
                command.extend(["--" + name.replace("_", "-"), str(value)])
        launch = launch_replay(command, log_path=root / f"process_{index+1}.log",
                               timeout_seconds=args.timeout_seconds)
        launches.append(launch)
        with (root / f"process_{index+1}.launch.json").open("x", encoding="utf-8") as stream:
            json.dump(launch, stream, indent=2)
        if launch["returncode"] != 0:
            raise RuntimeError(f"Independent replay {index+1} failed; see its launch record/log")
        paths.append(output / f"{args.stable_trial_dir.name}.json")
    result = {**compare_process_replays(paths, launches), "launches": launches,
              "receipts": [str(path) for path in paths]}
    with (root / "replication.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    if result["verified_process_replication"] is not True:
        raise RuntimeError("Independent replay contract or payload mismatch")


if __name__ == "__main__":
    main()
