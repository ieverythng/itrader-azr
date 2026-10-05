"""Replay claims, attributable completion receipts, and artifact validation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
from uuid import uuid4

import psutil

from .provenance import canonical_json_sha256, is_sha256, sha256_file


def episode_payload_sha256(episodes: list[dict]) -> str:
    """Order-independent episodes, retaining identity, all metrics and trace hashes."""
    # JSON strings retain duplicate episodes rather than deduplicating a set.
    return canonical_json_sha256(sorted(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for row in episodes
    ))


def evaluation_payload_from_result(result: dict, expected_episodes: int) -> str | None:
    evaluation = result.get("evaluation", {})
    hist = evaluation.get("hist_stats") or evaluation.get("env_runners", {}).get("hist_stats", {})
    encoded = hist.get("replay_episode_payload", [])
    if len(encoded) != expected_episodes:
        return None
    rows = [json.loads(value) for value in encoded]
    if not all(is_sha256(row.get("action_reward_trace_sha256")) for row in rows):
        return None
    return episode_payload_sha256(rows)


def replay_parity(selection: dict, *, observed_score: float,
                  episodes: list[dict], diagnostic_only: bool = False) -> dict:
    score_delta = float(observed_score) - float(selection["score"])
    score_parity = math.isfinite(score_delta) and abs(score_delta) <= 1e-8
    expected = selection.get("evaluation_payload_sha256")
    observed = episode_payload_sha256(episodes)
    trace_available = bool(episodes) and all(
        is_sha256(row.get("action_reward_trace_sha256")) for row in episodes)
    payload_parity = observed == expected if is_sha256(expected) and trace_available else None
    return {
        "replay_delta": None if diagnostic_only else score_delta,
        "selection_score_parity": None if diagnostic_only else score_parity,
        "payload_parity": None if diagnostic_only else payload_parity,
        "checkpoint_parity": None if diagnostic_only or payload_parity is None else score_parity and payload_parity,
        "evaluation_payload_sha256": observed,
        "parity_scope": "diagnostic_panel" if diagnostic_only else "source_episode_metrics_and_action_reward_trace_v1",
    }


class ReplayInvocation:
    """Reserve one output at entry; publish complete only after hashing artifacts."""

    def __init__(self, report_path: Path, *, operation: str):
        self.path = Path(report_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        started = psutil.Process().create_time()
        host = socket.gethostname()
        self.producer = {
            "invocation_id": str(uuid4()), "pid": os.getpid(), "host": host,
            "process_started_unix": started,
            "process_instance_id": canonical_json_sha256([host, os.getpid(), started]),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "launch_id": os.environ.get("ITRADER_REPLAY_LAUNCH_ID"),
        }
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump({"receipt_version": 1, "status": "running", "producer": self.producer}, stream)

    def complete(self, report: dict, *, contract: dict, artifacts: list[Path]) -> dict:
        if not artifacts:
            raise ValueError("Replay completion requires artifacts")
        records = []
        for path in artifacts:
            path = Path(path).resolve(strict=True)
            if path.parent != self.path.parent:
                raise ValueError("Replay artifacts must be beside their receipt")
            records.append({"file": path.name, "sha256": sha256_file(path),
                            "rows": len(path.read_text(encoding="utf-8").splitlines())})
        current = json.loads(self.path.read_text(encoding="utf-8"))
        if current.get("producer") != self.producer or current.get("status") != "running":
            raise ValueError("Replay output reservation changed")
        receipt = {**report, "receipt_version": 1, "status": "complete",
                   "producer": {**self.producer, "completed_at_utc": datetime.now(timezone.utc).isoformat()},
                   "evaluation_contract": contract,
                   "evaluation_contract_sha256": canonical_json_sha256(contract),
                   "artifacts": records}
        pending = self.path.with_name(f".{self.producer['invocation_id']}.pending")
        with pending.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, self.path)
        return receipt


def verify_receipt(path: Path) -> dict:
    path = Path(path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("receipt_version") != 1 or receipt.get("status") != "complete":
        raise ValueError("Replay receipt is incomplete or has unknown provenance")
    producer = receipt.get("producer", {})
    expected = canonical_json_sha256([producer.get("host"), producer.get("pid"), producer.get("process_started_unix")])
    if not producer.get("invocation_id") or producer.get("process_instance_id") != expected or not producer.get("completed_at_utc"):
        raise ValueError("Replay producer identity is incomplete")
    if receipt.get("evaluation_contract_sha256") != canonical_json_sha256(receipt.get("evaluation_contract")):
        raise ValueError("Replay contract hash mismatch")
    if not receipt.get("artifacts"):
        raise ValueError("Replay receipt has no artifacts")
    for artifact in receipt["artifacts"]:
        file = path.parent / artifact["file"]
        if file.resolve().parent != path.parent.resolve() or not file.is_file() or sha256_file(file) != artifact["sha256"]:
            raise ValueError("Replay artifact identity mismatch")
        if len(file.read_text(encoding="utf-8").splitlines()) != artifact["rows"]:
            raise ValueError("Replay artifact row count mismatch")
    return receipt


def _validate_episode_replay_contract(receipt: dict, path: Path) -> None:
    """A matching pair of arbitrary receipts is not checkpoint replay evidence."""
    contract = receipt.get("evaluation_contract", {})
    digest_keys = ("checkpoint_sha256", "dataset_sha256", "source_panel_sha256",
                   "evaluation_panel_sha256", "effective_env_config_sha256", "checkpoint_config_sha256")
    runtime = contract.get("runtime", {})
    execution = contract.get("execution", {})
    if (
        contract.get("version") != "deterministic_replay_v1"
        or receipt.get("producer", {}).get("operation") != "episode_replay"
        or not all(is_sha256(contract.get(key)) for key in digest_keys)
        or not is_sha256(contract.get("replay_source", {}).get("source_tree_sha256"))
        or not isinstance(contract.get("observation_schema"), dict)
        or not contract["observation_schema"]
        or not all(isinstance(runtime.get(key), str) and runtime[key]
                   for key in ("python", "platform", "ray", "gymnasium", "pyarrow", "torch", "pandas"))
        or not all(type(contract.get(key)) is int and contract[key] >= 0
                   for key in ("progress_steps", "restore_generation", "evaluation_seed"))
        or contract.get("explore") is not False
        or not isinstance(execution.get("device"), str) or not execution["device"]
        or type(execution.get("torch_deterministic_algorithms")) is not bool
        or type(execution.get("torch_num_threads")) is not int or execution["torch_num_threads"] < 1
    ):
        raise ValueError("Incomplete or incompatible checkpoint replay contract")
    artifacts = receipt["artifacts"]
    if (len(artifacts) != 1 or not artifacts[0]["file"].endswith(".episodes.jsonl")
            or type(receipt.get("episode_count")) is not int
            or receipt["episode_count"] < 1 or receipt["episode_count"] != artifacts[0]["rows"]):
        raise ValueError("Replay contract requires one complete episode artifact")
    rows = [json.loads(line) for line in (path.parent / artifacts[0]["file"]).read_text(encoding="utf-8").splitlines()]
    if (not all(is_sha256(row.get("action_reward_trace_sha256")) for row in rows)
            or episode_payload_sha256(rows) != receipt.get("evaluation_payload_sha256")):
        raise ValueError("Replay contract episode payload identity differs")


def compare_process_replays(paths: list[Path], launches: list[dict]) -> dict:
    """Launch records must come from a subprocess harness, not copied receipts."""
    if len(paths) != 2 or len(launches) != 2:
        raise ValueError("Exactly two independent subprocess replays are required")
    receipts = [verify_receipt(path) for path in paths]
    for receipt, path in zip(receipts, paths):
        _validate_episode_replay_contract(receipt, path)
    for receipt, launch in zip(receipts, launches):
        producer = receipt["producer"]
        process_observed = any(
            all(observed.get(key) == producer.get(key) for key in ("pid", "process_started_unix"))
            for observed in launch.get("observed_processes", [])
        )
        if (type(launch.get("returncode")) is not int or launch["returncode"] != 0
                or not process_observed or not producer.get("launch_id")
                or producer["launch_id"] != launch.get("launch_id")):
            raise ValueError("Replay producer does not match its successful process launch")
    left, right = receipts
    distinct = all(left["producer"][key] != right["producer"][key]
                   for key in ("invocation_id", "process_instance_id"))
    same_contract = left["evaluation_contract_sha256"] == right["evaluation_contract_sha256"]
    same_payload = [(row["sha256"], row["rows"]) for row in left["artifacts"]] == [
        (row["sha256"], row["rows"]) for row in right["artifacts"]]
    return {"distinct_processes": distinct, "matched_contract": same_contract,
            "payload_parity": same_payload,
            "verified_process_replication": distinct and same_contract and same_payload}
