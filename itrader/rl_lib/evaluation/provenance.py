from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from importlib.metadata import version
import platform
from pathlib import Path
from typing import Any

from itrader.rl_lib.api_stack import LEGACY_API_STACK_ID, NATIVE_API_STACK_ID


CHECKPOINT_AUTHORITY_FILES_BY_API_STACK = {
    LEGACY_API_STACK_ID: (
        "algorithm_state.pkl",
        "rllib_checkpoint.json",
        "policies/default_policy/policy_state.pkl",
    ),
    NATIVE_API_STACK_ID: (
        "algorithm_state.pkl",
        "rllib_checkpoint.json",
        "learner_group/learner/rl_module/default_policy/module_state.pkl",
    ),
}
CHECKPOINT_AUTHORITY_FILES = CHECKPOINT_AUTHORITY_FILES_BY_API_STACK[LEGACY_API_STACK_ID]


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_sha256(path: Path | str) -> str:
    """Fingerprint one data file or a partitioned dataset directory."""
    dataset = Path(path).expanduser().resolve(strict=True)
    if dataset.is_file():
        return sha256_file(dataset)

    files = sorted(item for item in dataset.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Dataset directory contains no files: {dataset}")

    digest = hashlib.sha256()
    for item in files:
        relative_path = item.relative_to(dataset).as_posix()
        digest.update(f"{relative_path}\t{sha256_file(item)}\n".encode("utf-8"))
    return digest.hexdigest()


def checkpoint_sha256(path: Path | str, *, api_stack_id: str) -> str:
    """Validate and content-address a complete RLlib checkpoint directory."""
    try:
        authority_files = CHECKPOINT_AUTHORITY_FILES_BY_API_STACK[api_stack_id]
    except KeyError as exc:
        raise ValueError(f"Unknown checkpoint API stack: {api_stack_id!r}") from exc
    checkpoint = Path(path).expanduser().resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint is not a directory: {checkpoint}")
    invalid = [
        name
        for name in authority_files
        if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size
    ]
    if invalid:
        raise ValueError(
            "Checkpoint is missing required non-empty files: " + ", ".join(invalid)
        )

    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in checkpoint.rglob("*") if candidate.is_file()):
        relative_path = item.relative_to(checkpoint).as_posix()
        digest.update(f"{relative_path}\t{sha256_file(item)}\n".encode("utf-8"))
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None and value != "0" * 64


def require_indexed_integer(value: Any, field: str) -> int:
    """Do not round, parse or coerce a recorded checkpoint clock."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative JSON integer")
    return value


def verify_checkpoint_digest(
    path: Path | str,
    expected: Any,
    *,
    api_stack_id: str,
) -> str:
    """Verify original scored bytes, never bless a replacement digest."""
    if not is_sha256(expected):
        raise ValueError("Missing or invalid indexed checkpoint digest")
    actual = checkpoint_sha256(path, api_stack_id=api_stack_id)
    if actual != expected:
        raise ValueError("Checkpoint digest differs from the indexed scored checkpoint")
    return actual


def runtime_identity() -> dict[str, str]:
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in ("ray", "gymnasium", "pyarrow", "torch", "pandas")}}


def evaluation_contract_sha256(env_config: dict, dataset_hash: str) -> str:
    """Protect evaluation semantics; state masking is the declared R4 factor."""
    cfg = deepcopy(env_config)
    cfg["df_path"] = f"sha256:{dataset_hash}"
    state_keys = ("solver_state_mode", "observation_schema_id", "base_obs_dim",
                  "solver_state_dim", "agent_obs_dim", "solver_state_features")
    for key in (*state_keys, "debug_resets", "debug_reset_every", "capture_replay_trace"):
        cfg.pop(key, None)
    if "semantic_observation_contract" in cfg:
        for key in state_keys:
            cfg["semantic_observation_contract"].pop(key, None)
    cfg.setdefault("t_scale", 1.0)
    if cfg.get("start_idx_mode") == "task_spec":
        for key in ("stream_jump_every_episodes", "stream_jump_on_wrap", "stream_stride"):
            cfg.pop(key, None)
    return canonical_json_sha256(cfg)


def source_tree_fingerprint(repo_root: Path | str) -> dict[str, object]:
    """Fingerprint runtime Python sources, including untracked modules."""
    root = Path(repo_root).resolve()
    source_root = root / "itrader"
    excluded_directories = {"__pycache__", ".pytest_cache", "outputs"}
    paths: list[Path] = []
    for current_root, directory_names, file_names in os.walk(source_root):
        directory_names[:] = sorted(
            name for name in directory_names if name not in excluded_directories
        )
        current = Path(current_root)
        paths.extend(
            current / name for name in sorted(file_names) if name.endswith(".py")
        )

    digest = hashlib.sha256()
    for path in sorted(paths):
        relative_path = path.relative_to(root).as_posix()
        digest.update(f"{relative_path}\t{sha256_file(path)}\n".encode("utf-8"))
    return {
        "algorithm": "sha256(sorted relative-path TAB file-sha256 LF)",
        "python_file_count": len(paths),
        "source_tree_sha256": digest.hexdigest(),
    }
