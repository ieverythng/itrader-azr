"""Guarded, one-shot provisioner for the reviewed CUDA test profile.

This command is intentionally separate from the normal repository launcher.
It never changes the caller's environment and never targets the root .venv.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import tomllib
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH = REPO_ROOT / "docs" / "coordination" / "_local" / "cuda-uv-bridge28"
PROOF_SCRATCH = REPO_ROOT / "docs" / "coordination" / "_local" / "cuda-uv-bridge27"
PROOF_STATE_PATH = PROOF_SCRATCH / "provision-state.json"
PROOF_LOG_PATH = PROOF_SCRATCH / "logs" / "wheel-acquire.log"
PROOF_REVIEW_PATH = PROOF_SCRATCH / "reviews" / "candidate-review.json"
CANDIDATE = SCRATCH / "candidate"
REVIEW_DIR = SCRATCH / "reviews"
LOG_DIR = SCRATCH / "logs"
UV_CONFIG_ROOT = SCRATCH / "uv-config"
UV_APPDATA = UV_CONFIG_ROOT / "user"
SYSTEM_APP_DATA = Path(r"C:\ProgramData")
STATE_PATH = SCRATCH / "provision-state.json"
EXPECTED_REPO = Path(r"C:\Users\Admin\PROJECTS\iTRADER\itrader-azr")
NATIVE_PYTHON = Path(r"C:\Users\Admin\itrader\python.exe")
UV_EXE = Path(r"C:\Users\Admin\itrader\Scripts\uv.exe")
LOCAL_APP_DATA = Path(r"C:\Users\Admin\AppData\Local")
CACHE_DIR = LOCAL_APP_DATA / "uv" / "cache"
PROFILE = LOCAL_APP_DATA / "iTrader" / "envs" / "cuda-tested-cu128"
WHEEL_DIR = LOCAL_APP_DATA / "iTrader" / "wheel-proof"
WHEEL_NAME = "torch-2.9.0.dev20250801+cu128-cp311-cp311-win_amd64.whl"
WHEEL = WHEEL_DIR / WHEEL_NAME
WHEEL_URL = (
    "https://download.pytorch.org/whl/nightly/cu128/"
    "torch-2.9.0.dev20250801%2Bcu128-cp311-cp311-win_amd64.whl"
)
WHEEL_SHA256 = "8ff2cbe2a88bebfc14b71fea17f4937d77f0ff70aa9e4cebd6836b5bbe82765d"
WHEEL_BYTES = 3_062_428_787
CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
DEFAULT_INDEX_URL = "https://pypi.org/simple"
CPU_DEFAULT_MARKER = "sys_platform == 'win32' or sys_platform == 'linux'"
CUDA_TESTED_MARKER = (
    "sys_platform == 'win32' and platform_machine == 'AMD64' and "
    "implementation_name == 'cpython' and python_version == '3.11'"
)
RAM_START = 4 * 1024**3
RAM_ABORT = 2 * 1024**3
DISK_START = 30 * 1024**3
DISK_ABORT = 15 * 1024**3
LOCK_TIMEOUT = 300
SYNC_TIMEOUT = 1200
SMOKE_TIMEOUT = 120
PRIOR_BUDGET_CHARGED_SECONDS = 87
CLEANUP_RESERVE_SECONDS = 300
TOTAL_TIMEOUT = 2400 - PRIOR_BUDGET_CHARGED_SECONDS - CLEANUP_RESERVE_SECONDS
LEASE_TIMEOUT = 2700
OFFICIAL_HOSTS = {"download.pytorch.org", "download-r2.pytorch.org"}
BASE_PROJECT_SHA256 = "11ce4f40f9973549d27cf740b437fe767755ad0d12f7ade74c704a2b0c4e9c16"
BASE_LOCK_SHA256 = "4abdc1e100e8e86712babcb01ada0a23741c1b29abd896034017f2aad711126e"
PINNED_UV_VERSION_OUTPUTS = (
    "uv 0.11.7",
    "uv 0.11.7 (9d177269e 2026-04-15 x86_64-pc-windows-msvc)",
)


class ProvisioningError(RuntimeError):
    """A fail-closed provisioning precondition or command failed."""


@dataclass(frozen=True)
class RuntimeIdentity:
    system: str
    machine: str
    implementation: str
    major: int
    minor: int
    pointer_bits: int


def require_cuda_target(identity: RuntimeIdentity) -> None:
    if identity != RuntimeIdentity("Windows", "AMD64", "CPython", 3, 11, 64):
        raise ProvisioningError(
            "cuda-tested supports only 64-bit CPython 3.11 on Windows AMD64; "
            f"observed {identity!r}. Refusing CUDA group selection."
        )


def require_uv_cli_version(returncode: int, stdout: str) -> str:
    if stdout.endswith("\r\n"):
        observed = stdout[:-2]
    elif stdout.endswith("\n"):
        observed = stdout[:-1]
    else:
        observed = stdout
    if returncode != 0 or observed not in PINNED_UV_VERSION_OUTPUTS:
        raise ProvisioningError(
            f"Expected uv 0.11.7; observed {observed!r} with exit code {returncode}."
        )
    return observed


def build_cuda_sync_command(uv_exe: str, project: str, python: str) -> list[str]:
    return [
        uv_exe,
        "sync",
        "--locked",
        "--project",
        project,
        "--python",
        python,
        "--no-default-groups",
        "--group",
        "dev",
        "--group",
        "cuda-tested",
    ]


def verify_review_receipt(
    receipt: dict[str, Any], expected_hashes: dict[str, str]
) -> None:
    if not isinstance(receipt, dict):
        raise ProvisioningError("Review receipt must be an object.")
    if receipt.get("artifact_sha256") != expected_hashes:
        raise ProvisioningError(
            "Review receipt does not bind the current artifact hashes."
        )
    reviewers = receipt.get("reviewers")
    if not isinstance(reviewers, list) or len(reviewers) != 2:
        raise ProvisioningError("Exactly two independent review receipts are required.")
    identities = set()
    models = set()
    for reviewer in reviewers:
        if not isinstance(reviewer, dict) or reviewer.get("verdict") != "APPROVE":
            raise ProvisioningError("Both independent reviewers must return APPROVE.")
        if reviewer.get("blocking_findings") != 0:
            raise ProvisioningError("A review receipt reports blocking findings.")
        identity, model = reviewer.get("id"), reviewer.get("model")
        if not isinstance(identity, str) or not isinstance(model, str):
            raise ProvisioningError("Review receipts must identify reviewer and model.")
        identity = identity.strip().casefold()
        model = model.strip().casefold()
        if not identity or not model:
            raise ProvisioningError("Review receipts must identify reviewer and model.")
        identities.add(identity)
        models.add(model)
    if len(identities) != 2 or len(models) != 2:
        raise ProvisioningError("Reviewers and model identities must be independent.")


def verify_approved_source_snapshot(
    reviewed_hashes: dict[str, str],
    current_hashes: dict[str, str],
    receipt: dict[str, Any],
) -> None:
    if reviewed_hashes != current_hashes:
        raise ProvisioningError("Reviewed source snapshot changed before sync.")
    verify_review_receipt(receipt, current_hashes)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity() -> RuntimeIdentity:
    return RuntimeIdentity(
        system=platform.system(),
        machine=platform.machine(),
        implementation=platform.python_implementation(),
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        pointer_bits=struct.calcsize("P") * 8,
    )


def _load_process_scope():
    source = REPO_ROOT / "itrader" / "rl_lib" / "evaluation" / "process_scope.py"
    spec = importlib.util.spec_from_file_location("itrader_cuda_process_scope", source)
    if spec is None or spec.loader is None:
        raise ProvisioningError("Could not load the repository process-scope owner.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ReplayProcessScope


ReplayProcessScope = _load_process_scope()


def _assert_no_reparse(path: Path, *, allow_missing_leaf: bool = False) -> Path:
    full = Path(os.path.abspath(path))
    current = Path(full.anchor)
    components = full.parts[1:]
    for index, component in enumerate(components):
        current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            if allow_missing_leaf:
                continue
            raise ProvisioningError(
                f"Required path component does not exist: {current}"
            )
        attributes = getattr(info, "st_file_attributes", 0)
        if current.is_symlink() or attributes & 0x400:
            raise ProvisioningError(
                f"Path contains a symbolic link or reparse point: {current}"
            )
        resolved = current.resolve(strict=True)
        if os.path.normcase(str(resolved)) != os.path.normcase(str(current)):
            raise ProvisioningError(f"Path is not canonical: {current} -> {resolved}")
        if index == len(components) - 1 and allow_missing_leaf and not current.exists():
            break
    return full


def _child_environment(**overrides: str) -> dict[str, str]:
    child = os.environ.copy()
    for name in tuple(child):
        if name.upper().startswith("UV_"):
            child.pop(name, None)
    for name in (
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
    ):
        child.pop(name, None)
    child.update(
        {
            "UV_PROJECT_ENVIRONMENT": str(PROFILE),
            "UV_CACHE_DIR": str(CACHE_DIR),
            "UV_PYTHON_DOWNLOADS": "never",
            "PYTHONNOUSERSITE": "1",
            "APPDATA": str(UV_APPDATA),
            "XDG_CONFIG_HOME": str(UV_CONFIG_ROOT),
        }
    )
    child.update(overrides)
    return child


def _resources() -> tuple[int, int]:
    import psutil

    free_ram = int(psutil.virtual_memory().available)
    free_disk = int(shutil.disk_usage(LOCAL_APP_DATA).free)
    return free_ram, free_disk


def _check_abort_floors() -> tuple[int, int]:
    free_ram, free_disk = _resources()
    if free_ram < RAM_ABORT:
        raise ProvisioningError(f"Free RAM fell below 2 GiB: {free_ram} bytes.")
    if free_disk < DISK_ABORT:
        raise ProvisioningError(f"Free C: disk fell below 15 GiB: {free_disk} bytes.")
    return free_ram, free_disk


def _assert_owner(path: Path) -> str:
    target = _assert_no_reparse(path)
    ps = (
        Path(os.environ.get("WINDIR", r"C:\Windows"))
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )
    script = (
        "$ErrorActionPreference='Stop'; "
        "$me=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value; "
        "$owner=(Get-Acl -LiteralPath $env:ITRADER_CUDA_OWNER_PATH).GetOwner("
        "[System.Security.Principal.SecurityIdentifier]).Value; "
        "[Console]::Out.WriteLine($me+'|'+$owner)"
    )
    process = _run_small(
        [str(ps), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
        env=_child_environment(ITRADER_CUDA_OWNER_PATH=str(target)),
        timeout=10,
        cwd=REPO_ROOT,
    )
    try:
        current_sid, owner_sid = process.stdout.strip().split("|", 1)
    except ValueError as exc:
        raise ProvisioningError(
            f"Could not establish path owner for {target}."
        ) from exc
    if process.returncode != 0 or current_sid != owner_sid:
        raise ProvisioningError(f"Path is not owned by the current user: {target}.")
    return current_sid


def _check_start_floors() -> tuple[int, int]:
    free_ram, free_disk = _resources()
    if free_ram < RAM_START:
        raise ProvisioningError(
            f"Start RAM floor not met: {free_ram / 1024**3:.2f} GiB."
        )
    if free_disk < DISK_START:
        raise ProvisioningError(
            f"Start disk floor not met: {free_disk / 1024**3:.2f} GiB."
        )
    return free_ram, free_disk


def _run_small(command: list[str], *, env: dict[str, str], timeout: int, cwd: Path):
    _check_abort_floors()
    process = None
    with ReplayProcessScope() as scope:
        process = scope.start(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        scope.resume()
        try:
            stdout, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            cleanup = scope.stop()
            if cleanup["cleanup_active_processes"] != 0 or cleanup["cleanup_errors"]:
                raise ProvisioningError(
                    f"Timed out and process-tree cleanup failed: {cleanup}"
                ) from exc
            raise ProvisioningError(
                f"Bounded identity command timed out: {command[0]}."
            ) from exc
        except BaseException as exc:
            cleanup = scope.stop()
            if cleanup["cleanup_active_processes"] != 0 or cleanup["cleanup_errors"]:
                raise ProvisioningError(
                    f"Identity command failed and cleanup failed: {cleanup}"
                ) from exc
            raise
        cleanup = scope.stop()
    if cleanup["cleanup_active_processes"] != 0 or cleanup["cleanup_errors"]:
        raise ProvisioningError(f"Process-tree cleanup failed: {cleanup}")
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout.decode("utf-8", errors="replace"),
        None,
    )


def _run_command(
    stage: str,
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    total_deadline: float,
    env: dict[str, str],
) -> str:
    stage_deadline = min(time.monotonic() + timeout, total_deadline)
    if time.monotonic() >= stage_deadline:
        raise ProvisioningError(f"Stage has no remaining budget: {stage}")
    _check_start_floors()
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True, exist_ok=False)
    _assert_no_reparse(LOG_DIR)
    _assert_owner(LOG_DIR)
    log_path = LOG_DIR / f"{stage}.log"
    if log_path.exists():
        raise ProvisioningError(
            f"Refusing to overwrite stage log; no retry: {log_path}"
        )
    scope = ReplayProcessScope()
    process = None
    with log_path.open("xb") as output:
        with scope:
            if time.monotonic() >= stage_deadline:
                raise ProvisioningError(
                    f"Stage deadline expired before launch: {stage}"
                )
            process = scope.start(
                command,
                cwd=str(cwd),
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            scope.resume()
            try:
                while True:
                    if time.monotonic() >= stage_deadline:
                        raise ProvisioningError(
                            f"Stage timed out after its fixed budget: {stage}"
                        )
                    if process.poll() is not None:
                        if time.monotonic() >= stage_deadline:
                            raise ProvisioningError(
                                f"Stage timed out after its fixed budget: {stage}"
                            )
                        break
                    _check_abort_floors()
                    remaining = stage_deadline - time.monotonic()
                    time.sleep(min(0.25, max(0.0, remaining)))
            except BaseException as exc:
                cleanup = scope.stop()
                if (
                    cleanup["cleanup_active_processes"] != 0
                    or cleanup["cleanup_errors"]
                ):
                    raise ProvisioningError(
                        f"Stage stopped and process cleanup failed: {cleanup}"
                    ) from exc
                raise
            return_code = process.returncode
            cleanup = scope.stop()
    if cleanup["cleanup_active_processes"] != 0 or cleanup["cleanup_errors"]:
        raise ProvisioningError(f"Contained process left survivors: {cleanup}")
    if return_code != 0:
        tail = log_path.read_bytes()[-4096:].decode("utf-8", errors="replace")
        raise ProvisioningError(
            f"Stage {stage} exited {return_code}; log tail:\n{tail}"
        )
    return str(log_path)


def _artifact_hashes(paths: dict[str, Path]) -> dict[str, str]:
    result = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ProvisioningError(f"Reviewed artifact is missing: {path}")
        _assert_no_reparse(path)
        result[name] = _sha256(path)
    return result


def _read_review(path: Path, hashes: dict[str, str]) -> dict[str, Any]:
    _assert_no_reparse(path)
    _assert_owner(path)
    if not path.is_file():
        raise ProvisioningError(f"Required review receipt is missing: {path}")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvisioningError(f"Invalid review receipt: {path}") from exc
    verify_review_receipt(receipt, hashes)
    return receipt


def _candidate_artifacts() -> dict[str, Path]:
    return {
        "pyproject.toml": CANDIDATE / "pyproject.toml",
        "README.md": CANDIDATE / "README.md",
        "driver.py": REPO_ROOT / "itrader" / "tools" / "provision_cuda_tested.py",
        "runtime_smoke.py": REPO_ROOT / "itrader" / "tools" / "cuda_runtime_smoke.py",
        "launcher.ps1": REPO_ROOT / "itrader" / "tools" / "invoke_cuda_tested.ps1",
        "identity_probe.py": REPO_ROOT / "itrader" / "tools" / "cuda_identity_probe.py",
        "contract_tests.py": REPO_ROOT
        / "itrader"
        / "tests"
        / "test_cuda_provisioning_contract.py",
        "process_scope.py": REPO_ROOT
        / "itrader"
        / "rl_lib"
        / "evaluation"
        / "process_scope.py",
    }


def _verify_profile_metadata(path: Path) -> dict[str, Any]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ProvisioningError(f"Candidate pyproject is invalid: {path}") from exc
    project = data.get("project", {})
    if not isinstance(project, dict):
        raise ProvisioningError("Candidate project metadata must be a TOML table.")
    package_dependencies = project.get("dependencies", [])
    if not isinstance(package_dependencies, list) or any(
        not isinstance(item, str) for item in package_dependencies
    ):
        raise ProvisioningError("Candidate project dependencies must be strings.")
    if any(
        re.match(r"\s*torch(?:\s|@|[<>=!~;\[]|$)", item, re.IGNORECASE)
        for item in package_dependencies
    ):
        raise ProvisioningError(
            "Torch must not remain in unconditional project dependencies."
        )
    groups = data.get("dependency-groups", {})
    required_groups = {
        "dev",
        "etl",
        "research",
        "llm-local",
        "cpu-default",
        "cuda-tested",
    }
    if not required_groups.issubset(groups):
        raise ProvisioningError("Candidate removed a preserved dependency group.")
    expected_cpu_group = [
        f"torch==2.3.1+cpu; {CPU_DEFAULT_MARKER}",
        "torch==2.3.1; sys_platform == 'darwin'",
    ]
    expected_cuda_group = [f"torch==2.9.0.dev20250801+cu128; {CUDA_TESTED_MARKER}"]
    if groups["cpu-default"] != expected_cpu_group:
        raise ProvisioningError(
            "CPU reproduction group must retain the exact Windows/Linux and Darwin pins."
        )
    if groups["cuda-tested"] != expected_cuda_group:
        raise ProvisioningError(
            "CUDA group must retain the exact tested Torch version and platform marker."
        )
    tool_config = data.get("tool", {})
    if not isinstance(tool_config, dict):
        raise ProvisioningError("Candidate tool metadata must be a TOML table.")
    uv_config = tool_config.get("uv", {})
    if not isinstance(uv_config, dict):
        raise ProvisioningError("Candidate uv metadata must be a TOML table.")
    if uv_config.get("default-groups") != ["dev", "cpu-default"]:
        raise ProvisioningError("Bare uv sync must remain the CPU-default profile.")
    conflicts = uv_config.get("conflicts", [])
    if conflicts != [[{"group": "cpu-default"}, {"group": "cuda-tested"}]]:
        raise ProvisioningError(
            "CPU and CUDA profiles are not declared mutually exclusive."
        )
    indexes = uv_config.get("index", [])
    if indexes != [{"name": "pytorch-cpu", "url": CPU_INDEX_URL, "explicit": True}]:
        raise ProvisioningError("CPU Torch index must be explicitly scoped.")
    sources = uv_config.get("sources", {})
    if not isinstance(sources, dict):
        raise ProvisioningError("Candidate uv sources must be a TOML table.")
    source_entries = sources.get("torch", [])
    expected_source_entries = [
        {
            "index": "pytorch-cpu",
            "group": "cpu-default",
            "marker": CPU_DEFAULT_MARKER,
        },
        {
            "url": WHEEL_URL,
            "group": "cuda-tested",
            "marker": CUDA_TESTED_MARKER,
        },
    ]
    if source_entries != expected_source_entries:
        raise ProvisioningError(
            "Candidate CPU/CUDA source routing differs from the approved contract."
        )
    return {
        "default_groups": uv_config["default-groups"],
        "groups": sorted(required_groups),
    }


def _check_base() -> dict[str, Any]:
    require_cuda_target(_identity())
    if Path(os.path.abspath(REPO_ROOT)) != EXPECTED_REPO:
        raise ProvisioningError(f"Unexpected repository root: {REPO_ROOT}")
    if Path(sys.executable).resolve() != NATIVE_PYTHON:
        raise ProvisioningError(
            f"Run this gate only with the exact interpreter {NATIVE_PYTHON}."
        )
    if (
        Path(os.path.abspath(NATIVE_PYTHON)) != NATIVE_PYTHON
        or not NATIVE_PYTHON.is_file()
    ):
        raise ProvisioningError(
            "The exact native CPython 3.11 interpreter is unavailable."
        )
    if Path(os.path.abspath(UV_EXE)) != UV_EXE or not UV_EXE.is_file():
        raise ProvisioningError("The exact reviewed uv executable is unavailable.")
    if Path(os.path.abspath(os.environ.get("LOCALAPPDATA", ""))) != LOCAL_APP_DATA:
        raise ProvisioningError(
            "LOCALAPPDATA differs from the exact approved profile root."
        )
    for path in (REPO_ROOT, SCRATCH, CANDIDATE, CACHE_DIR):
        _assert_no_reparse(path)
    if PROFILE.exists():
        raise ProvisioningError(
            f"Isolated target already exists; refusing overwrite: {PROFILE}"
        )
    _assert_no_reparse(PROFILE, allow_missing_leaf=True)
    _assert_owner(REPO_ROOT)
    _assert_owner(CANDIDATE)
    _assert_owner(SCRATCH)
    _assert_owner(CACHE_DIR)
    _assert_owner(LOCAL_APP_DATA)
    _assert_no_reparse(UV_CONFIG_ROOT, allow_missing_leaf=True)
    _assert_no_reparse(UV_APPDATA, allow_missing_leaf=True)
    if not UV_CONFIG_ROOT.exists():
        UV_CONFIG_ROOT.mkdir(exist_ok=False)
    if not UV_APPDATA.exists():
        UV_APPDATA.mkdir(exist_ok=False)
    _assert_no_reparse(UV_CONFIG_ROOT)
    _assert_owner(UV_CONFIG_ROOT)
    _assert_no_reparse(UV_APPDATA)
    _assert_owner(UV_APPDATA)
    if (UV_APPDATA / "uv" / "uv.toml").exists():
        raise ProvisioningError("Isolated uv user configuration is not empty.")
    system_uv_config = SYSTEM_APP_DATA / "uv" / "uv.toml"
    if system_uv_config.exists():
        raise ProvisioningError(
            f"Unreviewed system uv configuration would affect resolution: {system_uv_config}"
        )
    current = CANDIDATE
    while True:
        if (current / "uv.toml").exists():
            raise ProvisioningError(
                f"An unreviewed project uv.toml would shadow the pyproject profile: {current / 'uv.toml'}"
            )
        if current == current.parent:
            break
        current = current.parent
    protected_environment = (REPO_ROOT / ".venv").resolve(strict=True)
    profile_resolved = PROFILE.resolve(strict=False)
    if (
        profile_resolved == protected_environment
        or profile_resolved.is_relative_to(protected_environment)
        or protected_environment.is_relative_to(profile_resolved)
    ):
        raise ProvisioningError("Target profile aliases the protected root .venv.")
    if UV_EXE.name != "uv.exe" or NATIVE_PYTHON.name != "python.exe":
        raise ProvisioningError(
            "Native command identities differ from the reviewed executable names."
        )
    version = _run_small(
        [str(UV_EXE), "--version"],
        env=_child_environment(),
        timeout=10,
        cwd=REPO_ROOT,
    )
    uv_version_output = require_uv_cli_version(version.returncode, version.stdout)
    free_ram, free_disk = _resources()
    return {
        "identity": asdict(_identity()),
        "repo_root": str(REPO_ROOT),
        "candidate": str(CANDIDATE),
        "native_python": str(NATIVE_PYTHON),
        "uv": str(UV_EXE),
        "uv_version_output": uv_version_output,
        "uv_cache": str(CACHE_DIR),
        "target_profile_absent": str(PROFILE),
        "free_ram_gib": round(free_ram / 1024**3, 2),
        "free_disk_gib": round(free_disk / 1024**3, 2),
    }


def _write_state(state: dict[str, Any], *, create: bool = False) -> None:
    temporary = STATE_PATH.with_suffix(".json.tmp")
    _assert_no_reparse(STATE_PATH, allow_missing_leaf=True)
    _assert_no_reparse(temporary, allow_missing_leaf=True)
    if temporary.exists():
        raise ProvisioningError(
            f"State temporary exists; refusing overwrite: {temporary}"
        )
    if create and STATE_PATH.exists():
        raise ProvisioningError(
            "Provisioning state already exists; no retry is permitted."
        )
    if not create:
        if not STATE_PATH.is_file():
            raise ProvisioningError("Provisioning state is missing.")
        if _assert_no_reparse(STATE_PATH) != STATE_PATH:
            raise ProvisioningError("Provisioning state path is not canonical.")
    with temporary.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATE_PATH)


def _verify_lock(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as lock_file:
            lock = tomllib.load(lock_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ProvisioningError(f"Generated lock is unreadable: {path}") from exc
    packages = lock.get("package", [])
    if not isinstance(packages, list) or any(
        not isinstance(package, dict) for package in packages
    ):
        raise ProvisioningError("Generated lock package records are malformed.")
    required = {
        "ray": "2.58.0",
        "gymnasium": "1.2.2",
        "numpy": "1.26.4",
        "pandas": "2.1.4",
        "pyarrow": "17.0.0",
    }
    for name, version in required.items():
        matching = [package for package in packages if package.get("name") == name]
        if len(matching) != 1 or matching[0].get("version") != version:
            found = [package.get("version") for package in matching]
            raise ProvisioningError(
                f"Lock changed protected pin {name}: {found} != [{version}]"
            )
    cuda_entries = [
        package
        for package in packages
        if package.get("name") == "torch"
        and package.get("version") == "2.9.0.dev20250801+cu128"
    ]
    if len(cuda_entries) != 1:
        raise ProvisioningError(
            "Lock does not contain exactly one exact CUDA torch record."
        )
    cuda_entry = cuda_entries[0]
    source = cuda_entry.get("source")
    if not isinstance(source, dict):
        raise ProvisioningError("Lock CUDA torch source record is malformed.")
    source_url = source.get("url", "")
    if not _matches_approved_wheel_url(source_url):
        raise ProvisioningError(
            f"Lock CUDA torch source differs from reviewed URL: {source_url}"
        )
    wheels = cuda_entry.get("wheels")
    if (
        not isinstance(wheels, list)
        or len(wheels) != 1
        or not isinstance(wheels[0], dict)
        or not _matches_approved_wheel_url(wheels[0].get("url", ""))
        or wheels[0].get("hash") != f"sha256:{WHEEL_SHA256}"
    ):
        raise ProvisioningError(
            "Lock must bind exactly one CUDA wheel to the approved URL and SHA256."
        )
    cpu_entries = [
        package
        for package in packages
        if package.get("name") == "torch" and package.get("version") == "2.3.1+cpu"
    ]
    cpu_source = cpu_entries[0].get("source") if len(cpu_entries) == 1 else None
    if not isinstance(cpu_source, dict) or not _matches_registry(
        cpu_source.get("registry", ""), CPU_INDEX_URL
    ):
        raise ProvisioningError(
            "Lock does not preserve the explicit CPU control source."
        )
    darwin_cpu_entries = [
        package
        for package in packages
        if package.get("name") == "torch" and package.get("version") == "2.3.1"
    ]
    darwin_cpu_source = (
        darwin_cpu_entries[0].get("source") if len(darwin_cpu_entries) == 1 else None
    )
    if not isinstance(darwin_cpu_source, dict) or not _matches_registry(
        darwin_cpu_source.get("registry", ""), DEFAULT_INDEX_URL
    ):
        raise ProvisioningError("Lock does not preserve the Darwin CPU source branch.")
    return {"protected_versions": required, "cuda_source": source_url}


def _matches_approved_wheel_url(value: str) -> bool:
    if not isinstance(value, str):
        return False
    try:
        actual = urllib.parse.urlsplit(value)
        approved = urllib.parse.urlsplit(WHEEL_URL)
        return (
            actual.scheme.casefold() == approved.scheme.casefold() == "https"
            and actual.hostname is not None
            and approved.hostname is not None
            and actual.hostname.casefold() == approved.hostname.casefold()
            and actual.port == approved.port
            and actual.username is None
            and actual.password is None
            and urllib.parse.unquote(actual.path) == urllib.parse.unquote(approved.path)
            and not actual.query
            and not actual.fragment
        )
    except ValueError:
        return False


def _matches_registry(value: str, expected: str) -> bool:
    if not isinstance(value, str):
        return False
    try:
        actual = urllib.parse.urlsplit(value)
        approved = urllib.parse.urlsplit(expected)
        return (
            actual.scheme.casefold() == approved.scheme.casefold() == "https"
            and actual.hostname is not None
            and approved.hostname is not None
            and actual.hostname.casefold() == approved.hostname.casefold()
            and actual.port == approved.port
            and actual.username is None
            and actual.password is None
            and actual.path.rstrip("/") == approved.path.rstrip("/")
            and not actual.query
            and not actual.fragment
        )
    except ValueError:
        return False


def _candidate_review_hashes() -> dict[str, str]:
    return _artifact_hashes(_candidate_artifacts())


def _profile_child_env() -> dict[str, str]:
    return _child_environment()


def _new_state(
    preflight: dict[str, Any], candidate_hashes: dict[str, str]
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "scope": "DEV-CUDA28",
        "state": "reuse_proof_lock_started",
        "started_utc": now,
        "candidate_sha256": candidate_hashes,
        "preflight": preflight,
        "wheel_expected_sha256": WHEEL_SHA256,
        "prior_attempt_budget": {
            "cuda27_charged_seconds": PRIOR_BUDGET_CHARGED_SECONDS,
            "remaining_seconds": 2400 - PRIOR_BUDGET_CHARGED_SECONDS,
            "cleanup_reserve_seconds": CLEANUP_RESERVE_SECONDS,
        },
        "stages": {
            "acquire": "reused_verified_proof",
            "lock": "not_started",
            "sync": "not_started",
            "smoke": "not_started",
        },
    }


def _read_retained_wheel_proof() -> dict[str, Any]:
    for path in (PROOF_STATE_PATH, PROOF_LOG_PATH, PROOF_REVIEW_PATH, WHEEL):
        _assert_no_reparse(path)
        _assert_owner(path)
    try:
        state = json.loads(PROOF_STATE_PATH.read_text(encoding="utf-8"))
        log_receipt = json.loads(PROOF_LOG_PATH.read_text(encoding="utf-8"))
        review = json.loads(PROOF_REVIEW_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvisioningError("Retained CUDA27 wheel proof is unreadable.") from exc
    candidate_hashes = state.get("candidate_sha256")
    if (
        state.get("scope") != "DEV-CUDA27"
        or state.get("state") != "acquire_lock_started"
        or state.get("stages", {}).get("acquire") != "complete"
        or state.get("stages", {}).get("lock") != "started"
        or state.get("wheel_expected_sha256") != WHEEL_SHA256
        or state.get("wheel_log") != str(PROOF_LOG_PATH)
        or not isinstance(candidate_hashes, dict)
    ):
        raise ProvisioningError(
            "Retained CUDA27 state does not prove completed acquisition."
        )
    verify_review_receipt(review, candidate_hashes)
    receipt = state.get("wheel")
    if not isinstance(receipt, dict) or receipt != log_receipt:
        raise ProvisioningError(
            "Retained wheel state differs from its acquisition log."
        )
    if (
        receipt.get("path") != str(WHEEL)
        or receipt.get("bytes") != WHEEL_BYTES
        or receipt.get("sha256") != WHEEL_SHA256
        or receipt.get("source_url") != WHEEL_URL
        or receipt.get("final_host") not in OFFICIAL_HOSTS
        or WHEEL.stat().st_size != WHEEL_BYTES
        or _sha256(WHEEL) != WHEEL_SHA256
    ):
        raise ProvisioningError("Retained wheel does not match the approved proof.")
    return receipt


def _elapsed_budget(state: dict[str, Any]) -> float:
    try:
        started = datetime.fromisoformat(state["started_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProvisioningError("Provisioning start timestamp is invalid.") from exc
    if started.tzinfo is None or started.utcoffset() is None:
        raise ProvisioningError("Provisioning start timestamp must include a timezone.")
    elapsed = (datetime.now(UTC) - started.astimezone(UTC)).total_seconds()
    if elapsed < 0:
        raise ProvisioningError("Provisioning start timestamp is in the future.")
    total_budget = min(TOTAL_TIMEOUT, LEASE_TIMEOUT)
    remaining = total_budget - elapsed
    if remaining <= 0:
        raise ProvisioningError(
            f"Provisioning budget expired after {elapsed:.1f} seconds; no renewal."
        )
    return time.monotonic() + remaining


def _lock_with_retained_proof() -> dict[str, Any]:
    preflight = _check_base()
    root_project_hash = _sha256(REPO_ROOT / "pyproject.toml")
    root_lock_hash = _sha256(REPO_ROOT / "uv.lock")
    if root_project_hash != BASE_PROJECT_SHA256 or root_lock_hash != BASE_LOCK_SHA256:
        raise ProvisioningError(
            "Root pyproject/lock changed from the frozen pre-CUDA baseline."
        )
    profile_contract = _verify_profile_metadata(CANDIDATE / "pyproject.toml")
    wheel_receipt = _read_retained_wheel_proof()
    free_ram, free_disk = _resources()
    if free_ram < RAM_START or free_disk < DISK_START:
        raise ProvisioningError(
            f"Start floors not met: RAM {free_ram / 1024**3:.2f} GiB, disk {free_disk / 1024**3:.2f} GiB."
        )
    candidate_hashes = _candidate_review_hashes()
    _read_review(REVIEW_DIR / "candidate-review.json", candidate_hashes)
    state = _new_state(preflight, candidate_hashes)
    state["profile_contract"] = profile_contract
    state["wheel"] = wheel_receipt
    state["proof_source"] = {
        "scope": "DEV-CUDA27",
        "state": str(PROOF_STATE_PATH),
        "acquisition_log": str(PROOF_LOG_PATH),
        "sha256_verified_offline": WHEEL_SHA256,
    }
    _write_state(state, create=True)
    total_deadline = _elapsed_budget(state)
    state["stages"]["lock"] = "started"
    _write_state(state)
    lock_command = [
        str(UV_EXE),
        "lock",
        "--project",
        str(CANDIDATE),
        "--python",
        str(NATIVE_PYTHON),
    ]
    state["lock_log"] = _run_command(
        "uv-lock",
        lock_command,
        cwd=CANDIDATE,
        timeout=LOCK_TIMEOUT,
        total_deadline=total_deadline,
        env=_profile_child_env(),
    )
    state["lock_contract"] = _verify_lock(CANDIDATE / "uv.lock")
    state["candidate_lock_sha256"] = _sha256(CANDIDATE / "uv.lock")
    state["stages"]["lock"] = "complete"
    state["state"] = "awaiting_lock_review"
    _write_state(state)
    return state


def _sync_and_smoke() -> dict[str, Any]:
    _check_base()
    _assert_no_reparse(STATE_PATH)
    _assert_owner(STATE_PATH)
    if not STATE_PATH.is_file():
        raise ProvisioningError("No retained-proof/lock stage is recorded.")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    current_candidate_hashes = _candidate_review_hashes()
    candidate_receipt = _read_review(
        REVIEW_DIR / "candidate-review.json", current_candidate_hashes
    )
    verify_approved_source_snapshot(
        state.get("candidate_sha256"), current_candidate_hashes, candidate_receipt
    )
    total_deadline = _elapsed_budget(state)
    if (
        state.get("scope") != "DEV-CUDA28"
        or state.get("stages", {}).get("acquire") != "reused_verified_proof"
        or state.get("state") != "awaiting_lock_review"
        or state.get("stages", {}).get("lock") != "complete"
    ):
        raise ProvisioningError(
            "No reviewed CUDA28 retained-proof lock is ready for sync."
        )
    if _sha256(CANDIDATE / "uv.lock") != state.get("candidate_lock_sha256"):
        raise ProvisioningError("Candidate lock bytes changed after generation.")
    root_hashes = _artifact_hashes(
        {
            "pyproject.toml": REPO_ROOT / "pyproject.toml",
            "uv.lock": REPO_ROOT / "uv.lock",
        }
    )
    candidate_root_hashes = _artifact_hashes(
        {
            "pyproject.toml": CANDIDATE / "pyproject.toml",
            "uv.lock": CANDIDATE / "uv.lock",
        }
    )
    if root_hashes != candidate_root_hashes:
        raise ProvisioningError(
            "Root package metadata/lock do not exactly match the reviewed candidate."
        )
    _read_review(REVIEW_DIR / "lock-review.json", candidate_root_hashes)
    if PROFILE.exists():
        raise ProvisioningError(
            "Target CUDA environment already exists; no overwrite or retry."
        )
    if state["stages"].get("sync") != "not_started":
        raise ProvisioningError("Sync was already attempted; no retry is permitted.")
    state["stages"]["sync"] = "started"
    state["state"] = "sync_started"
    _write_state(state)
    state["sync_log"] = _run_command(
        "uv-sync",
        build_cuda_sync_command(str(UV_EXE), str(REPO_ROOT), str(NATIVE_PYTHON)),
        cwd=REPO_ROOT,
        timeout=SYNC_TIMEOUT,
        total_deadline=total_deadline,
        env=_profile_child_env(),
    )
    state["stages"]["sync"] = "complete"
    _write_state(state)

    target_python = PROFILE / "Scripts" / "python.exe"
    _assert_no_reparse(PROFILE)
    _assert_owner(PROFILE)
    state["stages"]["smoke"] = "started"
    _write_state(state)
    state["identity_log"] = _run_command(
        "target-identity",
        [
            str(target_python),
            str(REPO_ROOT / "itrader" / "tools" / "cuda_identity_probe.py"),
        ],
        cwd=REPO_ROOT,
        timeout=SMOKE_TIMEOUT,
        total_deadline=total_deadline,
        env=_profile_child_env(),
    )
    identity_text = Path(state["identity_log"]).read_text(encoding="utf-8").strip()
    identity_lines = [
        line for line in identity_text.splitlines() if line.startswith("{")
    ]
    if not identity_lines:
        raise ProvisioningError("Target interpreter identity output is missing.")
    identity_payload = json.loads(identity_lines[-1])
    expected_python = str(target_python.resolve()).casefold()
    if (
        str(Path(identity_payload["executable"]).resolve()).casefold()
        != expected_python
    ):
        raise ProvisioningError(
            "Installed target interpreter identity does not match the fixed profile."
        )
    if identity_payload["implementation"] != "CPython" or identity_payload[
        "version"
    ] != [3, 11]:
        raise ProvisioningError("Installed target interpreter is not CPython 3.11.")
    state["target_identity"] = identity_payload
    state["runtime_smoke_log"] = _run_command(
        "cuda-runtime-smoke",
        [
            str(target_python),
            str(REPO_ROOT / "itrader" / "tools" / "cuda_runtime_smoke.py"),
        ],
        cwd=REPO_ROOT,
        timeout=SMOKE_TIMEOUT,
        total_deadline=total_deadline,
        env=_profile_child_env(),
    )
    state["stages"]["smoke"] = "complete"
    state["state"] = "cuda_tensor_smoke_passed"
    state["finished_utc"] = datetime.now(UTC).isoformat()
    _write_state(state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "reuse-proof-lock", "sync-smoke"),
        default="preflight",
    )
    arguments = parser.parse_args(argv)
    try:
        if arguments.stage == "preflight":
            result = _check_base()
        elif arguments.stage == "reuse-proof-lock":
            result = _lock_with_retained_proof()
        else:
            result = _sync_and_smoke()
    except (ProvisioningError, OSError, ValueError) as exc:
        print(f"CUDA provisioning stopped: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
