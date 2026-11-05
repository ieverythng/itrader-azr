from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _absolute(path: str | Path) -> Path:
    """Normalize a path lexically without following junctions or symlinks."""
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _same_path(first: str | Path, second: str | Path) -> bool:
    return os.path.normcase(str(_absolute(first))) == os.path.normcase(str(_absolute(second)))


def _paths_overlap(first: str | Path, second: str | Path) -> bool:
    first_absolute = _absolute(first)
    second_absolute = _absolute(second)
    if os.name == "nt":
        try:
            first_absolute = first_absolute.resolve(strict=False)
            second_absolute = second_absolute.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ValueError("Could not canonicalize profile paths for overlap validation.") from exc
    first_path = os.path.normcase(os.path.normpath(str(first_absolute)))
    second_path = os.path.normcase(os.path.normpath(str(second_absolute)))
    try:
        common = os.path.commonpath((first_path, second_path))
    except ValueError:
        return False
    return common in (first_path, second_path)


def _assert_no_reparse_points(path: str | Path, *, required: bool) -> None:
    """Reject symlink/junction components without resolving the target path."""
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for segment in absolute.parts[1:]:
        current = current / segment
        try:
            details = os.lstat(current)
        except FileNotFoundError:
            if required:
                raise FileNotFoundError(f"Required profile path is missing: {current}")
            return
        attributes = getattr(details, "st_file_attributes", 0)
        if stat.S_ISLNK(details.st_mode) or attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError(f"Profile path contains a reparse point: {current}")


def validate_profile_identity(
    identity: Mapping[str, object],
    expected_python: str | Path,
    expected_profile: str | Path,
) -> None:
    """Require the isolated 64-bit CPython 3.11 environment identity."""
    try:
        version = identity["version"]
        identity_paths = {
            field: identity[field] for field in ("executable", "prefix", "base_prefix")
        }
        if any(
            not isinstance(path, str)
            or not path.strip()
            or "\x00" in path
            or not Path(path).is_absolute()
            for path in identity_paths.values()
        ):
            raise ValueError(
                "Interpreter identity paths must be nonempty absolute strings without NUL."
            )
        base_prefix = identity["base_prefix"]
        version_is_311 = (
            isinstance(version, list)
            and len(version) == 2
            and all(type(component) is int for component in version)
            and version == [3, 11]
        )
        valid = (
            identity["implementation"] == "CPython"
            and identity["machine"] == "AMD64"
            and type(identity["pointer_bits"]) is int
            and identity["pointer_bits"] == 64
            and version_is_311
            and _same_path(identity_paths["executable"], expected_python)
            and _same_path(identity_paths["prefix"], expected_profile)
            and not _same_path(base_prefix, expected_profile)
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ValueError("cuda-tested returned incomplete or invalid interpreter identity.") from exc
    if not valid:
        raise ValueError(
            "cuda-tested requires its isolated 64-bit CPython 3.11 AMD64 interpreter; identity differs."
        )


def _validate_profile_runtime(interpreter: Path, repo_root: Path) -> None:
    expected_profile = interpreter.parent.parent
    probe = repo_root / "itrader" / "tools" / "cuda_identity_probe.py"
    result = subprocess.run(
        [str(interpreter), str(probe)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            f"cuda-tested interpreter identity probe failed ({result.returncode}): "
            f"{result.stderr.strip()}"
        )
    try:
        identity = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("cuda-tested returned invalid interpreter identity JSON.") from exc
    if not isinstance(identity, Mapping):
        raise ValueError("cuda-tested returned invalid interpreter identity JSON.")
    validate_profile_identity(identity, interpreter, expected_profile)


def resolve_python(
    repo_root: Path,
    environ: Mapping[str, str] | None = None,
    *,
    os_name: str = os.name,
    current_executable: str | Path = sys.executable,
) -> Path:
    """Resolve a repository Python without crossing OS executable formats."""
    environment = os.environ if environ is None else environ
    raw_profile = environment.get("ITRADER_PYTHON_PROFILE")
    profile = "" if raw_profile is None else raw_profile.strip()
    override = environment.get("ITRADER_PYTHON_BIN", "").strip()
    conda_prefix = environment.get("CONDA_PREFIX", "").strip()

    if raw_profile is not None and not profile:
        raise ValueError("ITRADER_PYTHON_PROFILE cannot be empty.")
    if profile:
        if profile != "cuda-tested":
            raise ValueError(f"Unsupported Python profile: {profile!r}.")
        if os_name != "nt":
            raise ValueError("The cuda-tested Python profile requires Windows.")
        local_app_data = environment.get("LOCALAPPDATA", "").strip()
        if not local_app_data:
            raise ValueError("LOCALAPPDATA is required for the cuda-tested profile.")
        expected_profile = (
            Path(local_app_data)
            / "iTrader"
            / "envs"
            / "cuda-tested-cu128"
        )
        expected = expected_profile / "Scripts" / "python.exe"
        protected_environment = repo_root / ".venv"
        _assert_no_reparse_points(expected, required=True)
        _assert_no_reparse_points(
            protected_environment / "Scripts" / "python.exe", required=False
        )
        if _paths_overlap(expected_profile, protected_environment):
            raise ValueError("The cuda-tested profile must not overlap the protected project .venv.")
        if not expected.is_file():
            raise FileNotFoundError(f"cuda-tested interpreter is missing: {expected}")
        if override and not _same_path(override, expected):
            raise ValueError("ITRADER_PYTHON_BIN does not identify cuda-tested.")
        return expected

    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())

    if os_name == "nt":
        candidates.extend(
            (
                repo_root / ".venv" / "Scripts" / "python.exe",
                repo_root / ".itrader_env" / "Scripts" / "python.exe",
            )
        )
        if conda_prefix:
            candidates.append(Path(conda_prefix) / "python.exe")
    else:
        candidates.extend(
            (
                repo_root / ".venv" / "bin" / "python",
                repo_root / ".itrader_env" / "bin" / "python",
            )
        )
        if conda_prefix:
            candidates.append(Path(conda_prefix) / "bin" / "python")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return Path(current_executable)


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(os.path.abspath(__file__)).parents[2]
    arguments = sys.argv[1:] if argv is None else argv
    preserve_cwd = bool(arguments and arguments[0] == "--preserve-cwd")
    if preserve_cwd:
        arguments = arguments[1:]
    try:
        interpreter = resolve_python(repo_root)
        if os.environ.get("ITRADER_PYTHON_PROFILE", "").strip() == "cuda-tested":
            _validate_profile_runtime(interpreter, repo_root)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Python profile selection failed: {exc}", file=sys.stderr)
        return 2

    child_environment = None
    child_cwd = repo_root
    child_arguments = [str(interpreter), *arguments]
    if preserve_cwd:
        child_cwd = Path.cwd()
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        import_paths = [str(child_cwd), str(repo_root)]
        import_paths.extend(
            path
            for path in existing_pythonpath.split(os.pathsep)
            if path and path not in import_paths
        )
        child_environment = os.environ.copy()
        child_environment["PYTHONPATH"] = os.pathsep.join(import_paths)
        # Keep module resolution in this explicit caller/repository order; -P
        # prevents -m from adding a second implicit cwd entry ahead of it.
        child_arguments.insert(1, "-P")
    return subprocess.call(child_arguments, cwd=child_cwd, env=child_environment)


if __name__ == "__main__":
    raise SystemExit(main())
