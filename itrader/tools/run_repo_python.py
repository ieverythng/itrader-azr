from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


def resolve_python(
    repo_root: Path,
    environ: Mapping[str, str] | None = None,
    *,
    os_name: str = os.name,
    current_executable: str | Path = sys.executable,
) -> Path:
    """Resolve a repository Python without crossing OS executable formats."""
    environment = os.environ if environ is None else environ
    override = environment.get("ITRADER_PYTHON_BIN", "").strip()
    conda_prefix = environment.get("CONDA_PREFIX", "").strip()

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
    repo_root = Path(__file__).resolve().parents[2]
    arguments = sys.argv[1:] if argv is None else argv
    interpreter = resolve_python(repo_root)
    return subprocess.call([str(interpreter), *arguments], cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
