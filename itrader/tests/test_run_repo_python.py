from pathlib import Path


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_resolver_uses_platform_native_repo_venv(tmp_path: Path):
    from itrader.tools.run_repo_python import resolve_python

    windows_python = _touch(tmp_path / ".venv" / "Scripts" / "python.exe")
    posix_python = _touch(tmp_path / ".venv" / "bin" / "python")

    assert resolve_python(tmp_path, {}, os_name="nt") == windows_python
    assert resolve_python(tmp_path, {}, os_name="posix") == posix_python


def test_resolver_never_selects_windows_executable_on_posix(tmp_path: Path):
    from itrader.tools.run_repo_python import resolve_python

    _touch(tmp_path / ".venv" / "Scripts" / "python.exe")
    current_python = _touch(tmp_path / "system" / "python3")

    assert (
        resolve_python(
            tmp_path,
            {},
            os_name="posix",
            current_executable=current_python,
        )
        == current_python
    )


def test_resolver_prefers_explicit_existing_override(tmp_path: Path):
    from itrader.tools.run_repo_python import resolve_python

    override = _touch(tmp_path / "custom" / "python")
    _touch(tmp_path / ".venv" / "bin" / "python")

    assert resolve_python(
        tmp_path,
        {"ITRADER_PYTHON_BIN": str(override)},
        os_name="posix",
    ) == override
