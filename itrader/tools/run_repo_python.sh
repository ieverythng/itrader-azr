#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN_OVERRIDE="${ITRADER_PYTHON_BIN:-}"
UV_PYTHON_POSIX="${REPO_ROOT}/.venv/bin/python"
UV_PYTHON_WINDOWS="${REPO_ROOT}/.venv/Scripts/python.exe"
PYTHON_BIN_POSIX="${REPO_ROOT}/.itrader_env/bin/python"
PYTHON_BIN_WINDOWS="${REPO_ROOT}/.itrader_env/Scripts/python.exe"
CONDA_PYTHON_POSIX="${CONDA_PREFIX:-}/bin/python"
CONDA_PYTHON_WINDOWS="${CONDA_PREFIX:-}/python.exe"

if [[ -n "${PYTHON_BIN_OVERRIDE}" && -x "${PYTHON_BIN_OVERRIDE}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_OVERRIDE}"
elif [[ -x "${UV_PYTHON_POSIX}" ]]; then
  PYTHON_BIN="${UV_PYTHON_POSIX}"
elif [[ -f "${UV_PYTHON_WINDOWS}" ]]; then
  PYTHON_BIN="${UV_PYTHON_WINDOWS}"
elif [[ -x "${PYTHON_BIN_POSIX}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_POSIX}"
elif [[ -f "${PYTHON_BIN_WINDOWS}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_WINDOWS}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PYTHON_POSIX}" ]]; then
  PYTHON_BIN="${CONDA_PYTHON_POSIX}"
elif [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PYTHON_WINDOWS}" ]]; then
  PYTHON_BIN="${CONDA_PYTHON_WINDOWS}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Unable to resolve Python interpreter. Tried:" >&2
  echo "  ITRADER_PYTHON_BIN=${PYTHON_BIN_OVERRIDE:-<unset>}" >&2
  echo "  ${UV_PYTHON_POSIX}" >&2
  echo "  ${UV_PYTHON_WINDOWS}" >&2
  echo "  ${PYTHON_BIN_POSIX}" >&2
  echo "  ${PYTHON_BIN_WINDOWS}" >&2
  echo "  ${CONDA_PYTHON_POSIX}" >&2
  echo "  ${CONDA_PYTHON_WINDOWS}" >&2
  echo "Set ITRADER_PYTHON_BIN, run uv sync, activate conda, or create .itrader_env." >&2
  exit 1
fi

exec "${PYTHON_BIN}" "$@"
