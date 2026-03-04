#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.itrader_env/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing repo-local interpreter: ${PYTHON_BIN}" >&2
  echo "Create it first (example): python3 -m venv .itrader_env" >&2
  exit 1
fi

exec "${PYTHON_BIN}" "$@"
