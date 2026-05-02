# Other — itrader-tools

# itrader-tools: run_repo_python.sh

## Overview

`run_repo_python.sh` is a utility script that ensures Python commands execute using a repository-local virtual environment interpreter. It provides a consistent entry point for running Python scripts within the project, eliminating ambiguity about which Python interpreter to use.

## Purpose

This script solves a common problem in multi-developer environments: ensuring all Python execution uses the correct virtual environment. Rather than relying on developers to manually activate a virtual environment or remember the correct interpreter path, this script:

- Automatically locates the repository root
- Points to a dedicated `.itrader_env` virtual environment
- Validates the interpreter exists before execution
- Passes through all arguments to the Python interpreter

## Usage

### Basic Usage

```bash
# Run a Python script
./itrader/tools/run_repo_python.sh script.py

# Run Python with arguments
./itrader/tools/run_repo_python.sh -m pytest tests/

# Start an interactive Python session
./itrader/tools/run_repo_python.sh
```

### Integration with Other Scripts

```bash
#!/usr/bin/env bash
# Example: A script that needs Python
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${SCRIPT_DIR}/../itrader/tools/run_repo_python.sh"

$PYTHON -c "import sys; print(sys.prefix)"
```

## Prerequisites

Before using this script, create the virtual environment:

```bash
cd /path/to/repo/root
python3 -m venv .itrader_env
source .itrader_env/bin/activate
pip install -r requirements.txt  # or project-specific setup
```

## How It Works

```bash
set -euo pipefail
```
Enables strict mode:
- `-e`: Exit immediately if any command fails
- `-u`: Treat unset variables as an error
- `-o pipefail`: Pipeline fails if any command in it fails

```bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
```
Determines the repository root by navigating relative to the script location. This approach works regardless of where the script is called from.

```bash
PYTHON_BIN="${REPO_ROOT}/.itrader_env/bin/python"
```
Constructs the path to the virtual environment's Python interpreter.

```bash
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing repo-local interpreter: ${PYTHON_BIN}" >&2
  echo "Create it first (example): python3 -m venv .itrader_env" >&2
  exit 1
fi
```
Validates the interpreter exists and is executable. If not, provides a helpful error message with remediation steps.

```bash
exec "${PYTHON_BIN}" "$@"
```
Replaces the current shell process with Python, passing through all arguments. Using `exec` ensures Python inherits the same process ID and that signals are handled correctly.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Virtual environment doesn't exist | Prints error to stderr, exits with code 1 |
| Python binary not executable | Prints error to stderr, exits with code 1 |
| Python script fails | Exit code propagates from Python |

## Design Decisions

**Why `exec`?** Using `exec` replaces the bash process rather than spawning a child. This means:
- Python receives signals directly (no bash wrapper)
- Exit codes pass through without modification
- No zombie processes

**Why `.itrader_env`?** The hidden directory (leading dot) keeps the virtual environment out of sight during normal filesystem navigation while remaining discoverable for those who need it.

**Why validate existence?** Failing fast with a helpful message is better than a cryptic "command not found" error later in execution.