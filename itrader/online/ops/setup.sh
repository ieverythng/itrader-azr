#!/usr/bin/env bash
# One-shot setup for the iTRADER online stack on a Mac or Linux box. Idempotent.
#   bash itrader/online/ops/setup.sh            # venv, deps, data, bench, tests
#   bash itrader/online/ops/setup.sh --no-data  # skip the download (about 10 minutes)
set -euo pipefail
cd "$(dirname "$0")/../../.."          # repo root
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
. .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet pandas numpy pyarrow pytest requests pydantic certifi
echo "python: $(python --version)  pandas: $(python -c 'import pandas;print(pandas.__version__)')"
if [ "${1:-}" != "--no-data" ]; then
  python -m itrader.online sync
fi
python -m itrader.online bench
python -m pytest itrader/tests/test_online_*.py -q
echo
echo "Next: python -m itrader.online frame      (one dry framing pass)"
echo "      python -m itrader.online paper --once"
echo "      python -m itrader.online dashboard  (then open data/online/runs/dashboard.html)"
