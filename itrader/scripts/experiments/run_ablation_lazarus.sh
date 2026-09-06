#!/usr/bin/env bash
# Lazarus-style manual invocation: run the episode/regime alignment ablation
# ONLY after confirming the llama.cpp server is down (avoids OOM on 16GB RAM).
#
# Usage:
#   bash itrader/scripts/experiments/run_ablation_lazarus.sh [--force]
#
# What it does:
#   1. Checks if llama.cpp is listening on port 8080 (Windows host via WSL).
#   2. If UP and --force not given: aborts with instructions.
#   3. If DOWN (or --force): launches run_ablation.py with the ablation config.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${REPO_ROOT}/itrader/scripts/experiments/configs/episode_regime_alignment_ablation.json"
PYTHON="${PYTHON:-python3}"

LLAMA_HOST="${LLAMA_HOST:-172.24.16.1}"
LLAMA_PORT="${LLAMA_PORT:-8080}"

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    *) echo "Unknown arg: $arg (use --force to override the llama.cpp check)" >&2; exit 2 ;;
  esac
done

echo "[lazarus] Checking llama.cpp at ${LLAMA_HOST}:${LLAMA_PORT} ..."

# Try a lightweight HTTP probe; curl with short timeout.
if command -v curl >/dev/null 2>&1; then
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 \
    "http://${LLAMA_HOST}:${LLAMA_PORT}/v1/models" 2>/dev/null || echo "000")
else
  # Fallback: check if the port responds at all via bash /dev/tcp (WSL may not have curl)
  if (echo > /dev/tcp/"${LLAMA_HOST}"/"${LLAMA_PORT}") 2>/dev/null; then
    HTTP_CODE="200"
  else
    HTTP_CODE="000"
  fi
fi

if [[ "$HTTP_CODE" != "000" && "$FORCE" -eq 0 ]]; then
  echo "[lazarus] ERROR: llama.cpp server is UP at ${LLAMA_HOST}:${LLAMA_PORT} (HTTP ${HTTP_CODE})."
  echo "[lazarus] Training will OOM while the Qwen model is loaded."
  echo ""
  echo "  To proceed anyway:  $0 --force"
  echo "  Or stop the server: powershell.exe -Command \"& 'C:\\Users\\Admin\\PROJECTS\\iTRADER\\itrader-azr\\tools\\stop_llama_server.ps1'\""
  echo ""
  exit 1
fi

if [[ "$HTTP_CODE" == "000" ]]; then
  echo "[lazarus] llama.cpp is DOWN. Clear to train."
else
  echo "[lazarus] llama.cpp detected but --force given. Proceeding (OOM risk)."
fi

echo "[lazarus] Launching ablation: ${CONFIG}"
cd "${REPO_ROOT}"
exec "${PYTHON}" -m itrader.scripts.experiments.run_ablation \
  "${CONFIG}"
