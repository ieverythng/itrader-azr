#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

if [[ -n "${GITNEXUS_MCP_URL:-}" ]]; then
  exec codex \
    -C "${REPO_ROOT}" \
    -c "mcp_servers.gitnexus.url=\"${GITNEXUS_MCP_URL}\"" \
    "$@"
fi

if [[ "${GITNEXUS_USE_HTTP:-0}" == "1" ]]; then
  exec codex \
    -C "${REPO_ROOT}" \
    -c 'mcp_servers.gitnexus.url="http://127.0.0.1:4747/api/mcp"' \
    "$@"
fi

exec codex \
  -C "${REPO_ROOT}" \
  -c "mcp_servers.gitnexus.command=\"./tools/knowledge/gitnexus.sh\"" \
  -c 'mcp_servers.gitnexus.args=["mcp"]' \
  "$@"
