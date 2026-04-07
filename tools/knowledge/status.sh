#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

advisory_mode=0
if [[ "${1:-}" == "--advisory" ]]; then
  advisory_mode=1
  shift
fi

cd "${REPO_ROOT}"

if [[ ! -f "${REPO_ROOT}/.gitnexus/meta.json" ]]; then
  log "No GitNexus index found yet. Run tools/knowledge/index_repo.sh"
  exit "${advisory_mode}"
fi

run_gitnexus status "$@" || true

python3 - "${REPO_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
meta_path = repo_root / ".gitnexus" / "meta.json"
status_doc = repo_root / "docs" / "knowledge" / "INDEX_STATUS.md"
wiki_dir = repo_root / "docs" / "knowledge" / "wiki"
module_tree = repo_root / "docs" / "knowledge" / "module_tree.json"

meta = json.loads(meta_path.read_text())
stats = meta.get("stats") or {}

print("[knowledge] Local artifact summary")
print(f"[knowledge] Indexed at: {meta.get('indexedAt', 'unknown')}")
print(f"[knowledge] Indexed commit: {meta.get('lastCommit', 'unknown')}")
print(
    "[knowledge] Stats: "
    f"{stats.get('files', '?')} files, "
    f"{stats.get('nodes', '?')} nodes, "
    f"{stats.get('edges', '?')} edges, "
    f"{stats.get('processes', '?')} flows"
)

if status_doc.exists():
    stale = status_doc.stat().st_mtime < meta_path.stat().st_mtime
    print(f"[knowledge] INDEX_STATUS.md stale: {'yes' if stale else 'no'}")
else:
    print("[knowledge] INDEX_STATUS.md stale: missing")

if wiki_dir.exists():
    pages = sorted(p for p in wiki_dir.glob("*.md") if p.name != "README.md")
    print(f"[knowledge] Tracked wiki pages: {len(pages)}")
else:
    print("[knowledge] Tracked wiki pages: 0")

print(f"[knowledge] module_tree.json present: {'yes' if module_tree.exists() else 'no'}")
PY

staged_files="$(git diff --cached --name-only)"
if [[ -n "${staged_files}" ]]; then
  log "Staged files detected. After commit, the local GitNexus index may need a refresh."
fi
