# Agent Notes

## Knowledge Layer

This repo has a GitNexus knowledge layer under `tools/knowledge/`.

Guidance:

- For architecture, debugging, blast-radius, and rename work, prefer the GitNexus MCP tools when they are available.
- Keep raw grep and direct file reads as a fallback and as a spot-check path. If GitNexus and source disagree, trust the code and call out the mismatch.
- Before architecture-sensitive work, run `tools/knowledge/status.sh` if you need to confirm the index is fresh.
- Refresh the local index with `tools/knowledge/index_repo.sh` after major code changes or when GitNexus reports staleness.
- Wiki sync is manual and LLM-backed: `tools/knowledge/generate_wiki.sh`.
- Optional remote MCP serving is available through `tools/knowledge/serve_http_mcp.sh`.
