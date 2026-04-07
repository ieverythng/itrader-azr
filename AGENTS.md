# Agent Notes

## Knowledge Layer

This repo has a GitNexus knowledge layer under `tools/knowledge/`.

Advisory-mode guidance:

- Prefer GitNexus MCP tools for architecture, debugging, impact, and rename work when they are available.
- Keep raw grep and direct file reads as a fallback and as a spot-check path. If GitNexus and source disagree, trust the code and call out the mismatch.
- Run `tools/knowledge/status.sh` when you need to confirm whether the local index is fresh enough for architecture-sensitive work.
- Refresh the local index with `tools/knowledge/index_repo.sh` after major code changes or when GitNexus reports staleness.
- Do not auto-run wiki generation unless the task actually needs refreshed tracked docs and LLM config is available.
- GitNexus HTTP serving is optional on this Mac through `tools/knowledge/serve_http_mcp.sh`.
