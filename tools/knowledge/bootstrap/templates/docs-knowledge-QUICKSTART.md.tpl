# __REPO_NAME__ GitNexus Quickstart

Run:

- `tools/knowledge/setup_gitnexus.sh`
- `tools/knowledge/index_repo.sh`
- `tools/knowledge/status.sh`

Useful test paths:

- CLI: `./tools/knowledge/gitnexus.sh query <symbol>`
- Codex: `tools/knowledge/codex_with_gitnexus.sh`
- Web UI backend: `tools/knowledge/serve_http_mcp.sh`

Optional refresh automation:

- `tools/knowledge/post_commit_refresh.sh`

Wiki generation needs an OpenAI-compatible API key or saved GitNexus config.
