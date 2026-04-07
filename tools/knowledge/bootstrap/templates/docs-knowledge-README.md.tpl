# __REPO_NAME__ Knowledge Layer

This repo uses GitNexus as a local code-intelligence backend plus a tracked `docs/knowledge/` surface for humans and agents.

Primary commands:

- `tools/knowledge/setup_gitnexus.sh`
- `tools/knowledge/index_repo.sh`
- `tools/knowledge/status.sh`
- `tools/knowledge/generate_wiki.sh`
- `tools/knowledge/serve_http_mcp.sh`
- `tools/knowledge/codex_with_gitnexus.sh`
- `tools/knowledge/post_commit_refresh.sh`

Notes:

- first install can take a couple of minutes on Apple Silicon
- wiki generation needs an OpenAI-compatible API key or a saved `~/.gitnexus/config.json`
- the bootstrap skips existing guidance and config files instead of overwriting them
- `QUICKSTART.md` is the fastest path for a new repo
