# GitNexus Knowledge Layer

This repo uses GitNexus as the local code-intelligence backend and `docs/knowledge/` as the tracked human-readable surface.

## Why this exists

- give Codex and Cursor a graph-native path for architecture, debugging, impact, and rename work
- keep a repo-owned documentation surface that survives clone and handoff
- let us compare GitNexus-assisted work against the normal grep-first path before tightening any rules

## Primary commands

From the repo root:

- `tools/knowledge/setup_gitnexus.sh`
- `tools/knowledge/index_repo.sh`
- `tools/knowledge/status.sh`
- `tools/knowledge/generate_wiki.sh`
- `tools/knowledge/serve_http_mcp.sh`
- `tools/knowledge/codex_with_gitnexus.sh`
- `tools/knowledge/post_commit_refresh.sh`

If you want the shortest walkthrough first, start with [QUICKSTART.md](/Users/juanbendek/repos/itrader-azr/docs/knowledge/QUICKSTART.md).

## Default workflow on this Mac

1. Run `tools/knowledge/setup_gitnexus.sh` once.
2. Run `tools/knowledge/index_repo.sh` to build the local index.
3. Let Codex or Cursor use the tracked project MCP config.
4. Run `tools/knowledge/status.sh` whenever you want to check freshness.

The first GitNexus install on Apple Silicon can take a couple of minutes because it downloads parsers and the LadybugDB prebuilt binary. That longer first run is expected.

This rollout is intentionally advisory:

- GitNexus is preferred for architecture-sensitive tasks.
- Grep and direct file reads remain valid fallback tools.
- If GitNexus and the source disagree, the code wins.

## Tracked artifacts

- `docs/knowledge/INDEX_STATUS.md`
- `docs/knowledge/wiki/`
- `docs/knowledge/module_tree.json`
- `docs/knowledge/EVAL.md`
- `docs/knowledge/DECISIONS.md`

`docs/knowledge/wiki/` is synced from `.gitnexus/wiki/` when wiki generation is available. The local `.gitnexus/` database is never committed.

GitNexus may also generate local Claude skill files under `.claude/skills/gitnexus/`. Those stay local and are ignored in git.

## Wiki generation

`tools/knowledge/generate_wiki.sh` requires LLM access. On this machine, no default key is configured yet, so the script will refuse to run until an OpenAI-compatible API key or a saved `~/.gitnexus/config.json` is available.

ChatGPT Plus or Codex access is still useful for working in the repo, but it does not by itself provide the API key GitNexus wiki generation expects.

The simplest supported paths are:

- `OPENAI_API_KEY=... tools/knowledge/generate_wiki.sh`
- `GITNEXUS_API_KEY=... tools/knowledge/generate_wiki.sh`
- `AZURE_OPENAI_API_KEY=... tools/knowledge/generate_wiki.sh`
- `gitnexus wiki --api-key ...` once, then reuse the saved GitNexus config

Once GitNexus produces `.gitnexus/wiki/`, the repo-local sync step copies Markdown pages and `module_tree.json` into `docs/knowledge/`.

## Codex and Cursor

- Codex project config is tracked at `.codex/config.toml`
- Cursor project config is tracked at `.cursor/mcp.json`
- `tools/knowledge/codex_with_gitnexus.sh` is the deterministic Codex launcher if you want to force the repo-local GitNexus MCP wiring for a session

If a specific Cursor build ignores project-scoped MCP, copy `.cursor/mcp.json` into `~/.cursor/mcp.json` manually.

## Mac-hosted remote access

The default and most reliable mode is same-machine editor access.

If you want Watson-bot or another client on your main PC to hit this Mac, run:

```bash
tools/knowledge/serve_http_mcp.sh
```

That serves GitNexus on the LAN with MCP-over-HTTP at:

```text
http://<mac-ip>:4747/api/mcp
```

This is optional and not enabled by default, so it will not consume RAM on the main PC unless you later choose to host there instead.

## Refresh automation

We deliberately did not add a heavy auto-refresh step to normal `pre-commit` flow.

Why:

- `gitnexus analyze` can be non-trivial in time and memory
- wiki generation is LLM-backed and not deterministic enough for default commit hooks
- GitNexus freshness is commit-based, so `post-commit` is more accurate than `pre-commit`
- advisory mode is safer while we measure reliability against the current workflow

There is an opt-in manual pre-commit advisory hook script:

- `tools/knowledge/pre_commit_advisory.sh`

And there is an opt-in post-commit refresh script:

- `tools/knowledge/post_commit_refresh.sh`

The repo keeps the `pre-commit` integration advisory-only for now, and the higher-confidence automation path is documented in [QUICKSTART.md](/Users/juanbendek/repos/itrader-azr/docs/knowledge/QUICKSTART.md).

## Reuse in other repos

The reusable bootstrap layer lives under:

- `tools/knowledge/bootstrap/`

Use `tools/knowledge/bootstrap/init_repo_knowledge.sh <target-repo-path>` to seed another repo with the same runtime scripts, configs, and docs skeleton.

The bootstrap script is conservative about existing guidance and config files:

- runtime scripts are refreshed in `tools/knowledge/`
- existing `AGENTS.md`, `.codex/config.toml`, `.cursor/mcp.json`, and `docs/knowledge/*` files are left in place
- missing files are created from the tracked templates
