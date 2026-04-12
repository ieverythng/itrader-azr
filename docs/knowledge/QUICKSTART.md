# GitNexus Quickstart

This is the shortest path to test the GitNexus knowledge layer in `itrader-azr`.

## What already works without any API key

- local indexing
- Codex and Cursor MCP usage
- GitNexus CLI queries
- the GitNexus HTTP backend for the web UI
- tracked knowledge artifacts in `docs/knowledge/`

## What still needs an API-compatible model

- `tools/knowledge/generate_wiki.sh`

That is the only part of this rollout that needs an OpenAI-compatible chat endpoint. If `~/.gitnexus/config.json` is already configured, the repo-local wiki commands can use it directly.

## First-run bootstrap

From the repo root:

```bash
tools/knowledge/setup_gitnexus.sh
tools/knowledge/index_repo.sh
tools/knowledge/status.sh
```

Healthy output should show:

- `Status: up-to-date`
- node / edge / flow counts
- `module_tree.json present: yes`

## Fastest ways to use it

### 1. CLI checks

Try a few graph-native commands:

```bash
./tools/knowledge/gitnexus.sh list
./tools/knowledge/gitnexus.sh query TaskPool
./tools/knowledge/gitnexus.sh context TaskPool
./tools/knowledge/gitnexus.sh impact TaskPool
```

This is the quickest way to compare GitNexus against your normal grep workflow.

### 2. Codex with GitNexus forced on

```bash
tools/knowledge/codex_with_gitnexus.sh
```

Inside that session, GitNexus MCP is injected explicitly for this repo.

Use this when you want the most deterministic Codex test path.

If the GitNexus HTTP backend is already running, attach Codex to that shared backend instead:

```bash
GITNEXUS_USE_HTTP=1 tools/knowledge/codex_with_gitnexus.sh
```

Or point it at a custom GitNexus MCP endpoint:

```bash
GITNEXUS_MCP_URL=http://127.0.0.1:4747/api/mcp tools/knowledge/codex_with_gitnexus.sh
```

In that HTTP-backed mode, `codex mcp list` will show the shared backend as `gitnexus_http`.

### 3. Web UI / observability view

Start the local backend:

```bash
tools/knowledge/serve_http_mcp.sh
```

Then open:

- [GitNexus Web UI](https://gitnexus.vercel.app)

The local backend is exposed at:

```text
http://127.0.0.1:4747/api/mcp
```

When the UI backend is already live, this HTTP-attached Codex path is usually the cleanest setup.

If you want another machine on your LAN to use this Mac-hosted backend, use:

```text
http://<mac-ip>:4747/api/mcp
```

## How freshness works

GitNexus tracks freshness against the current git commit.

That means:

- indexing after code edits but before commit can still look fresh until the commit changes
- after a new commit, re-running `index_repo.sh` or `post_commit_refresh.sh` makes the graph match the new `HEAD`

This is why an automatic `post-commit` hook is more correct than a `pre-commit` hook for keeping the graph current.

## Optional automation

Manual refresh:

```bash
tools/knowledge/index_repo.sh
```

Optional post-commit refresh:

```bash
tools/knowledge/post_commit_refresh.sh
```

If you want git to run it after every commit, add this to `.git/hooks/post-commit`:

```bash
#!/usr/bin/env bash
exec "$(git rev-parse --show-toplevel)/tools/knowledge/post_commit_refresh.sh"
```

Then make it executable:

```bash
chmod +x .git/hooks/post-commit
```

## Wiki generation and model access

The current implementation issue is not with indexing, MCP, or the web UI. Those are already working.

The only unresolved part is wiki generation, because GitNexus wants an OpenAI-compatible chat endpoint for `gitnexus wiki`.

Important:

- a ChatGPT Plus subscription does not itself provide a reusable API key for GitNexus wiki generation
- ChatGPT/Codex usage and API usage are managed separately
- Codex can still help you work in this repo today; this limitation only affects `gitnexus wiki`

Supported practical options for this repo:

- OpenAI API key from the API platform
- Azure OpenAI key plus base URL
- an OpenAI-compatible self-hosted or remote endpoint
- Ollama, if exposed through an OpenAI-compatible `/v1` endpoint and paired with a capable chat model

Example shape:

```bash
tools/knowledge/generate_wiki.sh --base-url http://<host>:11434/v1 --api-key dummy --model <model-name>
```

If you prefer the lower-level wrapper, this also works and syncs the tracked Markdown wiki into `docs/knowledge/wiki/` afterward:

```bash
tools/knowledge/gitnexus.sh wiki --provider openai --base-url http://<host>:11434/v1 --api-key dummy --model <model-name>
```

## Bootstrap for another repo

Use the reusable bootstrap:

```bash
tools/knowledge/bootstrap/init_repo_knowledge.sh /absolute/path/to/other-repo
```

That copies the knowledge tooling and creates missing docs/config files without overwriting an existing `AGENTS.md` or Codex/Cursor config.
