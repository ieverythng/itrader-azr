# Knowledge Layer Decisions

## Primary backend

We are using GitNexus as the runtime knowledge backend for `itrader`.

Reason:

- it has a real standalone `analyze` CLI
- it exposes MCP tools directly for Codex and Cursor
- it supports local serving plus optional HTTP MCP
- it is a better fit for agent-first repo interaction than graphify's current skill-first build path

## Rollout mode

The rollout is advisory, not grep-forbidding.

Reason:

- Codex is already reliable here
- we want a measurable comparison before hard-switching repo-reading behavior
- graph-derived context should augment, not silently override, source inspection

## Tracked artifact surface

The tracked knowledge surface is:

- `docs/knowledge/INDEX_STATUS.md`
- `docs/knowledge/wiki/`
- `docs/knowledge/module_tree.json`

The local `.gitnexus/` database is intentionally not committed.

## Hosting default

This Mac is the primary host for now.

Reason:

- it lets us test locally without loading the main PC during live Watson sessions
- same-machine Codex/Cursor usage remains the default
- remote Watson-bot access can later use the Mac-hosted HTTP MCP endpoint if needed

## Pre-commit posture

We evaluated a pre-commit path but kept it opt-in.

Reason:

- indexing is heavier than normal hygiene hooks
- wiki generation is LLM-backed
- advisory mode fits the current trust model better
