# Other — AGENTS.md

# AGENTS.md — Agent Guidance Module

## Purpose

`AGENTS.md` provides operational guidance for AI agents working in this repository. It documents the GitNexus knowledge layer integration and establishes preferred workflows for common development tasks.

This module is **advisory documentation** — it contains no executable code and defines no APIs. Instead, it serves as a reference for agents to understand:

1. What knowledge tools are available
2. When to use knowledge-layer queries vs. direct source inspection
3. How to maintain knowledge-layer freshness during development

---

## Knowledge Layer

The repository includes a GitNexus knowledge layer located at `tools/knowledge/`. This layer provides indexed, queryable metadata about the codebase that enables faster exploration and more accurate impact analysis.

### Key Concept: Dual-Path Investigation

Agents should work with two parallel information sources:

| Source | Use Case | Trust Level |
|--------|----------|-------------|
| GitNexus queries | Architecture understanding, symbol relationships, impact analysis | Primary for exploration |
| Direct source reads | Verification, spot-checking, when GitNexus unavailable | Ground truth |

**Rule**: If GitNexus and source disagree, trust the code and report the mismatch.

---

## Advisory Guidance

### Index Freshness

The knowledge layer requires periodic refresh to remain accurate:

- **Check status**: `tools/knowledge/status.sh` — confirms whether the index is fresh enough for architecture-sensitive work
- **Refresh index**: `tools/knowledge/index_repo.sh` — run after major code changes or when GitNexus reports staleness

### Wiki Generation

Do not auto-run wiki generation. Only execute when:
- The task explicitly requires refreshed tracked docs
- LLM configuration is available

### HTTP Serving (Optional)

GitNexus HTTP serving is available on this Mac via `tools/knowledge/serve_http_mcp.sh`. This is optional and provides an alternative access method for the knowledge backend.

---

## Preferred Workflows

### Exploration

```
query → context → source reads
```

1. Start with a GitNexus `query` to understand the landscape
2. Use `context` to get detailed symbol information
3. Fall back to direct source reads for verification

### Debugging

```
query (symptom) → context (suspect) → source confirmation
```

1. `query` the symptom or error pattern
2. `context` the suspected symbol or function
3. Confirm findings in source code

### Refactoring

```
context + impact → edits → graph refresh
```

1. Use `context` and `impact` tools before making edits
2. Perform the refactoring
3. Refresh the graph with `index_repo.sh`

### Live UI Sessions

When the GitNexus HTTP backend is running:

```bash
GITNEXUS_USE_HTTP=1 tools/knowledge/codex_with_gitnexus.sh
```

In this mode, `codex mcp list` will show the shared backend as `gitnexus_http`.

---

## Reference Documentation

For detailed workflow documentation, see `docs/knowledge/WORKFLOWS.md`.

---

## Module Characteristics

| Attribute | Value |
|-----------|-------|
| Type | Advisory documentation |
| Executable code | None |
| Dependencies | GitNexus tools (external) |
| Call graph | N/A — no executable functions |
| Maintenance | Update when workflows change or new tools are added |