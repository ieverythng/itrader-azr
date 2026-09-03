# GitNexus Agent Workflows

This page ports the useful intent of the GitNexus Claude skill bundle into repo-owned guidance that any agent can follow here.

The generated GitNexus Claude skill files currently live under:

- `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md`
- `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md`
- `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md`
- `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md`
- `.claude/skills/gitnexus/gitnexus-cli/SKILL.md`
- `.claude/skills/gitnexus/gitnexus-guide/SKILL.md`

They are useful, but they are Claude-oriented and local-only. This document keeps the same workflows available in repo-owned form.

## Exploring unfamiliar code

Use this when the question is "how does this area work?"

Recommended path:

1. `tools/knowledge/status.sh`
2. `./tools/knowledge/gitnexus.sh query <concept-or-symbol>`
3. `./tools/knowledge/gitnexus.sh context <symbol>`
4. Read the source files GitNexus points to

Good `itrader` starting points:

- `TaskPool`
- `_generate_task_pool`
- `train_AZR`
- `TaskSampler00`

## Debugging

Use this when the question is "why is this failing?"

Recommended path:

1. `./tools/knowledge/gitnexus.sh query <error-text-or-symptom>`
2. `./tools/knowledge/gitnexus.sh context <suspect-symbol>`
3. If the server UI is live, inspect the surrounding process cluster visually
4. Confirm the root cause in source

Good rule:

- let GitNexus narrow the search space
- let source inspection prove the root cause

## Impact analysis

Use this before non-trivial edits.

Recommended path:

1. `./tools/knowledge/gitnexus.sh impact <symbol>`
2. Review direct dependents first
3. Check whether the affected files are tests, training code, samplers, or task definitions
4. Only then start editing

Interpretation:

- depth 1 = direct break risk
- depth 2 = likely follow-on impact
- depth 3 = test and verify

## Refactoring

Use this before renames, splits, or structural cleanup.

Recommended path:

1. `./tools/knowledge/gitnexus.sh context <symbol>`
2. `./tools/knowledge/gitnexus.sh impact <symbol>`
3. make the code change
4. `tools/knowledge/post_commit_refresh.sh` after the commit, or `tools/knowledge/index_repo.sh` while iterating

For live UI work, prefer attaching Codex to the same backend:

```bash
GITNEXUS_USE_HTTP=1 tools/knowledge/codex_with_gitnexus.sh
```

In that mode, `codex mcp list` will show the shared backend as `gitnexus_http`.

## CLI and freshness

Use these when the graph itself is the task:

- `tools/knowledge/setup_gitnexus.sh`
- `tools/knowledge/index_repo.sh`
- `tools/knowledge/status.sh`
- `tools/knowledge/post_commit_refresh.sh`
- `tools/knowledge/serve_http_mcp.sh`

## What we already have vs what the Claude skills add

Already implemented here:

- repo-owned scripts
- repo-owned docs
- AGENTS guidance
- reusable bootstrap for future repos
- Codex launcher for stdio or HTTP-backed MCP

Not yet implemented as a first-class Codex global skill:

- a `$CODEX_HOME` skill bundle mirroring the GitNexus Claude skill set one-for-one

If we want that later, this document is the cleanest starting point for a Codex-native port.
