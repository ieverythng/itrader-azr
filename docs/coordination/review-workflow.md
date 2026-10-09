# Review Workflow

Status: No-Mistakes is vendored, experimental advisory material, not activated.
Vendor instructions are not permission to execute/ship; discovery authorizes no pilot.

## Independent Gates

[REVIEW.md](../../REVIEW.md) remains authoritative; all eight sections apply.
Writer self-check is not independent review; fresh reviewers get diff/repo/REVIEW.
Default: `gpt-6.1-sol` / `max`; if unavailable, use the strongest supported model
at its highest supported effort and record the deviation. High-risk changes need
a second reviewer on a different model. Review every fix again.
Predeclare adversarial inputs before implementation inspection, execute in a
throwaway environment, and compare candidate behavior with the exact baseline.
For docs, use the nearest link/format/schema/rendering/workflow checks instead.
Retain every REVIEW probe and gate-attack requirement, honesty, exemption checks,
existing hooks, historical facts, and all five design-principle assessments.
Domain dual ownership, duplicated logic, and UI/template business logic block.
Reports classify `BLOCKING`/`NIT`, reproduce input/expected/actual results, number
findings `N of the K found so far.`, and state exact checks and coverage limits.
End each report with exactly `VERDICT: APPROVE` or `VERDICT: CHANGES`, nothing after.

Use `code-review` as two independent lanes on the same frozen snapshot:
- **Standards:** repo rules override smell heuristics; suspected smells are not rules.
- **Spec:** originating issue/PRD or explicit human requirements and exclusions.
Keep reports separate, missing spec unknown; no lane or green tests waive REVIEW.

## Test Evidence

Prefer public-contract integration and E2E over implementation-coupled unit growth.
Small deterministic tests must discriminate distinct invariants. Preserve the
existing suite; never delete tests to meet a count. Exercise real reset/restore/
runner failure paths with independent oracles and observable outcomes, not source
strings or mock internals. Ordinary CI must not depend on live models.
UAH's 2026-10-08 audit counted 549 cases, not 549 private-unit tests.

## Dirty And Untracked Snapshots

Base `3a3583d` alone is insufficient: committed history omits dirty/untracked bytes.
Before dispatch, retain a compact snapshot manifest with:
- Full base and HEAD SHAs, branch, scope ID, intent/spec and review-rule hashes.
- Separate target, read-dependency/context, and excluded path lists; context
  inclusion is not approval. Include relevant configuration and test inputs.
- Staged and unstaged deltas, untracked files, additions/deletions/renames,
  per-path SHA256 of actual submitted bytes, and file modes/link identities.
- Snapshot/manifest digest, capture time, and commands used to reconstruct it.
Review an immutable disposable copy; verify its bytes after copying, hooks or
line-ending conversion. Baseline and candidate must use their declared inputs.
Fail closed on missing paths, hash/scope drift or incomplete manifests. Recheck
dependencies/rules before integration or execution; stale approval cannot transfer.
Resnapshot/re-review changes and fixes; never reset the shared checkout for freshness.

## Model Telemetry

Record requested versus observed serving-model/effort, session/run IDs, telemetry
source and timestamps; dispatch labels do not attest identity. Absent telemetry:
`actual model/effort: unknown`; record deviations and unresolved model diversity.
Retain per-session counters and aggregation limits; do not invent billing totals.

## No-Mistakes Evidence And Boundary

Source: `C:/Users/Admin/PROJECTS/universal-agentic-harness/` plus
`docs/artifacts/reviews/2026-10-08_uah_no_mistakes_pilot.md`, not validation approval.
Test-policy provenance: `docs/artifacts/research/2026-10-08_uah_testing_strategy_and_no_mistakes.md` in UAH.
Pilot's two independent reviews returned CHANGES. Actual No-Mistakes v1.91.0 run
`01M4EHYDDZ8MBE4PGBQ76SZMF2` completed Intent, then FAILED/timed out in Review
after 485,084 ms (about eight minutes), with no structured final review verdict.
Test/Document/Lint/publication unreached; Rebase/Push/PR/CI skipped, not passed.
Client wait expiry was not terminal failure; no retry or pipeline fix occurred.
One launch/five rollouts is not an aggregate model/token/spend/CPU/time cap.
UAH native lane observed `gpt-6.1-sol` / `high`; review serving models were unknown.
No-Mistakes cannot replace independent reviews, approval or promotion evidence.

Any future actual tool pilot requires fresh, explicit human authorization:
- Dedicated clone with private `.git`; managed/linked worktrees are unsafe:
  init uses the common Git root/remotes. Verify isolation; never init main checkout.
- Reviewed, hash-verified snapshot; private state/config and controlled inputs.
  Do not install runtime, modify existing hooks or global config implicitly.
- Enforced aggregate model calls/sessions/tokens/spend, CPU/process/memory and
  wall-time budgets across all delegation and retries; stop on expiry, drift,
  unknown/unbounded usage, failed prerequisites or unauthorized writes/access.
- Explicitly disabled shipping stages (Push/PR/CI), Rebase and unsolicited
  repairs; finite allowed stages/write set, no `--yes` or automatic retry.
  Stop rules override vendor continuation guidance; verify child/daemon cleanup.
Native Windows only, WSL off. Nothing here authorizes tests/lint autofixes,
commits, reset/rebase, staging, push, PR/CI or a daemon in the shared repository.

## Durable Receipts

Keep compact verdicts, finding dispositions, snapshot/vendor hashes, telemetry,
commands/exit codes and limitations outside ignored directories. Verbose logs use
`docs/coordination/_local/` only after verifying its narrow ignore rule (parent-owned).
Never blanket-ignore review/research evidence or qualify from ignored logs alone.

Port receipt (2026-10-09): unchanged UAH local vendor bytes, pinned upstream
`kunchenguid/no-mistakes` commit `420adfd317e29e31bf6b5cd5d770529580084610`.
- [SKILL.md](../../.codex/skills/no-mistakes/SKILL.md) SHA256: `313374bcd2ba43da08f0ac7e179244f2804a9088b8cf44549f862e6bc9a122c5`.
- [LICENSE](../../.codex/skills/no-mistakes/LICENSE) SHA256: `945016bd37e1ba7211622ef60ee1d23ab727896ba7710edd21e8fbe983863969`.
Source/copied hashes match supplied pins; no refetch/binary attestation; pipeline NOT RUN.
Self-check: hashes/links/length/whitespace only; independent review pending; reverify LF/CRLF.
