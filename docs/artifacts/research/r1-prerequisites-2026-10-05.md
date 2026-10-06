# R1 prerequisite and browser closeout — 2026-10-05

Branch: `agent/Watson-dev`, base `d3eb4c5`; uncommitted work remains for review.
This follow-up does not rerun training, change rewards, implement R1, or integrate
the isolated native Ray migration. The eight corrected-control terminal replays
remain accepted. Historical returns and receipts are unchanged.

## Implemented and bounded

| Concern | Implemented | Still required |
| --- | --- | --- |
| Q21 completion | Disk-backed all-16 declaration/config/status/index/checkpoint verification; only predetermined terminal checkpoints returned | Frozen manifest, runner enforcement of completion plus independent freshness certificate |
| Q23 freshness | Metadata-only retained exposure inventory, with source hashes; never authorizes panel freeze | Reconstruct randomized starts and unknown legacy controls, qualify eligible calendar sessions, then freeze the panel |
| Q24 aggregation | Four horizons per anchor, anchors per symbol, then twelve symbols equally; paired candidate/control deltas | Invoke from terminal runner with matched producer/control identities and reviewed block uncertainty |
| Q25 duration | Net bps per actual bar; per-horizon, per-anchor, exposure, action, turnover and liquidation diagnostics; truncated/censored input rejected | Collect complete real 128-task receipts; do not relabel old 32-task means as this estimand |

Panel session IDs must be canonicalized by the upstream calendar audit. The
reporter treats them as opaque IDs; it cannot certify calendar aliases or
non-exposure. Reporting is descriptive, not a promotion gate. R1 remains
unimplemented. The accepted Q20–Q25 design is unchanged.

## Historical exposure evidence

**2026-10-06 follow-up:** the sixteen unknown-config control sidecars recover
recorded episode windows. The 26 randomized runs remain source-unbound; their
six dirty 82-file fingerprints cannot be reconstructed from the 72-file Git
tree. Conservative possible exposure leaves zero certified fresh candidates.
The counts below describe the original inventory, not the newer qualification
verdict. See the [bounded audit](r1-panel-qualification-2026-10-06.md).

The [inventory](r1-prerequisites-2026-10-05-evidence/exposure_inventory.json)
covers retained resolved configs under `C:/Users/Admin/ray_results`,
`C:/itrader-ray`, and repository experiment outputs. Resolved paths deduplicate
junctions. Dataset SHA256 is
`20aac96de6da77e20beeaff214d380ef06184faf301ff0dd53bed0ffd4c04ce9`.

- 163 matching configurations: 121 explicit-start, 26 randomized, 16 unknown.
- The randomized configurations contain 594 evaluation reports without retained
  top-level episode-start payloads. This is a metadata limitation, not evidence
  that every possible window was evaluated.
- The earlier 4,094 eligible slots do not by themselves establish non-exposure.
- No panel has been frozen/opened. Strict freshness remains the default. A user
  choice to run an exposure-unknown development panel would need explicit labeling
  and would not create fresh-validation or economic-promotion evidence.

## Dashboard and experiment access

The [Research Dashboard](../../research/itrader_research_dashboard.html#history)
now puts dated operational notices at the bottom behind a History navigation
link. The [Experiment Browser](../../research/itrader_experiment_browser.html#runs)
adds individual run selection, timeline plots, exact raw metric tables, source
hashes, and pairwise run comparison alongside experiment-level comparison.

Net return is displayed as percent; Sharpe remains per square-root bar, not
annualized. Actions, clip fraction, entropy, value fit and other available
diagnostics are exposed without filling missing values. The historical actor
search retains eight individual member traces and two population summaries.
Historical PBT records do not establish that every requested optimizer mutation
reached the policy; the corrected compatibility run remains separately labeled.

The staged R4 first seed remains +0.392761% net / +0.018892 Sharpe per square-root
bar and about 95.08% short. The second seed is flat; this is not replicated uplift.
Neither a more detailed chart nor the new reporter changes that conclusion.

## Validation and independent review

Fresh reviewers exercised persisted completion fixtures, malformed and conflicting
identities, checkpoint tampering, source mismatches, action-fraction boundaries,
unequal anchor counts, reordered episodes and extreme numeric inputs. Repairs
include duplicate JSON-key rejection, finite paired-rate checks, one metric-unit
owner, Windows-case run identity checks, and population/source binding.

Final commands and review verdicts are recorded in the
[review closeout](r1-prerequisites-2026-10-05-evidence/review_closeout.md).
Interactive local-file browser verification is blocked by the app's
browser policy; no workaround was used. Generated-content consistency, JavaScript
syntax and local-link checks are separate from visual verification.

## Next execution order

1. Resolve historical-start qualification without selecting on outcomes.
2. Freeze the calendar-qualified 128-task panel and 16-cell manifest.
3. Wire the terminal runner to require all-cell completion and panel qualification,
   matched replay/control producer identities, and the new weighted reporter.
4. Independently review that end-to-end gate before opening terminal evaluation.
5. Implement R1 only after this preparation boundary is accepted; compare the
   four representations with exact cost versus explicitly matched causal staging.

No additional R0 search, WSL restart, staging, commit or push was performed.
