# Independent Second Review: Evaluation Code

Date: 2026-10-06

Requested reviewer invocation: gpt-6-luna, xhigh (set by the parent task). This session exposes no trustworthy runtime-model metadata to independently verify the active model, so the different-model independence dimension remains unresolved unless verifiable elsewhere.

## Predeclared adversarial inputs

Recorded before opening implementation source, per `REVIEW.md`.

1. Config shape and spelling: missing config, empty object, `null`, list/scalar instead of object, unknown key, wrong-case key, alias/case variant for every accepted enum/status, and two such variants combined. Expected: invalid forms are rejected or surfaced as unknown; no silently inferred default may make an unsupported statistical qualification appear approved.
2. Config value forms: booleans where counts/durations are expected, numeric strings vs numbers, zero, negative, exact boundary, just below/above boundary, missing unit vs each supported unit, `NaN`/infinity, and a duration paired with contradictory controls. Expected: explicit validation and consistent units; displayed duration/control claims must match the actual configured values.
3. Provenance integrity: absent evidence, malformed JSON, wrong top-level type, missing/extra identity fields, duplicate identity, mismatched digest, stale digest after changing a file, relative and absolute paths to the same file, and a path outside the expected root. Expected: incomplete or unverifiable provenance remains unknown/ineligible and cannot be promoted by a label or path spelling.
4. Panel semantics: empty, one complete, one partial, one failed, and mixed complete/partial/failure evidence sets; reverse the order of two entities; combine a complete report with a missing required producer artifact. Expected: qualification is invariant to order, requires all stated prerequisites, and never upgrades partial evidence to scientific/executor closure.
5. Identity/metric reconciliation: two tasks and two metric records in both orders; duplicate task IDs; swapped task-to-metric mappings; missing task, extra metric, missing metric, and same metrics under different task identities. Expected: records join by verified identity rather than incidental order; ambiguous or absent identity is not guessed.
6. Environment reset plumbing: no task, one task, two distinct tasks in both orders, repeated reset, and a task override combined with a fixed seed. Expected: task-conditioned reset and provenance agree for each environment instance and do not leak state across resets or depend on iteration order.
7. User-visible completion/error path: each status/reason and malformed case from the producer through the report/CLI, including empty and boundary-duration cases. Expected: the displayed wording preserves the underlying state and reason; no success/qualification wording appears for unknown or failed prerequisites.

Probe execution will use public synthetic/persisted inputs only. No Ray jobs, training, replay, WSL, full-market/outcome reads, or broad full-suite run.

## Base comparison and scope

Base `d3eb4c517075b1b85506f5500a9a5552d6186e3f` resolves. All six named modules (`panel_qualification.py`, `screen_completion.py`, `run_terminal_screen.py`, `terminal_panel_report.py`, `replay_evidence.py`, and `provenance.py`) are absent from that tree, so they have no base implementation behavior to compare. The base `run_checkpoint_replay.py` had score-only parity and directly wrote episode/report files; current code adds per-episode trace payloads and producer receipts. No replay was launched, so that runtime path was reviewed statically and with pure helper probes only.

The base environment already selected `TaskSpec.start_idx` in `task_spec` mode and passed it through `DataPartition.clamp`. The current change adds explicit evaluation-panel loaders and hashes those requested tasks, but those loaders only require a non-null start. This makes the inherited clamping behavior newly relevant to a frozen-panel identity claim.

Active architecture/masterplan documentation says fresh-panel eligibility and execution gates remain pending. It also records that no panel is certified/opened, no actual terminal executor is available, and Q26 dependence-aware uncertainty is still under review. The new terminal CLI correctly reports `evaluation_authorized: false`; the Q24/Q25 functions are descriptive only and explicitly leave uncertainty unscored. This review does not approve a panel, Q26 defaults, scientific closure, or execution.

## Executed probes

- `.venv/Scripts/python.exe -m pytest -q itrader/tests/test_panel_qualification.py itrader/tests/test_screen_completion.py itrader/tests/test_terminal_panel_report.py itrader/tests/test_terminal_screen.py -k "not real_environment"` — **48 passed, 1 deselected**. These cases use synthetic metadata/JSON/checkpoint fixtures. The deselected test writes and reads synthetic Parquet; it was excluded under the no-Parquet constraint.
- Direct helper probe: legacy-only and `env_runners`-only episode payload shapes both produced the expected digest; when both were present and legacy `hist_stats` contained unrelated metrics, the helper returned `None` despite a valid nested payload.
- Direct interval probe: source `[40, 196)` and diagnostic `[196, 352)` were rejected as overlapping.
- Direct persisted-JSON probe: removing `trial_id` from both status and index records still let all 16 synthetic screen cells pass.
- Direct environment-selection probe used an in-memory `DataPartition` and the public evaluation-panel loader. A declared start of `999` loaded unchanged; the reset selector chose `25` for the valid range `[3, 25]`.

No market Parquet was read. No Ray process, training run, checkpoint replay, WSL command, real outcome data, or broad suite was run. The full `reset()` integration path was not executed because it reads Parquet; its consumer flow was inspected at `itrader/envs/itrader_env.py:578-610`.

## Findings

1 of the 4 found so far. **BLOCKING — explicit TaskSpec panel starts can be silently changed after hashing.** `itrader/rl_lib/train/train_solo_V2.py:545` accepts a task panel after checking only that `start_idx` is non-null; `itrader/rl_lib/train/train_solo_V2.py:431` hashes those requested tasks. `itrader/envs/itrader_env.py:454` sends a supplied start through `DataPartition.clamp`, and reset publishes the returned start. Reproduction: a valid one-task payload with `start_idx=999`, `horizon=5`, and a 30-row synthetic partition passes `_load_evaluation_task_panel`; with `window_size=3`, the valid range is `[3,25]` and `_choose_start_idx` returns `25`. Expected: reject an ineligible frozen start or ensure the runtime uses exactly the declared start. Actual: the recorded panel can hash `999` while evaluation uses `25`, so a fixed-panel evaluation identity can name a different market path than the one scored. The same loader is used by `run_policy_baselines.py`. The final `panel_qualification.validate_panel_metadata` does check `DataPartition.contains`, but it is a separate terminal-panel gate and does not protect these training/baseline panel loaders.

2 of the 4 found so far. **NIT — combined RLlib result shapes can drop valid replay evidence.** `itrader/rl_lib/evaluation/replay_evidence.py:28` chooses legacy `hist_stats` whenever that mapping is nonempty, even if it lacks `replay_episode_payload`, and then ignores a valid `env_runners.hist_stats` payload. Reproduction: provide both paths, with an unrelated `episode_return` entry in legacy stats and one valid traced episode in `env_runners`; expected the valid digest, actual `None`. This is fail-closed downstream (payload/checkpoint parity remains unknown), but loses available evidence.

3 of the 4 found so far. **NIT — adjacent half-open diagnostic windows are treated as overlapping.** `itrader/rl_lib/evaluation/run_checkpoint_replay.py:72` builds `[start_idx-window_size, start_idx+horizon)` intervals but compares endpoints with `<=`. Reproduction: source start `100`, horizon `96`, window `60` gives `[40,196)`; diagnostic start `256` gives `[196,352)`. Expected: accept these non-overlapping adjacent intervals; actual: raises “Diagnostic evaluation panel overlaps”. This only rejects a valid boundary arrangement.

4 of the 4 found so far. **NIT — missing trial identity passes the completed-screen gate.** `itrader/rl_lib/evaluation/screen_completion.py:126` compares terminal and status `trial_id` values without requiring either to be present. Reproduction: remove `trial_id` from both records in each otherwise complete synthetic cell; expected missing identity to remain unknown/rejected, actual `verify_completed_screen` returns all 16 terminals because `None == None`. Stable paths and config hashes still distinguish these fixtures, so this is a traceability/schema gap rather than a false score claim.

## Design principles

1. Separation of concerns — **OK**. Panel structure, historical qualification, completion checks, replay receipts, and descriptive aggregation have separate module boundaries; the terminal CLI stops at preflight.
2. Programming by intention — **Violation: `itrader/rl_lib/train/train_solo_V2.py:545` and `itrader/envs/itrader_env.py:454`**. The explicit frozen-panel path accepts one start and evaluates a clamped different start; see Finding 1.
3. Encapsulation — **OK**. Shared validation and provenance helpers are reused; aggregation does not duplicate the freshness gate.
4. High cohesion — **OK**. The new modules each focus on one evidence boundary or report.
5. Low coupling — **OK**. The terminal preflight composes the separate checks in order and does not invoke training or evaluation.

## Reviewer independence

The requested invocation was gpt-6-luna/xhigh, set explicitly by the parent task. This session exposes no trustworthy runtime-model metadata to verify the active model, so the different-model independence dimension remains unresolved.

VERDICT: CHANGES
