# DEV-REVIEW30: R1 and Evaluation Evidence Review

Date: 2026-10-12
Branch / HEAD: `agent/Watson-dev` / `8f9fcf94e028b51d883a1a484c2c9466a3d6b547`
Snapshot: `docs/coordination/_local/dev-review30/snapshot.md`, SHA256
`913de0c10c37ddf0f3a9306375bd671200e4a4726e7be4e7e1fcb3f590075ee6`
Review status: **CHANGES** — three blocking findings, three nits.

## Scope and execution boundary

This was a read-only review of the dirty R1/evaluation work and exact dependency
changes. The snapshot bound 48 source, test, and runtime hash entries; both
review lanes confirmed those entries still match. The previous R1 manifest
matches 27 of 30 entries only; its approval applies only to identical bytes and
the original scope. The `next-actions.md` context was rebound and checked at
SHA256 `dc9141b24abd7f09204151ab12f16d59f3156a0d889faad7c3c9a3df6f5feaee`.

The pre-read probe matrix covered missing/non-finite evidence, ambiguous cell
identities, accounting boundaries, selector/restore conflicts, altered
checkpoint identities, truthful multi-record presentation, and dependency
changes. Two independent review lanes inspected the same frozen snapshot and
`REVIEW.md`. Their observed serving models/effort were unavailable; model
diversity is therefore unknown.

Executed: branch/HEAD/status inspection, source/base diffs, prior-snapshot
comparison, SHA256 checks, static dependency comparison, relative-file-link
checks across 13 active Markdown documents, and scoped `git diff --check`
(exit 0). The dependency comparison found 76 shared non-Torch package version
changes and removal of `decorator`; a direct repository import was not found.
This static result does not prove runtime compatibility. Four large test files
currently contain 175 named `test_` functions total (`test_train_config.py` 98,
`test_policy_baselines.py` 26, `test_run_ablation.py` 20, and
`test_experiment_browser.py` 31). These are inventory counts, not newly added
tests or executed cases; parametrization changes case counts.

Not executed: behavioral probes or behavioral base comparison, pytest,
healthcheck, pre-commit, renderer/schema execution, Ray/Torch/GPU imports,
package operations, or GitNexus. They were deferred under the active
TRAINING-CUDA30 preparation/execution scope to avoid overlapping its conditional
resource lease. A human terminal reported six CPU pre-commit
hooks passed, but exact-byte/count/skip and CUDA-profile parity were not
independently verified here. No-Mistakes was not run and is not a substitute
for these review gates. No source, test, runtime, package, or Git-shipping
changes were made.

## Findings

### 1. BLOCKING — R4 acceptance can omit independent process evidence

**1 of the 6 found so far.** In
[`analyze_r4_v1_ablation.py`](../../../itrader/scripts/experiments/analyze_r4_v1_ablation.py:255),
`_terminal_replay_matches` verifies a receipt and local score/contract values
but does not require the shared process-replay comparison or a launch witness.
This conflicts with R1 readiness's requirement for distinct verified process
instances and matched execution identity.

- Input: the fixture at `itrader/tests/test_analyze_r4_v1_ablation.py:186-228`,
  whose invocations are constructed in one test process, have no launch
  witnesses, use `evaluation_seed=None`, and omit complete runtime/execution
  metadata.
- Expected: attribution and promotion remain ineligible until two distinct
  process instances are independently witnessed and the same contract,
  checkpoint, and payload are verified.
- Actual: source predicates allow the fixture's `promotion_passes=True` path;
  the test expects that result. This was source-observed, not re-executed.

Route terminal acceptance through the full process-replay gate or a receipt
that carries equivalent verified per-invocation launch evidence.

### 2. BLOCKING — replay receipts accept incomplete producer identity/time fields

**2 of the 6 found so far.** `verify_receipt` in
[`replay_evidence.py`](../../../itrader/rl_lib/evaluation/replay_evidence.py:104)
checks a self-consistent identity digest and non-empty completion string, but
does not enforce the required producer field types, UTC timestamp syntax, or
chronology.

- Input: remove `started_at_utc` from an otherwise valid receipt; separately,
  replace `completed_at_utc` with arbitrary non-empty text. Missing host/PID/
  process-start fields also pass if the self-hash is recomputed.
- Expected: reject incomplete or malformed identity/time evidence; keep older
  records readable only as explicitly unknown-process evidence.
- Actual: the generic verifier returns the receipt. This was source-observed,
  not dynamically executed.

Validate the producer contract strictly before it can support replay or
promotion acceptance.

### 3. BLOCKING — impossible action fractions enter ex-post regime reports

**3 of the 6 found so far.**
[`regime_conditioned_summary`](../../../itrader/rl_lib/evaluation/block_uncertainty.py:240)
checks action-fraction shape and finiteness but not bounds or sum-to-one,
unlike the shared validator in
[`evidence_values.py`](../../../itrader/rl_lib/evaluation/evidence_values.py:11).

- Input: use the public checkpoint-evidence fixture and change a bullish
  candidate's `action_fractions` to `[2, -1, 0]`.
- Expected: reject before reporting action fractions.
- Actual: source-observed analysis accepts and reports short/flat/long fractions
  `2/-1/0`. This violates truthful reporting and duplicates a weaker domain
  validation rule; the probe was not executed.

Reuse the canonical action-fraction validator and preserve the malformed-input
rejection as a public analysis scenario.

### 4. NIT — cross-panel uncertainty checks compare counts, not identities

**4 of the 6 found so far.** In
[`block_uncertainty.py`](../../../itrader/rl_lib/evaluation/block_uncertainty.py:100)
and the seed-pooling path near line 158, crossed horizons/anchors are checked
by cardinality, not exact identity sets.

- Input: crossed rows for anchors with horizon sets `{96,128}` and `{96,200}`;
  separately, two seeds with different anchor identities but equal counts.
- Expected: either require matching identity sets or label this explicitly as
  count-only exploratory pooling.
- Actual: source-observed checks pass because counts match. Active docs already
  disclose the limitation and exclude this exploratory method from qualified
  Q26 inference; it is not a hidden qualification pass. The probe was not run.

Do not use these intervals as a Q26-qualified inference method without the
separate support/method/confirmation gates.

### 5. NIT — early replay-launch failure can discard cleanup diagnostics

**5 of the 6 found so far.** `ReplayProcessScope.__exit__` in
[`process_scope.py`](../../../itrader/rl_lib/evaluation/process_scope.py:134)
does not retain `stop()`'s result. An initial process-identity lookup or
`resume()` failure can therefore hide cleanup errors; the launcher can exit
before writing a launch receipt.

- Input: cause initial process-identity lookup to raise, then make cleanup
  report a non-empty scope or stop error.
- Expected: preserve the primary launch exception and cleanup outcome together.
- Actual: source-observed control flow returns only the identity exception and
  no launch receipt. No surviving process was observed or claimed by this
  review; no failure injection was executed.

Extend the existing initial-identity-failure case to check this second failure
and keep normal completion's strict cleanup behavior.

### 6. NIT — active next-actions text conflates CPU lifecycle with CUDA/performance

**6 of the 6 found so far.** The historical-boundary summary at
[`next-actions.md`](../../project-state/next-actions.md:677) says R1 lacks a Ray
lifecycle/performance qualification, while the current project record contains
the completed bounded CPU lifecycle22 receipt. CUDA qualification and training
performance remain pending, but these are distinct statuses.

- Input: read the current blocker sentence alongside the lifecycle22 receipt.
- Expected: distinguish completed bounded CPU lifecycle evidence from pending
  CUDA and performance qualification.
- Actual: the sentence groups the completed CPU lifecycle with the still-open
  qualifications. This is a documentation/source consistency finding, not a
  claim that lifecycle22 proves PPO performance.

Correct the active summary while retaining dated historical statements.

## Design principles

- Separation of concerns: **OK**. Solver, reward, evidence, qualification,
  and presentation ownership remain distinct.
- Programming by intention: **violation**, finding 3 at
  `block_uncertainty.py:240`; fraction-labeled values must obey the fraction
  contract. Count-only cross-panel checks are additionally a disclosed nit.
- Encapsulation: **nit**, finding 5 at `process_scope.py:134`; cleanup outcomes
  disappear on exceptional exit.
- High cohesion: **OK** in the inspected modules.
- Low coupling: **OK**; no additional blocking coupling issue was found.

## Test ROI and R1 boundary

The test base is large in the highest-churn areas, but the counts alone do not
show waste. Avoid an arbitrary reduction target and do not fold every invariant
into one brittle end-to-end scenario. Keep small deterministic unit tests for
distinct boundaries: canonical evidence/ranges/hashes, exact accounting,
selector/restore identity, PPO diagnostic calculation, and process containment.
Where the same contract is tested repeatedly, share fixtures and table-drive
input variants rather than reconstructing near-identical JSON and assertions.

Use a small named set of public integration oracles to prove complete user
paths: the checkpoint lifecycle through update/save/restore/frozen replay;
accounting plus independent episode evidence; causal complete-window behavior;
matched aggregation; and receipt-to-terminal-report/browser disposition,
including truthful unknowns. These align with
[`docs/evaluation/README.md`](../../evaluation/README.md)'s named public
scenarios. Keep the four encoder/state lifecycle cells separate because they
represent distinct architecture contrasts. Synthetic E2E proves lifecycle
wiring, not market edge. Retain each test only when its defect boundary and
independent oracle are clear; do not delete existing tests merely to lower a
count.

R1 implementation is present, but this review does not qualify its Ray
lifecycle, CUDA profile, PPO performance, economic behavior, Q23 fresh panel,
or Q26 inference. The R1 representation comparison remains blocked on the
explicit evidence and resource gates recorded in active docs. This DEV review
performed no R1 training or terminal panel run and does not confer execution
authority; Q23/Q26 remain open under the active plan.

## Review receipts

Spec and Standards were independent read-only lanes on the same frozen
snapshot. Each assessed the five principles and disclosed non-executed probes.
The Standards lane also checked relative file links across 13 active Markdown
documents. Scoped `git diff --check` passed; it does not replace behavior tests.
Actual reviewer serving-model/effort telemetry is unknown.

VERDICT: CHANGES
