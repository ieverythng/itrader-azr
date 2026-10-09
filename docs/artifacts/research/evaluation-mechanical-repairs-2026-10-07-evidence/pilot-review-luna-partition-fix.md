# Fresh focused review: economics partition and numeric-scale fixes

Date: 2026-10-07. Reviewer: Luna review pass. The coordinator records an accepted
collaboration dispatch requesting GPT 6 Luna / xhigh. Actual runtime model and
effort telemetry are unavailable; the app's separate host model catalog does
not prove a fallback for this agent. This corrects the earlier unsupported
availability/fallback assertion without changing probes or verdict. The prior
`pilot-review-luna.md` and `pilot-review-luna-fixes.md` remain unchanged.

## Scope and predeclared probes

Re-read `REVIEW.md` and reviewed the current gate in
`itrader/rl_lib/evaluation/economic_diagnostics.py`, the corresponding pilot
tests, and `DataPartition` split semantics. The requested base
`3a3583d109af7a4c094124e73ad84008f371d8e3` is available and equals current
`HEAD`; the economics helper and tests do not exist at that base, so there is no
base pilot result for numerical comparison.

Before reopening the implementation, I predeclared these synthetic probes:
exact `train` and `validation` with eligible task starts should run; consistent
`all`, `test`, `TEST`, `Train`, `VALIDATION`, whitespace, empty, and unknown
values should reject before reset; and actual/raw split disagreement should
reject in both directions, including a test partition paired with raw `train`,
an `all` partition paired with raw `validation`, and two individually permitted
but unequal scopes. I also predeclared `lagged_volatility=10**400` with an
explicit method: it must become unavailable without an exception or invented
scaled value, while ordinary return accounting remains intact. All data were
synthetic; no real market data, Ray, WSL, training, replay, or full suite was
used.

## Correctness result

The previous partition-disagreement bypass is fixed. The current gate at
`itrader/rl_lib/evaluation/economic_diagnostics.py:164-166` requires task-owned
starts, an exact raw `train` or `validation`, and equality with the effective
`env.data_partition.split` before reset.

In a 100-row synthetic panel with window 60 and horizon 2, exact `train` at
start 60 and exact `validation` at start 70 each returned one episode. Consistent
`all`, `test`, `TEST`, `Train`, and `VALIDATION` all rejected with zero reset
calls. Leading/trailing whitespace, empty, and unknown splits were rejected by
environment construction.

I then verified both mismatch directions with task starts valid in their
effective partitions. Actual test/raw train at start 90, actual validation/raw
train at start 70, actual train/raw validation at start 60, and actual
train/raw test at start 60 all rejected with zero reset calls. Actual `all`/raw
validation also rejected before reset. The earlier actual test/raw train
reproduction now fails closed.

The extreme numeric-scale fix also holds. With prior 0, target +1, asset return
0.01, cost 1 bp, nonterminal, explicit method, and `lagged_volatility=10**400`,
`step_accounting` returns `volatility_status="unavailable"`, reason
`"unrepresentable"`, and no scale or scaled return. Gross return remains 0.01
and continuation net remains 0.0099.

## Test ROI audit

One optional consolidation opportunity is the positive-only overlap between
`test_episode_attribution_reconciles_direction_timing_and_existing_evidence`
and `test_opt_in_pilot_uses_actual_env_arithmetic_without_changing_headline_evidence`.
Both exercise a two-step long/short path and assert net/timing attribution; the
manual fixture additionally checks the full direction decomposition, while the
environment fixture verifies transition capture and unchanged headline reward.
If reducing duplicated setup is useful, the public environment test could
assert the complete decomposition and absorb the overlapping positive checks.
Keep the standalone mismatched-evidence rejection, wrong-position rejection,
duplicate/reordered trace checks, stale-context rejection, and all split-gate
negative cases. The `(-1, +1)` nonterminal row also appears in the nine-transition
matrix; retain the standalone terminal/liquidation assertions and the matrix's
other transition, reference-action, and scaled-return cases. These are partial
overlaps, not grounds to drop the distinct controls. The current suite is small,
so the likely runtime savings are modest.

## Findings

Finding 1 of the 1 found so far — NIT: optional consolidation of overlapping
positive attribution checks. Reproduction: run the manual two-step
`attribute_episode` fixture and the synthetic long/short `ITraderEnv` pilot.
Expected for a consolidation: the public-seam test asserts the shared
net/timing/decomposition invariants once, while direct unit tests retain the
distinct arithmetic rejection cases. Actual: the two test functions partly
repeat positive net/timing checks, but use different returns and each protects
additional behavior. No incorrect runtime behavior was found.

## Review principles

1. Separation of concerns — OK. Effective start eligibility stays owned by
   `DataPartition`; economics remain secondary to environment evidence/reward.
2. Programming by intention — OK. The gate now rejects unsupported or
   inconsistent scope before reset; the extreme scale stays unknown instead of
   becoming a misleading score.
3. Encapsulation — OK. The evaluator checks both the declared split and the
   effective partition before invoking the environment reset seam.
4. High cohesion — OK. Accounting, attribution, and the bounded reporting loop
   remain focused in the evaluation module.
5. Low coupling — OK. The pilot remains outside Ray, training, proposer, and
   executor paths.

## Executed evidence

- `.venv/Scripts/python.exe -m pytest -q itrader/tests/test_economic_diagnostics.py`
  — 28 passed in 0.77 seconds.
- A separate `.venv/Scripts/python.exe -` synthetic public-seam harness
  exercised accepted train/validation scopes, the split alternatives, both
  disagreement directions, reset-call counts, and the prior test/raw-train
  bypass. A direct synthetic `step_accounting` call exercised `10**400` and
  verified unavailable reason plus unchanged arithmetic.
- The base commit equals `HEAD`; the pilot helper is absent from the base tree.
  No production files or existing review reports were changed; this pass wrote
  only this artifact.

VERDICT: APPROVE
