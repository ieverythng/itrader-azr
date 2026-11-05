# Focused fix review: economics split gate

Date: 2026-10-07. This is a new independent review of the split-gate fix; the
original `pilot-review-luna.md` remains unchanged and records the earlier
`CHANGES` verdict. The requested Luna/xhigh pair is unavailable in this host's
model catalog; Luna supports up to `max`, so this pass used its highest
supported effort.

## Scope and predeclared probes

Re-read `REVIEW.md` and reviewed the current split gate in
`itrader/rl_lib/evaluation/economic_diagnostics.py`, its pilot tests, and the
effective split behavior in `itrader/rl_lib/evaluation/data_partition.py`.
The base is available: `3a3583d109af7a4c094124e73ad84008f371d8e3`, equal to
current `HEAD`. The helper and its tests are absent from that base, so there is
no base pilot result to compare numerically.

Before reading the implementation, I fixed these probes: exact `train` and
`validation` should run; `all`, `test`, `TEST`, `Train`, `VALIDATION`, leading
or trailing whitespace, empty, and unknown split inputs should reject without
calling reset; and both directions of disagreement between the exposed raw
`env.data_split` and the effective `env.data_partition.split` should reject
before reset. All inputs were synthetic. No real market data, Ray, WSL,
training, replay, or full suite was used.

## Finding 1 of the 1 found so far — BLOCKING — the gate ignores the effective partition

`itrader/rl_lib/evaluation/economic_diagnostics.py:160-161` allowlists only the
raw `env.data_split` value. It does not confirm that the partition used by
`env.reset()` has the same scope. The reset path samples and clamps against
`env.data_partition`; its `split` is the effective authority. A stale or
otherwise inconsistent raw field therefore lets the pilot pass its gate and
run on the test partition.

Reproduction: create a 100-row synthetic panel, window 60, horizon 2, and
explicit task start 90. Construct the environment with `data_split="test"`;
the effective partition has valid test starts 85–98. Then set the exposed raw
`env.data_split` to `"train"` while leaving `env.data_partition.split` as
`"test"`. Wrap `reset()` with a call counter and invoke the pilot. Expected:
fail closed before reset because effective and declared scopes disagree.
Actual: the pilot returns one episode from start 90 and calls reset once.

The inverse disagreement (actual partition `train`, raw field `test`) rejects
before reset. Exact `train` at start 60 and exact `validation` at start 70 both
return one episode. `all`, `test`, `TEST`, `Train`, and `VALIDATION` reject with
  zero reset calls, confirming the earlier bypass fix works for ordinary
  consistent environments. The remaining finding is specifically the
  authoritative-partition mismatch. Leading/trailing-space, empty, and unknown
  split strings are rejected by `ITraderEnv` construction before a reset can be
  called.

The gate should require the effective partition to be exactly `train` or
`validation` and fail if the environment's declared scope disagrees with it,
before calling reset.

## Review principles

1. Separation of concerns — OK. `DataPartition` still owns start eligibility;
   the issue is the pilot's read of a stale mirror, not a second accounting or
   reward owner.
2. Programming by intention — violation at
   `itrader/rl_lib/evaluation/economic_diagnostics.py:160-161` (Finding 1): the
   gate says train/validation scope while an effective test partition can run.
3. Encapsulation — violation at
   `itrader/rl_lib/evaluation/economic_diagnostics.py:160` (Finding 1): the
   consumer trusts the raw config field without checking the environment's
   effective partition.
4. High cohesion — OK. Economic accounting, trace reconciliation, and the
   bounded pilot remain together in the evaluation module.
5. Low coupling — OK. The pilot remains isolated from Ray, training, proposer,
   and executor paths.

## Executed evidence

- `.venv/Scripts/python.exe -m pytest -q itrader/tests/test_economic_diagnostics.py`
  — 24 passed in 1.12 seconds.
- A synthetic Parquet/public-seam matrix confirmed exact train/validation
  success; `all`, `test`, `TEST`, `Train`, and `VALIDATION` rejection with zero
  reset calls; test-partition/raw-train mismatch acceptance with one reset;
  and inverse train-partition/raw-test rejection with zero resets. Leading or
  trailing whitespace, empty, and unknown split strings raised constructor
  `ValueError` before reset.
- Git confirmed the base commit is available and equals `HEAD`; the economics
  helper is not present in the base tree. The previous CHANGES report was left
  unchanged, and this review wrote only this artifact.

VERDICT: CHANGES
