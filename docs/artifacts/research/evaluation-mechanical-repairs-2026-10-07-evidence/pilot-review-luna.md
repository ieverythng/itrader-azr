# Independent review: experimental economics pilot

Date: 2026-10-07. Reviewer: Luna review pass. The requested Luna/xhigh pair is
not available in this host's model catalog; Luna supports up to `max`, so this
review used its highest supported effort. This report makes no claim that
producer telemetry or checkpoint provenance was verified.

## Scope and predeclared evidence

Reviewed `REVIEW.md`, `itrader/rl_lib/evaluation/economic_diagnostics.py`, the
`capture_step_economics` / `economic_decision_context` /
`economic_transition` environment seams, `itrader/tests/test_economic_diagnostics.py`,
and `docs/artifacts/research/evaluation-mechanical-repairs-2026-10-07.md`.

The requested base is available as commit
`3a3583d109af7a4c094124e73ad84008f371d8e3`; it is also the current `HEAD`.
The economics helper and its test file do not exist at that commit, and the base
environment has none of the three named capture seams. There is therefore no
base pilot output for a numerical side-by-side comparison. The current synthetic
pilot does reconcile its calculations to the environment's existing episode
evidence; this does not establish producer provenance or terminal-panel
qualification.

Before opening the implementation, I fixed this synthetic-only matrix: flat
100-to-100 at zero cost; long and short returns of one percent at 0, 1, and 2
bp; a long-to-short reversal across 100, 101, and 99; repeated pre-action
context reads; absent cost units and malformed/non-finite values; and an empty
episode. I also tested a 120-row synthetic panel with a task start in its
70/15/15 test interval, both as `all` and as uppercase `TEST`. No real market
data, Ray, WSL, training, replay, or full suite was used.

## Finding 1 of 1 — BLOCKING — the sealed-test gate admits `all` and case variants

`itrader/rl_lib/evaluation/economic_diagnostics.py:160-161` rejects only when
`env.data_split == "test"`, using a case-sensitive comparison against the raw
environment config. `DataPartition` lowercases split names
(`itrader/rl_lib/evaluation/data_partition.py:40-42`) and `split="all"` selects
the complete per-symbol interval (`data_partition.py:132-135`). The pilot can
therefore evaluate outcomes from the test interval while its error text says
that sealed test is unavailable.

Reproduction: create 120 synthetic rows, use the default 70/15/15 fractions,
window 60, horizon 2, task start 105, task-owned starts, and economics capture.
The test interval starts at index 102, so start 105 is eligible there. With
`data_split="all"`, the expected result is rejection before evaluation; the
actual result is one returned episode. With `data_split="TEST"`,
`DataPartition.split` normalizes to `test`, its valid start range is 102–118,
and the pilot again returns one episode because `env.data_split` retains the
uppercase spelling. Lowercase `data_split="test"` is rejected. Require an
explicit permitted non-test partition and validate task starts against that
partition before reset.

## Review principles

1. Separation of concerns — OK. The helper reports secondary arithmetic in the
   evaluation layer; the environment supplies opt-in context/transitions, and
   optimization reward and `EpisodeEvidence` remain separate.
2. Programming by intention — violation at
   `itrader/rl_lib/evaluation/economic_diagnostics.py:160-161` (Finding 1).
   The stated sealed-test boundary is weaker than the accepted split forms.
3. Encapsulation — OK for this bounded pilot. The evaluator uses the named
   environment reset/step/context seams and exposed task/config state; it does
   not write environment state or alter reward ownership.
4. High cohesion — OK. The module contains step accounting, episode
   reconciliation, and their bounded evaluation loop; it does not load datasets
   or decide promotion.
5. Low coupling — OK. The pilot depends on the `ITraderEnv` public seams and
   evaluation evidence/hash contracts, with no Ray, training, proposer, or
   executor dependency.

## Executed evidence

- First targeted run, before the shared-tree trace-consistency patch:
  `.venv/Scripts/python.exe -m pytest -q itrader/tests/test_economic_diagnostics.py`
  produced 1 failure and 12 passes. The wrong-position trace was accepted.
- After the root's minimal trace-consistency patch, the same targeted command
  passed: 23 tests. The initial failure is resolved in the reviewed snapshot;
  it is retained here as review chronology, not as a current finding.
- Additional synthetic public-seam probes rejected an empty episode, omitted or
  `None` cost units, NaN returns, boolean positions, and negative costs. Long
  and short net returns were 0.0100, 0.0098, and 0.0096 at 0/1/2 bp. A
  long-to-short reversal returned gross 0.0298019802, cost 0.0004, net
  0.0294019802, with switching costs 0.0001/0.0002 and terminal liquidation
  0/0.0001. Repeated pre-action context was identical. Synthetic timestamp
  probes preserved leap-day and UTC year-boundary values.
- Future-row perturbation and the nine transition/reference-vector cases also
  passed in the targeted test module. No multi-asset batch accounting API exists
  in this pilot; each environment episode is a single-symbol sequence.
- `tools/knowledge/status.sh` reports the local GitNexus index stale at
  `1d5ba82`; source and tests were used as authority. The pre-existing dirty
  tree was preserved, and this review wrote only this artifact.

VERDICT: CHANGES
