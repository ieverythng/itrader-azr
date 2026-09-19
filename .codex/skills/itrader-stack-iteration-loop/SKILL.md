---
name: itrader-stack-iteration-loop
description: Run a bounded, evidence-gated run-test-patch loop over iTrader training, evaluation, PBT, and experiment artifacts. Use for Phase A/B measurement regressions, environment/task/reward seam failures, PBT validity, or stack stabilization that requires TDD and a comparable full-stack rerun.
---

# iTrader Stack Iteration Loop

Use this workflow to turn a training result into evidence that can survive a
promotion decision. Preserve `TaskSpec`, `ITraderEnv`, `RewardEngine`, solver,
and deterministic online-shell ownership from `$itrader-architecture-check`.

## Freeze the round

Record the exact loadout, dataset/checkpoint hashes, target behavior, protected
controls, non-goals, iteration budget, and dirty-tree boundary. Separate these
claim classes before editing:

- optimization reward learnability;
- raw gross and cost-adjusted return;
- held-out generalization;
- PBT search effectiveness;
- artifact and lineage completeness.

One metric must never stand in for another claim class.

## Establish a red gate

Use `$tdd` at the narrowest public seam. Prefer, in order:

1. terminal environment `info` evidence;
2. callback/report metric records;
3. PBT/loadout construction;
4. ablation summary and persisted provenance;
5. a bounded official-stack probe.

Keep one known-good control. Classify the failure as contract loss, sampling or
split leakage, invalid search mutation, runtime wiring, or observability.

## Patch the owning seam

Apply the smallest structural correction and run `$itrader-architecture-check`.
Keep economic evidence out of `RewardEngine`, broker/risk semantics out of the
solver, proposer output behind validation, and LSTM/recurrent work out of scope
unless the user explicitly includes it.

Prefer a deep module with a small interface when evidence normalization or
validation would otherwise be duplicated across the environment, callbacks,
training, and reporting.

## Widen validation

Run focused tests, affected suites, the repository healthcheck, config-schema
validation, and `git diff --check`. A source-only pass is insufficient when the
claim concerns Ray/PBT wiring or generated artifacts.

## Run the comparable stack probe

Use the checked-in ablation JSON and official path:

```text
loadout -> run_ablation.py -> train_solo_V2 -> Ray Tune/PBT
-> ITraderEnv -> TaskSampler00 or validated pool -> RewardEngine
-> stable metrics/provenance -> all-member ablation summary
```

Require independent population seeds, constrained mutations, aligned
checkpoint/perturbation cadence, symbol-safe chronological splits, deterministic
held-out evaluation, and non-empty manifests. Reject a PBT round that performs
no actual search even when it completes successfully.

## Decide and record

Accept only when red tests, protected controls, and the comparable stack probe
agree. Update the active Markdown/HTML audit and masterplan with exact commands,
hashes, metrics, residual risks, and a next probe. Do not stage, commit, or push
unless asked.

Run a bounded SkillOpt-style iteration (baseline -> mutate -> holdout gate -> accept/reject log) before finalizing major wording changes.

## Stop conditions

Stop when the frozen gates pass, a bounded run exposes a new discriminating
failure, or an external dependency prevents truthful scoring. Mark missing
evidence `not_scored`; never convert it into a performance failure or success.
