# GitNexus Evaluation

Use this file to compare the current grep-first path against the new GitNexus-assisted path without destabilizing normal work.

## Evaluation goals

- check whether GitNexus reduces missed dependencies
- see whether it narrows the right files faster than grep
- measure whether it over-trusts graph context where direct code reads would be safer
- decide later whether advisory mode should stay advisory or become stricter

## Prompt set

Use the same questions twice:

1. Baseline:
   - raw grep
   - direct file reading
2. GitNexus-assisted:
   - `query`
   - `context`
   - `impact`
   - `detect_changes` when relevant

Suggested prompts:

- "How does task generation flow from proposer selection to `TaskSpec` validation in `itrader`?"
- "If we change `TaskSpec`, what is the likely blast radius?"
- "Where does solver training pick up task-conditioned information?"
- "What would be the safest path to rename a proposer backend?"
- "Which files are most central to the current AZR loop scaffolding?"

## What to record

- correctness of the answer
- key files found
- dependencies missed
- time to first useful answer
- whether the answer felt overconfident
- whether direct reads changed the conclusion

## Decision rule

Keep advisory mode unless GitNexus consistently:

- finds the right files faster
- misses fewer dependencies than the baseline
- stays well calibrated when the graph is stale or incomplete
