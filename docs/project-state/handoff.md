# Handoff

Last updated: 2026-09-14
Branch: `agent/Watson-dev`
Base HEAD before this evidence commit: `bec5c77`

## Resume point

The Phase A measurement, PBT checkpoint/evaluation, and iteration-aligned
reporting seams are implemented in the dirty working tree. The corrected
four-member, 200-iteration run
`pbtA26_parity200_checkpoint_valid_20260914_041907_pbt200_checkpoint_valid`
completed in 49 minutes 57 seconds. It used R0 `SolverMLPFiLM`, `stream_worker`
training, random/generated tasks with no LLM backend, two rollout workers per
member, fixed batch 4,800, one deterministic 16-task held-out panel, 1 bp cost,
validation net return as the PBT objective, and checkpoint/perturbation cadence
10 after burn-in 20.

All members reached iteration 200 and about 990k retained environment steps.
Terminal net returns were -0.0556%, -0.1069%, -0.1866%, and -0.3906%; terminal
Sharpes per square-root bar were -0.00060, -0.03145, -0.04089, and -0.05376.
The least-negative member was 91.3% flat. Its durable terminal checkpoint
reproduced net return, Sharpe, and turnover exactly in a fresh process. This is
valid negative candidate-generation evidence, not proof of solver uplift or
reward learnability.

PBT recorded 23 population exploit/restore mutations: member `00000` restored
3 times, `00001` 2 times, and `00002`/`00003` 9 times each. Member `00001` was
strongest before its second restore: mean held-out net return was +0.1211% over
iterations 80-100 and +0.0636% over 101-120, with a +0.3495% peak at iteration
107. It moved from about 60% flat/39% long over 80-100 to 72% flat/24% long over
101-120 at 6.2-7.9 turnover. Over 121-140 it flipped toward short and fell to
-0.1200% mean net. Its iteration-130 exploit changed lambda 0.90 to 0.95,
minibatch 256 to 128, SGD passes 10 to 20, value-loss coefficient 1 to 2, and
learning rate 1.79e-5 to 9.47e-5. The following segment therefore cannot be
interpreted as uninterrupted learning by the same optimizer configuration.

The fixed panel contains 16 `pnl_linear` tasks: horizons 96/128/160/200 occur
7/3/1/5 times, and turnover weights 0/0.0005/0.001/0.002 occur 6/4/3/3 times.
During member `00001`'s positive 80-120 window, mean reported PnL reward terms
were about +0.0106, drawdown terms -0.0023 to -0.0036, and turnover terms only
about -0.00003. PnL dominates the shaped reward, drawdown is material, and the
task-level turnover penalty is negligible relative to both; separate 1 bp
economic costs still make high-turnover policies unattractive.

The aggregate outputs, plots, replay receipt, final loadout, and documentation
are committed as evidence. The raw Ray trial/checkpoint tree remains local. Code,
scripts, tests, and the new stack-iteration skill remain staged for user review;
no recurrent-solver implementation or dependency upgrade was included.

Validation passed: repository healthcheck, 97-test full suite, focused reporting
tests, JSON-schema validation, `git diff --check`, and the complete pre-commit
suite. GitNexus was readable but stale relative to the dirty implementation and
should be refreshed after the code commit.

## User-review boundary

Phase B includes offline/curriculum work and online replay/shadow validation;
Phase C is later operationalization. The shared recurrent trunk/family-specific
policy-value head proposal, detailed FiLM placement/order, and any R-ID
reconciliation are not accepted decisions. Families `00`–`04` are the roadmap;
`00`–`02` sharing `Discrete(3)` width does not settle inputs, task/reward
semantics, or state ownership. `RewardEngine` remains the environment-owned
reward seam, not a learned reward-head requirement.

## Current truth

- R0 `SolverMLPFiLM` is the active/default model.
- Its implemented FiLM is `tanh(gamma)*hidden + beta` with task dropout.
- Non-`film` conditioning in that model does not implement concat.
- Legacy `SolverLSTMModel` is shape-incompatible with appended task embeddings
  and does not carry state.
- R1–R3 and R4-v1 are design/plan cells only.
- Original R2a/R2b/R2c/R2d meanings are preserved from `4732a98`.
- R2c per-step FiLM is the active candidate; R2b init-only and R2d value-only
  are deferred; R4-v1 is matched explicit solver-state observability.
- `R2c-CARRY` versus `R2c-RESET` is the required same-input control.
- R1 versus newest-bar R2 remains an unequal-information engineering comparison.

## AZR boundary

The local AZR v2 paper synthesis is in
`docs/artifacts/research/azr-stack-review-2026-09-12.md`. Eq. 4 is zero at
observed `p=0` or `p=1`, otherwise `1-p`; it is not a symmetric 50% target or
measured learning uplift. iTrader has historical OHLCV, a language-model
proposer, and numerical PPO. `train_AZR.py` is a bootstrap handoff, not a
co-adaptive AZR loop. Keep proposer frozen/adaptive reference-buffer studies
separate from solver recurrence.

## Remaining Phase A performance sequence

1. Preserve R0 and the repaired evidence/PBT contracts.
2. Add flat/long/short/random/untrained baselines on identical validation starts,
   multiple costs, and block-aware uncertainty.
3. Run the multi-seed overnight PBT study; select on validation net return and
   evaluate once on the untouched final test.
4. Run matched alignment/reward controls; do not call reward necessary for
   recurrence or claim longer horizons automatically win.
5. Resolve the user's open placement/memory-conditioning decision before choosing
   or implementing R2a/R2c.
6. Implement the selected recurrent control using the native state lifecycle and
   sequence contract.
7. Run `R2c-CARRY` versus `R2c-RESET` only if R2c remains selected; preserve the
   same-input/same-conditioning attribution control.
8. Run R4-v1 with identical explicit state fields on FF/recurrent controls.
9. Treat R3 as a later cell-type comparison. Proposer and online/replay/shadow
   work remain within Phase B and can proceed alongside the offline track;
   recurrence experiments do not impose a blanket gate on that work.

Ray/RLlib 2.9.0 is validated in the Windows-native environment for the Phase A
path. Recurrent APIs still require their own pinned-runtime lifecycle tests; the
earlier WSL executable mismatch is not the current validation environment.

## Validation commands for a compatible environment

```text
./itrader/tools/run_repo_python.sh itrader/tools/healthcheck.py
./itrader/tools/run_repo_python.sh -m pytest -q itrader/tests
./itrader/tools/run_repo_python.sh -m pre_commit run --all-files
git diff --check
```

Use `uv.lock`/`pyproject.toml` as authority; do not use the historical Ray 2.47.1
plan line to choose a recurrent API.
