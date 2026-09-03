# iTrader Architecture Entry Point

Last updated: 2026-06-17

This file is the compact compatibility entrypoint for iTrader architecture. For
the fuller review document, see:

- `docs/architecture/itrader_system_architecture.md`
- `docs/architecture/itrader_system_architecture.html`

For phase ordering and roadmap status, see:

- `docs/plans/itrader_masterplan.md`
- `docs/iTRADER_Codex_Implementation_Spec.md`

## Current Baseline

The reliable executable path is still:

```text
itrader/rl_lib/train/train_solo_V2.py
-> ITraderEnv
-> TaskSpec / TaskSampler00
-> RewardEngine
-> SolverMLPFiLM
-> PPO/GAE metrics and manifests
```

`SolverMLPFiLM` is the implemented baseline solver. A recurrent-window
`SolverWindowLSTMFiLM` remains a Phase B development frontier and is not part of
this branch's operational API yet.

## Core Runtime Roles

- Data: `itrader/data/ohlcv.parquet` supplies local OHLCV features.
- Environment: `ITraderEnv` emits `flattened_price_window + task_embedding`.
- Task layer: `TaskSpec` remains the canonical task/problem contract.
- Reward layer: `RewardEngine` computes finite, decomposed solver rewards.
- Solver: PPO + FiLM-conditioned model learns action/value behavior.
- Proposer: `itrader/proposers/` provides backend abstractions and fallbacks.
- Observability: stable metrics, Ray artifacts, run manifests, and GitNexus docs
  support evaluation and handoff.

## Phase B Target

Phase B should improve solver quality and task-contract discipline without
destabilizing family `00`:

1. Preserve the MLP+FiLM baseline.
2. Evaluate recurrent-window solver heads as opt-in experiments.
3. Track capability by family, regime, horizon, and calibration.
4. Keep `CandidateTask`, `ExecutionIntent`, `CapabilityProfile`, and
   `DecisionRecord` explicit for future online/shadow-mode work.

## Phase C Target

The online/shadow stack should remain deterministic at the actioning shell:

```text
market window
-> MarketFramer
-> CandidateTask set including WAIT
-> solver scoring
-> DecisionGate
-> RiskEngine
-> Executor / shadow recorder
-> DecisionRecord + CapabilityProfile update
```

LLM output may frame candidates, but it must not bypass solver scoring,
decision thresholds, risk limits, or executor constraints.

## Refactor Principles

- Preserve working training paths.
- Prefer extension over rewrite.
- Keep typed contracts central.
- Keep risk and execution deterministic.
- Treat GitNexus as advisory and confirm important architecture claims in source.
