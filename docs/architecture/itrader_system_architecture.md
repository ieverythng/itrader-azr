# iTrader System Architecture

Last updated: 2026-09-03

## Current Implemented Path

The reliable execution path is still the Phase A/B solver stack:

```text
OHLCV parquet
-> ITraderEnv
-> flattened market window + task embedding
-> SolverMLPFiLM
-> PPO/GAE update
-> stable metrics and manifests
```

Implemented anchors:

- `itrader/rl_lib/train/train_solo_V2.py` is the primary PPO training entrypoint.
- `itrader/envs/itrader_env.py` owns observation assembly and episode stepping.
- `itrader/rl_lib/tasks/spec.py` owns `TaskSpec`.
- `itrader/rl_lib/tasks/rewards.py` owns solver reward calculation.
- `itrader/models/solver_mlp_film.py` owns the current FiLM-conditioned solver
  baseline.
- `itrader/proposers/` owns proposer abstractions and backends.

## Task And Online Contracts

`TaskSpec` remains the canonical problem statement. Online trading semantics
should stay in separate contracts:

- `ExecutionIntent` describes how a candidate could be traded if accepted.
- `CandidateTask` combines a `TaskSpec` with optional execution framing.
- `CapabilityProfile` records empirical solver capability by family, regime,
  horizon, and calibration.
- `DecisionRecord` captures the outcome of a decision cycle.

This separation prevents live broker semantics from leaking back into the core
training task contract.

## Solver Architecture

Current baseline:

```text
price_window_flat + t_embed
-> split price/task features
-> MLP trunk over price window
-> FiLM conditioning from task embedding
-> policy logits + value
```

Phase B frontier:

```text
price_window_flat + t_embed
-> reshape price window as time x features
-> LSTM window encoder
-> FiLM conditioning from task embedding
-> policy logits + value
```

`SolverWindowLSTMFiLM` is not available on this branch yet. It should be treated
as an in-development candidate that must be opt-in, tested, and compared against
the MLP baseline before it becomes operational guidance.

A true stateful recurrent policy is a separate candidate. It must use RLlib's
sequence/state contract and be compared against the window encoder before any
gain is attributed to episode memory. The current observation does not expose
position or episode progress, so those features require a later matched
observation ablation.

## Reward Architecture

`pnl_linear` is the only reward proven in training and remains the enabled
family `00` default. Two challengers are available only for ablation:
`sharpe_only` uses bounded incremental unannualized strategy Sharpe, while
`pnl_sharpe` adds that component to the `pnl_linear` base. Neither is a family
default. Reward and solver-head experiments should be run independently before
testing their interaction.

Canonical family semantics are `00` directional prediction, `01` breakout
expansion, `02` mean reversion, `03` trade management, and `04` abstention.

## Offline Loop

Target Phase B offline loop:

```text
historical market window
-> summarizer / feature assembly
-> CurriculumProposer
-> TaskSpec candidates
-> validation / task pool
-> solver training or evaluation
-> capability profile update
-> proposer feedback dataset
```

The proposer should improve task quality without destabilizing family `00` or
the existing PPO training path.

## Online Shadow Loop

Target Phase C online/shadow loop:

```text
live or replayed market window
-> market summary + current CapabilityProfile
-> MarketFramer frames capability-conditioned candidate tasks
-> CandidateTask set including WAIT
-> solver scoring
-> DecisionGate filters with solver scores, CapabilityProfile, and thresholds
-> RiskEngine
-> Executor or shadow recorder
-> DecisionRecord and CapabilityProfile update
```

Rules:

- `WAIT` / `NO_ACTION` is first-class.
- `CapabilityProfile` is the empirical truth layer for what solver heads can
  currently handle by family, regime, horizon, and calibration.
- `MarketFramer` should use the live/replayed market window plus current
  capability profile to propose tasks inside a sliding Goldilocks range for the
  available solver heads.
- `DecisionGate` remains the deterministic chooser after solver scoring; it may
  also use `CapabilityProfile` for thresholds, routing, and WAIT selection.
- LLM output may frame candidates but cannot bypass solver scoring, deterministic
  decision thresholds, risk, or executor constraints.
- `RiskEngine` and `Executor` remain deterministic.
- Shadow/replay mode comes before any live actioning.

## AB / Neural Workbench Lens

The AB framework is useful as a design language:

- solver heads are learned policy objects,
- PPO/GAE are training deltas,
- LoRA/adapters may later specialize proposer/framer behavior,
- CapabilityProfile is the empirical truth layer.

Do not force AB terminology into runtime APIs unless it makes a seam clearer.

## Knowledge And Observability

GitNexus provides graph-guided exploration and generated wiki pages. It is a
narrowing tool, not the final authority. Check `tools/knowledge/status.sh`
before relying on graph output, then confirm important findings in source.

Operational outputs to preserve:

- Ray trial artifacts under `~/ray_results/<experiment_name>/`
- stable metrics mirror under `~/ray_results/_stable_metrics/`
- run manifests with git commit, config hash, dataset hash, and seed
- generated task pools and proposer outputs for later replay or SFT/QLoRA
