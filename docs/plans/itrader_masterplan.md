# iTrader Masterplan

Last updated: 2026-06-17

## North Star

iTrader is moving from a stable PPO/RLVR research baseline toward a coupled
offline/online proposer-solver system. The core principle is separation of
learning, framing, deterministic gating, and execution.

```text
offline: market data -> curriculum proposer -> TaskSpec -> solver -> capability
online: market window -> MarketFramer -> CandidateTask -> solver score -> gate -> risk -> execution
```

## Phase A - Stable Solver Baseline

Status: mostly complete.

- PPO + GAE training runs through `itrader/rl_lib/train/train_solo_V2.py`.
- `ITraderEnv` emits flattened market windows plus task embeddings.
- `SolverMLPFiLM` is the current implemented solver baseline.
- Rewards are explicit, finite, and decomposed through `RewardEngine`.
- Stable metrics, manifests, plotting, healthcheck, and PBT-oriented operations
  are already present.

Success standard:

- Existing smoke tests, reward tests, config tests, and healthcheck remain
  green.
- Family `00` remains the compatibility path while richer heads are explored.

## Phase B - Solver Quality And Task Contracts

Status: active.

- Preserve the MLP+FiLM baseline while investigating a recurrent-window solver
  frontier.
- Treat `SolverWindowLSTMFiLM` as in development until code, tests, and run
  comparisons land on this branch.
- Keep PPO feed-forward semantics unless a deliberate recurrent RLlib policy
  migration is planned and tested.
- Strengthen contracts around `TaskSpec`, `CandidateTask`, `ExecutionIntent`,
  `CapabilityProfile`, and `DecisionRecord`.
- Evaluate solver capability by task family, regime tags, horizon, calibration,
  and held-out rolling windows rather than training reward alone.

Near-term implementation order:

1. Keep docs and AGENTS aligned with implemented seams.
2. Run baseline solver experiments and record capability by family/regime/horizon.
3. Port or reimplement the recurrent-window solver only after the branch is ready
   for code changes.
4. Add an ablation harness for MLP+FiLM versus window-LSTM+FiLM on identical task
   pools and seeds.

## Phase C - Offline/Online Proposer Split

Status: planned with contracts partially present.

- Offline `CurriculumProposer` should generate Goldilocks-difficulty `TaskSpec`
  candidates and improve task-pool quality.
- Online `MarketFramer` should generate a small set of `CandidateTask` objects,
  including a first-class `WAIT` / `NO_ACTION` candidate.
- `DecisionGate` should select only candidates that clear solver confidence and
  expected utility thresholds.
- `RiskEngine` and `Executor` should stay deterministic and final for live or
  shadow-mode actioning.
- `DecisionRecord` should capture candidate set, solver scores, risk decisions,
  executor action, and realized outcome for later analysis.

Success standard:

- Shadow-mode replay can run market window -> framer -> candidate tasks -> solver
  scoring -> deterministic decision/risk record without placing live trades.
- Capability profiles can later drive online gating without being inferred from
  raw training reward alone.

## Phase D - Market-JEPA / World Model Frontier

Status: future optional expansion.

Market-JEPA belongs inside iTrader as a possible latent market representation or
candidate-window prior, not as a WatsonOW component. It should remain behind
Phase B/C seams until solver capability and shadow-mode gating are measurable.

Potential route:

1. Define context/target patch dataset.
2. Train latent market representation on historical windows.
3. Add downstream heads for regime, uncertainty, difficulty, and WAIT prior.
4. Feed MarketFramer with JEPA priors for candidate-window proposal.
5. Compare against non-JEPA framing in replay.

## Cross-Cutting Operating Rules

- Prefer extension over rewrite.
- Keep typed contracts central.
- Keep risk and execution deterministic.
- Preserve `WAIT` as a valid outcome.
- Track capability empirically before trusting online decisions.
- Keep imported AB/Neural Workbench research as design language, not as a
  mandatory implementation framework.
