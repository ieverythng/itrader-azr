# iTrader Masterplan

Last updated: 2026-09-03

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
- `pnl_linear` is the only reward proven in training and remains the sole
  default. `sharpe_only` and the `pnl_sharpe` blend are experimental challengers
  that must outperform it in matched held-out evaluation.
- Stable metrics, manifests, plotting, healthcheck, and PBT-oriented operations
  are already present.

Success standard:

- Existing smoke tests, reward tests, config tests, and healthcheck remain
  green.
- Family `00` remains the compatibility path while richer heads are explored.

## Phase B - Solver Quality And Task Contracts

Status: active.

- Preserve the MLP+FiLM baseline while separately testing a corrected
  window-LSTM-FiLM encoder and a true stateful recurrent policy.
- Treat both recurrent variants as unimplemented until code, state-lifecycle
  tests, and matched run comparisons land on this branch. The legacy
  `SolverLSTMModel` is neither task-embedding-safe nor stateful.
- Keep PPO feed-forward semantics unless a deliberate recurrent RLlib policy
  migration is planned and tested.
- Strengthen contracts around `TaskSpec`, `CandidateTask`, `ExecutionIntent`,
  `CapabilityProfile`, and `DecisionRecord`.
- Evaluate solver capability by task family, regime tags, horizon, calibration,
  and held-out rolling windows rather than training reward alone.

Current experiment seams:

- The alignment config contains mixed-horizon controls plus a matched
  random-128 versus stream-128 pair.
- Sharpe shaping and recurrence are orthogonal variables. Establish each against
  `SolverMLPFiLM` before testing their interaction.
- Reward-parameter sampling uses an RNG separate from task identity, preserving
  horizon and market-window seed sequences across reward variants.
- The current observation does not expose position, previous action, PnL, or
  episode progress. Test those as a matched observation-feature ablation rather
  than attributing plan tracking to recurrence alone.
- Family IDs remain source-defined: `00` directional, `01` breakout, `02` mean
  reversion, `03` trade management, `04` abstention.

Near-term implementation order:

1. Keep docs and AGENTS aligned with implemented seams.
2. Run the five-cell alignment ablation from the validated pinned runtime.
3. Compare `pnl_linear`, `sharpe_only`, and the `pnl_sharpe` blend on fixed data,
   horizon, seeds, and MLP-FiLM head.
4. Ensure task metadata reaches metrics before claiming capability slices.
5. Implement R1, a corrected window-LSTM-FiLM head, through TDD.
6. Implement R2a, a native RLlib recurrent LSTM with task concatenation, then
   R2b per-step FiLM as a separate cell.
7. Add explicit solver-state observation features only as a matched follow-up.
8. Lock families `01` and `02` only after baseline evaluation is trustworthy.

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

JEPA should be trained and evaluated as a representation/framing component, not
as a replacement for deterministic risk or execution. Candidate objectives
include masked future latent prediction, regime-transition prediction, and
uncertainty calibration. The first comparison must preserve the same
MarketFramer output contract and downstream solver.

## Cross-Cutting Operating Rules

- Prefer extension over rewrite.
- Keep typed contracts central.
- Keep risk and execution deterministic.
- Preserve `WAIT` as a valid outcome.
- Track capability empirically before trusting online decisions.
- Keep imported AB/Neural Workbench research as design language, not as a
  mandatory implementation framework.
