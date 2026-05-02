# Decisions

## 2026-05-02 - Treat Episode-Aligned PPO Optimization As An Experiment, Not A Default

- Decision: Keep the current PPO `batch_mode = "complete_episodes"` baseline, and evaluate stricter episode/regime alignment as an ablation instead of making it the default training path.
- Why: RLlib already collects complete episodes, so rewards/returns/GAE are not being computed across episode boundaries. The remaining mixing happens when complete episodes are aggregated into `train_batch_size` and shuffled into SGD minibatches, which is normal PPO behavior and often improves gradient variance.
- Consequences: Solver quality work should focus first on controlled evaluation by market regime, horizon, reward family, and start-index mode rather than forcing every gradient update to represent only one episode.
- Revisit when: A/B runs show the baseline mixed-batch solver underperforms a stream/regime-aligned variant on held-out rolling windows, or a true recurrent solver is introduced that needs stricter sequence semantics.

## 2026-05-02 - Capability Profiles Should Be Updated From Evaluation Slices

- Decision: `CapabilityProfile` should represent solver capability on held-out or rolling evaluation slices by family/regime/horizon, not raw training-batch returns alone.
- Why: Training batches are influenced by curriculum sampling and PPO updates; online gating needs calibrated estimates of what the solver can handle in comparable market windows.
- Consequences: Add a capability updater/report before making online `DecisionGate` thresholds depend on solver scores.
- Revisit when: Shadow-mode online replay produces enough `DecisionRecord` outcomes to calibrate live candidate selection.

## 2026-05-02 - Proper Recurrent Solver Work Needs A Separate Check

- Decision: Do not assume the current `SolverLSTMModel` behaves like a stateful RLlib recurrent policy across environment steps.
- Why: The current default config uses `SolverMLPFiLM`; the existing LSTM model consumes the flattened rolling window and returns no recurrent state. Episode-bound minibatching is therefore less relevant than it would be for a true RLlib recurrent policy.
- Consequences: If LSTM becomes the main solver head again, validate sequence handling, `max_seq_len`, hidden-state reset behavior, and minibatch sequencing before attributing performance changes to episode alignment.
- Revisit when: Solver head selection changes for task families `00`, `01`, or `02`.

## 2026-05-02 - Test A Window Encoder Before A Fully Stateful Recurrent Solver

- Decision: Add a corrected time-series window encoder head before investing in a fully stateful recurrent PPO policy.
- Why: The current environment already provides a rolling OHLCV window plus task embedding. A native window encoder can test whether temporal inductive bias improves solver quality without adding recurrent rollout-state complexity immediately.
- Consequences: Keep `SolverMLPFiLM` as the baseline, then compare it against a window LSTM/GRU/attention head that explicitly splits price features from task embedding. Implement true recurrent state only after that comparison shows benefit or exposes a memory limitation.
- Revisit when: Window-encoder results are available on held-out rolling windows and capability slices.
