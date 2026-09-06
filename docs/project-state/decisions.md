# Decisions

## 2026-09-03 - Isolate Reward RNG From Task Identity

- Decision: task samplers and the rule-based proposer use a dedicated RNG for
  reward parameters, separate from price, horizon, and task-seed sampling.
- Why: reward functions consume different numbers of random draws. A shared RNG
  changes later task seeds and market windows, invalidating a reward ablation.
- Consequences: matched reward variants preserve task identity while still
  sampling reward-specific parameters deterministically.
- Revisit when: task generation adopts immutable pre-generated evaluation pools.

## 2026-09-03 - Keep PnL Default And Test Two Sharpe Challengers

- Decision: `pnl_linear` remains the sole proven/default reward. `sharpe_only`
  tests bounded incremental strategy Sharpe alone; `pnl_sharpe` adds the same
  component to normalized `pnl_linear`. Both challengers are unannualized during
  optimization, have a warmup floor, and are not family defaults.
- Why: a full rolling Sharpe level paid every step double-counts prior path
  quality and minute-scale annualization saturates a bounded signal. The
  incremental form is finite, inspectable, and approximately telescoping.
- Consequences: family `00` can run a three-way matched comparison without
  changing its default or semantics. Family-specific reward assignments remain
  deferred until a challenger wins held-out evaluation.
- Revisit when: held-out multi-seed reward comparisons are available.

## 2026-09-03 - Treat Recurrent FiLM As A Factorized Hypothesis

- Decision: compare R0 MLP-FiLM, R1 reset-every-step window-LSTM-FiLM, R2a
  stateful LSTM plus task concatenation, and R2b stateful LSTM plus per-step FiLM
  in that order.
- Why: this separates within-window temporal bias, persistent state, and task
  modulation. The current observation also lacks explicit trade state, so plan
  tracking cannot be attributed to recurrence without a later matched feature
  ablation.
- Consequences: the legacy `SolverLSTMModel` is not a valid recurrent baseline.
  RLlib 2.9 state lifecycle and truncated BPTT need TDD coverage before training
  claims. The newest bar is the last row of the chronological market window.
- Revisit when: the alignment and reward baselines are measured.

## 2026-09-03 - Preserve Canonical Family Semantics

- Decision: documentation and experiments use the source registry: `00`
  directional, `01` breakout, `02` mean reversion, `03` trade management, and
  `04` abstention.
- Why: prior research prose swapped `01`/`02` and treated `03` as trend
  following, which would make reward and head routing ambiguous.
- Consequences: no reward is assigned by an obsolete family label.
- Revisit when: a deliberate, migrated family-registry change is proposed.

## 2026-09-03 - Keep All Ablation Variants On FiLM; Defer LSTM Head Fix

- Decision: The episode/regime alignment ablation runs entirely on `SolverMLPFiLM`. It now contains mixed-horizon controls plus a matched random-128 versus stream-128 pair. The `SolverLSTMModel` obs-shape fix (`flat_dim // win` breaking with task embeddings) is deferred to the solver-head ablation.
- Why: Mixing an uncorrected LSTM head into the alignment ablation would confound two variables at once (alignment + head architecture). The FiLM baseline handles arbitrary flat obs via `flat_dim - t_dim`, so all 4 variants get a clean comparison.
- Consequences: fixed-horizon alignment is measured by comparing v3 and v4,
  rather than comparing a mixed-horizon control with a single-horizon cell. If
  alignment helps, solver-head experiments use the winning mode.
- Revisit when: Solver-head ablation (item 4) is scoped, or if alignment results show horizon-aligned tasks benefit disproportionately from temporal inductive bias.

## 2026-09-03 - Stash@{0} proposer_llm.py Is Superseded; Stash Can Be Dropped After Verification

- Decision: The 199-line Transformers+bitsandbytes `proposer_llm.py` that exists only in `stash@{0}` is considered superseded by the lighter ollama/openai_compat proposers integrated in commits `4921708` and `bd6453e`.
- Why: The old proposer required a local 4-bit quantized Qwen2.5-7B via HuggingFace Transformers, which is heavier and less flexible than the OpenAI-compatible endpoint path already wired into the proposer factory.
- Consequences: No branch needs the stashed file. After Juan verifies nothing else in stash@{0} is needed, all 3 stashes can be dropped to clean up the reflog.
- Revisit when: If a future task requires offline Transformers inference without a server (e.g., edge deployment), the file can be recovered via `git show "stash@{0}:itrader/llm/proposer_llm.py"`.

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
