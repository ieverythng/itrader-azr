# Decisions

## 2026-09-03 - Sharpe Reward Design For Task Families 00-03

Full reward architecture designed for all four task families. `pnl_sharpe` exists in code but is not wired to any family — this is the gap. Design: Family 00 stays PnL-dominant (short horizon, insufficient data for stable Sharpe), Family 01 gets PnL+Sharpe blend (mean-reversion tracking), Family 02 gets Sharpe-dominant + vol-expansion bonus (breakout patience), Family 03 gets Sharpe + trend-persistence bonus (trade duration memory). The reward shifts from PnL→Sharpe as horizon increases, co-dependent on the solver shifting from FiLM MLP→recurrent LSTM. Implementation roadmap: Phase 1 (wire pnl_sharpe to family 01, ~2h) → Phase 2 (sharpe_dominant for family 02, ~1d) → Phase 3 (sharpe_trend for family 03, ~1-2d) → Phase 4 (R2c recurrent solver integration, ~3-5d) → Phase 5 (proposer model selection + loop closure = Phase B closure). Full design: `docs/research/sharpe-reward-design.md`.

## 2026-09-03 - FiLM + True Recurrence: The Novel Architecture

The FiLM + true per-episode recurrence combo is the project's key architectural contribution. Full theory, ablation design (R0-R4 + R2a-R2d sub-variants), implementation code, and success criteria documented in `docs/research/recurrent-solver-ablation.md`.

Core insight: persistent hidden state compounds across steps within an episode, giving the solver a working model of *where it is in the trading process* — not just what the market looks like. FiLM keeps task identity alive at every step (not just init), enabling timescale separation (fast components → short-term patterns, slow components → regime/trend) and implicit episode progress tracking. This is verification-aware trading: the solver tracks progress toward the task's definition of success, which no feed-forward architecture can do.

## 2026-09-03 - Keep All Ablation Variants On FiLM; Defer LSTM Head Fix

- Decision: The episode/regime alignment ablation (v0-v3) runs entirely on `SolverMLPFiLM`. The `SolverLSTMModel` obs-shape fix (`flat_dim // win` breaking with task embeddings) is deferred to the solver-head ablation (next-actions item 4).
- Why: Mixing an uncorrected LSTM head into the alignment ablation would confound two variables at once (alignment + head architecture). The FiLM baseline handles arbitrary flat obs via `flat_dim - t_dim`, so all 4 variants get a clean comparison.
- Consequences: v3 (horizon-aligned) uses FiLM, not LSTM. If alignment results are positive, the solver-head ablation can then test whether a corrected window encoder or true recurrent policy adds value on top of the best alignment mode.
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
