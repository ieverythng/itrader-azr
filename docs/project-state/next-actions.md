# Next Actions

Last updated: 2026-09-03

## Now

1. Finish the five-cell episode/regime alignment run.
   - Config:
     `itrader/scripts/experiments/configs/episode_regime_alignment_ablation.json`.
   - Mixed controls: random episode, contiguous stream, and stream-level return
     normalization.
   - Fixed-horizon matched pair: random-128 versus stream-128.
   - The launcher now passes the config through the runner's positional CLI and
     consumes `--force` locally.
   - Hold solver head, reward, data, seeds, and PPO budget constant.

2. Run the first Sharpe reward ablation on family `00`.
   - Config: `itrader/scripts/experiments/configs/sharpe_reward_ablation.json`.
   - Compare proven `pnl_linear`, experimental `sharpe_only`, and experimental
     `pnl_sharpe` blend with `SolverMLPFiLM`.
   - Use fixed-horizon matched cells before pooling horizons.
   - Track held-out return, Sharpe, drawdown, turnover, action entropy,
     `sharpe_delta` saturation, and throughput.
   - Keep `pnl_linear` as the sole default unless a challenger passes promotion
     criteria.
   - Reward-parameter RNG is isolated from task/horizon/seed RNG so the two
     variants receive matched task identities.

3. Make capability slices truthful.
   - Ensure family, horizon, and regime metadata reaches each stable metrics
     record consumed by `generate_capability_report.py`.
   - Do not describe the current reporter as family/regime calibrated when
     `task_breakdown_available` is false.
   - Replace dataset-wide fallback tagging with episode-window-aware regime
     attribution for streamer runs.

## Next

4. Implement solver variant R1 through TDD.
   - Add failing tests for explicit `[window * features | task embedding]`
     splitting.
   - Implement a reset-every-step window-LSTM encoder with FiLM policy/value
     heads.
   - Add an opt-in model selector to `train_solo_V2`; preserve MLP-FiLM default.
   - Compare R0 and R1 on identical tasks, seeds, reward, and data.

5. Implement true recurrent variants only after R1 is measurable.
   - R2a: native RLlib 2.9 recurrent LSTM with task concatenation.
   - R2b: per-step FiLM as a separate comparison.
   - Test initial state, state carry, sequence shapes, terminal reset, finite
     backward pass, state norms, and truncated BPTT settings.
   - Feed the newest row of the chronological market window, not its oldest row.

6. Evaluate solver-state observability separately.
   - Candidate fields: current position and normalized episode progress first;
     unrealized PnL and position age only when their semantics are explicit.
   - Add fields to matched feed-forward and recurrent cells to avoid confounding
     recurrence with extra information.

7. Lock families `01` and `02`.
   - `01` is `breakout_expansion`; `02` is `mean_reversion`.
   - Both retain the three-action shape, but need real samplers, validation,
     evaluation slices, and reward comparisons before being enabled.
   - Do not make Sharpe a family default based on name or horizon alone.

## Later

8. Build the shared trunk plus family-routed policy/value heads after two
   families are operational. One TaskSpec routes to one family head; mixed PPO
   batches may route samples to different heads.

9. Complete the Phase B proposer loop in `train_AZR` after the solver baseline,
    reward choice, and capability reporting are credible. Keep
    `train_solo_V2` independently runnable.

10. Add proposer-on/off and proposer-model comparisons with task acceptance,
    invalid-output, diversity, difficulty, solver-uplift, and LLM-throughput
    metrics.

11. Build Phase C replay/shadow flow through `MarketFramer -> CandidateTask ->
    solver -> DecisionGate -> RiskEngine -> Executor -> DecisionRecord`, with
    `WAIT` first class and risk/execution deterministic.

12. Explore Market-JEPA behind the MarketFramer representation seam. Compare
    masked latent prediction and regime-transition objectives against the same
    downstream candidate and solver contracts before adding online influence.

## Blocked Evidence

- Solver-quality conclusions require completed matched training and held-out
  evaluation on the target workstation.
- Recurrent claims require a real RLlib stateful implementation; the legacy
  `SolverLSTMModel` is not evidence.
- Online thresholds require calibrated capability records, not aggregate reward
  curves.
