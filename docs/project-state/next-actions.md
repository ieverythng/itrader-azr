# Next Actions

Last updated: 2026-09-12

## Now — evidence and controls

1. **Done:** implement a versioned terminal `EpisodeEvidence` contract that reports
   optimization reward separately from raw gross return, explicit costs/net
   return, turnover including liquidation, drawdown, actions, task identity,
   symbol/split, checkpoint, commit, and dataset hash. Rename the current shaped
   `episode_return` metrics.
2. **Done:** make reset and evaluation symbol-safe. Define per-symbol chronological
   train/validation/final-test ranges with lookback and outcome-horizon purge; fit
   observation normalization on training only and freeze it for evaluation.
3. **Done:** repair PBT before another long run: independently seed members, sample diverse
   initial hyperparameters, validate every mutation (`0 < gamma <= 1`), align
   checkpoint and perturbation cadence, and persist lineage. Make the ablation
   summary enumerate all population members.
4. **Done:** persist manifests, resolved configs, trial status, and checkpoint indices
   atomically outside Tune-owned paths; fail completion when required evidence is
   empty. Repair `generate_capability_report.py` to consume the same evidence
   schema.
5. **Next:** keep R0 `SolverMLPFiLM` as the compatibility control; add matched
   flat/long/short/random/untrained baselines, then rerun the Phase A
   random/generated pipeline across multiple training seeds. Promote only on an
   untouched final test with flat/long/short/random/untrained baselines, block-aware
   uncertainty, and multiple cost assumptions.
6. Run alignment and reward comparisons independently on matched task identities.
   Keep `pnl_linear` default; do not treat reward changes as necessary for
   recurrence. Incremental Sharpe telescopes only as an undiscounted, pre-clipping
   sum and is not automatically policy-invariant when `gamma < 1`.
7. Use the working Windows `.venv` for current validation. The earlier WSL
   `Exec format error` is historical and does not describe this session.

## Next — solver controls and true recurrence

8. Resolve the open solver trunk/head and FiLM-placement decisions through the
   user's grill-me review; do not implement proposed architecture before approval.
9. Implement the selected R1/R2 control through TDD with the lifecycle contract.
10. Run same-input/same-conditioning `R2c-CARRY` versus `R2c-RESET` if R2c remains
   selected; keep the attribution caveat explicit.

## Phase B replay/shadow track

11. In parallel with offline/curriculum evidence, define replay/shadow coverage
   through `MarketFramer -> CandidateTask -> solver -> DecisionGate -> RiskEngine
   -> Executor`, with WAIT first class and deterministic risk/execution. Do not
   defer this track to Phase C or authorize live execution.

## Later — matched observability and proposer

12. Add `R4-v1` explicit position/progress/PnL fields identically to FF and
   recurrent controls. Do not infer action/fill/position from OHLCV-only state.
13. Run the separate proposer study: rule/random control, frozen LLM, frozen LLM
    plus solver-feedback/reference selection, then adaptive proposer. Include
    frozen-versus-adaptive reference-buffer factor, cost-aware verifier, accepted
    solver timesteps, rejects, calls, tokens/latency, cost, turnover, drawdown,
    chronological validation, and untouched final test.
14. Build shadow mode only after capability profiles have held-out calibration
    evidence. PPO logits/value outputs are not automatically confidence or
    expected net utility. Keep candidate-specific recurrent state/replay; never
    mutate one shared state across candidate scoring.
15. Defer R3 GRU, family expansion, MoE, and JEPA until the causal solver and
    proposer evidence is credible.

## Remaining blockers to a performance claim

- Historical PBT scores are shaped optimization reward, not economic return; both
  restored targets fail the 1 bp net-return diagnostic and remain quarantined.
- The repaired PBT population now starts distinctly, validates mutations, aligns
  checkpoints, matches validation tasks, and records restore-aware lineage.
- Ray-owned manifests and policy logs still reproduce as zero-byte artifacts;
  non-empty stable manifests/config history/status/checkpoint indices now replace
  them as authoritative provenance.
- No R1/R2/R3/R4 implementation exists on this branch.
- No recurrent training result exists on this branch. The bounded Phase A PPO
  probe validates wiring/search only, not performance.
- New runs use `semantic_v2`; `legacy_v1` remains only for checkpoint compatibility.
- `train_AZR.py` has an independently audited unbounded backend-exception retry
  path; track it as a separate implementation issue, not as evidence of a
  successful AZR loop.
