# Next Actions

Last updated: 2026-09-17

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
   checkpoint and perturbation cadence, persist lineage, and schedule on a
   trial-owned sampled-environment-step clock rather than restored RLlib
   iteration. Make the ablation summary enumerate all population members.
4. **Done:** persist manifests, resolved configs, trial status, and checkpoint indices
   atomically outside Tune-owned paths; fail completion when required evidence is
   empty. Repair `generate_capability_report.py` to consume the same evidence
   schema.
5. **Controls and both matched bridges complete; promotion blocked:** keep R0
   `SolverMLPFiLM` as the compatibility control. The frozen controls, bounded
   probe, and matched four-member-per-arm 120-update bridges completed on
   2026-09-16 and 2026-09-17. The latest synchronous bridge generated 10 and
   8 genuine config changes in the exact 1 bp and PnL-linear arms. Mean terminal
   net returns were -0.14219% and -1.00048%; every terminal Sharpe was negative.
   An early PnL-linear checkpoint replayed at +0.06718% net but Sharpe -0.02301
   and turnover 26.1875. Its 14-block exploratory paired interval against
   long-only crossed zero; its same-action 2 bp net return was -0.19470%.
   Neither reward arm passed the matched long-only plus Sharpe gate. Do not
   spend the multi-seed overnight budget until a bounded
   challenger clears net, Sharpe, and turnover controls. Promote only on an
   untouched final test with the same controls, block-aware uncertainty, and
   multiple cost assumptions.
6. **Next bounded solver study:** diagnose the inactivity attractor under 1 bp
   costs without weakening the evaluation truth. Compare a no-cost learning
   control, a staged cost curriculum, and exact 1 bp `net_return` on identical
   tasks and seeds; always score all three at 1 bp. Gate first on gross edge over
   long-only, then on positive net return, positive Sharpe, and controlled
   turnover. Preserve historical `pnl_linear` as a named reference rather than
   silently changing it. Incremental Sharpe telescopes only as an undiscounted,
   pre-clipping sum and is not automatically policy-invariant when `gamma < 1`.
7. **Use synchronous PBT for promotion studies:** set `pbt_synch: true` on Ray
   2.9 so upper-quantile donor checkpoints exist before lower-quantile exploit.
   Count actual mutations from stable configuration histories, not raw restores,
   because synchronous pauses resume every member. Keep asynchronous mode only
   for explicit legacy reproduction.
8. Use the working Windows `.venv` for current validation. The earlier WSL
   `Exec format error` is historical and does not describe this session.
9. **Before another overnight PBT:** implement a coordinated all-member
   checkpoint/pause protocol with exact progress/report and scheduler-state
   recovery, or explicitly budget a fresh-arm restart. Preserve the interrupted
   2026-09-17 PnL-linear prefix as provenance, not comparison data. A storage
   A/B on the same short loadout can follow with matched warm/cold cache,
   isolated temp/results directories, and no relocation during the study.

## Next — solver controls and true recurrence

10. Resolve the open solver trunk/head and FiLM-placement decisions through the
   user's grill-me review; do not implement proposed architecture before approval.
11. Implement the selected R1/R2 control through TDD with the lifecycle contract.
12. Run same-input/same-conditioning `R2c-CARRY` versus `R2c-RESET` if R2c remains
   selected; keep the attribution caveat explicit.

## Phase B replay/shadow track

13. In parallel with offline/curriculum evidence, define replay/shadow coverage
   through `MarketFramer -> CandidateTask -> solver -> DecisionGate -> RiskEngine
   -> Executor`, with WAIT first class and deterministic risk/execution. Do not
   defer this track to Phase C or authorize live execution.

## Later — matched observability and proposer

14. Add `R4-v1` explicit position/progress/PnL fields identically to FF and
   recurrent controls. Do not infer action/fill/position from OHLCV-only state.
15. Run the separate proposer study: rule/random control, frozen LLM, frozen LLM
    plus solver-feedback/reference selection, then adaptive proposer. Include
    frozen-versus-adaptive reference-buffer factor, cost-aware verifier, accepted
    solver timesteps, rejects, calls, tokens/latency, cost, turnover, drawdown,
    chronological validation, and untouched final test.
16. Build shadow mode only after capability profiles have held-out calibration
    evidence. PPO logits/value outputs are not automatically confidence or
    expected net utility. Keep candidate-specific recurrent state/replay; never
    mutate one shared state across candidate scoring.
17. Defer R3 GRU, family expansion, MoE, and JEPA until the causal solver and
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
