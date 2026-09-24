# Next Actions

Last updated: 2026-09-24

## Dataset-v2 Phase A reset

1. **Done:** authenticate Alpaca SIP historical access and build the immutable,
   split-adjusted 12-symbol snapshot through 2026-09-22. Preserve all sessions
   physically, but select RTH only for Phase A.
2. **Done:** make Parquet the authority, DuckDB the filtered catalog/query layer,
   and Ray the consumer of only requested symbols/sessions/columns. Enforce
   session and exact one-minute episode boundaries.
3. **Done:** run the 32-start flat/long/short/random/untrained RTH controls and
   matched fresh 12-update no-cost/1-bp R0 bridges. PPO is active, but both arms
   fail the economic gate; the 1 bp arm reduces turnover by becoming 95.2% short.
4. **Ready and paused before launch — matched three-arm P0:** Gate D completed
   all four members at 170,152--170,416 steps with 140 report/checkpoint pairs,
   five genuine exploits, and no trial error or score/checkpoint lag. A direct
   forensic pass also matched every retained restore copy to its selected donor
   tree hash. New arms go further: every scheduler selection must bind to exactly
   one target restore receipt, a same-generation target report, and a
   same-generation retained checkpoint. Legacy Gate D inspection requires an
   explicit compatibility flag.

   The original 32-task panel is historical because its horizons used different
   starts. The canonical P0 now uses eight explicit RTH starts crossed with all
   four horizons (32 tasks). Its 1 bp controls are flat 0.00000%, long-only
   -0.09785%/+0.00827 Sharpe, short-only +0.05785%/-0.01558 Sharpe, random
   -1.31143%/-0.25857 Sharpe, and untrained FiLM exactly equal to short-only.
   This removes the horizon/start shortcut without pretending that a
   selection-informed development panel is an independent test.

   Launch only `--variant no_cost` from the frozen canonical JSON when the user
   resumes the goal. Keep `stream_worker`, family 00, random/generated tasks,
   four members, 60 updates, the explicit crossed panel, and 1 bp evidence fixed.
   Jointly search lower learning rates, bounded decaying entropy schedules, and
   the existing PPO/GAE parameters. At 20 updates/~96k steps, require four live
   members, at least two genuine exploits, strict generation-bound lineage,
   finite/non-empty evidence, and at least 20 GB free on C. Stop between arms;
   no gate result launches staged-cost or exact-cost automatically. **Do not
   launch PBT while the current goal is paused.**
5. **Next P1 after P0 selection:** move selection to predeclared non-overlapping chronological
   blocks. Require positive 1 bp net and Sharpe, aggregate uplift over flat and
   the strongest matched directional control, paired block evidence, exact
   replay, 2 bp survival, and an independent population seed before final test.
   Also report dominant action and logit margin by start/horizon. If crossed-panel
   candidates remain horizon-static with weak value fit, run named market-only,
   horizon/task-only, and matched explicit-state diagnostics rather than spending
   another blind PBT budget.
6. **Parallel only:** R-series lifecycle implementation may proceed without a
   performance claim. Frozen Qwen3.8 and families 01/02 begin only after the R0
   data/evidence baseline is frozen; adaptive proposer reward, dividends, and
   extended-hours training remain later named factors.

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
5. **Controls, matched bridges, and P0 discovery/replication complete; promotion blocked:** keep R0
   `SolverMLPFiLM` as the compatibility control. The frozen controls, bounded
   probe, and matched four-member-per-arm 120-update bridges completed on
   2026-09-16 and 2026-09-17. The latest synchronous bridge generated 10 and
   8 genuine config changes in the exact 1 bp and PnL-linear arms. Mean terminal
   net returns were -0.14219% and -1.00048%; every terminal Sharpe was negative.
   An early PnL-linear checkpoint replayed at +0.06718% net but Sharpe -0.02301
   and turnover 26.1875. Its 14-block exploratory paired interval against
   long-only crossed zero; its same-action 2 bp net return was -0.19470%.
   Neither bridge reward arm passed its gate. The later no-cost P0 discovery
   produced two exact-replay, 1--2 bp-positive checkpoints above matched
   short-only net and Sharpe, but they share ancestry. Two independent
   populations failed replication. Do not spend the multi-seed overnight
   budget or open the final test until an independent challenger clears net,
   Sharpe, and turnover controls. Promote only on an
   untouched final test with the same controls, block-aware uncertainty, and
   multiple cost assumptions.
6. **P0 cadence study complete; promotion still blocked:** eligibility-aware
   24k PBT produced no exploit, while raw-net 24k PBT produced two exploits and
   a replay-exact member-02 transient at 372,216 steps (+0.39258% 1 bp net,
   +0.02959 Sharpe, +0.30008% at 2 bp). It beat short-only for two consecutive
   reports, then regressed; its selection-panel block interval versus short-only
   crossed zero. Retain the checkpoint as a P1 challenger, not a winner.
7. **P1 next:** train independent lineages on genuinely new, non-overlapping
   chronological validation blocks. Require a sustained baseline-relative gate,
   selection-aware block uncertainty, regime-conditioned action attribution,
   and fresh per-step 0.5/1/2 bp return and Sharpe replay. Keep direction
   diagnostic rather than punitive; only effectively all-flat behavior is an
   automatic action gate. Preserve historical `pnl_linear` as a named reference.
8. **Use synchronous PBT for promotion studies:** set `pbt_synch: true` on Ray
   2.9 so upper-quantile donor checkpoints exist before lower-quantile exploit.
   Count actual mutations from stable configuration histories, not raw restores,
   because synchronous pauses resume every member. The adapter now recovers an
   empty scheduled save from the durable checkpoint index, reattaches an upper
   donor's own checkpoint before resume, and suppresses terminal-exposure
   mutations. Synchronous upper donors use the exact same-step immutable
   sidecar checkpoint as the donor authority; a missing exact checkpoint is a
   contract failure. An actor awaiting its first post-restore report reuses its
   attached checkpoint rather than accepting a concurrent Tune save request.
   Keep asynchronous mode only for explicit legacy reproduction.
9. Use the working Windows `.venv` for current validation. The earlier WSL
   `Exec format error` is historical and does not describe this session.
10. **Before another overnight PBT:** keep `workers: 1`, declare a memory-safe
    `ray_num_cpus` cap, and use the validated durable donor/tail adapter. For an
    external pause, still capture coordinated all-member model/progress and
    scheduler state or explicitly budget a fresh-arm restart. A storage A/B can
    follow with matched warm/cold cache, isolated temp/results directories, and
    no relocation during the study. Active Ray outputs resolve to D; require at
    least 20 GB free on C and keep WSL crash dumps on E. Do not retry GitNexus
    analysis before a coordinated WSL restart activates `crashDumpFolder`; its
    2026-09-20 Node crash created an 82.46 GB dump that was recovered to E only
    after briefly filling C, and the graph remains stale.

## Next — solver controls and true recurrence

11. Resolve the open solver trunk/head and FiLM-placement decisions through the
   user's grill-me review; do not implement proposed architecture before approval.
12. Implement the selected R1/R2 control through TDD with the lifecycle contract.
13. Run same-input/same-conditioning `R2c-CARRY` versus `R2c-RESET` if R2c remains
   selected; keep the attribution caveat explicit.

## Phase B replay/shadow track

14. In parallel with offline/curriculum evidence, define replay/shadow coverage
   through `MarketFramer -> CandidateTask -> solver -> DecisionGate -> RiskEngine
   -> Executor`, with WAIT first class and deterministic risk/execution. Do not
   defer this track to Phase C or authorize live execution.

## Later — matched observability and proposer

15. Add `R4-v1` explicit position/progress/PnL fields identically to FF and
   recurrent controls. Do not infer action/fill/position from OHLCV-only state.
16. Run the separate proposer study: rule/random control, frozen LLM, frozen LLM
    plus solver-feedback/reference selection, then adaptive proposer. Include
    frozen-versus-adaptive reference-buffer factor, cost-aware verifier, accepted
    solver timesteps, rejects, calls, tokens/latency, cost, turnover, drawdown,
    chronological validation, and untouched final test.
17. Build shadow mode only after capability profiles have held-out calibration
    evidence. PPO logits/value outputs are not automatically confidence or
    expected net utility. Keep candidate-specific recurrent state/replay; never
    mutate one shared state across candidate scoring.
18. Defer R3 GRU, family expansion, MoE, and JEPA until the causal solver and
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
