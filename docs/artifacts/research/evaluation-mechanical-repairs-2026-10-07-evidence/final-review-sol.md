# Fresh final review: Q27 mechanics and experimental economics pilot

Review date: 2026-10-07. Repository branch: `agent/Watson-dev`.
Fixed base: `3a3583d109af7a4c094124e73ad84008f371d8e3`; HEAD equals that base,
so the subject is the scoped working tree, including new pilot files.
Reviewer: fresh independent agent `/root/q27_pilot_final_review_sol`.
Dispatch metadata confirmed by the coordinator: GPT 6.1 Sol, max reasoning,
`fork_turns=none`. This is dispatch metadata, not observed model telemetry.
`REVIEW.md` was read completely before this declaration. No scoped implementation
or tests had been read when the following inputs were written.

## Predeclared adversarial inputs

### Q27 mechanics

1. Metadata eligibility: synthetic OHLCV-only Parquet with a valid footer row
   count; timestamp and symbol metadata in differing row/symbol orders; boundaries
   at the validation interval and physical EOF; explicit negative, fractional,
   boolean, string, missing, and out-of-range starts; invalid combined start and
   horizon; source starts and override starts; ordinary clamping as a control.
2. Shared partition construction: compare metadata preflight eligibility with
   real environment eligibility for the same synthetic source/configuration,
   including session/cadence declarations, empty data, and one readable row.
3. EOF/reset: 100 constant-price rows with `(start, horizon)` values `(99, 1)`,
   `(98, 2)`, `(96, 4)`, plus a requested horizon extending beyond physical EOF.
   Probe flat/long/short, switching, final liquidation, start/end rows, terminal
   R4 tails, and the next reset. Exact-boundary horizons must complete; censored
   episodes must remain censored.
4. Terminal selection: empty index, one unique maximum, tied highest update
   with equal or differing sampled steps in both row orders, repeated identical
   rows, missing historical status, completed/noncompleted statuses, and
   mismatched completion step/iteration coordinates. An ambiguous maximum must
   reject rather than select by input order.
5. R4 intervention: wrong schema version, wrong state feature order/count,
   base/agent width mismatch, unsupported mode, and actual reset observation
   shape mismatch. Valid historical initialization remains a control. Exercise
   both helper and CLI/preflight consumers without Ray or checkpoints.

### Experimental pilot

6. All nine prior/selected position transitions from `{-1, 0, 1}`; zero,
   positive, and negative asset returns; zero and positive transaction costs;
   terminal and nonterminal rows; flat/long/short/hold-prior references. Switching,
   continuation, hypothetical roundtrip, and actual terminal liquidation must
   reconcile to existing environment evidence.
7. Trace integrity: wrong positions, broken prior-position chain, duplicated
   or skipped rows, stale post-terminal context, changed task/action/schema/policy
   identity, remapped starts, censored horizons, and sealed-test requests. Reject
   malformed evidence instead of presenting attribution as valid.
8. Optional volatility estimator: absent, zero, negative, nonnumeric, nonfinite,
   extremely small, and unrepresentable scales; future-row perturbation with an
   identical frozen past-close context. Missing scales must remain unavailable
   with truthful coverage/reasons. No test-only estimator becomes a default.
9. Additive attribution: synthetic alternating long/short return path and pure
   directional controls; gross/cost/net/reward reconciliation; directional gross
   plus timing gross equals realized gross. Check producer-label honesty and
   the absence of a new pooled promotion score, single-step Sharpe, or training
   feedback.

## Execution and base comparison

All Python executions used `.venv/Scripts/python.exe -B`. Synthetic frames,
Parquet, configs and index files lived in process-owned temporary directories.
The only durable reviewer-written file is this report. No real market data,
Ray import/start, WSL, training, checkpoint replay launch, full-suite duplicate,
production/test edits, staging or commit occurred.

Before reading implementation, I ran:

```powershell
& .venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider itrader/tests/test_env_core.py itrader/tests/test_policy_baselines.py itrader/tests/test_economic_diagnostics.py
```

Initial result: **90 passed in 4.69 s**. After the partition/scale corrections,
the identical command returned **94 passed in 3.93 s**. A final collection of
`test_economic_diagnostics.py` confirms 28 cases. These counts identify execution
snapshots; the independent assertions below supply the behavioral evidence.

`git diff --check 3a3583d --` the five tracked scoped source/test paths passed.
`tools/knowledge/status.sh` through explicit Git Bash reports stale index
`1d5ba82` and exits 1 at `python3: command not found`. No graph refresh or WSL
fallback was attempted. Direct source remains authoritative.

### Independently constructed public probes

- **162 accounting cases:** Cartesian product of prior/target positions
  `{-1,0,1}`, returns `{-0.02,0,0.03}`, cost-bps `{0,1,2.5}` and terminal
  `{false,true}`. For actual, hold-prior, flat, long and short, independently
  assert `position * return - cost_bps * 1e-4 * abs(position-prior)` minus
  `cost_bps * 1e-4 * abs(position)` only at actual termination. All matched.
- **1,440 eligibility comparisons:** two symbols, 20 rows per symbol/date,
  dates 2024-02-28, leap day 2024-02-29, 2024-12-31 and 2025-01-01; the
  December session is after-hours and the others regular. Uppercase symbol,
  timestamp and session columns, per-session boundaries, one-minute cadence,
  validation fractions `[0.1,0.8,0.1]`, window 5. Forward, reverse and shuffled
  row orders each produce the same metadata/runtime ranges for horizons
  `{1,2,4,15}`. A `pandas.read_parquet` spy observes metadata-only projections,
  excluding all five OHLCV columns. OHLCV-only footer-backed eligibility also
  equals runtime eligibility across all 100 starts and horizons `{1,2,4}`.
- **Strict starts:** public reset rejects `{-1,0,4,99,100,100000,60.5,"60",
  true,null}` for window 5/horizon 2 on 100 rows. Eligible starts 5 and 98 stay
  exact. The existing public control verifies ordinary clamping still works.
- **Terminal selection:** persisted empty, tied, reverse-tied, identical
  duplicate and equal-step/different-path indices reject. A unique historical
  maximum without status and a matching completed status pass. Running status,
  mismatched sampled steps and mismatched iteration each reject. Messages are
  respectively the explicit empty-index, ambiguous-maximum, completed-status
  and completion-coordinate errors; no input-order selection survives.
- **Diagnostic context:** a metadata-only 6,000-row synthetic source with
  window 5/horizon 2, source start 1,000 and validation fractions
  `[0.1,0.8,0.1]` accepts replacement start 2,000 with strict task starts.
  Starts 999,999, 599 and 5,400 reject before outcome evaluation.
- **R4 consumers:** five malformed schema/mode/width/feature-order contracts,
  wrong observed array width, and wrong reset-array width reject before policy
  scoring. Public CLI `main()` rejects state-free schema, reversed feature order
  and config/schema width conflict before any Ray import. CLI checkpoint loading
  and invocation construction were inert substitutes in this narrow preflight
  probe; actual eligibility, reset/step and accounting probes used real owners.
- **Real pilot:** synthetic 100 closes, 60 initial closes at 100, next closes
  110 and 99, explicit start 60/horizon 2, actions long then short. Public
  `evaluate_economic_diagnostics` yields gross `0.2`, cost `0.0004`, net `0.1996`,
  optimization reward `199.60000000000002`. Directional gross is zero and timing
  gross `0.2`; this is an algebraic fixture, not forecasting or market evidence.
- **Trace integrity:** mutate one actual public trace at a time: wrong target,
  broken prior-position chain, duplicate row, skipped row, wrong action, wrong
  task hash, changed second-row schema, changed second-row policy. All eight
  reject through `attribute_episode`; changed declared horizon and stale
  post-terminal context also reject. Effective `test`, `all`, or `validation`
  under declared `train` rejects before reset (`current_task` remains absent).
- **Causal context and controls:** perturb the next close from 110 to 150;
  initial observations and complete pre-action contexts stay equal. An explicitly
  named test estimator receives the frozen 60-close tuple, then the newly past
  150 only at the following decision. Flat/long/short public controls have zero
  timing residual and net `0/-0.0002/-0.0002`. Producer verification remains
  false, with no default estimator, single-step Sharpe or pooled promotion score.
- **Unavailable scales:** absent, zero, negative, boolean, string, NaN, infinity,
  smallest positive float and integers `+/-10**400` exercised directly. The
  originally failing large integer now returns unavailable/unrepresentable;
  tiny positive float returns unavailable/nonfinite-scaled-outcome. Ordinary
  raw economics stay available and unchanged.

### Fixed-base comparison

Loaded the exact Git source with `git show 3a3583d:<path>`, compiled into
separate in-memory modules, and executed base/current side by side. Unchanged
runtime dependencies were held common, so this isolates the reviewed source
behavior rather than claiming a complete historical environment reproduction.

| Start / horizon | Base reset / evidence start / end / steps | Current evidence start / end / steps |
|---|---|---|
| 60 / 4 | 60 / 60 / 64 / 4 | 60 / 64 / 4 |
| 99 / 1 | 99 / 98 / 99 / 1 | 99 / 100 / 1 |
| 98 / 2 | 98 / 98 / 99 / 1 | 98 / 100 / 2 |
| 96 / 4 | 96 / 96 / 99 / 3 | 96 / 100 / 4 |

For long-only constant prices, both produce net `-0.0002` and reward sum `-0.2`;
the repaired rows/identities and complete horizons are the changed behavior.
All nine current EOF combinations (three edge horizons times short/flat/long)
have exact evidence bounds, terminal R4 `[selected_position,0]`, and next-reset
R4 `[0,1]`.

The interior long/short public pilot agrees exactly with base environment on
gross, cost, net, turnover, optimization reward, start, end, steps and horizon.
Base/current no-override `load_validation_env` configs also agree exactly.
The base has no `load_checkpoint_selection`, `counterfactual_state_details`,
`evaluate_checkpoint_state_diagnostics`, or `economic_diagnostics.py`; those
new public surfaces cannot honestly receive a historical output-parity claim.
Their behavior was instead exercised directly and checked against the explicit
review inputs and protected environment evidence.

## Findings and design principles

**BLOCKING at discovery; resolved — 1 of the 1 found so far.**
`itrader/rl_lib/evaluation/economic_diagnostics.py:52` converted an unrepresentable
optional volatility integer to float without containment. Input:

```python
step_accounting(prior_position=0, target_position=1, asset_return=0.01,
                cost_bps=1, terminal=False, lagged_volatility=10**400,
                volatility_method="predeclared_probe")
```

Expected: unavailable normalized outcomes with an explicit reason, preserving
raw accounting. Actual before correction: `OverflowError: int too large to
convert to float`. The writer extended the existing public unavailable-scale
matrix and added the narrow conversion guard. Fresh independent re-execution
of positive/negative large integers and the tiny float passes; the scoped suite
then passed 94 cases. No unresolved blocking finding remains in this review.
The pilot module did not exist at the fixed base, as independently checked.

| REVIEW.md principle | Mechanics | Experimental pilot |
|---|---|---|
| Separation of concerns | OK: DataPartition owns eligibility; env adapter projects metadata; evaluator owns checkpoint selection and gates. | OK: env owns actual transitions and context; reporting owns secondary decomposition; RewardEngine and headline evidence remain authoritative. |
| Programming by intention | OK: strict starts, unique maximum and R4 contract checks state their intended rejection conditions. | OK: continuation, hypothetical roundtrip and actual liquidation are distinct; unavailable scaling is explicit. |
| Encapsulation | OK: shared `market_partition` construction avoids consumers rebuilding split/session/cadence rules; state mutations follow validation. | OK: estimator receives a frozen past-close tuple; attribution must reconcile to episode evidence. |
| High cohesion | OK: reset/EOF correction stays in env; replay identity/eligibility stays in evaluation. | OK: the new module contains additive diagnostics and bounded opt-in orchestration without changing learning. |
| Low coupling | OK: historical defaults remain controls; R4 validation does not require zero initialization or alter task embeddings. | OK: the pilot uses env/policy public seams and caller-named optional estimator; no proposer, trainer, broker or promotion dependency is added. |

No new domain with two owners, duplicated production business rule, or business
logic in a template/UI was found. Recomputing a reporting invariant against the
owning environment evidence is intentional verification, not a competing
execution/reward authority.

## Test value assessment and consolidation recommendations

The Parquet -> DataPartition -> actual env reset/step -> RewardEngine ->
EpisodeEvidence -> diagnostics chain has the strongest value here. Persisted
terminal JSONL/status/config inputs cover the selection seam directly. Passing
helper-only tests cannot replace either integration.

1. `test_episode_attribution_reconciles_direction_timing_and_existing_evidence`
   and `test_opt_in_pilot_uses_actual_env_arithmetic_without_changing_headline_evidence`
   repeat positive decomposition assertions. Consolidate shared positive
   assertions into a parameterized public-env pilot covering flat/long/short and
   long-short. Keep wrong-evidence and wrong-position rejection cases distinct;
   coincident aggregate totals must never substitute for valid transitions.
2. The nine-transition helper matrix plus separate terminal accounting fixture
   could become an actual-env transition matrix: establish each prior position
   on a setup bar, choose each target on the scored bar, cross terminal versus
   nonterminal, and compare reported costs with terminal EpisodeEvidence. Keep
   one small helper boundary matrix for unavailable/nonnumeric/unrepresentable
   scales, where malformed estimator outputs are the distinct fault class.
3. The footer-only, uppercase, boundary-only and partition-filter fixtures can
   share one synthetic frame builder and a metadata -> strict reset/step matrix.
   Retain the projection spy, symbol/session/cadence gap rejection, and final-row
   identity assertions. A shared factory alone adds no evidence; real consumers
   must still execute, and historical protected tests should stay intact until
   equivalent coverage is demonstrated.
4. Add persisted CLI-preflight coverage to the same R4 malformed-contract matrix
   when that consumer is next touched. Helper tests currently give limited
   evidence of wiring; this review's independent public `main()` probes showed
   rejection before Ray, but they are not a committed regression. Preserve the
   valid historical initialization control and actual reset-array mismatch case.

These are optional consolidation/coverage recommendations, not additional
correctness findings or authorization to edit/delete tests. Duplicate setup and
overlapping positive assertions are modest; the exact EOF thresholds, schema
aliases, reversed terminal rows, malformed trace identities and partition
disagreement cases each retain a distinct adversarial failure or provenance
boundary. Case count is not the acceptance criterion.

## Documentation, scope and final assessment

The final dated repair note was reread completely. It distinguishes prior
failures, current experimental reporting, earlier test snapshots, pending
final validation, and future Q23/Q26/terminal execution. It explicitly records
the conservative adjacent-window rejection and unsupported source `start_index`
alias. Those acknowledged fail-closed limitations are not complete alternative
input support, leakage permission, or an instruction to broaden this repair.

Mechanics: **APPROVE** for the scoped Q27 corrections and protected controls.
Pilot: **APPROVE** as bounded synthetic secondary reporting after the resolved
scale correction and effective-partition gate. This says nothing about market
performance, producer attestation, calibrated volatility, fresh-panel
qualification, Q24/Q26 statistics, or the future terminal-only role/executor.
No real terminal panel or sealed test was opened.

Reviewed source snapshot SHA-256 values (LF/CRLF bytes as read):

```text
itrader/envs/itrader_env.py cb262c630cbdefbf19b9a797bcc90e392234ae6cb771004b18dd4902c355a268
itrader/rl_lib/evaluation/policy_baselines.py 9a13e18a049d59edcfecd0d3491659c8ae1ded5cdb89b340bbfe78150765ca52
itrader/rl_lib/evaluation/run_checkpoint_state_diagnostics.py 6ee19f5ec4086529c95a7d6af8c3e1e2a853a16e34ce21fc46e7de55ad968c9f
itrader/rl_lib/evaluation/run_checkpoint_replay.py 025c1b2ddaea24e41517886c91b08fcafb8816d3d4f6db860c33e1f460a8fedb
itrader/rl_lib/evaluation/economic_diagnostics.py 9fdcbaa1e5233edd6926736d2054645cab2946226da27de4d4e46423e27f33b8
itrader/tests/test_env_core.py 219e0cfddcc2d480b16fca80fba2e0b426891436dbc69e67c40a931593987703
itrader/tests/test_policy_baselines.py a1f6411b1db0c817bae73afad47ead2bfea176fe22b7f0f658a8b68a92286d63
itrader/tests/test_economic_diagnostics.py 0717f2b33bf771f4cc5ef0838696e25e1cc6e3a0570600417f3f0050f26ecaf6
docs/artifacts/research/evaluation-mechanical-repairs-2026-10-07.md 149953afbeab48199475e2606d5bac88cb47a4fcdbf4c25e8ccc5a8b7bd36730
```

VERDICT: APPROVE
