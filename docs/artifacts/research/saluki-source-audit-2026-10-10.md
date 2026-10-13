# Saluki advisory source audit

**Date:** 2026-10-10  
**Scope:** Static comparison of the imported Saluki trading-RL advisory against current source, tests, and the R0/R4-v1 loadout.  
**Status:** Advisory review only. No source or test files changed; no tests, model, Ray, or training jobs run; no market outcomes or raw data accessed. No proposal in the advisory is adopted by this audit.

## Executive conclusion

The repository has explicit seams for target-position actions, environment-owned reward context, registered per-step reward calculation, independent episode evidence, chronological/session-bounded sampling, model selection, and solver-state inputs. The imported advisory often describes related ideas, but several of its concrete claims do not match those seams. In particular:

- the current action is a target position (`short`, `flat`, `long`), not an order or a “trade count”;
- configured cost is charged per unit of position turnover, plus terminal liquidation in the relevant accounting path—not once per round trip;
- the evidence ledger is not the PPO optimizer reward in general;
- the R0/R4-v1 static loadout fixes cost at 1 bp and has PBT disabled; its gamma is omitted and resolves to the training CLI default 0.99, not the advisory’s three-value gamma set;
- the suggested five-action WAIT/EXIT expansion changes the action contract and introduces state-dependent aliases; it is not a no-op extension;
- the literal v2 transform concatenation has a 2-D/1-D rank mismatch and raises; reshaping its repeated scalar list into a row would instead make 61×5, still violating the present 300-value market contract.

No current production defect was established by this static pass. Several advisory recommendations are plausible experiments, but they need named hypotheses, matched recipes, and evidence gates before implementation. Claims about dataset row counts/coverage, performance, uncertainty intervals, the root runtime, and runtime GAE activation remain **UNKNOWN** here because this audit did not inspect dataset artifacts, outcomes, or initialize software.

## Inputs and method

Read the advisory Markdown and HTML as untrusted research material; checked source, nearby public tests, active architecture/masterplan, and the loadout configuration. The HTML was read as static text with embedded script/style removed; it was not executed in a browser. The available architecture-check, codebase-design, TDD integration-testing, and targeted deslop instructions informed this source/test review. This was not an independent runtime review or a numerical qualification.

Imported advisory identity (verified SHA-256):

| Input | SHA-256 |
| --- | --- |
| `docs/research/trading-rl-evaluation-advisory-2026-10-10.md` | `0dd526633e95bcf47525e68d55abc40f4734bbbf035cdb5ecdd065be3fc7b707` |
| `docs/research/trading-rl-evaluation-advisory-2026-10-10.html` | `fbef9a5db93272bed7d05df12f23ea5dd498163b6e4c38b72d9acbecb6c6dc30` |

The HTML companion contains the same material claims and does not provide independent evidence. Active-doc precedence remains source/tests/config first, then architecture/masterplan and project-state; this audit is a dated artifact, not a replacement contract.

## Claim ledger

| Advisory claim or recommendation | Verdict | Source-grounded correction |
| --- | --- | --- |
| The asserted 4.4 bp relative gap is smaller than a 1 bp cost. | **FALSE as written** | 4.4 is numerically greater than 1. Aggregate episode costs may be more than 1 bp because cost applies to turnover units, but that does not make 4.4 bp “smaller than 1 bp.” Neither this audit nor a source diff validates the 4.4 bp outcome itself. Do not infer significance from this arithmetic correction. |
| Configured 1 bp is a cost per round trip. | **FALSE** | `ITraderEnv.step` computes `abs(target_position - prior_position)`; `net_return` and `EpisodeEvidence` multiply that turnover by bps. A flat→long→flat path incurs 1+1 turnover units; long→short incurs 2 in one transition. Remaining exposure also incurs terminal liquidation turnover. This is a configurable simulation cost, not validated broker spread/slippage/commission. |
| Repeating the long/short action increases a position unit. | **FALSE** | Actions set the target to -1/0/+1; repeated target means zero turnover. The action does not scale in. |
| PPO always optimizes `EpisodeEvidence.net_return`. | **FALSE** | Environment reward is returned by `RewardEngine`; evidence is separately derived. The current R0+R4-v1 loadout selects `net_return` with matching 1 bp reward/evidence cost and scale 1000, which gives an arithmetic relation for that recipe. Other registered objectives (e.g. `pnl_linear`) differ. Evidence is not inferred from reward components. |
| Repository code computes GAE in `RewardEngine` or proves it was active for a run. | **FALSE / UNKNOWN** | RewardEngine computes per-step reward. Training passes gamma/lambda into RLlib and the diagnostic PPO policy consumes advantages/value targets; GAE postprocessing belongs to the Ray/RLlib runtime. Runtime activation/defaults and historical run identity are not established by this static source check. |
| This static R0+R4-v1 recipe sweeps gamma over {0.98, 0.99, 0.995}. | **FALSE** | The named loadout sets `pbt: false`, fixed 1 bp optimization/evidence cost, 40 iterations and 4,800 train batch size; it has no gamma override. The training CLI defaults gamma to 0.99 and lambda to 0.95. The 40×4,800 product is a nominal batch target, not a measured count of environment transitions. |
| Reducing training cost for the same trained policy reveals its “unmasked” behavior. | **FALSE as a causal claim** | Retraining with 0.1/0.01 bp changes the optimized objective and can produce a different policy. Re-accounting a frozen checkpoint at multiple evidence costs is a different experiment and isolates accounting sensitivity, not the effect of training cost. Keep the two estimands separate. |
| A cost-response curve alone diagnoses whether architecture or reward is the bottleneck. | **UNSUPPORTED** | Lower cost can change turnover, policy learning, and measured net return; these possibilities are confounded in a retrained cost sweep. Frozen-checkpoint replay plus a matched retraining factorial can separate some effects, but neither proves an architectural bottleneck by itself. |
| The current 200-row purge is insufficient and should become 500–1,000. | **UNSUPPORTED recommendation** | `DataPartition` splits by symbol, chronology, expected interval and boundary columns, then applies configured purge around split outcome ranges; eligibility also respects window and horizon. Current R0 horizon max is 200 and window is 60. A larger purge needs a specified dependency/leakage rationale and a declared data-retention tradeoff; no source evidence here requires the proposed value. |
| Current `market_relative_v1` z-scores every OHLCV channel. | **FALSE** | OHLC prices are normalized relative to the final close in the window; volume is log1p-transformed and standardized per window; other feature channels are standardized. The result is sanitized/clipped to [-10,10]. |
| Suggested `market_relative_v2` snippet preserves the R1 input schema. | **FALSE for the shown construction** | The literal `np.concatenate([window_v1, [vol] * len(window_v1[0])])` passes a 2-D 60×5 array and a 1-D length-5 list to default-axis concatenation, so NumPy raises a dimension-mismatch error; it does not produce a 61×5 array. If the list were reshaped to one row, concatenation would produce 61×5 and still violate R1's 60×5 market contract. Repeating one scalar across OHLCV slots is not a defined per-bar volatility feature. The snippet's “precomputed manifest” is also not inherently leak-free unless the fitting cutoff and provenance exclude validation/final-test information. Any new transform needs a versioned schema, exact shape/order, fit provenance, and past-only construction. |
| Family 01/02 and “pairs” behavior are currently active. | **FALSE / UNKNOWN** | Registry metadata labels 01 breakout and 02 mean-reversion, but both are disabled; active family is 00. Their metadata does not establish trained family-specific features/rewards. The inspected contract remains a three-action target-position path; no multi-leg/pairs execution semantics are shown. |
| Discrete(5) adds WAIT and EXIT without changing the action contract. | **FALSE / design risk** | Five outputs are a different contract; current R1 explicitly requires three. Under target-position semantics, WAIT can duplicate “keep the same position”; EXIT aliases flat for a nonflat position, and in a flat state may duplicate flat/no-op. Duplicate state transitions can still receive separate categorical probability/entropy mass. Require a versioned action definition and matched restore/control identity before considering this. |
| Checkpoint/model identity versions the action-ID vocabulary as well as output width. | **FALSE for same-width remaps; prospective compatibility gap** | `model_identity` serializes selector plus custom model config, and the semantic observation contract versions the input schema but has no action-schema ID. R1 rejects an output count other than 3, so a 3→5 width change fails. A permutation/reinterpretation that still has three outputs is not represented in this identity. The current source mapping is fixed, so this is a future-change risk, not an observed current runtime defect. Before any action-vocabulary change, version it in the manifest and assert restore rejection for a same-width remap. |
| Online `WAIT`/`NO_ACTION` requires a solver action alias. | **FALSE** | The online shell treats WAIT/NO_ACTION as an explicit outcome; it is not evidence for duplicating an offline target-position action. Keep action semantics separate from broker/risk/executor contracts. |
| 0.1/0.01 bp “unmasks” signal, and >10 turnovers implies reward structure is wrong. | **UNSUPPORTED** | Those costs and turnover threshold are proposed diagnostics, not source facts or accepted gates. A 1 bp result remains the economic anchor in current docs; lower-cost cells can be diagnostic only with predeclared interpretation and the frozen policy/retrained-policy distinction above. No arbitrary turnover cutoff is adopted. |
| R0 is necessarily undertrained by 10–100×; negative at 0.01 bp would prove architecture is the bottleneck. | **UNSUPPORTED** | The 10–100× step multiplier and the asserted architectural diagnosis are hypotheses. A larger budget is a new experimental decision requiring compute/stopping limits, matched controls, and non-overlapping evidence. The source audit cannot certify a “true” training budget or diagnose performance without valid results. |
| The advisory's row counts, date range, session composition, or 4.9-year generalization narrative are verified here. | **UNKNOWN** | The advisory cites a dataset manifest, but data artifacts were out of scope. The loadout establishes configured symbols/sessions/path, not actual row counts, market coverage, or realized episode distribution. |
| A 96–200 “horizon” is the solver's forward prediction lead or extra market-history length. | **FALSE** | The configured horizon is episode length. `ITraderEnv` advances one bar per action and rebuilds the trailing observation window each decision; R1 re-encodes the same 60-bar window with fresh in-window state. R4 remaining-horizon state is episode progress, not lookback. |
| A fresh walk-forward re-fit is simply a stronger version of the current holdout evaluation. | **FALSE as a protocol description** | Training on the old train+validation range changes the training cutoff and model; it is a new rolling-origin/retraining evaluation. It can be useful if predeclared, but must not be reported as the same fixed R0/R4 attribution or reopen a sealed test for repeated selection. |
| The new research claims demonstrate Sharpe near annualized 1.0 or a confidence interval crossing zero. | **UNKNOWN** | No experiment outputs or interval computations were read. Current `EpisodeEvidence` reports per-√bar Sharpe and can produce annualized Sharpe only when periods/year is supplied; annualization assumptions must be tied to the actual sampling/calendar contract. |

## Verified implementation boundaries

- `itrader/envs/itrader_env.py:17-22, 298-321` maps actions 0/1/2 to short/flat/long and emits a flattened 60×5=300 market vector, then task embedding and optional solver-state tail.
- `itrader/envs/itrader_env.py:351-370` implements the channel-specific `market_relative_v1` transform. Reset builds a trailing window from prior rows (`:432-450`); `DataPartition` additionally enforces symbol, split, time/session boundaries and interval continuity. The advisory v2 snippet at its lines 398-416 fails literal array-rank compatibility before any shape change; a row-shaped repair would be 61×5 and schema-incompatible.
- `itrader/envs/itrader_env.py:673-768` applies target position before next-bar mark-to-market, computes turnover from position delta, calls RewardEngine per step, and accumulates terminal episode information.
- `itrader/envs/itrader_env.py:342-349, 432-450, 673-717` returns the rolling trailing window at each decision, sets the selected target before the next-bar return, and treats configured horizon as episode termination/progress rather than a forecast lead.
- `itrader/rl_lib/tasks/rewards.py:266-293` charges configured `net_return` cost per turnover unit and terminal liquidation, scaled for optimization. Other reward IDs have their own semantics (`:296-371`).
- `itrader/rl_lib/evaluation/episode_evidence.py:17-23, 74-102` intentionally derives evidence separately, counts liquidation, and reports per-√bar Sharpe; annualization is conditional on periods/year.
- `itrader/rl_lib/evaluation/data_partition.py:73-164` forms contiguous symbol/time/boundary segments and applies split/purge/window/horizon eligibility.
- Split fractions are calculated from each symbol's row count (`data_partition.py:132-144`), then intersected with the timestamp/session segments. This is chronological per-symbol partitioning, not one shared calendar cut date across all symbols; reports should retain the resolved boundaries if calendar-regime alignment is a study factor.
- `itrader/rl_lib/tasks/families.py:27-80` has only Family 00 enabled; 01/02 are bootstrap metadata, 03/04 future metadata.
- `itrader/models/solver_window_lstm_film.py:15-18, 35-54, 92-131` specifies an in-window stateless LSTM, semantic 60×5 contract, exactly three Family-00 outputs and optional state residual. `itrader/models/registry.py:6-34` versions explicit model selection/identity and preserves historical restore selectors separately.
- `itrader/models/solver_mlp_film.py:35-66, 70-95` keeps the R0 market trunk and task FiLM head; R4-v1 adds a two-value, width-8 state path with zero-initialized actor/value residuals. R1 uses a 156-hidden LSTM and 256 projection but remains stateless across decisions. These are architecture/configuration facts, not a measured performance or runtime capacity claim.
- `itrader/models/observation_contract.py:11-39` pins feature order, window order/end, transform and observation schema, but it has no action-vocabulary identity; `itrader/models/registry.py:23-34, 93-124` checks model/policy/restore model identity, not same-width action-ID meaning.
- `itrader/scripts/experiments/configs/phase_a_r0_plus_r4_v1_1_dec2025_4x3_40_20260925.json:7-47` is a fixed, PBT-off loadout; this is configuration identity, not proof of completed training or performance.

### Architecture-principle check

| Principle | Assessment |
| --- | --- |
| Separation of concerns | **OK in reviewed seams.** Environment owns state/observation/context, RewardEngine owns objective, EpisodeEvidence owns independent economics, DataPartition owns eligible chronology, model registry owns solver identity. |
| Programming by intention | **OK.** Transform, cost, action, family, and model identities are named/configured; do not silently reinterpret a 3-action task as 5 actions or treat evidence cost as broker execution. |
| Encapsulation | **OK.** Mutable target position is environment-owned; task and solver identities are explicit; do not leak future regime labels or broker policy into TaskSpec. |
| High cohesion | **OK.** Reviewed modules each have clear responsibilities. A new fitted transform or action codec would need its own explicit, versioned contract if approved. |
| Low coupling | **OK with a prospective risk.** Current contracts are explicit across environment/model/registry. `Discrete(5)` or new feature widths would couple loadouts, heads, evaluation, checkpoints and controls; version rather than mutate legacy identity. |

No source/test defect was confirmed in this bounded pass. The imported advisory itself needs corrections before its numerical claims or snippets are used as implementation instructions. This is not an independent two-reviewer code approval and not a release decision.

## Test ROI: integration first, without a test-count target

The current suite is not simply a growing pile of assertion-per-field unit tests. Several focused tests protect genuinely different public contracts. Best return comes from a small set of end-to-end scenarios with literal independent oracles, while preserving compact parameterized checks for mathematical invariants and fail-closed boundaries.

| Existing coverage | Recommendation | Why |
| --- | --- | --- |
| `test_economic_diagnostics.py::test_nine_transitions_and_frozen_reference_vectors` (9-case parameterized matrix) | **Keep.** Do not replace with nine separate files/tests or delete for a quota. | It is a cheap independent arithmetic oracle for the complete prior/target state transition matrix. An environment test is a valuable additional public-path check, but it couples the arithmetic oracle to observation/env plumbing. |
| `test_env_core.py::test_terminal_info_separates_optimization_reward_from_economic_evidence` and `test_economic_diagnostics.py::test_opt_in_pilot_uses_actual_env_arithmetic_without_changing_headline_evidence` | **Potential consolidation only after a coverage map.** If test churn is material, one scenario could exercise a fixed synthetic task through public reset/step and the evaluation diagnostics adapter, asserting reward, evidence, action trace, context, and session identity. Keep both if each remains the clearest oracle for distinct paths. | They overlap on environment arithmetic, but the second also exercises the diagnostics-policy/evaluator boundary and per-step provenance; merging them can create one opaque failure-prone mega-test. No removal is approved by this audit. |
| `test_data_partition_sessions.py` exact range/session test and missing-minute split test | **Keep two distinct scenarios** or parameterize if only fixture/setup duplicates. | Valid session containment and a missing expected interval are different segmentation causes. A single named public partition test can assert both only if failure diagnostics remain legible. |
| `test_semantic_observation_contract.py` tagged schema, chronology, state neutrality, trace, conflict cases, and forward semantics | **Preserve public-boundary integration and the small rejection matrix.** Consolidate fixture construction, not distinct failure reasons. | Feature order/timing, state-tail separation, model selection, R0 parity, and trace identity guard different contracts. A single giant scenario would obscure which boundary regressed. |
| Family registry checks repeated between `test_task_contracts.py` and `test_task_families.py` | **Small candidate for a later edit:** centralize roadmap/active-family assertions in one registry-contract test; keep TaskSpec alias/canonicalization and active-family runtime enforcement separately covered. | Some metadata expectations overlap; no economic or solver integration is needed to test static registry metadata. It is low-risk cleanup, not a priority fix. |
| Selector identity tests vs actual model forward/restore tests | **Keep separate trust levels.** One loadout→selector→checkpoint/restore round trip is the high-value integration; keep small unknown/conflicting-selector rejection matrix. | A successful class lookup does not prove serialized identity, logits/value/action parity, or restore correctness. |
| Action semantics vs checkpoint identity | **Add only when action semantics are changed:** a same-output-width action-schema mismatch restore case. | Width checks catch a 3→5 head change; they cannot distinguish reassigning the meanings of three output IDs unless the vocabulary is versioned in checkpoint identity. This is a distinct compatibility oracle, not a reason for a broad test ladder today. |

### Recommended compact public oracles (future, not run here)

1. **Accounting lifecycle integration:** build a tiny public environment with close sequence `[100, 100, 110, 121, 110]`, `window_size=2`, fixed start index 2, horizon 3, actions `[2,2,0]`, `net_return`, reward/evidence cost 10 bp, reward scale 1,000, and the existing public step-economics capture enabled. Independent expectations: positions `[+1,+1,-1]`; step turnover `[1,0,2]`; liquidation turnover `1`; gross return `0.2909090909…`; total turnover `4`; cost return `0.004`; net `0.2869090909…`; summed reward `286.9090909…`. Assert the per-step public trace plus terminal evidence. This catches hold/flip/liquidation and cost basis mistakes in one meaningful path.
2. **Causal observation/partition integration:** tiny tagged multi-symbol, multi-session minute data with one missing bar; assert exact ordered history ending at `idx-1`, expected window shape, no crossing symbol/session/split/gap, and invariance to perturbing future rows. Keep a compact incompatible-schema/order rejection matrix.
3. **Model identity lifecycle integration:** loadout → explicit model selector → model construction → save/restore → logits/value/action parity, with R0 and R1 identities never relabeled. Keep small adversarial unknown/conflicting/stale-config cases. If the action vocabulary ever changes, include its version and reject a same-width remap on restore.

These are proposals for future scoped work. They are not permission to edit, remove tests, run training, or claim solver/economic validity. The accounting oracle is a test design; this audit did not execute it.

## Follow-up decisions, not adopted here

Before any code change, GRILL/human should decide whether lower cost is (a) fixed-checkpoint re-accounting, (b) retraining at a changed objective, or both as separately named arms; preserve 1 bp as the named current anchor unless explicitly changed. Treat 0.1/0.01 bp as diagnostics until gates are approved. Do not add an independent turnover penalty without an explicit objective/scale ablation. Keep family 00 and the three-action contract for R0/R1 parity. Defer market_relative_v2 and discrete(5) until a versioned semantic/action design and public restore/control path are accepted. Any budget increase, purge change, or rolling retraining protocol needs a new matched experiment specification and resource stop criteria.

### Reproducibility identities

SHA-256 values for current working-tree bytes read in this audit:

| Path | SHA-256 |
| --- | --- |
| `itrader/envs/itrader_env.py` | `cb262c630cbdefbf19b9a797bcc90e392234ae6cb771004b18dd4902c355a268` |
| `itrader/rl_lib/tasks/rewards.py` | `dc1b2eb69364a29d4fe781ff6da994d954cfc511f1890e403f20a67364daf20f` |
| `itrader/rl_lib/evaluation/episode_evidence.py` | `1a73f2ea89f1b15ee5fbd004ca51d35bfa47dd582639028ff498211a96f19e23` |
| `itrader/rl_lib/evaluation/data_partition.py` | `0038e80b11840b7c1973a685f8caa4643f66bf3d1424fd1aa2bb692c2e605941` |
| `itrader/rl_lib/tasks/families.py` | `93c29c93da98863ba086e69f084bb6c9fa7d7525ffebf4d0f48fe80a2a5aea49` |
| `itrader/rl_lib/tasks/samplers.py` | `9f40d2208954574ab090d386a71ae5afdbafcf2a8785e6c12246798818c14f1f` |
| `itrader/rl_lib/tasks/spec.py` | `8839e0c7c7f66bdb105f92f9d85ecd6628ffa0cd75b8722ea2d7274e5b493f87` |
| `itrader/models/solver_mlp_film.py` | `8473b004c3197ffccbddac8f587f3ba7c54b71dfeb32ae85cd20aee48903d145` |
| `itrader/models/solver_window_lstm_film.py` | `0f808c1db2908dc8a23a289ab23cd7353f53ee6f676c93ef4aa3daa99546b5e2` |
| `itrader/models/registry.py` | `4bfc0e84088f85acc47d20762e1766a019c2ef45b35f0843a88715438129ee3a` |
| `itrader/models/observation_contract.py` | `e3deb74a2b1c82513383797e871d10784a61b398b5608af0649963d193a4d741` |
| `itrader/rl_lib/train/train_solo_V2.py` | `58a26a1308b38bff0e3dd503be7e622d54e59895db9d37a5a7bdb5160ad8beb0` |
| `itrader/rl_lib/train/ppo_diagnostics.py` | `b96d433ada705e89dfecd167a1d3791e91ac1c17441c66c3e15c0ca1e27296ac` |
| `itrader/scripts/experiments/configs/phase_a_r0_plus_r4_v1_1_dec2025_4x3_40_20260925.json` | `4c9b37f2ef65e53e25f5566be2126231676e133e8fcfbd3281609d632dc11d7e` |
| `itrader/tests/test_economic_diagnostics.py` | `0717f2b33bf771f4cc5ef0838696e25e1cc6e3a0570600417f3f0050f26ecaf6` |
| `itrader/tests/test_env_core.py` | `219e0cfddcc2d480b16fca80fba2e0b426891436dbc69e67c40a931593987703` |
| `itrader/tests/test_data_partition_sessions.py` | `698c8611c73446d68def4a528d5e596784ec950fb5a054747acc2e62d52364b6` |
| `itrader/tests/test_task_families.py` | `d05f3bfb26f176541e4e4a2ec13cfe15dee0b7c3a3fad84046157381801391f9` |
| `itrader/tests/test_task_contracts.py` | `9365f52d652e89c2721ea5556d61da8420bffa786cfacb5fe68ee88ed8a2a62a` |
| `itrader/tests/test_semantic_observation_contract.py` | `7027e85031998370ec1adb583b0bc6a0784b0645000e5c6c03f670dbfa9842ea` |
| `itrader/tests/test_solver_selection.py` | `554991f9270e03bb1198e6a2e8c95f3ff6f80a909d70aa68fc09cbbceb4cd7c4` |

No `git` status/diff was run, so branch/index/HEAD are not certified by this DEV audit. Source identities describe the bytes read; external concurrent changes after this capture are not ruled out. No tests were executed. Review is static and limited to the listed material; the claim pool is not exhaustive.
