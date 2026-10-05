# R1 development readiness

Updated: 2026-10-05. Compatibility/evidence prerequisites passed scoped independent
Sol/Astra review; R1 itself remains unimplemented. Mainline evidence base:
`d3eb4c5`. Current active API architecture is superseded by the full native
cutover underway on the isolated `feat(R)/Ray_upgrade` worktree.

The earlier instructions below about a default ModelV2/TorchPolicy stack are
historical compatibility context for the corrected Ray 2.58 runs, not the active
trainer contract. New source uses a PyTorch-only solver core and native
RLModule/Learner/EnvRunner APIs. Historical adapters/configs are isolated under
`itrader/compat/rllib_legacy/`; Ray 2.9 remains a distinct runtime profile.

## Current implementation boundary

- Implemented and exercised on synthetic data: strict semantic observations,
  structural model selection, fixed-recipe override protection, R0 stateless
  lifecycle, same-forward PPO clipping diagnostics, original indexed-checkpoint
  digest checks, replay trace/receipt evidence and fail-closed comparison gates.
- The real builder → PPO → callback → checkpoint → frozen replay probe covers
  both R0 and R0 + R4-v1. This is mechanical validation, not performance evidence.
- Integration exposed two additional Ray-2.58 compatibility defects: obsolete
  optimizer config names left active minibatch/epoch settings at defaults;
  donor restore overwrote policy settings and optimizer LR while algorithm config
  and reported LR retained target values. Runtime key translation is repaired;
  restore reconciliation passes local/remote-policy integration probes. Historical mutation logs alone do not
  establish that requested settings reached training.
- **No R1 encoder in this pass.** Fresh Sol/Astra review approved the fixes;
  406 tests pass after the explicit fixed-run entropy-schedule increment. The user subsequently resumed fresh R0/R4 controls with
  GPT-6 Luna/xhigh supervision. The fixed-recipe, PBT-off declaration and
  resource preflight are tracked in the
  [control audit](../artifacts/research/corrected-compatibility-r0-r4-controls-2026-10-05.md).
  No new performance result is available yet. See also the
  [implementation audit](../artifacts/research/r1-foundation-hardening-2026-10-05.md).
- The source-grounded **legacy RLlib API stack** means ModelV2/TorchPolicy and
  is retained only for explicit parity against historical runs. The native
  `feat(R)/Ray_upgrade` worktree is now the active engineering gate; this does
  not qualify the four representation cells or change their economic evidence.
  Do not use legacy recipes through the native trainer.

## Evidence and decision

G0 is closed without economic promotion. All five member-00002 checkpoints
lost money and had negative Sharpe on the fresh November panel. Paired episode
and state artifacts match exactly; historical receipts lack producer identities
and therefore cannot independently establish separate process execution.
Preserve the raw evidence and report that limitation explicitly.

R0 + R4-v1 remains the architecture-qualified feed-forward baseline. G0 supports
moving to a representation experiment; it does not prove that an MLP cannot
learn an economic edge or that an LSTM will. Keep the final test sealed.

## Historical compatibility repair work order (superseded for active API)

The initial review reproduced three additional pre-training blockers: replay parity
compared only the many-to-one selection score, and the R4 attribution
analyzer records differing provenance counts without making them disqualifying;
PBT donor recovery recomputes a changed checkpoint hash without checking it
against the scored/indexed digest. Add full trace/metric parity, fail-closed
matched provenance and indexed-digest verification as part of the
gates below. The Sol reviewer also reproduced a string `"false"` being accepted
as replay success and control baselines losing their identities before promotion
comparison. Require strict boolean receipts and matched control provenance.
These findings motivated the current repair pass. They do not change G0's
negative economic result; the stronger gates now have scoped independent approval.

| Order | Owning seam | Required outcome before advancing |
|---|---|---|
| 0 | Independent review under `REVIEW.md` | Resolve or explicitly track blockers; preserve dirty-tree boundary |
| 1 | Legacy Torch PPO policy diagnostics | Correct clipping statistic with unchanged losses, gradients, RNG and one-update weights |
| 2 | Shared replay provenance | Verified checkpoint/config/data/panel identities and separately attributable process receipts |
| 3 | Model selection and observation contract | Explicit R0/R1 selector through loadout, training, manifests and replay; no fallback for unknown names |
| 4 | R1 window encoder and R4-v1 cross | Chronological market-only LSTM, reset each decision; matched task/readout and state residual semantics |
| 5 | Bounded smoke | Initialization, finite PPO/GAE update, save/restore, diagnostic metrics and two-process replay checks |
| 6 | Fixed-recipe representation study | Matched cells and seeds; PBT disabled; attribution before further search |

## 1. Capture clip fraction at its producer

Source inspection of installed Ray 2.58 `PPOTorchPolicy.loss` and `stats_fn`
confirms that the legacy policy computes the PPO likelihood ratio but exports
no clipping fraction. `train_solo_V2._extract_report` already looks for
`clip_fraction` / `clip_frac`, so changing the plot or adding another alias
cannot recover it. The legacy API stack remains explicitly selected.

For the actions actually sampled in the rollout, define:

```text
r = exp(current_action_logp - rollout_action_logp)
outside = (r < 1 - clip_param) OR (r > 1 + clip_param)
clip_fraction = sum(outside AND valid) / sum(valid)
```

This measures ratios outside the clipping interval. It differs from the
advantage-sign-dependent fraction where clipping determines the chosen
surrogate term. If the latter is added, name it separately.

Implementation contract:

- Own the instrumentation in a small version-pinned legacy PPO policy adapter,
  not in ITraderEnv, the reward engine, evaluation callbacks or the dashboard.
- Capture detached counts from the same forward/distribution used for loss.
  Do not call `super().loss()` and then perform a second model forward to infer
  the ratio: the current model uses task dropout during learning, so that can
  measure another policy realization and consume RNG.
- Reuse the loss validity mask. R1 has no cross-step rollout state and does not
  introduce padded RLlib sequences; later R2 must exclude sequence padding.
- Reduce clipped and valid counts across towers/minibatches/SGD passes before
  division. Label the interval as all optimizer sample-visits in one train
  report, not unique environment transitions or a post-update evaluation.
  Repeated SGD passes intentionally count repeated visits. Do not average
  unequal minibatch fractions without weighting. Exclude initialization/dummy
  loss calls and reset the report accumulator after consumption.
- Preserve missing versus zero: old reports remain unavailable, empty valid
  sets are unavailable, non-finite ratios are diagnostic failures. Keep metric
  definition/version, valid count and reduction scope in the evidence contract.
- Wire the chosen policy class through training and restore registration, then
  through learner stats -> report rows -> summaries -> browser. Do not switch
  to RLModule/EnvRunner as part of this instrumentation.

Required probes: ratios below/at/inside/at/above the interval; positive and
negative advantages; masked padding; unequal minibatches; repeated epochs;
zero-valid rows; non-finite values; real legacy learner propagation. Compare
instrumented versus original policy with identical weights, batch and RNG:
loss, gradients, sampled actions/RNG state and one-update weights must agree.
Include task dropout in that parity test.

Reference: the PPO implementers' [implementation-details account](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
describes clip fraction and the importance of implementation-level controls;
the installed pinned Ray source is the authority for the adapter.

## 2. Stamp and validate replay provenance

`run_checkpoint_replay.py` and `run_checkpoint_state_diagnostics.py` now share
invocation/completion receipt construction in `evaluation/replay_evidence.py`.
`evaluation/provenance.py` owns checkpoint, dataset and source hashing. Historical
reports without producer identities remain unknown; they are not rewritten.

Receipt acceptance contract (implementation coverage is tracked in the audit):

- fresh invocation UUID and process-instance identity established at process
  entry, with PID, process creation time, host identifier and UTC start/end;
- operation (episode replay or state diagnostic), checkpoint tree SHA-256,
  selected progress/generation, dataset hash, source/evaluation panel hashes,
  resolved effective config hash and dirty source-tree hash;
- runtime versions, device/determinism settings, observation schema and seeds;
- artifact filenames, byte hashes, row counts, completion status and metric
  contract version.

An invocation UUID alone cannot distinguish two calls inside the same process.
The harness should launch fresh subprocesses, record their process identities
and exit results, and bind each invocation receipt to the launch record. An
episode replay and a state diagnostic are separate invocations even if grouped
under a single human-readable replication directory. These records support
auditable provenance; they are not cryptographic proof against fabrication.

Verify the indexed checkpoint digest against the actual tree before restore;
verify dataset and effective panel/config identities before evaluation. Commit
the completion receipt atomically only after artifact hashes are known. A
failed/interrupted run must not look complete, and output paths must not
silently overwrite another invocation. Compare deterministic payload hashes,
not the whole receipts, because execution IDs and times should differ.

Source-panel parity must compare episode identities, action/reward traces and
the underlying metric vector. Matching the scalar selection score alone is
insufficient: the current non-flat selection score maps distinct near-flat
policies to zero. Use separate names for score parity, payload parity and
verified process replication. Where historical reference traces are absent,
report the stronger checks as unavailable rather than inferring exact parity.

The comparison gate must require distinct verified process instances, identical
effective evaluation contracts/checkpoint identity and matching payloads.
Copying a directory, changing a panel, reusing an invocation ID or modifying
the checkpoint must fail. Legacy receipts remain readable as unknown-process
provenance; do not backfill invented IDs into G0 evidence.

## 3. R1 seams and decisions to freeze

GRILL coordination received 2026-10-04 reports Q15 acceptance of one
unidirectional `LSTM(5,156)` followed by `Linear(156,256) + ReLU`, using the
final chronological output and zero h/c at each decision. The existing task
FiLM/dropout and 256-wide readout remain. GRILL coordination on 2026-10-05
confirms explicit human acceptance of Q18: one shared market encoder,
projection and task-FiLM path, with separate pi/vf heads. Both losses update
that shared representation. Q19 selection and Q20 screening budget were also
accepted through GRILL on 2026-10-05; Q21 two-tier evaluation was subsequently
approved the same day. Approval is a protocol decision, not evidence that its
fresh panel or runner gates are implemented.

Source-grounded parameter check, using actual PyTorch modules and the current
R0 class (60 x 5 market input, task dimension 32, task projection 64,
MLP widths 256/256, three action logits, default LSTM biases):

| Component / cell | Trainable parameters |
|---|---:|
| R0 market MLP | 142,848 |
| R1 window LSTM | 101,712 |
| R1 156-to-256 projection | 40,192 |
| Common task FiLM | 35,392 |
| Separate pi/vf heads combined | 1,028 |
| R0 total | 179,268 |
| R1 total, accepted shared encoder/FiLM design | 178,324 |
| R0 + R4-v1 total | 179,316 |
| R1 + R4-v1 total, accepted shared encoder/FiLM design | 178,372 |

R1 is 944 parameters (0.527%) smaller than R0 under the accepted shared-path design.
The unchanged width-8 R4 residual adds 48 parameters. A disposable forward
check produced `[2,60,156]` LSTM outputs and `[2,256]` final readouts. This is
capacity/shape verification, not R1 implementation or a learning result.

Accepted Q18: share the market encoder, projection and task FiLM
between actor and critic, with separate pi/vf heads, matching R0's ownership.
Both losses then train the same representation; a separate-encoder study would
change sharing and capacity together. Shared weights do not imply equal actor
and critic gradient contributions. Keep value-loss weighting and task-dropout
semantics fixed, and report the respective gradients/critic fit before adding
a new sharing ablation. This is an accepted design, not implemented R1 code.

The structural selector now reaches the ablation schema/runner, training builder,
resolved manifests, untrained controls and checkpoint/diagnostic paths.
`solver_window_lstm_film` fails explicitly as unimplemented; it is not registered
to a placeholder or the historical `solver_lstm_v2` class.

Accepted Q19: one structural `solver_model` selector with names
`solver_mlp_film` and `solver_window_lstm_film`, independent of R/S experiment
labels and solver-state/reward/curriculum factors. The first name already
matches the current registered R0 key; the second has no current registration
or implementation collision. Bind the selected model and versioned observation,
preprocessing and state contracts to the resolved manifest and restore path.
Prefer one validated selection interface shared by schema/runner/training,
model-matched baselines and replay; a plugin framework is unnecessary here.

Compatibility concerns for Q19: define how a missing selector in historical
configs resolves from their recorded `model.custom_model` without changing old
weights or observations; reject conflicts between the outer selector and that
nested key after override merges. Keep `solver_lstm_v2` and other historical
registrations available only for faithful restores; never reinterpret them as
R1. New R1 loadouts require explicit selection. An intentional transfer between
architectures would require a separate future migration contract; it is not
ordinary restore. These are accepted requirements, not permission to reinterpret old checkpoints.

R1 input is the same chronological `60 x 5` OHLCV window plus the same task
embedding. Split `[market | task | optional state]` explicitly before reshaping.
Only market bars enter the window LSTM. Initialize its hidden/cell state to
zero per decision, select the final chronological output, then apply the
existing task FiLM and policy/value readouts. Return no carried RLlib state.
The episode horizon and 60-bar input window are distinct; `max_seq_len` does
not supply within-window chronology for this model.

For `R1 + R4-v1`, ITraderEnv still owns `[position, remaining_horizon_fraction]`.
Preserve the existing additive actor/value residual connector with `zero_v1`
heads; do not add nonlinear state-market fusion (R4-v2) in this experiment.
Require initial-policy/value parity within each base/state-augmented pair.
R0 and R1 need matched inputs and budgets, not identical initial weights across
different architectures.

Remaining execution freeze: construct/hash the Q21 fresh panel, pin concrete
paired seeds and resource/wall-time declaration for the complete target screen.
Q21's evaluation protocol is accepted; Q20 approves the sampled-step budget. Q15 fixes hidden size/layer
count and readout shape; preserve the FiLM formula and task-dropout treatment.
Measure parameter count, wall time and memory separately
from matched sampled environment steps. Do not describe equal steps as equal
compute, or a parameter tolerance as satisfied without counts.

Tests must include tagged chronological bars, malformed layout, batch-size and
batch-order invariance in evaluation mode, repeated identical observations
despite different prior calls, actor/critic gradients, state-mask parity,
checkpoint class/config restoration and replay. Truncation/termination GAE
behavior remains owned by environment/RLlib; do not redefine it for R1.

### Contract-freeze acceptance checklist (GRILL audit, 2026-10-04)

These six planning concerns are acceptance criteria for implementation. They
are not demonstrated R1 runtime defects: R1 remains unimplemented. They do not
reopen Q15. Q18 sharing, Q19 selectors and Q20 budget were subsequently accepted
on 2026-10-05; Q21's two-tier evaluation protocol is also accepted, with panel
construction and enforcement still pending.

1. **Semantic observation validation.** Pin the ordered
   `[OPEN, HIGH, LOW, CLOSE, VOLUME]` features, oldest-to-newest 60-bar window
   ending at `idx-1`, task dimension 32 and optional state dimension 2. Include
   task-encoder and preprocessing identities. The current environment's
   `_resolve_feature_columns` defaults to dataframe order, so width alone does
   not establish semantics. Validate at the env/model/config boundary; reject
   incompatible feature order, timing or schema. In particular, a 334-wide
   observation with state disabled must not become a 302-wide market slice by
   subtraction. Use tagged bars/features and conflicting-schema tests.
2. **Exact accepted Q15 computation.** Preserve
   `final LSTM output -> Linear(156,256,bias=True) -> ReLU ->
   tanh(gamma) * hidden + beta -> separate pi/vf heads`.
   Task conditioning occurs at the existing FiLM readout. No pooling, task
   injection per market bar, identity-centered FiLM or new recurrent dropout
   belongs to this cell. Test the numerical formula with controlled weights;
   actor-only and value-only backward passes must reach the intended paths.
   Under accepted Q18, verify that each loss reaches the shared encoder/FiLM
   while the policy-only loss does not train the value head and vice versa.
3. **Stable selector across every consumer.** Carry an explicit versioned R1
   identity through loadouts, training, resolved manifests, checkpoint restore,
   diagnostics and model-matched untrained controls. `UntrainedFiLMPolicy`
   previously constructed `SolverMLPFiLM` directly; the shared registry now owns
   selection for controls too. Preserve the legacy
   `solver_lstm_v2` registration for old restores; never alias it to R1. Reject
   unknown selectors and cross-architecture restore requests. Check round-trip
   logits, value predictions and actions, not just successful checkpoint load.
4. **Protect architecture after overrides.** `_apply_best_params` historically
   merged arbitrary `custom_model_config` keys. Restrict architecture overrides
   for fixed-recipe cells or validate the fully merged config against the
   declared contract before construction. The manifest, checkpoint and restored
   model must agree. Test stale best-parameter files that change state width,
   task/window dimensions, hidden size, preprocessing or initialization.
5. **Explicit stateless RLlib lifecycle.** Return empty RLlib state; allocate
   zero h/c for each decision, with no cross-call hidden caches. Reject
   unexpected carried state instead of ignoring it. Verify evaluation-mode
   batch A/B permutation, batch-size invariance and A -> B -> A repeatability,
   plus finite actor/critic backward behavior. A within-window LSTM still
   differentiates through the 60 visible bars; it does not carry episode state.
6. **Pin preprocessing and initialization in every fixed-recipe cell.** Explicitly
   set `market_relative_v1`, `NoFilter`, and the `zero_v1` initialization profile
   (where R4 state is enabled), even with PBT disabled. Current generic defaults
   are `raw`, `MeanStdFilter`, and `random_v1`; the PBT-specific validation does
   not protect non-PBT arms. Preserve common parameter initialization within
   each base/state pair. Raw zero-state neutrality must be tested after the
   actual preprocessing path; it relies on the selected `NoFilter` contract.
   At zero_v1 initialization the state encoder correctly has zero first-pass
   gradients through zero output heads; the residual heads may receive nonzero
   gradients. Test this expected pattern instead of requiring every tensor to
   have a nonzero gradient on the first backward pass.

Keep these criteria in the eventual implementation checklist alongside the five
existing evidence-gate fixes. No performance launch can bypass those fixes.

## 4. Complete the representation comparison

The previous three-cell proposal omitted the current forward FF baseline.
The 2026-10-04 GRILL coordination confirms human acceptance of this complete
2 x 2 comparison under each declared cost curriculum:

| Market encoder | No explicit solver state | Position + remaining horizon |
|---|---|---|
| MLP | R0 | R0 + R4-v1 |
| Reset-per-decision window LSTM | R1 | R1 + R4-v1 |

The two named curriculum factors are exact 1 bp and causal
`0 -> 0.25 -> 0.5 -> 1 bp`; held-out evaluation is always exact 1 bp.
GRILL Q20 accepts the compressed screen with exact sampled-step boundaries
14,400 / 28,800 / 43,200 within 57,600 nominal steps per cell. Keep a separately named
compressed profile. Do not modify the historical 48k/96k/144k profile or scale
the entropy schedule along with cost.
Keep dataset-v2 RTH, Family 00, random/generated proposer, stream_worker,
Ray 2.58 local-worker topology, task information and environment-step exposure
matched. Freeze the member-00002 lower-LR recipe (approximately 7.1517e-6),
including entropy schedule and PPO/GAE settings, rather than launching a new
hyperparameter search. Historical checkpoints are references, not substitutes
for freshly matched R0 controls.

Accepted Q20 budget: smoke every cell first, then four cells x two paired seeds
x two curricula = 16 runs, 921,600 nominal sampled steps total. A full three-seed
replication is a later declaration. A short screen does not traverse the
historical 48k/96k/144k schedule: label the compressed curriculum explicitly.
Freeze the integer boundaries in the loadout, not a runtime rescaling formula. Record
actual cost used for each completed update and apply crossings to subsequent
sampling. The original entropy schedule remains on its original clock unless
separately approved. Run sequentially within memory
headroom. Declare a common step target and
staging boundaries before launch. Use fresh chronological validation for any
new selection; January and November are now inspected development panels.
Log within-seed encoder, solver-state and interaction deltas with anchor-aware
uncertainty. Crossed horizons at one anchor are not independent samples.

### Accepted Q21 evaluation and reporting contract

Human approval verified in GRILL-IMPLEMENT on 2026-10-05:

| Tier | Contract |
|---|---|
| Progress monitoring | Same inspected 32-task panel at initialization and every update, evaluated deterministically at 1 bp; development curves only |
| Terminal screening | Fresh validation-only 32 eligible symbol/session anchors crossed with frozen horizons 96/128/160/200: 128 tasks, identical hashed panel for every target-screen cell and control |
| Panel declaration | Select and hash before training using deterministic eligibility/calendar/symbol rules, without inspecting outcomes; verify chronology, session boundaries and non-overlap with inspected development anchors |
| Checkpoint selection | Predetermined terminal checkpoint, never the best transient; freeze identity before opening terminal results |
| Opening rule | Evaluate the fresh terminal panel only after all 16 target-screen runs finish; do not adapt later cells to early terminal results |
| Final test | Remains sealed and unused |

The two-tier runner/panel gates are **requirements, not implemented claims**.
Initialization monitoring must use the actual seeded model before its first
update, including arm-matched untrained controls. Failed/incomplete cells stop
the all-16 completion gate; do not silently substitute retained-best checkpoints
or treat a partial screen as complete.

First-class report sections must distinguish:

- Per-episode and per-run gross/net return, Sharpe with its exact unit, turnover,
  drawdown, exposure and short/flat/long fractions.
- Matched flat, long-only, short-only, random and architecture/state-matched
  seeded untrained controls; show the strongest matched directional control
  without changing checkpoint selection or inventing a new promotion threshold.
- Within-seed encoder, solver-state and interaction deltas.
- Policy/value losses, clip fraction and its denominator, approximate KL,
  entropy, explained variance and other available critic diagnostics.
- Actual causal-bp-stage exposure and cost applied, sampled steps, throughput,
  memory, checkpoint lineage and source/data/panel/producer identities.

Retain raw episode/action/reward evidence. Crossed horizons share anchors and
are not independent samples; use anchor/calendar-block-aware summaries and do
not infer statistical significance from two training seeds. Evaluation steps,
cost scenarios and evaluation wall time are separate from the approved 921,600
nominal training steps. This screen is attribution/feasibility, not automatic
economic promotion.

The separately declared corrected-compatibility eight-cell R0/R4 preflight
remains a development-panel bridge, not a retroactive fulfillment of this
16-cell target-screen protocol. Reusing any earlier cell requires explicit
contract/source/initialization/panel equivalence; otherwise rerun it. Historical
January/November diagnostics are not relabelled fresh terminal evidence.

Accepted platform ordering (2026-10-05): finish compatibility repairs and fresh
R0 controls, implement/test R1 on that same stack, then integrate and qualify
RLModule/Learner/EnvRunner with the complete four-cell bridge. Native development
may proceed now in the separate migration worktree with bounded mechanical
tests, without changing this screen's API stack or running concurrent full-data
jobs. Commits/pushes there do not authorize commits in the main dirty tree. Preserve
historical checkpoints and manifests; new stack/source/settings are explicit
factors, never relabelled historical parity.

Require expected dataset, effective task panel, source, configuration, runtime
and seed identities for every cell before computing a promotable attribution.
Observed uniqueness counts alone are not gates. Permit only the declared
representation/curriculum differences; missing or differing protected identities
must make the comparison ineligible even when every economic threshold passes.

R1 may advance to the R2 lifecycle study after correctness and measurement
gates even if its economics remain negative. Economic promotion remains a
separate held-out decision. PBT resumes only after this attribution study.

## Review and navigation

Confirmed open findings are preserved in the
[independent review record](../artifacts/research/r1-readiness-independent-review-2026-10-03.md).

Reviewer default: GPT-6.1 Sol at MAX in [REVIEW.md](../../REVIEW.md).
The [stack iteration skill](../../.codex/skills/itrader-stack-iteration-loop/SKILL.md)
owns invocation of the fresh-reviewer ladder; high-risk review uses a second
model. Review the actual dirty diff, including untracked source, against the
declared base. No staging or commit is required.

See [next actions](next-actions.md), the [masterplan](../plans/itrader_masterplan.md)
and the [experiment browser](../research/itrader_experiment_browser.html).
