# R1 development readiness

Reviewed: 2026-10-04. Planning contract; the diagnostics and R1 work below are
not implemented by this review pass. Base: `c13af18`, dirty `agent/Watson-dev`.

## Evidence and decision

G0 is closed without economic promotion. All five member-00002 checkpoints
lost money and had negative Sharpe on the fresh November panel. Paired episode
and state artifacts match exactly; historical receipts lack producer identities
and therefore cannot independently establish separate process execution.
Preserve the raw evidence and report that limitation explicitly.

R0 + R4-v1 remains the architecture-qualified feed-forward baseline. G0 supports
moving to a representation experiment; it does not prove that an MLP cannot
learn an economic edge or that an LSTM will. Keep the final test sealed.

## Work order and acceptance

The fresh review reproduced three additional pre-training blockers: replay parity
currently compares only the many-to-one selection score, and the R4 attribution
analyzer records differing provenance counts without making them disqualifying;
PBT donor recovery recomputes a changed checkpoint hash without checking it
against the scored/indexed digest. Add full trace/metric parity, fail-closed
matched provenance and indexed-digest verification as part of the
gates below. The Sol reviewer also reproduced a string `"false"` being accepted
as replay success and control baselines losing their identities before promotion
comparison. Require strict boolean receipts and matched control provenance.
These are newly identified code gaps, not changes to G0's negative
economic result. They are recorded for implementation, not fixed by this pass.

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

`run_checkpoint_replay.py` and `run_checkpoint_state_diagnostics.py` currently
write separate reports. Neither stamps execution identity. The shared
`evaluation/provenance.py` already owns checkpoint, dataset and source hashing;
extend that boundary rather than duplicate receipt construction.

Proposed versioned receipt fields:

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
FiLM/dropout and 256-wide readout remain. The actor/critic sharing decision is
explicitly still pending the human answer in GRILL; budget is also pending.

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
| R1 total, assuming shared encoder/FiLM | 178,324 |
| R0 + R4-v1 total | 179,316 |
| R1 + R4-v1 total, assuming shared encoder/FiLM | 178,372 |

R1 is 944 parameters (0.527%) smaller than R0 under the shared-path proposal.
The unchanged width-8 R4 residual adds 48 parameters. A disposable forward
check produced `[2,60,156]` LSTM outputs and `[2,256]` final readouts. This is
capacity/shape verification, not R1 implementation or a learning result.

Recommendation to GRILL: share the market encoder, projection and task FiLM
between actor and critic, with separate pi/vf heads, matching R0's ownership.
Both losses then train the same representation; a separate-encoder study would
change sharing and capacity together. Shared weights do not imply equal actor
and critic gradient contributions. Keep value-loss weighting and task-dropout
semantics fixed, and report the respective gradients/critic fit before adding
a new sharing ablation. Do not treat this recommendation as accepted yet.

`_model_config` currently selects `solver_mlp_film` unconditionally. Registering
the old `solver_lstm_v2` model does not provide R1. The next change needs a
validated model selector in the ablation schema/runner, training builder,
resolved manifest, checkpoint loader and diagnostic paths.

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

Remaining GRILL freeze: actor/critic sharing, exact selector/config identity,
panel/seeds and step/wall-time/resource budget. Q15 fixes hidden size/layer
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
reopen Q15 or settle pending actor/critic sharing and budget.

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
   Sharing remains pending, so these tests must reflect the final GRILL answer.
3. **Stable selector across every consumer.** Carry an explicit versioned R1
   identity through loadouts, training, resolved manifests, checkpoint restore,
   diagnostics and model-matched untrained controls. `UntrainedFiLMPolicy`
   directly constructs `SolverMLPFiLM` today; replacing training selection alone
   would leave an MLP labelled as an untrained R1 control. Preserve the legacy
   `solver_lstm_v2` registration for old restores; never alias it to R1. Reject
   unknown selectors and cross-architecture restore requests. Check round-trip
   logits, value predictions and actions, not just successful checkpoint load.
4. **Protect architecture after overrides.** `_apply_best_params` currently
   merges arbitrary `custom_model_config` keys. Restrict architecture overrides
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
GRILL Q17 accepts a compressed cost curriculum for the screen. Its exact
sampled-step boundaries remain part of the pending budget declaration; the
coordination message supplies no numerical boundaries. Keep a separately named
compressed profile. Do not modify the historical 48k/96k/144k profile or scale
the entropy schedule along with cost.
Keep dataset-v2 RTH, Family 00, random/generated proposer, stream_worker,
Ray 2.58 local-worker topology, task information and environment-step exposure
matched. Freeze the member-00002 lower-LR recipe (approximately 7.1517e-6),
including entropy schedule and PPO/GAE settings, rather than launching a new
hyperparameter search. Historical checkpoints are references, not substitutes
for freshly matched R0 controls.

Budget proposal, awaiting the run declaration: smoke every cell first. The
dashboard's two-seed 12-update screening convention becomes four SRU per
curriculum, eight SRU / 921,600 nominal sampled steps for both curricula.
A full three-seed replication is a later declaration (24 cells across both
curricula). A short screen cannot traverse the original 48k/96k/144k cost
boundaries fully. Use the accepted compressed profile after declaring its exact
sampled-step boundaries; never imply a complete original-schedule comparison
from 12 updates. As a budget proposal only, equal quarters of a 57,600-step
target would use 14,400/28,800/43,200 steps. Confirm the target first and freeze
integer boundaries in the loadout, not a runtime rescaling formula. Record
actual cost used for each completed update and apply crossings to subsequent
sampling. The original entropy schedule remains on its original clock unless
separately approved. Run sequentially within memory
headroom. Declare a common step target and
staging boundaries before launch. Use fresh chronological validation for any
new selection; January and November are now inspected development panels.
Log within-seed encoder, solver-state and interaction deltas with anchor-aware
uncertainty. Crossed horizons at one anchor are not independent samples.

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
