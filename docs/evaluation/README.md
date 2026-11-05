# Bounded Evaluation

Markdown is authoritative; the [HTML companion](README.html) is a review surface.
Scope reconciled on 2026-10-13. This section documents the existing evaluator,
tests and runner, not a new framework, simulator, interval method or run permit.

Use [AGENTS](../../AGENTS.md), [coordination](../coordination/README.md),
[architecture](../architecture.md), [detailed architecture](../architecture/itrader_system_architecture.md),
[masterplan](../plans/itrader_masterplan.md), [current state](../project-state/current-state.md)
and [decisions](../project-state/decisions.md) for authority and live sequencing.
[REVIEW](../../REVIEW.md) governs independent review. GRILL owns index/active-status
integration; this section does not change boards, leases or existing research wings.

## Current Evidence, Not Promotion

CUDA37 now records all four R0/R1 x R4-state/off synthetic GPU cases passing
PPO/update/save/restore/frozen replay, zero failures/errors/skips and cuda:0
after construction/update/restore. [Execution receipt](../coordination/training-testing.md#training-cuda37-release1---complete)
binds the tested runtime and commands; total supervised work119.078s, including
transport, with clean owned cleanup. This qualifies that bounded GPU lifecycle,
not real-market evaluation, uncertainty, general throughput or economic edge.
The CPU control remains unchanged. Earlier CUDA28/29 facts below describe their
historical scopes; CUDA37 supersedes their missing GPU PPO result only.

The [R1-LIFECYCLE22 receipt](../artifacts/research/r1-lifecycle22/receipt.json)
records all four R0/R1 x R4-v1 state/off CPU synthetic cases passing:
PPO update -> save -> restore -> deterministic frozen replay, zero
failures/errors/skips, 37.234 owned seconds, cleanup survivors zero.
The [existing public checkpoint test](../../itrader/tests/test_representation_checkpoint_integration.py),
`test_representation_survives_ppo_checkpoint_and_frozen_replay`, defines the
600-row synthetic input, 12 sampled training steps, one update/one SGD epoch,
four-step tasks and one frozen validation episode per cell. The compact receipt
is not the whole recipe; bind the source/config snapshot as well.

This certifies that bounded CPU lifecycle, not market edge, PPO throughput,
the 16-run representation screen, uncertainty qualification or CUDA.
GPU opt-in/device assertions in that same test are now exercised by CUDA37. The
[CUDA component receipt](../artifacts/research/cuda26-review-and-test-roi-2026-10-10.md)
does not qualify provisioning or the production packet.

CUDA28 completed on2026-10-11 at02:08UTC: independently reviewed root uv profiles/
lock applied, one locked sync passed, and real tiny GPU tensor forward/backward
passed on RTX5070Ti with Torch2.9.0.dev20250801+cu128/CUDA12.8/cuDNN91002.
See [the durable receipt](../artifacts/research/cuda28-known-error-repair-2026-10-11.md)
and [profile operations](../environment_strategy.md). The root CPU control remains
unchanged. The refreshed lock includes76 other shared package-version changes;
this is not blanket dependency compatibility, PPO throughput or market evidence.
TRAINING-CUDA29 reached zero cells: an added wrapper outside its reviewed
packet failed PowerShell parsing before PPO. No owned workload survives.
This is historical failed-launch evidence, not a failed GPU experiment;
CUDA37 subsequently passed the public four-cell GPU lifecycle. No automatic
retry is authorized.
Earlier low-RAM/preparation-only deferrals are historical, not a current install
blocker or a reason to skip fresh preflight. Follow the current owned boards.
CUDA is the preferred first-class forward target only after reproducible
provisioning and proper-stack public PPO/update/save/restore/frozen-replay
qualification. Retain CPU control; kernel speed is not stack qualification.
Do not lower resource gates or start jobs from intake facts.

## Existing Evaluation Interface

- [ITraderEnv](../../itrader/envs/itrader_env.py) owns reset, causal observation assembly, transitions and terminal [EpisodeEvidence](../../itrader/rl_lib/evaluation/episode_evidence.py). [RewardEngine](../../itrader/rl_lib/tasks/rewards.py) owns optimization reward; economic evidence is independent of shaped reward.
- [Policy baselines](../../itrader/rl_lib/evaluation/policy_baselines.py) provide frozen deterministic policy replay and flat/long/short/random/model-matched untrained controls. [Replay provenance](../../itrader/rl_lib/evaluation/provenance.py) and [replay evidence](../../itrader/rl_lib/evaluation/replay_evidence.py) bind identities; supplied labels alone are not producer proof.
- [Panel qualification](../../itrader/rl_lib/evaluation/panel_qualification.py) checks metadata/exposure evidence; [screen completion](../../itrader/rl_lib/evaluation/screen_completion.py) checks all declared cells and predetermined terminals. [Terminal runner](../../itrader/rl_lib/evaluation/run_terminal_screen.py) is currently read-only preflight, explicitly returning `evaluation_authorized: false`, not a completed terminal executor.
- [Terminal panel report](../../itrader/rl_lib/evaluation/terminal_panel_report.py) owns strict Q24/Q25 reduction. [Economic diagnostics](../../itrader/rl_lib/evaluation/economic_diagnostics.py) add trace accounting and attribution without reward/promotion authority. [Disposition](../../itrader/rl_lib/evaluation/evaluation_disposition.py) validates supplied status objects; the [Experiment Browser](../research/itrader_experiment_browser.html) is an evidence navigator, not a new ledger.

Keep this small public interface and its independent oracles. Reuse the current
modules; do not add a parallel evaluator, loader, reward, registry or private-method
test tree. TaskSpec stays the canonical problem contract. Capability calibration
and deterministic DecisionGate/RiskEngine/Executor authority remain separate;
WAIT/NO_ACTION and flat are legitimate outcomes.

## Four Separate Decisions

- **Evidence validity:** are data, task/window, checkpoint/config/source/runtime, producer and control identities complete and bound, with correct causal/accounting semantics? Preserve invalid, incomplete or unbound attempts as raw/quarantined evidence with reasons, not certified comparisons.
- **Metric availability:** does this metric have the required finite observations and dispersion? Valid positive, zero and negative net remain descriptive. Unavailable Sharpe is not measured zero; availability must have evidence and a reason when unavailable/unknown.
- **Uncertainty qualification:** has a predeclared method passed support, independent synthetic calibration and confirmation for this estimand? Unresolved inference is not invalid arithmetic. A qualified interval crossing zero is inconclusive; support/method failure must not delete an otherwise valid descriptive estimate.
- **Promotion eligibility:** eligible-for-review is not promoted. Held-out economics, matched controls, chronology, robustness, resource/runtime qualification and human review remain separate gates. Relative paired uplift does not establish positive absolute profitability.

`build_evaluation_disposition` is **only a shape/vocabulary validator for
caller-supplied statuses**. It neither calculates cross-axis validity, derives
inference, inspects metric values nor grants promotion. Callers must justify
each axis; a well-shaped object alone is not a qualification certificate.

Current implementation limits matter: the terminal reporter requires finite
Sharpe scalars, but marks Sharpe availability unknown because count/dispersion
are absent. Legacy EpisodeEvidence returns scalar zero for fewer than two
returns or near-zero sample SD; some legacy browser records retain that zero.
Explicit availability exists in selected report/registry contracts, not every
owner. Do not claim universal unavailable-Sharpe handling or infer availability
from a legacy scalar.

## Four Named Public Scenarios

These are accepted scenario labels, not four new pytest entry points.
Pointers below identify implemented mechanics and independent expected values;
they do not certify the entire dirty tree, close pending execution-backed Spec
review, or establish statistical qualification.

### AccountingLifecycle

Public reset -> step -> terminal evidence -> attribution must reconcile entry,
hold, reversal and exactly-once liquidation, with no extra terminal bar.
The current literal test uses closes 100 -> 110 -> 121 -> 110, targets
long/long/short, 10 bp: gross `0.1 + 0.1 + 11/121`, turnover
`1 + 0 + 2 + 1 = 4`, cost `0.004`, net `gross - 0.004`.
Reward equals net x1000 only because this fixture explicitly selects that reward.

[Accounting tests](../../itrader/tests/test_economic_diagnostics.py):
`test_opt_in_pilot_reconciles_literal_hold_reverse_liquidation_path`,
`test_directional_controls_have_zero_timing_residual_on_identical_steps`
and `test_real_trace_identity_rejects_duplicated_steps_and_stale_context`.
The constant-direction two-step fixture has long gross zero/net -0.0002;
flat has zero economics. [EOF/reset test](../../itrader/tests/test_env_core.py)
`test_last_eligible_start_consumes_full_horizon_with_exact_identity`
covers horizons 1/2/4, exact bounds and long entry/liquidation cost +0.0002,
giving net -0.0002 on flat prices.

### CausalCompleteWindow

A future suffix change must not alter the earlier observation, task embedding,
pre-action context or lagged scale. Reject missing minutes, session/split
crossings, incomplete warmup and incomplete maximal outcome windows rather
than remapping starts or shortening the primary horizon.

[Causal-context test](../../itrader/tests/test_economic_diagnostics.py)
`test_future_row_perturbation_cannot_change_pre_action_context_or_lagged_scale`;
[tagged observation test](../../itrader/tests/test_semantic_observation_contract.py)
`test_tagged_parquet_preprocessing_preserves_chronology_and_stateless_calls`;
[continuity tests](../../itrader/tests/test_data_partition_sessions.py)
`test_partition_splits_on_missing_minute_inside_session`;
[qualification tests](../../itrader/tests/test_panel_qualification.py)
`test_full_metadata_window_rejects_exposure_of_lookback_bar` and
`test_calendar_close_must_contain_the_entire_final_minute`.
These exercise distinct mechanics; a complete execution-backed scenario
receipt, including the explicit 210-minute early-close/maximal-window oracle,
is not claimed here.

### WeightedMatchedPanel

Exact task/control joins and fixed weights precede aggregation. The literal
fixture gives one two-anchor symbol net 0.012 and eleven symbols zero:
equal-symbol net 0.001 versus raw episode mean 0.00075. Row order is irrelevant;
missing/duplicate/wrong-seed/control identities reject, never renormalize.

[Panel tests](../../itrader/tests/test_terminal_panel_report.py):
`test_equal_symbol_and_duration_normalize_before_aggregation`,
`test_invalid_coverage_or_economics_is_rejected`,
`test_matched_controls_are_compared_once_over_whole_panel` and
`test_liquidation_cost_adds_no_extra_bar`.
[Environment-to-reporter integration](../../itrader/tests/test_panel_qualification.py)
`test_real_environment_and_reporter_agree_on_crossed_control_receipts`
uses synthetic complete crossed controls. The design oracle averaging fixed
pair effects +0.002 and -0.001 to +0.0005 is not a claimed executed Q26
two-pair integration or an interval.

### EvidenceDisposition

Preserve valid positive/zero/negative net when inference is unresolved.
Keep unknown/unavailable Sharpe distinct, corrupt comparisons quarantined, and
all-cell/data gates fail-closed before terminal access. Never convert a failure
to zero or a qualified-inconclusive result to promotion.

[Descriptive report test](../../itrader/tests/test_terminal_panel_report.py)
`test_disposition_keeps_descriptive_panel_results_separate_from_inference`;
[browser contract test](../../itrader/tests/test_experiment_browser.py)
`test_terminal_gate_keeps_steps_and_renders_explicit_disposition`;
[runner rejection test](../../itrader/tests/test_terminal_screen.py)
`test_partial_screen_rejects_before_panel_read_or_output_creation` and
`test_read_only_preflight_disposition_does_not_authorize_evaluation`.
Report/registry fixtures and preflight checks exist. Full receipt -> report ->
browser execution-backed Spec closure and qualified interval behavior remain
separate pending gates, not a blanket pass from these pointers.

## Accounting And Trace Identity

The existing Discrete(3) actions select target positions -1/0/+1; repeated long
means hold, not another buy. Starting flat, for arithmetic asset return r_t,
target position p_t, H actual decisions and cost c in bps:

```text
gross = sum(p_t * r_t)
turnover = sum(abs(p_t - p_(t-1))) + abs(p_H)
cost = turnover * c * 1e-4
net = gross - cost
direction = mean(p_t) * sum(r_t)
timing = gross - direction
net = direction + timing - cost
```

Direction uses the same-window matched long gross, not a hindsight best side;
timing is an accounting residual, not independently identified forecasting alpha.
Liquidation costs enter once on the last actual decision, not a fictitious bar.
Continuation net, hypothetical round-trip net and actual terminal net are
different quantities. Frozen-action repricing changes costs on the same trace;
it is not retraining or proof the policy would react identically at another cost.

Bind source/panel/checkpoint/config/runtime/producer receipts, exact task hash,
seed/symbol/session/row bounds, observation schema, causal past-window endpoint,
actions/positions, return and cost trace, and terminal flag. Trace reconciliation
checks contiguous reset-flat positions and exact ordering; the diagnostics'
`producer_provenance_verified: false` remains explicit. Arithmetic identity
does not authenticate a supplied checkpoint label.

One-bar net is well-defined; one-bar realized Sharpe is not. Flat/zero-SD series
also cannot support measured Sharpe from a zero fallback. Optional past-only
volatility scaling is not realized Sharpe and remains off/nonblocking here.
Episode Sharpe units are per square-root bar; annualization needs an explicit
periods contract. Overlapping episodes are not a stitched portfolio.
Reported current drawdown is gross additive fractional-return episode drawdown,
not compounded peak-relative or net drawdown. Gross returns +0.1/-0.1/-0.1
produce additive drawdown 0.2, not compounded peak-relative drawdown 0.19.

## Fixed Terminal Contracts

### Q23 Exposure And Maximal Support

The accepted 2020-01-01 through 2021-12-31 candidate envelope and retrospective
clearance approach are not access rights, external/manual non-exposure
attestation, a frozen panel or forward-time trading validation.
Qualification needs complete retained exposure/source proofs and reviewed
calendar metadata; unknown histories stay unknown. New bytes, dates, seeds or
fresh weights do not establish freshness.

The frozen design requires 32 distinct symbol/session anchors across 12 symbols,
at least two each plus eight seeded allocations, RTH offsets 60/120 minutes,
60 same-session past minute bars and the full 200-bar maximal future support.
Cross every anchor with horizons 96/128/160/200: 128 tasks. Eligibility and
non-overlap cover the whole `[start-60, start+200)` window, not just a shorter
selected horizon. A 210-minute early-close session cannot fit offset60 plus
200 future bars. Use full-minute calendar bounds, continuity, split and purge;
select on metadata, never future returns/regime/model outcomes.

Approved `terminal_screen_only`, separate training/terminal dataset identities
and certified eligible intervals remain prerequisites. Current qualification/
report/preflight source still requires validation and a common dataset identity;
it does not implement that external dual-dataset role end to end. Do not relabel
external terminal evidence as ordinary validation or silently alter train/test
partitions. See [R1 readiness](../project-state/r1-development-readiness.md)
and the [Q23/Q26 decision chain](../artifacts/research/q23-q26-decision-chain-2026-10-09.md).

### Q24 Weights And Q25 Denominator

The implemented reporter is **strictly** 128 tasks / 32 anchors / 12 symbols,
horizons 96/128/160/200, exact 1 bp and full horizon:
`steps == horizon == end_idx - start_idx`.
It is not a generic small-panel/variable-horizon robustness reporter.

Q24 averages four horizons within anchor, anchors within symbol, then 12 symbols
equally, with identical weights for policies and paired controls. With n_s
anchors for symbol s, each episode weight is `1/(12 * n_s * 4)`.
Retain labeled raw means and per-horizon diagnostics; missing coverage rejects.
Show flat/long/short/random/architecture-and-state-matched seeded untrained
controls. Choose the stronger constant direction once across the whole panel,
not per-anchor hindsight. Control labels require caller-owned provenance.

Q25 is secondary `net_bps_per_bar = 10000 * episode_net_return / actual_steps`.
Normalize each episode first, then apply Q24 weights. Require positive integral
steps and consistent endpoints; never substitute lookback, exposure/training
steps, a clamp or a pooled-return/pooled-step ratio. Liquidation adds cost,
not a bar. A shortened episode is censored and rejected from this fixed-panel
primary, not normalized into a complete result. Neither weighting nor this
diagnostic changes rewards, PBT scores or the primary terminal-net estimand.

### Q26 Qualification, Not A Passing Method

Primary: fixed-1-bp R1 + R4-v1 minus R0 + R4-v1 Q24 terminal net,
averaged over **two frozen checkpoint pairs**, reporting each pair and retaining
negative/zero effects. This is conditional fixed-checkpoint/calendar evidence,
not training-seed-population inference, positive absolute net or Sharpe inference.

Accepted policy is a two-sided nominal 95% interval, with central equal-tailed
2.5%/2.5% first candidate. Every required predeclared synthetic scenario must
meet simultaneous Monte Carlo coverage **LOWER >=94%** and each directional-miss
**UPPER <=3%**, including positive-uplift declarations under no uplift.
Only after confirmation may it be labeled nominal95%, tolerance-qualified
on the declared stress class; never validated exact95%/exact2.5% or a universal
market guarantee. A positive lower bound supports relative uplift only after
method qualification.

The bounded fixed-symbol-mean process claim and three roles are accepted:
supported scenarios must pass; observable structural defects must reject with
reasons; out-of-class regime breaks/long memory are diagnostics.
**BOTH exact finite-lag and bounded stable-decay families are now mandatory
supported tests.** Strong dependence inside declared bounds cannot be dropped
or relabeled diagnostic after failure. These are synthetic interval reliability
tests with imposed truth, not OHLCV shocks or known real-market effects.

Calendar-synchronized dependent multipliers are the first qualification
candidate, not a qualified scored method. [CLOCK27's delivered both-clock investigation](../artifacts/research/q23-q26-decision-chain-2026-10-09.md#127-clock27-exchange-session-versus-elapsed-day-distance)
compares full exchange-session ordinals with elapsed exchange-local civil-day
distance. **Human accepts session distance as the first inference-clock
candidate, challenged by both mandatory dependence families under BOTH DGP
clocks.** CLOCK27 is completed research; RESEARCH has verified its response
against the exact dispatch after a wrong-scope chat answer. The accepted direction
is not qualification.
Do not compress unselected open sessions or equate civil days with UTC hours/24.
Keep DGP clock, inference distance, correlation kernel and
bandwidth distinct. Same-date symbols/horizons/pairs share calendar perturbation;
changing distance cannot change Q24 weights or create observations.

Qualified outcome-free support, finite lag/persistence ranges and moments,
centering, kernel/bandwidth and interval law, MC confidence/simultaneous
allocation, failure cap, precision/budget, multiplicity and independent frozen
development/confirmation streams remain open. No calibration or market access
is authorized by these decisions. The [legacy block uncertainty](../../itrader/rl_lib/evaluation/block_uncertainty.py)
is exploratory per-symbol positional-row resampling, not a qualified Q26
calendar method or an interval for the changed Q24 estimand.

### All-Cell Opening And Promotion

Retain the 16-run screen: four representations x two cost curricula x two seeds,
PBT off, matched recipe/runtime, predetermined terminals and all-run-complete
opening gate. The 12-step lifecycle smoke is not that screen.
Checking the primary R4 pair mechanically first does not permit interim fresh
market inspection. Q23 clearance/role, Q26 qualification, final source/config/
data/review seals and an explicit resource lease precede terminal execution.
A passing read-only preflight grants none of these missing authorities.
Economic promotion and operational capability remain further decisions.

## Chronology, Fits And Isolation

Chronological train -> later validation -> final sealed test is necessary, not
a freshness certificate. Current [DataPartition](../../itrader/rl_lib/evaluation/data_partition.py)
uses per-symbol row-fraction boundaries, not shared calendar cutoffs; common-
calendar folds and rolling refits are proposals, not current implemented behavior.
Declare complete causal input/feature/reward/outcome intervals and permitted
past context before selecting purge and eligibility rules.

Fit any learned transform/statistic/feature selection only on the appropriate
earlier training fold; freeze it for validation/test. Current visible-window
market-relative normalization is fitted-state-free, not a globally fitted
z-score. Redo training fits within each genuinely refitted fold, and report fit
lineage, selection attempts, inspected outcomes and researcher/agent exposure.
Fresh checkpoint/optimizer initialization cannot reset that information history.
More SGD reuse adds optimizer experience, not independent calendar support.

Golden/literal synthetic test fixtures supply independent expected arithmetic,
reset, observation and identity values. They are software oracles, not market
qualification or the final market `test` split. Keep live models/data outside
ordinary CI. Proposer examples/reference buffers, reward/credit feedback and
curriculum selection may use authorized training/validation evidence, never
the sealed final test. Terminal-screen-only evidence must not feed PPO, tuning,
proposer reward/curriculum or operational capability calibration. Invalid proposer
output stays behind TaskSpec validation; it cannot redefine evaluation economics.

Later chronological redevelopment/confirmation and any release of the reserved
envelope require separate decisions; dates/access/splits are not selected here.
See the [delivered selection-exposure and fit-lineage investigation](../artifacts/research/saluki-methodology-research-2026-10-10.md).
Do not reinterpret historical inspected panels as untouched confirmation.

## PLANNED Stress-Only Evaluation

Preserve the separately registered wing in the
[research dashboard](../research/itrader_research_dashboard.md) and reuse the
[delivered market-scenario stress note](../artifacts/research/market-scenario-stress-2026-10-10.md).
This is a later ablation proposal, **not implemented simulator behavior,
Q26 dependence calibration, fresh-market independence or authorized curriculum**.
R/C/S, reward/cost/optimizer, AZR/proposer, PF/WM/TTS and the separate examiner
wings retain their own dashboard/masterplan gates; this subsection merges none.

Decaying price-level displacement injects an opposing reversion component;
a fading return disturbance need not remove its accumulated price offset.
Do not treat injected recovery as discovered forecasting ability. No amplitude,
decay law, duration, clock or other parameter is adopted.

A future path-stress adapter must use the **same altered path for observations
and economic accounting**, retaining immutable original/derived identities.
Sensor-only corruption is a different study. Put scenario identity/seed/schedule
in an **external manifest**, not `TaskSpec.metadata`: semantic_v2 encodes that
metadata and could reveal the scenario. Hold TaskSpec/embedding fixed for hidden
path stress; future task-visible scenarios need separate review.

First use literal fixed-action traces for arithmetic, then compare the **same
frozen policy/checkpoint** on identity versus shock paths with matched starts,
horizons, costs, preprocessing and state initialization. Policy actions may
change in this robustness comparison. Retain target-position Discrete(3), fixed
cost rules, exactly-once terminal liquidation and existing rewards/`pnl_linear`;
different reactions can change turnover/cost totals without changing accounting.
Withheld stress seeds/families test the imposed assumptions, not new independent
market evidence. Any later curriculum/training arm needs separate authorization.

Six **proposed, not implemented/run** public oracles:
`IdentityControl`, `PrefixNoLeak`, `CoherenceEligibility`,
`EconomicReconciliation`, `ResetLineage`, `MatchedPolicyControl`.
Their detailed definitions and remaining design choices stay in the stress note,
not a duplicate report here.

## Review And Remaining Closure

[The accepted common contract and original scenario designs](../artifacts/research/q23-q26-decision-chain-2026-10-09.md)
and [mechanical repair receipt](../artifacts/research/evaluation-mechanical-repairs-2026-10-07.md)
are provenance, not blanket approval of current dirty source.
A receipt qualifies only its declared source/config/runtime/data snapshot.
Relevant drift invalidates that review/run claim until compared and requalified;
elapsed time alone does not. Writer static checks are not independent review.
High-risk changes retain REVIEW's fresh two-reviewer gate.

Source-grounded distinctions: the dated two-step 1-bp accounting oracle is not
the current three-step 10-bp test recipe; strict reporting still requires finite
Sharpe; disposition does not infer statuses; terminal preflight is not execution.
Pending closure includes execution-backed Spec probes, Q23 external role/dual
identities and exposure clearance, Q26 support/ranges/method/confirmation,
the actual all-16 terminal executor and proper-stack CUDA qualification.
Stress parameters/implementation/curriculum and later chronology remain
separately unapproved. Static documentation checks cannot close those gates.
