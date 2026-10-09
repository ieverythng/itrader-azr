# Q27 mechanical repairs and bounded diagnostic pilot

Date: 2026-10-07, Europe/Madrid. Branch `agent/Watson-dev`; review base
`3a3583d` (Alpaca ingestion commit). Other dirty changes are preserved.
Status: implemented; independent Sol and Luna reviews APPROVE. Final core
suite, healthcheck, clean scoped pre-commit rerun and diff checks pass.
Requested reviewer model/effort is not runtime telemetry. No panel-opening authority.

## Approved scope

The human approved Q27 in GRILL-IMPLEMENT on October 7: reject invalid explicit
diagnostic starts rather than remap them, reject ambiguous terminal selection,
repair physical-EOF/reset horizon consistency, and guard the actual R4 state
tail before counterfactual interventions. No broad sampler redesign, termination
API change, training, native migration, R1 or real terminal-panel opening.

Secondary episode return attribution and a bounded one-step diagnostic pilot
are approved as experimental reporting only. Additive economics must reconcile
to existing environment evidence; no headline/reward replacement, single-step
Sharpe, proposer/training feedback or invented volatility default.

Q23's first-class `terminal_screen_only` role, separate training/terminal
dataset identities and explicitly certified eligible intervals are approved;
that owning-seam implementation is the next distinct reviewed pass. Preserve the
512 recovered rows / 48 starts; no acquisition, outcome opening, appending or
repartitioning dataset_v2 or sealed-final-test access is authorized. Metadata
feasibility, source exposure and calendar/normalization declarations remain
required. Q26 statistics/support/calibration/seed decisions remain unapproved.

## Round and protected controls

Public seams: validation replay context and retained checkpoint selection;
ITraderEnv reset/step terminal evidence; state diagnostic helper and CLI before
Ray startup; opt-in economic diagnostics. Budget: four mechanical repair slices
and one synthetic reporting pilot, followed by independent REVIEW.md rounds.
Stop on failed correctness/provenance review; no comparable training run is
implicitly authorized by the stack-iteration skill.

Historical evaluator/receipts/checkpoints are not rewritten. Interior episode
starts, generic sampler clamping, task embeddings, state initialization and
entry/switch/liquidation accounting remain controls. Random/generated proposer,
Family 00, dataset_v2 RTH, compatibility Ray 2.58 are unchanged.

## Mechanical red/green evidence so far

- Terminal selection: two update-12 rows at 57,600/58,000 steps, in either order,
  previously selected by JSONL order (two expected-rejection tests failed).
  Highest update must now be unique; where retained completion status exists,
  completed status, sampled steps and RLlib iteration must agree. Historical
  indices without status retain an explicitly bounded highest-update selector;
  this is not the all-16 predetermined-terminal completion certificate.
- EOF/reset: synthetic 100-row constant prices, starts 99/98/96, horizons 1/2/4:
  all three public regressions failed before correction. The next unread index
  now reaches physical EOF only at length, not length minus one; the reset clamp
  permits the last readable row. All horizons complete with exact start/end and
  state reset/terminal tails. Long-only net stays -2 bp including liquidation,
  with reward sum -0.2 under reward scale 1000. 23 env/session tests passed.
- Diagnostic panel: start 999999 passed declaration hashing and would remap into
  the source path (expected-rejection regression failed). Override preflight now
  builds DataPartition from metadata only, using the environment's sorting,
  split/session/cadence semantics. Source and diagnostic starts must be eligible;
  runtime overrides opt into strict task-owned starts. Ordinary clamping remains.
- State tail: helper tests first failed on the new required contract interface.
  Known r4_v1_min, two tail values, valid mode/base/agent widths and feature order
  are required before mutation; actual observation shape is checked. CLI guards
  declared schema/config before Ray; declared observation-space shape is checked
  before reset, and the returned array shape after reset, before policy scoring.
  This does not require zero_v1 initialization or relabel historical random_v1.

Commands: `.venv/Scripts/python.exe -m pytest -q itrader/tests/test_env_core.py
itrader/tests/test_data_partition_sessions.py`; `.venv/Scripts/python.exe -m
pytest -q itrader/tests/test_policy_baselines.py` (30 passed before the additional
protected-control matrix). Broader checks and independent verdicts pending.

## Review-driven correction and secondary pilot

The first Sol mechanical review reproduced a new regression: count-backed but
columnless metadata frames are `pandas.empty`, so OHLCV-only Parquet was rejected
despite valid row eligibility. A public regression failed, then passed after the
fallback acquired a uint8 metadata marker column. It still uses only footer row
counts, never prices or volumes. Original findings and the fresh verification
are retained in the review; this is not hidden by the passing suite.

Pilot interface: `capture_step_economics` is strictly boolean, default false.
The env supplies actual raw transitions and a pre-action context (past close
tuple, decision-asof, row bounds, symbol/session, TaskSpec hash and observation
schema). `evaluate_economic_diagnostics` only accepts explicit task starts,
rejects remapping, censored episodes and the sealed test, and requires a
caller-supplied policy/control identity. That label is not producer attestation:
the report explicitly says `producer_provenance_verified: false`. A future gated
runner must bind the actual checkpoint bytes/runtime/producer receipts.

The optional estimator consumes the frozen past-close tuple before the action
or next bar, and must return a named method/scale. No estimator, epsilon, ddof,
annualizer or economic threshold is adopted. Missing, nonpositive, nonnumeric,
nonfinite or unrepresentable scales remain unavailable with reasons/coverage.
Both scaled gross and scaled continuation net are explicitly named; neither is
single-step Sharpe. Hold-prior, flat, long and short reference vectors use the
same frozen outcome, prior position, evidence cost and terminal treatment.

Per-step accounting separates switching, continuation, hypothetical roundtrip
and actual terminal liquidation. Episode gross is the additive sum of selected
positions times actual returns; `pbar * sum(asset_returns)` is directional gross,
and the difference is timing gross, not causal forecasting alpha. Each episode
reconciles gross/cost/net to existing EpisodeEvidence before attribution; no
cross-episode raw reducer replaces Q24's horizon/anchor/equal-symbol weighting.
The Q24/Q26 executor remains pending, so this pilot does not calculate approved
block intervals or a new pooled promotion score.

The wrong-position trace regression initially passed despite inconsistent
positions: root and pilot reviewer observed that failure before the added
step recomputation and contiguous-position checks. Actual EpisodeEvidence
also requires nonduplicated contiguous row/task/action/schema/policy identities.
Pure arithmetic fixtures without market identities are marked identity-unchecked;
no fixture or textual identity is a provenance certificate. Stale post-terminal
contexts now reject.

23 pilot tests pass: actual env/RewardEngine/evidence arithmetic, 9 transitions,
directional controls, future-row perturbation invariance, unavailable scales,
wrong positions, duplicated rows and stale contexts. A test-only 60-close / 59
return sample-standard-deviation estimator demonstrates the optional interface;
it is not a configured default. Synthetic long-short gross is +0.2, cost 0.0004,
net +0.1996 and reward 199.6; this is not market-performance evidence.

Initial wider snapshot: 90 focused tests and required healthcheck pass; full
native suite 508 passed in 126.30 s before the final expanded pilot/metadata
regressions. These are not claimed as final-snapshot results. Final revalidation
and independent mechanical/pilot verdicts follow below.

## Final review fixes and remaining nits

Luna's mechanical fix review approves the current scope after two additional
public failures were repaired: fractional/string/boolean starts now reject
instead of coercing, and schema-resolved metadata names preserve uppercase
symbol/session filters and boundary-only files without forced lowercase fields.
The actual reset-array tail is checked after reset, not claimed before it.

The pilot's first split fix still trusted the raw configuration field. A fresh
review reproduced an actual test partition disguised by raw `train`. Three
public mismatch cases failed before correction; declared/effective partition
identity must now agree before reset. Exact train/validation remain accepted;
all/test/case variants and inconsistent scopes remain unavailable. The pilot
does not gain terminal-screen access by this fix.

Sol reproduced an integer volatility scale too large to convert to float.
An extension to the existing unavailable-scale matrix failed, then passed when
the opt-in pilot reported `unrepresentable`, preserving ordinary accounting.
The current pilot module passes 28 cases. This does not select an estimator.

Fresh mechanical and pilot Luna reviews and the final Sol review return APPROVE.
Earlier CHANGES findings remain in their original reports:
[mechanical fix review](evaluation-mechanical-repairs-2026-10-07-evidence/review-luna-fixes.md),
[pilot partition/scale fix review](evaluation-mechanical-repairs-2026-10-07-evidence/pilot-review-luna-partition-fix.md).
[Final combined Sol review](evaluation-mechanical-repairs-2026-10-07-evidence/final-review-sol.md)
records 162 independently checked accounting combinations, 1,440
metadata/runtime eligibility comparisons, CLI guards before Ray import and
base/current public-seam comparison. Its 94-case focused suite passed. Those
bounded synthetic probes do not establish market performance or freshness.
Requested reviewer identities are recorded as dispatch metadata, not observed
runtime model telemetry.

Two mechanical nits remain deliberately bounded: supported source-panel
`start_index` aliases reject before execution, and exact adjacent panel windows
are conservatively rejected. Neither admits leakage or silently remaps a task.
They are not silently represented as fully equivalent public input support.

## Test value and integration-first follow-up

Acceptance rests on executed contracts and independent probes, not case count.
The key integrations use synthetic Parquet, real DataPartition/ITraderEnv,
RewardEngine and EpisodeEvidence, plus persisted checkpoint/config/index files.
They jointly exercise reset identity, actual bar accounting, terminal costs,
state observations and the reporting consumer. No mocks substitute for those
owners; fixture outcomes are not market evidence.

Review identified modest positive-assertion duplication between manual
two-step attribution and the environment pilot. A later focused consolidation
can put their common decomposition assertions in the public environment test,
retaining wrong-evidence and wrong-position rejection, duplicate/stale trace
identity and split disagreement checks. EOF horizons, metadata formats, terminal
ambiguity and nine position transitions remain compact parameterized matrices:
each case has a distinct failure mode, not a new copy of implementation logic.
Do not delete protected historical tests or refactor green checks cosmetically.
For the next role pass, prioritize one snapshot -> metadata eligibility ->
terminal-preflight integration with adversarial role/identity mutations, rather
than duplicating the same validation assertions in each consumer.

## Remaining prerequisite chain

1. Q27/pilot review and scoped validation are closed; preserve protected controls.
2. Implement/review the approved first-class terminal-only role, dual dataset
   identities and eligible-interval adapter as its own change. The current
   qualifier/preflight still require shared dataset identity and ordinary
   validation; relabeling external data is forbidden.
3. Metadata-first envelope planning is now approved by the human in GRILL.
   RESEARCH must recommend exact defensible bounds/calendar support and retained
   exposure/access clearance before acquisition. No price acquisition or outcome
   opening is approved by that planning decision.
4. Finalize Q26 against metadata and synthetic calibration; only then seal the
   panel/method and implement/review all-16 terminal execution with Q24/Q25 and
   matched producer/control identities.

Optional volatility normalization is not a prerequisite for base Q24 net-return
execution when disabled. No source here freezes a real panel, establishes fresh
market data, runs R1, opens the sealed final test or grants economic promotion.

## Final-snapshot validation record

Native `.venv/Scripts/python.exe` required healthcheck passes. Scoped source
hashes in the final Sol report match the author's current eight source/test
files. The final core suite passes **530 cases in 110.59 s** through the
pre-commit core hook. The first outer invocation nevertheless exits 1 with
`files were modified by this hook`: the author edited tracked
`docs/project-state/decisions.md` while the hook was active. This does not make
the first pre-commit invocation green. Freeze edits and rerun before closure.
No owned pytest/pre-commit process remained after that invocation. Scoped hooks
were then rerun, including the decision log, with author edits frozen: exit **0**, all
applicable hooks passing. See the retained [final validation receipt](evaluation-mechanical-repairs-2026-10-07-evidence/final-validation.md)
for the exact command and scope. Final administration-only closure text is
checked separately without repeating the full suite or changing reviewed code.

Final touched-code deslop self-check preserved the shared partition seam and
existing evidence/reward ownership; no cosmetic rewrites, new framework,
unrelated cleanup or test deletion. The independent review is not replaced by
that self-check. All work remains dirty, with no staging/commit/push.

## Skills actually applied

- itrader-stack-iteration-loop: froze bounded repair scope, public red/green
  slices, protected accounting/parity controls and mandatory independent review.
- itrader-architecture-check: DataPartition owns eligibility, ITraderEnv owns
  assembly/transitions, evidence remains separate from RewardEngine optimization.
- tdd: each repair follows executed public failure then minimal owning fix;
  synthetic Parquet and persisted indices exercise real consumers, not mocks.
- codebase-design: one shared partition-construction interface keeps metadata
  preflight and the real environment on the same eligibility implementation.
- deslop-refactor: touched first-party cleanup only after green; no generated,
  vendored or historical evidence cleanup, no speculative generic framework.
- itrader-experiment-handoff: active docs will distinguish approval,
  implementation, review and remaining launch gates. Autoresearch optimization
  is a later separately approved single-factor experiment, not this repair pass.

GitNexus is stale (index 1d5ba82); explicit Git Bash status fails because python3
is absent. No WSL fallback or graph refresh. Source/tests are authoritative.
