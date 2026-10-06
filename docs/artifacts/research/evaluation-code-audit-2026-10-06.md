# Evaluation mechanics and historical-exposure audit

Date: 2026-10-06, Europe/Madrid. Branch: `agent/Watson-dev`.
Status: **bounded audit complete; independent verdicts CHANGES; Q27/Q23/Q26 pending**.
[HTML companion](evaluation-code-audit-2026-10-06.html).
No panel freeze/opening, data collection, training, replay, R1 implementation,
native migration, staging, commit, push or artifact deletion occurred in this round.

## Decision boundary

The eight corrected-compatibility controls remain accepted replay evidence,
not replicated economic promotion. This audit does not rewrite their frozen
evaluator, checkpoints, receipts, rewards, configuration or observed outcomes.
It investigates current evaluation mechanics and recoverable historical exposure
before the real sixteen-cell representation screen is permitted.

The separate [primary-source methodology synthesis](evaluation-quant-methodology-2026-10-06.md)
is owned by RESEARCH;
its grouping, block length, interval, coverage and training-seed proposals require
individual GRILL decisions. Descriptive `not_scored` output is not acceptance of
missing uncertainty in the real screen. No new metric or statistical default is
adopted here.

Its leading Q26 candidate uses synchronized calendar blocks, keeps paired
crossed horizons together and recomputes Q24's exact equal-symbol hierarchy in
every resample. This is not an approved method: outcome-blind metadata support
and synthetic null calibration must precede the human grouping/block/interval/
coverage/seed decisions. Two training seeds remain descriptive screen evidence,
not a reliable training-population interval. Direction-versus-timing attribution
and causal local-action vectors are separately named diagnostic proposals,
not new reward terms or a substitute for terminal evaluation.

The [prior qualification audit](r1-panel-qualification-2026-10-06.md) remains
dated evidence. Its scoped preflight approval is not a whole-evaluation-stack
approval, an executable terminal runner or a freshness certificate.

## Frozen audit scope

Base revision: `d3eb4c517075b1b85506f5500a9a5552d6186e3f`.
Current runtime source: **118 Python files**, aggregate
`237efa2716b20568403f66bc56e20f6a1c9f2526d96b170a1ed798cf0a4c7dc4`.
The source boundary and individual file digests are in
[audit-boundary.json](evaluation-code-audit-2026-10-06-evidence/audit-boundary.json).
Audit scripts are under `docs/`, outside that runtime fingerprint.

Python 3.11.13; Ray 2.58.0; Gymnasium 1.2.2; Torch 2.3.1 CPU;
pandas 2.1.4; PyArrow 17.0.0; NumPy 1.26.4 from the healthcheck.
`runtime_identity()` currently omits NumPy, even though start sampling uses
NumPy's RNG. Recording it prospectively is a provenance-hardening requirement;
adding it now cannot retrospectively supply missing historical runtime evidence.
No replacement runtime identity was injected into accepted receipts.

GitNexus status was inspected using explicit Git Bash, not WSL: stale index
`1d5ba82` versus `d3eb4c5`; script then fails `python3: command not found`.
Direct source is authoritative. No graph refresh was attempted.

## Ownership and execution chain

```text
immutable Parquet + projected metadata/order
  -> DataPartition: symbol/split/session/cadence eligibility
  -> TaskSampler00 + ITraderEnv.reset: task and actual start
  -> ITraderEnv.step: action, position, horizon, episode accounting
  -> RewardEngine: optimization objective only
  -> EpisodeEvidence + callbacks: economic identity and trace
  -> checkpoint index / replay receipt: bytes, producer and contract
  -> all-16 completion + Q23 qualification preflight
  -> future terminal executor: candidates and matched controls
  -> Q24/Q25 descriptive report + separately approved Q26 uncertainty
```

The final two invocation/uncertainty seams are not implemented end-to-end.
`run_terminal_screen.py` is explicitly read-only and returns
`evaluation_authorized: false`. Its successful preflight cannot be used as
terminal-execution, independence or economic-promotion authority.
`terminal_panel_report.py` assumes provenance has already been verified by its
caller. `CapabilityProfile` is a future online eligibility/matching contract,
not an established offline calibration estimator, reward or solver input.

## Reset/start semantics established by public probes

The retained [synthetic probe](evaluation-code-audit-2026-10-06-evidence/public_env_probe.py)
uses a temporary 100-row constant-price Parquet, not real market outcomes.
Its [results](evaluation-code-audit-2026-10-06-evidence/public_env_probe_result.json)
are not historical start reconstruction.

- `random_episode` constructs a fresh RNG from the current task's seed at
  each reset. Two fixed tasks with seeds 17/29 repeat starts 73/90 in a six-reset
  cycle. Worker 1 does not change those already-fixed task-owned seeds.
- `stream_worker` advances its cursor/RNG; seed 7 gives starts 55 then 59,
  and an explicit `reset(seed=7)` restarts at 55. Horizon changes resample;
  jumps and invalid-boundary wraps also restart the stream location.
- Ordinary episode resets still clear position, episode PnL, action counts,
  horizon progress and evidence. A market cursor continuing is not portfolio,
  position or recurrent-state carry. Reward normalization may separately carry
  across a configured stream; the fixed screen uses episode normalization.
- Starts are constrained by the projected per-symbol frame and DataPartition.
  A task seed alone is not enough without the historical data projection,
  source semantics and RNG version. Stream-worker reconstruction additionally
  needs cursor/RNG state and reset/horizon/jump history.
- Explicit `task_spec` starts are currently clamped. Qualification must reject
  an invalid declared start before replay rather than claim the clamped location
  is the declared held-out task.

## Inherited physical-EOF/reset conflict

**BLOCKING before general boundary-correct evaluation; not a demonstrated
accepted-control contamination.** Public reproduction:

| Declared legal task, 100-row frame | Reset-reported start | Evidence start | Expected steps | Actual steps | Terminal remaining fraction |
|---|---:|---:|---:|---:|---:|
| start 99, horizon 1 | 99 | 98 | 1 | 1 | 0 |
| start 98, horizon 2 | 98 | 98 | 2 | 1 | 0.5 |
| start 96, horizon 4 | 96 | 96 | 4 | 3 | 0.25 |
| interior start 95, horizon 4 | 95 | 95 | 4 | 4 | 0 |

`DataPartition.valid_ranges()` permits `last = outcome_end - horizon`,
including the last usable bar. `step()` consumes `close[idx]`, then increments
`idx`; `_episode_done()` tests the incremented index against `len(df)-1` and
ends one action early at physical EOF. A second generic `_clamp_start_idx()`
caps at `len(df)-2`, so a one-step task also gets a reset/evidence identity
mismatch. Changing only the termination comparison would leave that mismatch.
The EOF condition is also present in base `d3eb4c5`.

All synthetic long-only cases charge exactly 2 bp in total (one entry and
one terminal liquidation), with zero gross return; net is `-0.0002`.
Liquidation adds no decision bar. No out-of-bounds access occurred during the
returned legal episodes. No new terminated-versus-truncated design is adopted.
The Q25 reporter rejects censored `steps != horizon`, but rejection does not
repair producer behavior. Correction/tests/review remain pending in this audit.

### Scope check of the accepted eight controls

[control_boundary_probe.py](evaluation-code-audit-2026-10-06-evidence/control_boundary_probe.py)
reads only existing configs, episode location/count identities and Parquet
footers. No return, Sharpe, action/regime or price values are used.
[control_boundary_result.json](evaluation-code-audit-2026-10-06-evidence/control_boundary_result.json)
records all eight cells and input hashes:

- actual configured projection: twelve symbols, regular sessions, validation;
- current 60 Parquet footers contain **5,515,061** projected rows;
- all **256** retained episode identities match their fixed starts/seeds/horizons;
- every episode satisfies `steps == horizon == end_idx-start_idx`;
- minimum start **849,750**, maximum end **4,985,377**, below physical EOF.
- every terminal index has exactly one highest-update entry, at update 12;
  the newly reproduced generic-selector tie defect did not trigger here.

This supports non-contamination by this particular EOF/reset defect for those
terminal episodes. It is not proof of every historical training episode or
every current evaluator invariant. Dataset bytes were not rehashed in this
footer-only probe. The first invocation of the audit helper used an incorrect
repository-parent offset and failed before reading evidence; that helper-only
path was corrected and the successful check is retained. No original artifact
was changed.

## Additional independent review findings — corrections not yet applied

Both independent reviews are complete and return **VERDICT: CHANGES**. Sol
reproduced four blockers: the three below plus the physical-EOF/reset conflict
above. Luna independently confirmed the explicit-panel start-remapping blocker.
GRILL's Q27 asks for approval of these four bounded mechanical repairs before
production changes; approval is not inferred from the earlier general request.

1. **Declared versus effective diagnostic location.** A declared diagnostic
   start 999999 on a 2,000-row validation frame passes the non-overlap check,
   then the environment clamps it to source start 1604/end 1700. The declared
   interval is not the actual replay interval. Validate eligibility and reject
   any remapped explicit start before making non-overlap claims.
2. **Order-dependent terminal selection.** Two index rows at the highest update
   12, with different clocks 57,600/58,000, are accepted by the generic replay
   selector; reversing JSONL rows changes its choice. Reject ambiguity and
   bind predetermined completion rather than using file-order ties. The
   all-sixteen preflight has stricter uniqueness checks but is not that selector.
3. **State diagnostics on a state-free model.** `evaluate_checkpoint_state_diagnostics`
   accepts `r0_base`, state dimension zero, then treats the final two task
   embedding values as mutable position/time. Require the explicit state-tail
   schema before intervention; unavailable is preferable to a mislabeled probe.
4. **Conservative half-open adjacency nit.** The older diagnostic overlap check
   uses `<=` for end-exclusive intervals and rejects adjacent eligible windows.
   It rejects too much rather than admitting overlap; defer unless needed for a
   proven unsafe-acceptance correction.

A separately tested initial wrong-recipe declaration can satisfy hashed
completion matching: hashes alone do not enforce Q22. This is a pending
semantic pre-training seal requirement, not proof that an already externally
sealed declaration was bypassed. Optional action-count fields ignored by the
current Q24 reporter are not its input contract; it validates action fractions.
Neither probe is silently elevated into an artifact-tampering claim.

### Bounded retained state-diagnostic check

The known `solver-state-r4-v1-causal-forensics-2026-09-29-evidence` directory
contains **11** retained `*.state_diagnostics.json` reports. A metadata-only
inventory found all eleven declare `solver_state_mode: both`,
`observation_schema_id: r4_v1_min`, state dimension 2. No R0/off diagnostic
claim was found within this explicit scope; no state-step/outcome file was
opened and no report was rewritten. This is not a workstation-wide assertion
that such artifacts cannot exist. File hashes and scope are in
[retained_state_diagnostic_metadata.json](evaluation-code-audit-2026-10-06-evidence/retained_state_diagnostic_metadata.json).
This inventory does not retroactively add producer provenance to historical
G0 receipts or reinterpret random-v1 weights as zero-v1 initialization.

## Historical recovery and freshness

The previous retained inventory has 163 matching configurations: 121 explicit
starts, 26 randomized and 16 legacy controls. The randomized configurations
have 594 retained reports without start payloads. Six dirty source aggregates
claim 82 Python files, while their named Git commit reconstructs only 72;
nominal source reconstruction is not yet source-bound proof.

The completed [recovery audit](evaluation-code-audit-2026-10-06-evidence/exposure-recovery.md)
recovers **512 recorded rows across sixteen controls, reducing to 48 starts**:
32 unpanelled starts, eight September crossed-panel starts and eight January
crossed-panel starts. Their sidecar symbols match the canonical regular-only,
alphabetically grouped twelve-symbol frame in **512/512** records. Every
recorded episode has its complete declared horizon within one symbol/session.
The coordinate/provenance ledger is
[exposure-recovery-starts.json](evaluation-code-audit-2026-10-06-evidence/exposure-recovery-starts.json).

**Correction to the prior qualification interpretation:** all sixteen configs
specify the same twelve `data_symbols`; six/eight describe task-symbol coverage
of the two panels, not six/eight-symbol market-frame projections. Symbol
identity is directly cross-checked, but historical within-symbol ordering is
still conditional because the exact dirty source snapshots were not recovered.
Recorded locations must not be conflated with source-bound RNG reconstruction.

The bounded search of the 26 exact randomized stable-trial directories and
source/snapshot/manifest-named immediate siblings at eight parent locations
found **zero candidate source snapshots**. None of the six aggregate runtime
hashes matches the named Git tree. Therefore zero randomized starts are admitted
as source-bound reconstruction. Seed-derived plausible locations cannot narrow
the conservative possible-exposure domain.

The 2,052 provisional calendar-eligible anchors remain **zero certified fresh
candidates** under conservative possible-exposure exclusion. That bound does
not claim every bar was actually visited. Repeating old runs does not restore
freshness. Older/newer dates alone do not prove non-exposure. A separately
declared evaluation-only dataset would require a reviewed exposure/calendar
contract; it must not silently append/repartition the existing snapshot or
open its sealed final test.

## Independent review and validation

Two fresh reviews were invoked without inherited conversation history:
requested `gpt-6.1-sol/max` and `gpt-6-luna/xhigh`, following `REVIEW.md`.
They predeclare probes, execute public behavior, compare base availability and
report all five design principles. Final verdicts and bounded reproductions:

- [Sol review](evaluation-code-audit-2026-10-06-evidence/independent-review-sol.md):
  **CHANGES**, four blockers and one adjacency nit; 99 focused head tests and
  19 isolated-base tests executed, plus public adversarial reproductions.
- [Luna review](evaluation-code-audit-2026-10-06-evidence/independent-review-luna.md):
  **CHANGES**, one overlapping start-identity blocker and three nits;
  48 synthetic tests executed, one Parquet-backed test deselected, plus helper
  probes. Requested model/reasoning pairs are recorded; the sessions expose no
  trustworthy runtime-model telemetry to independently certify model diversity.

Luna's additional fail-closed/schema nits are tracked, not silently repaired:
combined RLlib result shapes can hide a valid nested replay payload; absent
`trial_id` in both completion records compares equal; adjacent half-open windows
are rejected conservatively. The missing-trial-ID fixture retains distinct
stable paths/config hashes and is not evidence of a score or checkpoint swap.
These do not relax Q27's four-repair scope or become scientific defaults.

Root self-check, not an independent review:

```powershell
.venv/Scripts/python.exe docs/artifacts/research/evaluation-code-audit-2026-10-06-evidence/public_env_probe.py
.venv/Scripts/python.exe docs/artifacts/research/evaluation-code-audit-2026-10-06-evidence/control_boundary_probe.py
.venv/Scripts/python.exe -m pytest -q itrader/tests/test_env_core.py itrader/tests/test_data_partition_sessions.py itrader/tests/test_terminal_panel_report.py itrader/tests/test_terminal_screen.py itrader/tests/test_screen_completion.py
.venv/Scripts/python.exe itrader/tools/healthcheck.py
```

The focused suite executes **57 tests**; all pass. Required healthcheck packages
pass; optional `llama_cpp`/`transformers` are absent, CUDA unavailable. Synthetic
adversarial probes above expose a gap outside the current passing examples;
green tests alone are not whole-stack evidence. The prior round's 480-test
pre-commit result remains historical. RESEARCH separately executed the shared
Windows full suite in this round: **480 passed in 129.95 s**, exit code 0,
verified through its thread execution output; root did not duplicate that run.
The runtime aggregate above remained unchanged afterward. No new pre-commit
execution is claimed for this audit-only round.

## Ordered continuation — not launch permission

1. Settle Q27's four mechanical repairs in GRILL. If approved, reproduce public
   failures, patch the owning boundaries and obtain fresh independent review of
   each correction. Protect interior episodes, cost/liquidation accounting,
   initialization and the frozen evaluator/historical receipts.
2. Settle Q23's data role/coverage: retained history remains unresolved despite
   48 recovered recorded starts; a separately declared evaluation-only route
   needs explicit approval, provenance, calendar and exposure/purge rules.
3. Settle Q26 grouping, block length, interval, coverage, multiplicity and seed
   treatment individually, using outcome-blind metadata support and synthetic
   calibration. Preserve accepted Q24/Q25 descriptive semantics.
4. Implement and review the real terminal executor: all-16 completion first,
   predeclared hashes, verified calendar/freshness, predetermined terminal bytes,
   source/runtime/producer identity, exact 1 bp, matched flat/long/short/random/
   architecture-matched untrained controls, complete 128-task coverage, Q24/Q25
   aggregation and accepted Q26 uncertainty. No `not_scored` shortcut for closure.
5. Only then freeze the reviewed scientific/config/runtime boundary and advance
   R1 under its accepted stateless window-encoder contract. R0 and R0+R4-v1
   remain matched controls; native migration remains a separate later factor.
