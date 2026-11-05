# R1 panel qualification and terminal preflight — 2026-10-06

[HTML review companion](r1-panel-qualification-2026-10-06.html).

Branch `agent/Watson-dev`, base `d3eb4c5`. Work remains uncommitted. This is a
prerequisite audit, not a new training run, a freshness certificate, or economic
promotion. The eight corrected-compatibility control replays remain accepted.

## Decision boundary

No fresh 128-task panel is frozen or opened. Physical calendar eligibility is
ample, but historical-start provenance must be closed before strict non-exposure
can be certified. Q26's equal-symbol calendar-block uncertainty method also
awaits the human decision in GRILL-IMPLEMENT. A descriptive reporter is not a
substitute for that requirement.

R1 remains unimplemented. The separate native Ray migration does not change this
compatibility-stack experiment. No new training, replay, WSL startup, staging,
commit, push, or artifact deletion was performed.

## Outcome-independent calendar audit

GPT-6 Luna/xhigh projected only `symbol`, `ts`, `session`, and `session_id` through
the existing environment loader. It did not read OHLCV, return or regime values.
The projection contains 5,515,061 regular-session rows across twelve symbols.

Dataset: `dataset_v2_20220103_20260922_alpaca_sip_split/bars`, Alpaca SIP,
split-adjusted, one-minute data. Bars-tree SHA256:
`20aac96de6da77e20beeaff214d380ef06184faf301ff0dd53bed0ffd4c04ce9`.

Eligibility uses validation fractions 0.7/0.15/0.15, 200-row purge, a 60-bar
lookback, same-session one-minute continuity, and capacity for the longest
200-bar horizon. Permitted decisions begin 60 or 120 minutes after 09:30 New
York time. The audit found 2,084 eligible symbol/session anchors; excluding the
32 known explicit panel anchors leaves 2,052 across 176 dates,
2025-04-24 through 2026-01-08.

Remaining candidate counts: AAPL 172, AMD 170, AMZN 171, AVGO 168, GOOGL 171,
IWM 172, META 171, MSFT 171, NVDA 170, QQQ 171, SPY 172, TLT 173. These are
provisional eligibility counts, not certified fresh samples. No anchors were
selected. The final audit must bind authoritative exchange-session/calendar
provenance as well as historical non-exposure; weekday labels alone are not an
exchange calendar.

The snapshot retains Alpaca's raw calendar at
`raw/alpaca/000000_calendar.json.gz`: 1,185 session records, 2022-01-03 through
2026-09-23, SHA256
`0bb92722b805a05af29f11db1922081b279e6b5142265f4ad371e3266ac44a72`.
Only calendar payload metadata was inspected. This is available provenance for
a future reviewed normalization/seal, not a newly certified panel or permission
to collect data. The snapshot's `session_id` is `SYMBOL:DATE:regular`; it is not
a bare calendar date. Qualification now preserves opaque frame/anchor session
identity and looks up the calendar using the aware timestamp's New York date.

The retained inventory's `D:/iTRADER-offload/active/ray_results` is the resolved
target of `C:/Users/Admin/ray_results`, not a separate omitted scan root. The
other recorded roots are `C:/itrader-ray` and repository config outputs.

## Historical-start qualification

The retained inventory contains 163 matching configurations: 121 explicit-start,
26 `random_episode`, and 16 controls whose start configuration was not recorded
at the inventory's top level. The random configurations have 594 evaluation
reports without retained episode-start payloads.

The random-start implementation examined in source creates a new NumPy RNG
from each fixed task's seed on reset. If the historical source and partition
were proven identical, repeated evaluations would reuse a finite set of starts,
not necessarily visit every eligible bar. However, those runs record six
distinct aggregate source fingerprints under the same Git commit. The commit
alone does not pin dirty source or prove actual historical reset semantics.
Do not turn seed-based reconstruction into a freshness certificate without
that binding. Conservative possible-exposure exclusion is not a claim that
every excluded bar was actually observed.

All 26 runs identify Git commit `891fcaa2ba2200d0affe2c97a6d9ada592dbcfaa`, but
their six recorded 82-file aggregate source hashes are:

```text
0368366b50363546f8ddeefda188cb09ea522c48faa1d80779564f603f239df5
17ce9ff1fa5248a2940f201aa78ac07782f4cc0e76ad784ebcadd36e6d4cabcc
35e103d40e3c53bc853f6f721afe409db743759d3aa9575e153a9158f93366d9
5eb39fb7516dfe3e2a5481ae82f6106ed6c50308db8719424f8017f2ca6520af
9374f7710665a4c557de9784a33909265592ea70ee25b21b9094caf30a66e7de
adea033f1c7b85b27ff0f9ca444b1f9a18a5f23d18459293907dd69034f860b9
```

The Git tree has 72 tracked Python files; its reconstructed fingerprint is
`f284c7381d9cb0a5023b69d7231be63c85bbc2946c1947c1dce35a710879f482`, not one
of these six. Aggregate fingerprints retain neither per-file hashes nor source
contents. The bounded search covered the retained inventory/listed roots,
26 configs, their metrics identity fields, and one representative Ray trial.
It was not an exhaustive scan of every offload directory. No sufficient source
snapshot was found within that bound.

The sixteen control configs have sibling `episodes.jsonl` files, each with 32
validation records containing symbol, task seed, horizon, start, end and steps.
All checked records satisfy `steps == horizon` and `end == start + steps`.
They form three repeated identity panels across controls. This recovers recorded
control windows, not independent source-bound RNG reconstruction. Their episode
sidecars are not hashed by the old inventory, and the earliest five configs lack
panel/evaluation-seed identities; do not claim a complete qualification merely
from their presence.
Historical controls used twelve-, six- and eight-symbol projections. Any exact
exclusion certificate must map their local row indices through symbol/timestamp
identity into the new twelve-symbol canonical frame; raw indices are not
interchangeable across projections.

**Freeze verdict: blocked.** Conservatively exclude every possible start in each
configured symbol's validation split, with its 60-bar lookback and up to 200
future bars. That possible-exposure domain covers all 2,052 provisional anchors:
zero candidates can be certified fresh under this bound. This does not say every
window was actually evaluated. GRILL received the exact findings and the
recovery-versus-separately-declared-new-data/split decision. No relaxed freshness,
final-test access or repartition was authorized.

Inventory bytes checked this turn:
`74b4fff809a85ebb22c055b3fd0a33e9c2c3e145a044e366d296622ded81162f`.
One malformed historical search accidentally surfaced a whole old `result.json`
line; its economic values were not used, and further searches used identity-only
JSON projections. No candidate outcomes informed selection; no panel was selected.

## Implemented preparation

The scientific loadout draft is
[`phase_a_r1_corrected_compat_fixed_4x2x2_20261006.json`](../../../itrader/scripts/experiments/configs/phase_a_r1_corrected_compat_fixed_4x2x2_20261006.json).
It declares four representations, two seeds and two curricula, with PBT off:
R0, R0 + R4-v1, R1 and R1 + R4-v1; exact 1 bp versus causal staging.
Each cell has 57,600 nominal training steps; total 921,600. Causal cost stages
remain at 0/14,400/28,800/43,200 steps. The original entropy clock is unchanged.
The January 32-task panel remains inspected progress evidence only.

This is **not** a sealed executable manifest: R1 is unavailable, the terminal
panel and qualification are not frozen, and effective configs/source must be
sealed after implementation and independent review, before training.

`run_terminal_screen` is an executable **read-only preflight**, not yet the
128-task evaluation launcher. It requires all sixteen disk-backed completed
cells before panel access, original indexed checkpoint bytes, predeclared panel
and qualification hashes, full retained-inventory qualification coverage, and
matching dataset identity. Passing preflight never authorizes evaluation.

The metadata validator requires a separately hashed exchange calendar artifact
within the sealed qualification evidence, with provider provenance and explicit
aware open/close bounds. Membership, exact 09:30 New York opening, longest
window fit and exact 60/120-minute decision offsets are checked. A weekday or
`regular` label cannot substitute for an exchange session. No actual calendar
artifact is certified in this pass; fixture calendars are explicitly synthetic.

The metadata validator rejects historical input/outcome overlap, malformed
exposure intervals, split/session conflicts, naive timestamps, non-one-minute
TaskSpecs, wrong cost/scale or observation/action widths, and injected regime
tags, metadata/difficulty hints or task embeddings. Per-record start/source proofs remain the independent
audit's responsibility; hashing an assertion does not prove its truth.

Future invocation, **only after real artifacts exist**:

```text
.venv/Scripts/python.exe -m itrader.rl_lib.evaluation.run_terminal_screen
  --declaration <pre-training-sealed-screen.json>
  --panel <qualified-128-task-panel.json>
  --qualification <reviewed-retained-history-qualification.json>
```

The actual terminal executor still needs the approved Q26 method and matched
candidate/control producer receipts. It must evaluate all five controls and the
predetermined terminal, invoke Q24/Q25, and not reuse the old 32-task replay
score as parity for a different panel. No real terminal runner was invoked.

## Executed integration checks

Persisted incomplete/unsealed screens were rejected by both API and subprocess
CLI without panel reads or output creation. A synthetic Parquet dataset drove
the actual `ITraderEnv` through all 128 crossed tasks for flat and long controls.
The equal-symbol reporter agreed: flat net was zero; long turnover included
entry plus liquidation, producing exactly 0.0002 cost return at 1 bp. These
fixtures are not market-performance evidence.

Independent review identified task/time/hint gaps plus stale completion,
boolean-seed, calendar-membership/full-final-bar and composite-session defects.
The public regressions were
observed red before the owning validators were corrected. GPT-6.1 Sol/MAX and
GPT-6 Luna/xhigh review the final snapshot under
`REVIEW.md`. Final verdicts and repository checks will be recorded at closeout.

## Remaining order

1. Close the bounded source/start audit, or explicitly settle recovery versus
   separately predeclared new data/split in GRILL; preserve strict freshness.
2. Settle Q26 and qualify exchange-session provenance; select/hash the terminal
   panel without outcomes only when the exclusion audit permits it.
3. Complete terminal execution/receipt wiring and independent integration review.
4. Implement and review R1 on corrected compatibility APIs, then seal effective
   sixteen-cell configs and runtime/source identities before training.
5. Train all sixteen matched cells; only then open terminal evaluation and its
   equally weighted/duration-normalized controls. Final test stays sealed.

## Closeout

The two Luna/xhigh audits completed. Source qualification remains blocked, not
silently accepted. The first Sol/MAX scoped approval was followed by Luna's
`VERDICT: CHANGES` for three additional gate defects. Those defects and the
subsequent full-bar/session-identity defects are repaired. Final Sol/MAX and
Luna/xhigh fix reviews both emit `VERDICT: APPROVE`. Synthetic full-screen CLI
preflight returned sixteen terminals and explicitly `evaluation_authorized:
false`; this never establishes real panel readiness.

Executed checks:

- Focused completion, qualification and CLI tests: 31 passed with source-shaped
  composite session IDs and full final-minute containment.
- Repository healthcheck: required dependencies pass; optional local LLM
  libraries are absent, unrelated to the random/generated proposer study.
- Final pre-commit all-files run passed every configured hook, including
  healthcheck and core pytest on the final 480-test collection. Earlier 476/479
  hook snapshots are superseded by this run, not evidence for unfixed code.
- `git diff --check`: pass; new audit/companion local-link checks: no broken links.
- Knowledge status: stale; explicit Git Bash status script reports unavailable
  `python3`. No WSL fallback was used. Direct source/tests remain authoritative.

Sol's optional nit is retained: timezone-aware timestamps stored in microsecond,
millisecond or second units can fail closed because `DataPartition` assumes
nanoseconds. The current dataset uses nanoseconds. Supporting equivalent storage
units is a separately tracked normalization improvement, not an unsafe acceptance.

Reviewed qualification source SHA256:
`6455516f37800f060090b51e753823e195c85e18b78bd33f8ad738a24034d825`.
Reviewed completion source SHA256:
`e159818f3db744016ee18d4c4582e23a847a20908cc8a823ce984410c3c0b0ce`.
Reviewed preflight entrypoint SHA256:
`4cbc034bd9f4942d5eab84601ec9e0b46b3228f9d12b75872b289fef928127a5`.
Both reviewers report separation of concerns, programming by intention,
encapsulation, high cohesion and low coupling OK. Sol executed all 31 focused
tests and a complete synthetic CLI; Luna executed 30 with the real-environment
fixture deliberately deselected, plus independent persisted/calendar probes.
Their approval covers the preparation code, not historical non-exposure,
provider/calendar truth, real terminal execution or economic promotion.

Do not interpret this implementation closeout as real-panel or execution
acceptance. Runtime RAM was
about 5.47 GiB free at the final collection spot-check. No training was launched.

## Follow-up scope correction — 2026-10-06

The later [evaluation-wide audit](evaluation-code-audit-2026-10-06.md) corrects
the historical-start interpretation above: all sixteen legacy-control configs
use the same twelve-symbol input frame. Six/eight are task-symbol coverage of
two crossed panels, not different frame projections. The recovery audit maps
512 recorded rows to 48 unique starts; 512/512 sidecar symbols agree with the
canonical regular-session frame. Exact historical within-symbol ordering remains
conditional without the dirty source snapshots. The expanded bounded search
found no source/snapshot candidates for the 26 randomized configs. The zero
certified-fresh verdict therefore remains unchanged.

Fresh evaluation-wide Sol/MAX and Luna/xhigh reviews both return **CHANGES**.
Their broader public probes expose four mechanical boundaries: explicit-start
remapping, ambiguous terminal-index selection, physical-EOF/reset agreement and
state-tail diagnostic schema. They do not negate the earlier scoped preflight
fix approvals or the eight accepted control receipts. Q27 repair approval and
Q23/Q26 decisions remain pending; no panel or executor is authorized.
