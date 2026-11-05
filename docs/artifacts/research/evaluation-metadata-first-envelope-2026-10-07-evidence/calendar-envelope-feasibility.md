# Q23 metadata-first calendar-envelope feasibility

Date: 2026-10-07, Europe/Madrid

Status: **outcome-blind planning evidence; one retrospective candidate is ready
for declaration review, but no bar acquisition, Q26 choice, panel freeze, or
terminal access is authorized**

## Verdict

The tightest defensible historical candidate found under an outcome-blind rule
is an Alpaca request from **2020-01-01 through 2021-12-31**, whose actual exchange
sessions run from 2020-01-02 through 2021-12-31. The rule is fixed before any
bar request: choose the **nearest two complete calendar years immediately before
the retained dataset-v2 start**. It was not chosen because it fits 32 anchors,
because a Q26 method performed well, or because of any return, volatility,
regime, solver, or control result.

The provider calendar contains **505 sessions**, so the proposed twelve-symbol
request has **6,060 requested symbol/session slots** before availability and gap
checks. This is calendar capacity, not bar completeness. No candidate-period bar
endpoint was called, so the number of slots with any bar, an exact regular
schedule, or both eligible decision offsets remains **unknown until separately
authorized acquisition and metadata-only qualification**.

The interval is a **recommended candidate for declaration review; retrospective
clearance is pending**. A complete retrospective exposure clearance must still
establish that no training,
proposer/curriculum, tuning, calibration, checkpoint selection, diagnostic,
plotting, manual inspection, or agent/tool access consumed value-bearing data
from this interval through any provider or local source. Current SIP historical-
bar entitlement and all-twelve-symbol availability are separate open facts. If
either fails, do not silently change feed, provider, dates, or symbols.

## Inspection boundary

This pass used only:

- provider exchange-calendar dates and aware open/close clocks;
- retained `symbol`, UTC `ts`, `session`, opaque `session_id`, and
  `trading_date` projections;
- Parquet schema/row-count metadata, file identities, configuration declarations,
  and retained exposure classifications; and
- timestamp existence, duplicates, gaps, and calendar-relative eligibility.

It did **not** request or read OHLCV values, returns, regimes, labels, task
difficulty, policy/control actions, rewards, PnL, Sharpe, or model outputs. The
provider call used the existing read-only
[`AlpacaMarketDataSource.fetch_calendar`](../../../../itrader/inputs/market_data/alpaca.py)
adapter with locally configured credentials. Credential values were neither
printed nor persisted. The bar endpoint was not called.

The machine-readable receipt is
[`calendar-metadata-receipt.json`](calendar-metadata-receipt.json). Its calendar
request was canonicalized as:

```json
{"end":"2026-10-07","endpoint":"calendar","start":"2019-01-01"}
```

- canonical request SHA-256:
  `f1978c5153d32c5504f8ce59689dde7ff7dae4a2ae3d30e0c569f055ae3ff5c7`;
- canonical response-payload SHA-256:
  `360d219baff281f84b1185b9febb2590d176b8c2e4447bda540daf034ece7b9a`;
- response: 1,952 sessions, 2019-01-02 through 2026-10-07; and
- retrieval time: 2026-10-07T21:45:53.521105Z.

This successful calendar call proves current authenticated access to that
calendar endpoint only. It does not prove SIP bar entitlement, bar availability,
or that SIP is free.

## Retained all-symbol timestamp benchmark

The existing dataset-v2 snapshot is **not** a Q23 candidate. Its timestamp-only
projection is useful as a same-stack benchmark for what post-acquisition
qualification must report. The query selected only `symbol`, `ts`, `session`,
`session_id`, and `trading_date`, filtered to `regular`, and read no feature or
outcome column.

The retained snapshot has 5,515,061 regular rows, zero duplicate
symbol/timestamps, and at least one row for all twelve symbols on every one of
the 1,184 exchange sessions from 2022-01-03 through 2026-09-22. A gap means an
expected provider-calendar regular minute was absent. Eligibility requires the
entire 260-minute half-open information/outcome interval—60 past minutes plus
the longest 200-minute outcome—to be consecutive within one symbol/session.

| Symbol | Rows | Sessions | Sessions with gaps | Missing minutes | Max gap (min) | T+60 eligible | T+120 eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| AAPL | 460,140 | 1,184 | 0 | 0 | 1 | 1,175 | 1,175 |
| AMD | 460,140 | 1,184 | 0 | 0 | 1 | 1,175 | 1,175 |
| AMZN | 460,009 | 1,184 | 52 | 131 | 3 | 1,126 | 1,125 |
| AVGO | 455,685 | 1,184 | 534 | 4,455 | 6 | 662 | 650 |
| GOOGL | 458,122 | 1,184 | 132 | 2,018 | 4 | 1,046 | 1,044 |
| IWM | 460,136 | 1,184 | 1 | 4 | 5 | 1,174 | 1,175 |
| META | 460,139 | 1,184 | 1 | 1 | 2 | 1,174 | 1,174 |
| MSFT | 460,140 | 1,184 | 0 | 0 | 1 | 1,175 | 1,175 |
| NVDA | 460,140 | 1,184 | 0 | 0 | 1 | 1,175 | 1,175 |
| QQQ | 460,140 | 1,184 | 0 | 0 | 1 | 1,175 | 1,175 |
| SPY | 460,136 | 1,184 | 1 | 4 | 5 | 1,174 | 1,175 |
| TLT | 460,134 | 1,184 | 6 | 6 | 2 | 1,173 | 1,170 |

Across all twelve symbols, 625 dates have a complete provider-calendar schedule,
636 support T+60, 625 support T+120, and **622 support both offsets**. The
calendar has nine early closes in this range—2022-11-25, 2023-07-03,
2023-11-24, 2024-07-03, 2024-11-29, 2024-12-24, 2025-07-03, 2025-11-28, and
2025-12-24. None can fit the required 60/120 offset plus 200-minute outcome for
all symbols. Early closes must remain explicit exclusions; a short session is
not a generic gap.

These counts show why calendar session count cannot stand in for twelve-symbol
bar support. In particular, AVGO is the limiting retained-snapshot symbol even
though it has at least one row on every date. The new snapshot must report the
same dimensions before any anchor is selected.

Source identities:

- build request file SHA-256:
  `e2fd032d9af90dc74a2ae2b1762a96ef8b7d47c49bd957706f9e646ceee4df62`;
- manifest file SHA-256:
  `4c6202883a46cbe5bfbf4986003dcd1eb05b3738ac2d48eeb42d9a57fe974ca5`;
- immutable dataset content SHA-256:
  `fac41f6d42f8c5b1eac739755667bb38407461f82c168d7955de26bb2e5fb79c`;
  and
- retained raw calendar gzip SHA-256:
  `0bb92722b805a05af29f11db1922081b279e6b5142265f4ad371e3266ac44a72`.

The manifest predates the current raw-file/retrieval-time/usage ledger. Its
calendar file is useful retained evidence, not a complete modern provenance
certificate. See the [dataset-v2 audit](../dataset-v2-r0-audit-2026-09-24.md)
and current [market-data architecture](../../../architecture.md#market-data-authority).

## Historical exposure boundary

### Dataset v2: ruled out

The entire 2022-01-03 through 2026-09-22 retained-v2 chronology is ruled out for
the separate Q23 route:

- the per-symbol training outcome interval reaches from 2022-01-03 through
  2025-04-23 for most symbols and through 2025-04-28 for AVGO;
- validation begins from 2025-04-24 to 2025-04-29 and ends from 2026-01-07 to
  2026-01-09; the 26 unreconstructed randomized evaluation configurations force
  conservative possible-exposure exclusion over this domain; and
- the remaining 2026-01-08/12 through 2026-09-22 region belongs to the old
  sealed final test. The approved route preserves that test and does not recycle
  it into `terminal_screen_only`.

The dates above describe current 70/15/15 plus 200-row partition metadata. They
do not turn fractional splits into the new terminal role, and they do not claim
that every possible row was actually consumed. The controlling fact is the role:
training/validation exposure is disallowed, while the old final test stays
sealed.

### Legacy dataset v1: possible exposure, not a clean interval

The current `itrader/data/ohlcv.parquet` identity has SHA-256
`5f2e449548532ff71f419431df86962a2aedf7275d7bff5592729bbb8a900f73`,
719,855 rows, and three symbols:

| Symbol | Rows | Trading dates | First raw naive timestamp | Last raw naive timestamp |
|---|---:|---:|---|---|
| AAPL | 378,521 | 502 | 2023-07-05 09:00 | 2025-07-03 21:59 |
| AMZN | 188,431 | 250 | 2024-07-05 09:00 | 2025-07-03 21:59 |
| MSFT | 152,903 | 250 | 2024-07-05 09:00 | 2025-07-03 21:55 |

Those legacy timestamps are `datetime64[us]` and timezone-naive. They are raw
date-bound evidence only; the earlier audit found timezone defects, so this note
does not reinterpret them as UTC or use their clock time for eligibility.

Thirty-seven checked-in experiment configs reference that path. Thirty-one
explicitly declare the 70/15/15 split and 200-row purge; the remainder omit one
or both fields rather than proving a different historical runtime. Declared
start modes include `stream_worker` and `random_episode`. A checked-in config is
not execution proof, and dataset v1 lacks dataset-v2's immutable
manifest/session provenance. The safe interpretation is possible historical
value exposure across its full timestamp span. That span is already inside the
dataset-v2 exclusion, but it must remain in the ledger rather than disappearing
behind the newer source.

The retained exposure inventory SHA-256 is
`74b4fff809a85ebb22c055b3fd0a33e9c2c3e145a044e366d296622ded81162f`.
It covers 163 records, including 42 with unresolved start provenance, explicitly
labels itself an incomplete exposure certificate, and leaves 26 randomized
configs start-unbound within the exact dataset-v2 source. All 163 remain bound
by the inventory-level dataset SHA and therefore cannot reach 2020-2021. See the
[exposure recovery](../evaluation-code-audit-2026-10-06-evidence/exposure-recovery.md)
and [R1 qualification audit](../r1-panel-qualification-2026-10-06.md).

## Candidate-envelope disposition

| Envelope | Disposition | Evidence-based reason |
|---|---|---|
| 2022-01-03 through 2026-09-22 | **Ruled out** | Dataset-v2 training/validation possible exposure plus the preserved old sealed final test; dataset v1 reinforces overlap. |
| 2026-09-23 through 2026-10-07 | **Ruled out as the complete envelope today; prospectively accruing** | Only 11 sessions exist as of the decision date. This cannot provide a broad historical calendar support surface now. No future sufficiency threshold is inferred. |
| 2020-01-01 through 2021-12-31 | **Conditionally cleared for declaration review; recommended candidate** | Nearest two complete calendar years before dataset v2; 505 sessions and 6,060 requested symbol/session slots. Requires complete retrospective exposure clearance, current SIP access, and post-ingestion all-symbol/gap qualification. |
| 2019-01-01 through 2021-12-31 | **Unknown / broader reserve** | 757 sessions, 9,084 requested slots, and six early closes. Q26 support thresholds are unchosen, so the extra year is neither accepted nor rejected. Any use must be chosen before bar acquisition, never added after outcomes. |
| Any other retrospective interval | **Unknown** | Absence from the two known dataset files is not a complete exposure certificate; provider-overlapping sources, manual access, and unresolved retained roots must be audited first. |

“Requested symbol/session slots” is `exchange sessions × 12`; it does not assert
that a symbol has a bar, a full session, or an eligible 260-minute interval.

## Prospective versus retrospective clearance

Prospective-only collection after a sealed declaration gives the strongest
access proof: the ledger begins before the data exist locally. As of 2026-10-07,
however, only eleven post-v2 sessions have accrued. It cannot supply the intended
broad terminal support now, and this note does not invent a minimum wait.

The recommended candidate is therefore retrospective and needs a stronger
clearance standard. At minimum, clearance must be semantic rather than path-
based:

1. enumerate all retained data sources, experiment roots, task pools, notebooks,
   plots, reports, caches, and agent/tool access records that could contain the
   interval;
2. treat the same timestamps obtained through another provider or file as the
   same possible exposure;
3. distinguish mere acquisition/storage from value-bearing use, but record both;
4. require explicit coverage of training, proposer/curriculum, tuning,
   calibration, checkpoint choice, diagnostics, plots, and human inspection;
5. mark absent or unresolved roots as unknown, not clean; and
6. clear the full `[decision - 60 minutes, decision + 200 minutes)` envelope for
   every eventual anchor.

A new dataset hash proves new bytes, not fresh information. An earlier date is
not inherently unexposed.

## Candidate acquisition declaration for review

This is a concrete recommendation for the next declaration, **not permission to
execute it**:

| Field | Candidate value |
|---|---|
| Role | `terminal_screen_only`; separate immutable root and dataset identity |
| Provider/feed | Alpaca historical stock bars / explicit `sip`; no fallback |
| Request dates | 2020-01-01 through 2021-12-31 inclusive |
| Universe | `AAPL, MSFT, AMZN, NVDA, AMD, AVGO, GOOGL, META, SPY, QQQ, IWM, TLT` |
| Bar contract | `1Min`, `split`, `America/New_York`; dividends not applied |
| Terminal session | Reviewed regular-session calendar only; retain raw source pages under the current snapshot contract |
| Eligibility | Exact T+60/T+120 offsets; 60 prior + 200 future consecutive minutes; same symbol/session; early closes fail naturally |
| Prohibited consumers | Training, proposer/curriculum, tuning, calibration on real data, checkpoint choice, development/final panels, manual value inspection, and future capability feedback |
| Opening gate | Q27/role review complete; retrospective clearance accepted; current entitlement confirmed; immutable request/calendar/raw/manifest hashes sealed; Q26 and panel identities frozen; all sixteen cells complete |

Pre-acquisition stop rules:

- stop if the retrospective exposure audit is incomplete or finds value-bearing
  access; do not slide the dates to obtain a clean result without a new decision;
- stop if explicit SIP access for the whole range is unavailable; do not use
  IEX, a paid upgrade, or another provider without separate approval;
- stop if the first-class `terminal_screen_only` role and dual dataset identity
  have not landed and passed independent review; and
- stop if the canonical request, usage classification, sealed storage location,
  allowed metadata projection, and access ledger are not fixed before the first
  bar request.

Post-ingestion metadata-only stop rules:

- stop and keep values sealed if symbols, timestamps, session/calendar binding,
  duplicate/gap accounting, raw-page hashes, or manifest identity do not
  reconcile;
- stop if the fixed eligibility algorithm cannot produce the approved 32
  distinct anchors with at least two per symbol and all four horizons attached
  to each anchor; and
- stop if no predeclared candidate Q26 procedure qualifies on the locked layout
  under its metadata and synthetic-calibration requirements. Preserve each
  method-specific failure; do not inspect returns, choose a favorable subperiod,
  weaken the method, or add 2019 after seeing market/model outcomes.

## Q26 boundary

Nothing here selects an adjacency rule, block length, resampling family,
confidence level, alpha, seed, draw count, support threshold, or multiplicity
correction. The 505-session count is not a power calculation.

Candidate Q26 methods may be tested first on synthetic returns with predeclared
serial dependence, cross-symbol dependence, and heavy tails. Those nulls assess
type-I error, coverage, and stability; they do **not** assess solver
profitability. Simulated-effect power is a separate, explicitly named test. A
method must not be tuned to manufacture the desired significance, and a failed
null design must not be discarded without being reported.

## Next fact-backed GRILL decision

Before any bar acquisition, GRILL should approve or reject the **retrospective
clearance standard** for the 2020-2021 candidate: is an enumerated, hashed scan
of every retained source/experiment root plus explicit manual/agent value-access
attestation sufficient, with every unresolved root failing closed? If not, the
candidate remains unknown and the project must wait for a prospective envelope
or specify stronger evidence. This decision does not select Q26.

## Limitations

- The calendar response can establish session breadth and early closes, not SIP
  bar entitlement or all-symbol bar presence.
- The retained v2 table is a qualification benchmark, not evidence about the
  unacquired 2020-2021 bars.
- The exposure inventory is bounded and incomplete; a clean date comparison is
  not a substitute for a complete ledger.
- No actual bar availability, eligible anchor count, panel, Q26 method, model
  result, or economic claim was produced.
- DEV owns the first-class role implementation and active documentation. This
  dated artifact records research evidence only and does not override source,
  tests, architecture, or the canonical masterplan.
