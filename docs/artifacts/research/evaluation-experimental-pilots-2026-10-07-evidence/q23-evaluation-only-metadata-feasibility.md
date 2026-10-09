# Q23 terminal-screen-only dataset: metadata feasibility contract

Date: 2026-10-07, Europe/Madrid

Status: **route and first-class terminal-screen-only interval role approved,
with code/review pending. Metadata-first planning is approved and the exact
2020-01-01 through 2021-12-31 request is now a reviewed recommendation awaiting
the bounded-clearance/declaration decision. This is not acquisition or terminal-
execution authority.**

The human-selected Q23 route is a **separately declared terminal-screen-only
dataset** with first-class certified terminal-screen-only
eligible intervals and separate training/terminal dataset identities. It is not
adaptive validation, training data, a replacement final test, or an
extension/repartition of the existing snapshot.
The existing final test remains sealed. This note uses no return or model
outcome to choose a date, anchor, layout, or Q26 method.

## Feasibility verdict and current seam gap

The existing read-only Alpaca adapter and immutable snapshot builder can create
a separately hashed source with the required provider semantics and retained
raw provenance. The first-class role and dual identities are approved and DEV
has been dispatched for scoped TDD/review, but the route is **not executable on
the source inspected for this note until that qualification/preflight change
lands and passes review**:

1. [`run_terminal_screen.py`](../../../../itrader/rl_lib/evaluation/run_terminal_screen.py)
   requires the terminal panel dataset hash to equal the completed training
   screen's common dataset hash. A genuinely separate evaluation dataset must
   have a different identity.
2. [`panel_qualification.py`](../../../../itrader/rl_lib/evaluation/panel_qualification.py)
   requires the historical inventory and candidate panel to share one dataset
   hash and requires a non-empty qualified-record inventory. That cannot express
   a newly acquired dataset with no prior adaptive consumers while retaining the
   old exposure ledger as separate evidence.
3. The same qualifier hard-codes `validation`, `0.7/0.15/0.15`, and a 200-row
   purge. The methodology evidence explicitly rejects applying a generic
   percentage split to a short terminal-screen-only slice.
4. `terminal_panel_report` rejects episode rows whose `split` is not
   `validation`; `DataPartition`/`ITraderEnv` accept only the existing split
   vocabulary and `EpisodeEvidence` copies it; the experimental diagnostic
   runner currently admits only train/validation. A preflight-only role would
   therefore fail later or require the forbidden validation relabel.

Therefore the data adapter is reusable, but the approved role must travel
end-to-end: bind `terminal_screen_only` to terminal dataset and certified-
interval identities; use those intervals rather than fractional splitting for
eligibility; retain `DataPartition` continuity/session validation; and preserve
the role/certificate through environment reset/evidence, diagnostics,
qualification, preflight, and reporting. Do not alias the role to `validation`.
The implementation must be independently reviewed before acquisition can lead
to a usable panel. This note does not
reassess the final DEV audit's Q27 findings or claim that a real terminal
executor exists. See the [final DEV evaluation-code audit](../evaluation-code-audit-2026-10-06.md)
and [Q23/Q26 methodology synthesis](../evaluation-quant-methodology-2026-10-06.md).
The accepted panel and reporting semantics remain those in the
[Q23/Q24/Q25 readiness contract](../../../project-state/r1-development-readiness.md).

## Recommended same-stack parity contract

This is the lowest-confound recommendation for the selected route, not proof of
freshness and not authority to acquire data. Any human-approved deviation must
be declared as an acquisition/provider shift factor before outcomes are opened.

| Dimension | Required declaration and check |
|---|---|
| Data role | Approved first-class `terminal_screen_only` usage with certified eligible intervals in a new immutable snapshot root and separate training/terminal identities. Never append it to the training snapshot or relabel the sealed final test. |
| Provider/feed | Alpaca historical stock bars, SIP feed, through the existing read-only adapter. A different provider or feed is a new experimental factor, not freshness by substitution. |
| Bars/corporate actions | One-minute bars, ascending source order, split-adjusted prices, dividends not applied. Retain raw pages and request metadata. Provider omissions remain gaps; never forward-fill them. |
| Universe | Exactly `AAPL, MSFT, AMZN, NVDA, AMD, AVGO, GOOGL, META, SPY, QQQ, IWM, TLT`, with the same spelling and no survivorship-driven replacement. |
| Session/calendar | Retain the builder-supported raw `pre_market`, `regular`, and `after_hours` rows; only `regular` is terminal-eligible. Use `America/New_York` and the same Alpaca exchange-calendar source and normalization semantics. Retain and hash the raw calendar response plus a reviewed aware open/close artifact. Weekdays are not a calendar. |
| Canonical order | Sort the projected frame by symbol and UTC timestamp. Bind every anchor by dataset hash **and** semantic coordinates `(symbol, opaque session_id, aware UTC timestamp)`; never rely on a row index alone. |
| Observation parity | `OPEN,HIGH,LOW,CLOSE,VOLUME`, 60 bars, oldest-to-newest ending at `idx-1`, `market_relative_v1`, and `NoFilter`, exactly as `ohlcv_window_v1` declares. Price columns are relative to the newest close; volume uses `log1p` then within-window standardization; sanitization/clipping stays unchanged. No dataset-fitted normalization is introduced. |
| Task/economics identity | Family `00`, one-minute `TaskSpec`, the frozen horizons `96/128/160/200`, and the accepted 1 bp evidence/task semantics remain unchanged. Q24/Q25 and the named controls are reporting contracts, not panel-selection inputs. |
| Snapshot authority | Immutable Parquet partitioned by session/symbol/year. The manifest records provider, feed, adjustment, timezone, symbols, counts, missing-bar policy, raw-file hashes, creation time, usage/role, and content hash; each hashed raw record retains its own retrieval time and request metadata. DuckDB remains a rebuildable query surface, not authority. |

Source contracts: [market-data contracts](../../../../itrader/inputs/market_data/contracts.py),
[snapshot builder](../../../../itrader/inputs/market_data/snapshot.py),
[Alpaca adapter](../../../../itrader/inputs/market_data/alpaca.py), and
[semantic observation contract](../../../../itrader/models/observation_contract.py).

Provider/feed parity does not prove distributional parity. Retrieval time,
provider revisions, entitlement, calendar revisions, and any acquisition gap
remain recorded limitations even when every declared field matches.

### Local entitlement and retained-metadata check

The 2026-09-24 provider audit records a successful authenticated request for old
explicit Alpaca `feed=sip` bars on this account. On 2026-10-07 the existing
calendar-only adapter call also succeeded with the ignored repository `.env`;
no secret or market value was emitted, and no bar, quote, trade, or account
endpoint was called. This confirms the current credential/calendar path, not
the account plan, present SIP bar entitlement, or a claim that SIP is free. The
first post-approval sealed explicit-SIP bar request must prove the declared feed
and stop on mismatch. No paid upgrade or provider switch is authorized.

The retained snapshot request/manifest fixes Alpaca/SIP, split adjustment,
one-minute bars, the twelve symbols, and 2022-01-03 through 2026-09-22. Its raw
calendar contains 1,185 sessions through 2026-09-23. That older manifest
predates the current builder's per-raw-record retrieval timestamp,
usage-classification, and per-raw-file hash ledger, so a new snapshot must use
the current contract rather than copying the old manifest shape.

## Retained exposure and access history

The old evidence must remain inspectable rather than disappearing when the new
route is used. The DEV recovery audit preserves **512 recorded control rows
reducing to 48 unique starts**. Sidecar symbol identity matched the canonical
twelve-symbol regular-session projection in all 512 records, but within-symbol
ordering remains conditional on an unrecovered dirty source snapshot. The 26
randomized configurations remain start-unbound within the exact dataset-v2
source, so their old-snapshot possible exposure is not narrowed inside that
source. These facts are inspection/exposure evidence, not
candidate rows for the new dataset. See the
[recovery note](../evaluation-code-audit-2026-10-06-evidence/exposure-recovery.md)
and its [coordinate ledger](../evaluation-code-audit-2026-10-06-evidence/exposure-recovery-starts.json).

The new snapshot needs its own append-only access ledger from declaration
through opening. Its content hash/path must be absent from every adaptive or
selection consumer:

| Consumer | Before all sixteen cells complete |
|---|---|
| Solver training and progress monitoring | No bars, features, tasks, metrics, or checkpoints may be derived from the terminal dataset. Training continues to name only the training dataset and inspected development panel. |
| Curriculum/task sampling and proposer | No market summary, task prompt, task pool, reward feedback, filtering, or proposer update may use it. Metadata-only panel construction is a separate sealed process. |
| Hyperparameter tuning, PBT, early stopping, checkpoint choice | No access. Terminal checkpoints remain predetermined; terminal evidence cannot rank or retune cells. |
| Q26 method/calibration | Calendar/layout metadata and synthetic nulls are allowed. Real OHLCV, returns, regimes, policy outputs, and control outputs are forbidden until method and panel identities are sealed. |
| Human/manual inspection | Only the metadata allowlist below. No chart, bar-value, return, direction, volatility, regime, or difficulty inspection. |
| Terminal executor | Access only after the all-sixteen completion gate and after dataset, calendar, exposure, panel, method, checkpoint, and control identities are sealed. |
| Future capability/proposer feedback | Terminal-screen records stay excluded. Any later adaptive capability-to-curriculum source must be separately declared and versioned. |

An acquisition process may mechanically write licensed bar payloads into sealed
storage, but feasibility, selection, and Q26 support jobs may project only:
`symbol`, aware `ts`, `session`, opaque `session_id`, calendar open/close,
row-existence/count, duplicate/gap flags, and provenance hashes. OHLCV,
trade-count/VWAP, returns, labels, regimes, task difficulty, and model/control
outputs are outside the allowlist.

## Outcome-blind eligibility and panel contract

- Preserve the accepted Q23 shape: 32 distinct symbol/session anchors across all
  twelve symbols, at least two per symbol, with the remaining eight assigned by
  a predeclared recorded seed and deterministic canonical ordering.
- Candidate decision offsets remain exactly 60 or 120 minutes after reviewed
  RTH open. Every anchor needs 60 preceding and 200 following one-minute bars,
  all in one symbol and one exchange session, with no duplicates, reversals, or
  missing minute. The final bar must fit wholly before the reviewed session
  close, including early-close semantics.
- Eligibility is assessed at the longest 200-bar outcome. Each accepted anchor
  is crossed with 96/128/160/200, yielding exactly 128 tasks; horizon never
  chooses a different start.
- Treat the full information/outcome envelope as the half-open interval
  `[start - 60, start + 200)`. It must remain inside the declared
  `terminal_screen_only`
  role and be disjoint from every training, development, curriculum, proposer,
  tuning, calibration, and manually inspected value-bearing window.
- Preserve the accepted 200-row symbol-local purge/embargo at an adjacent
  adaptive-data boundary, computed on the combined semantic timeline rather
  than by comparing unrelated file indices. Do not recreate a percentage split
  merely to obtain that purge.
- Select using only calendar, symbol, offset, continuity, eligibility, exposure
  ledger, and the recorded seed. Do not balance direction, return, volatility,
  regime, benchmark difficulty, or model behavior. Post-January-2026 chronology
  is **not** presumed fresh; provenance and zero adaptive access must prove the
  role regardless of recency.
- Freeze the selection algorithm/version, metadata projection hash, exclusions,
  seed, semantic anchor coordinates, row-index mapping, calendar, dataset, and
  final panel hashes before any market or model outcome is opened.

These checks retain [`DataPartition`](../../../../itrader/rl_lib/evaluation/data_partition.py)
continuity/session semantics while requiring the approved first-class
`terminal_screen_only` role rather than silently reusing its fractional split.

## Calendar breadth and Q26 support diagnostics

This section is **candidate Q26 design input only**. Q26 remains wholly pending:
the human has not adopted a grouping, adjacency rule, resampling family,
calibration method, or threshold. If GRILL elects to compare calendar-support
designs before panel freeze, a useful outcome-blind report could show:

- available and selected exchange sessions, unique dates, anchors, symbols,
  offsets, chronological span and gaps, including early-close/DST handling;
- per-symbol anchor counts and exact twelve-symbol/four-horizon completeness;
- maximal 60-lookback-plus-200-outcome windows and the ordered calendar
  components formed under each **candidate** adjacency rule;
- component sizes, spans, gaps, symbol coverage, share of fixed Q24 weight, and
  leave-one-component influence diagnostics;
- the number of candidate contiguous blocks/subsamples over a predeclared grid,
  and whether every replicate can retain all twelve symbols and four horizons;
- a fully defined descriptive effective-support diagnostic, without converting
  it into an unapproved threshold; and
- synthetic-null coverage/type-I-error and instability results for whichever
  synchronized block, fixed-support multiplier, fixed-b subsampling, and/or HAC
  families the human elects to compare.

The accepted Q23 panel identity already keeps the four horizons attached to one
anchor. Whether a Q26 resampler also moves them together, and whether
same-calendar components use synchronized cross-symbol weights, are recommended
candidate rules rather than adopted requirements. Two training seeds remain
descriptive learning replications; a future calendar-resampling choice cannot
manufacture seed power. No grouping, adjacency, block rule/length, interval,
confidence level, coverage tolerance, bootstrap seed/draw count, multiplicity
family, or failure threshold is selected here. The evidence basis is the
[statistical protocol](../evaluation-quant-methodology-2026-10-06-evidence/statistical-protocol-primary-sources.md)
and [benchmark/environment survey](../evaluation-quant-methodology-2026-10-06-evidence/benchmark-environment-primary-sources.md).

## Required receipts before outcome access

1. A pre-acquisition declaration with the chosen role/partition authority,
   provider/feed/adjustment/timeframe/calendar semantics, fixed symbols,
   acquisition-envelope policy, prohibited consumers, storage/access policy,
   and stop/failure rules.
2. A post-ingestion immutable snapshot manifest, raw page/calendar hashes,
   normalized calendar artifact, metadata-only quality report, and independent
   parity review. Acquisition itself is outside this note.
3. A complete access/exposure ledger covering training, development monitoring,
   curricula, proposers, task pools, tuning/PBT, checkpoint selection,
   calibration, manual inspection, and later consumers.
4. A metadata eligibility receipt and Q26 support/calibration report. If either
   fails its predeclared rule, no anchor is selected and no fallback is invented.
5. A hashed 128-task panel and qualification certificate, plus a screen
   declaration that binds **separate** training and terminal dataset identities.
   Independent review must confirm the all-sixteen opening gate and that the
   terminal runner reads neither dataset before its authorized stage.

## Limitations

- The exact 2020-01-01 through 2021-12-31 request is now recommended by the
  follow-on metadata report, but has not passed bounded retrospective clearance
  or been authorized. No anchor, seed, final panel, Q26 default, dataset value,
  or final test is opened.
- Metadata feasibility does not guarantee that 32 anchors can also provide
  enough non-dominant calendar support for confirmatory Q26 inference.
- A separate snapshot removes the old snapshot's unresolved randomized-start
  ambiguity only if declaration time, acquisition provenance, storage isolation,
  and the complete access ledger establish zero adaptive exposure.
- The 512-row/48-start recovery remains bounded, conditional historical
  evidence. It is neither discarded nor promoted into full source-bound truth.
- Same-provider parity does not remove temporal/provider shift, corporate-action
  revision risk, licensed-data constraints, sparse sessions, or the limits of
  a two-seed screen.
- Q24/Q25 point reporting, benchmark identities, and metadata support do not
  substitute for an approved Q26 uncertainty contract or economic promotion.

## One next human decision before acquisition

Metadata-first sequencing is approved. The follow-on calendar/access/exposure
audit recommends the nearest two complete pre-v2 calendar years: request
2020-01-01 through 2021-12-31, whose first returned session is 2020-01-02. It is
a backward-time, cross-period retrospective representation screen, not
prospective forward-time validation or live-economic promotion.

**Approve `bounded_retrospective_clearance_v1` and the exact
`q23_terminal_screen_only_20200101_20211231_v0` declaration, subject to a human
manual/external-access statement and every stop rule; or require prospective-
only accumulation instead?**

The complete evidence, declaration fields, failure rules, and single decision
are in the [metadata-first envelope report](../evaluation-metadata-first-envelope-2026-10-07.md).
Known v1/v2 dates do not overlap the candidate, but opaque retained sources and
unlogged manual/external access prevent a freshness certificate. Approval of the
declaration would not bypass the pending Q23 role review, current SIP proof,
post-ingestion metadata qualification, synthetic Q26 calibration, final
panel/Q26 seal before training, or all-sixteen completion gate.
