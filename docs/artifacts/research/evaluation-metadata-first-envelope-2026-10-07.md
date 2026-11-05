# Q23 metadata-first historical envelope: feasibility and draft declaration

Date: 2026-10-07, Europe/Madrid

Branch: `agent/Watson-dev`

Status: **metadata-first planning is approved; the exact 2020-01-01 through
2021-12-31 acquisition declaration below is a research recommendation awaiting
human review. No bar data was acquired, no market or model outcome was opened,
and no Q26 method or numerical gate was selected.**

## Decision result

The recommended historical envelope is the two complete exchange-calendar years
immediately before dataset-v2:

```text
2020-01-01 through 2021-12-31, inclusive request bounds
Alpaca / SIP / 1Min / split-adjusted / America/New_York
builder-supported sessions retained; regular session only is terminal-eligible
fixed twelve-symbol universe
```

The first returned exchange session is 2020-01-02. The interval contains 505
exchange sessions and three venue-calendar early closes, hence 6,060 requested
symbol/session slots before any symbol-level bar qualification. It is the
closest contiguous pair of complete pre-v2 calendar years. The rule uses only
source boundaries and calendar topology; it does not use returns, volatility,
regimes, actions, checkpoint scores, or method performance. Choosing two years
is a reviewable design judgment, not evidence that this duration will support a
Q26 method; synthetic calibration must test that separately.

The interval is a **recommended candidate; retrospective clearance is still
pending**. It is disjoint from the retained dataset-v2 interval and the raw date
coverage of legacy dataset-v1, but retained artifacts cannot prove the absence
of deleted copies, unlogged external tools, or human chart inspection.
Acquisition cannot begin until the bounded retrospective-clearance standard and
exact declaration are approved and the first-class `terminal_screen_only` role
passes DEV review. The first post-approval sealed explicit-SIP request is the
entitlement check and first acquisition step; outcome inspection or use remains
blocked until all downstream gates pass.

| Item | Current decision | Boundary |
|---|---|---|
| Metadata-first sequence | Approved | Envelope metadata first; synthetic Q26 calibration second; real outcomes last |
| Exact envelope | Recommended request: 2020-01-01 through 2021-12-31; first session 2020-01-02 | Human review pending; not collection authority |
| Q23 role | First-class `terminal_screen_only` with separate dataset identity is approved | DEV implementation and independent review pending |
| Current calendar access | Confirmed on 2026-10-07 | Does not prove SIP bar entitlement |
| Current SIP entitlement | Unknown by design | Calendar has no feed/plan claim; no bar endpoint was called; the first post-approval sealed request must prove the declared feed |
| Q26 calibration sequence | Synthetic null calibration, then separately named simulated-effect power, is approved as the research order | No method, calendar unit, block choice, alpha, coverage target, seed, draw count, threshold, or multiplicity rule is adopted; no calibration run occurred |
| Optional volatility pilot | Separate and still unset | Not a base-Q24 blocker when disabled |
| `CapabilityProfile` | Future online/shadow contract only | Offline work may emit identity-compatible evidence for a later versioned transfer; it does not create an operational profile or feed `r_prop` |
| Acquisition / outcome access | Not authorized | No bars, final test, terminal panel, replay, or model run was opened |

## Architecture and authority boundary

This is a dated research artifact. The
[masterplan](../../plans/itrader_masterplan.md),
[R1 readiness contract](../../project-state/r1-development-readiness.md), and
current source/tests remain operational authority. Some deeper active-doc
passages still use the older `validation-only` label; the newest decision log
and top-level status correctly require first-class `terminal_screen_only`, and
DEV must reconcile those passages without aliasing the roles. DEV owns the role
contract, qualification code, terminal runner, active documentation, and
implementation review. This report owns only metadata/access/exposure evidence
and a draft declaration for the next human decision.

```text
retained metadata + exposure ledger + calendar-only provider response
                              |
                              v
                 reviewed acquisition-envelope seal
                              |
                              v
            sealed terminal_screen_only snapshot (future)
                              |
                              v
             metadata-only candidate-layout lock
                              |
                              v
                  synthetic Q26 calibration
                              |
                              v
         human Q26 decision + final panel/Q26 joint seal
                              |
                              v
             all-16 training completion gate -> outcomes
```

No artifact from this path may enter `TaskSpec` generation, proposer/curriculum
updates, PPO reward/loss, tuning, PBT, early stopping, checkpoint selection,
the old sealed final test, or an operational `CapabilityProfile`. A later
offline-to-online calibration adapter would require a separate versioned
decision and an adaptive-validation source that excludes terminal evidence.

## Outcome-blind source and calendar receipts

### Retained identities

| Evidence | Identity and scope |
|---|---|
| Dataset-v2 build request | SHA-256 `e2fd032d9af90dc74a2ae2b1762a96ef8b7d47c49bd957706f9e646ceee4df62`; Alpaca/SIP, 1Min, split, twelve symbols, 2022-01-03 through 2026-09-22 |
| Dataset-v2 manifest file | SHA-256 `4c6202883a46cbe5bfbf4986003dcd1eb05b3738ac2d48eeb42d9a57fe974ca5` |
| Dataset-v2 content | SHA-256 `fac41f6d42f8c5b1eac739755667bb38407461f82c168d7955de26bb2e5fb79c`; 10,370,064 total rows and 5,515,061 regular-session rows |
| Retained raw calendar | Gzip SHA-256 `0bb92722b805a05af29f11db1922081b279e6b5142265f4ad371e3266ac44a72`; 1,185 sessions through 2026-09-23 |
| Legacy dataset-v1 parquet | SHA-256 `5f2e449548532ff71f419431df86962a2aedf7275d7bff5592729bbb8a900f73`; raw naive timestamps span 2023-07-05 through 2025-07-03 across only AAPL, AMZN, and MSFT |
| Retained exposure inventory | SHA-256 `74b4fff809a85ebb22c055b3fd0a33e9c2c3e145a044e366d296622ded81162f`; 163 configs, including 42 with unresolved start provenance; all 163 are bound by the inventory-level dataset SHA to dataset-v2 and therefore date-disjoint from 2020-2021; explicitly not a complete exposure certificate |
| Recovered control evidence | 512 episode rows reduce to 48 starts; 26 randomized configurations remain start-unbound within the exact dataset-v2 source |

The current builder is stronger than the old v2 manifest: every raw response
records retrieval time and request metadata; the final manifest records usage,
missing-bar policy, raw-file hashes, creation time, and content identity. A new
terminal snapshot must use that contract rather than copy the old manifest.

### Fresh calendar-only provider checks

The existing
[`AlpacaMarketDataSource.fetch_calendar`](../../../itrader/inputs/market_data/alpaca.py)
path succeeded with locally configured credentials on 2026-10-07. No bar,
quote, trade, account, or model endpoint was called and no credential value,
prefix, length, or digest was emitted.

| Calendar receipt | Result |
|---|---|
| 2016-01-01 through 2026-10-07 | 2,706 sessions; 2,685 standard sessions; 21 early closes; no non-09:30 opens; payload SHA-256 `1b64dc77a9dada951d4a5fa3d3d0460b2224c3836c48d7e42ce00a5be7c3ad1f` |
| Exact retained-calendar re-fetch | 1,185 rows; current and retained canonical payload SHA-256 both `82b35d5b218cb02dc364b7de9a680eab89955c8d4aa1154de807005d37e4d080` |
| 2020-01-01 through 2021-12-31 calendar slice | First/last sessions 2020-01-02 / 2021-12-31; 505 sessions; three early closes |
| 2019-01-01 through 2021-12-31 comparison slice | 757 sessions; six early closes; broader reserve, not an approved fallback |
| 2026-09-23 through 2026-10-07 | 11 sessions; clean chronology alone does not supply a broad current terminal envelope |

The current calendar response proves that the adapter/credential path works and
that Alpaca returned exchange-session topology. It cannot prove all-twelve bar
presence, minute completeness, a market-data subscription, or freshness.
[The access evidence](evaluation-metadata-first-envelope-2026-10-07-evidence/alpaca-access-and-calendar.md)
records the request and response hashes plus the first-party Alpaca references.

## All-twelve-symbol support: known versus unknown

The fixed universe remains:

```text
AAPL, MSFT, AMZN, NVDA, AMD, AVGO, GOOGL, META, SPY, QQQ, IWM, TLT
```

The retained v2 metadata provides a useful qualification baseline without making
v2 eligible for the new terminal role. All twelve symbols have rows on all
1,184 exchange dates through the snapshot end, but minute completeness differs
by symbol. Across all twelve, 622 v2 dates support both accepted decision
offsets with a continuous 60-bar lookback and 200-bar longest outcome. None of
the nine early closes supports either full all-twelve offset. Those facts show
why calendar membership alone is insufficient and why no forward fill is
permitted. Same-date all-twelve completeness is a useful diagnostic, not a
panel-selection requirement: the accepted contract is 32 distinct
symbol/session anchors across all twelve symbols, with at least two per symbol.

For the recommended 2020-2021 envelope, only venue-calendar facts are known:
505 sessions, 6,060 requested symbol/session slots, and three early closes.
Actual rows, symbol-session presence, gaps, duplicates, complete 260-minute
windows, and offset eligibility are unknown until a declared sealed acquisition.
The metadata-only post-ingestion audit must report all of them for every symbol.
Failure is reported; the interval is not silently widened and missing bars are
not imputed.

The complete per-symbol v2 comparison is retained in the
[calendar receipt](evaluation-metadata-first-envelope-2026-10-07-evidence/calendar-metadata-receipt.json).
It is context for the validator, not a candidate terminal panel.

## Candidate contiguous-envelope disposition

| Envelope | Metadata disposition | Reason |
|---|---|---|
| Legacy/v2 overlap, 2022-01-03 through 2026-09-22 | **Ruled out** for the separate terminal role | Dataset-v2 fed training/development/evaluation consumers; randomized validation starts remain unresolved within that source; the existing final partition remains sealed |
| Legacy dataset-v1 raw dates, 2023-07-05 through 2025-07-03 | **Ruled out** | Known historical value-bearing source, only three symbols, weak provenance, and overlap with v2 |
| Post-v2, 2026-09-23 through 2026-10-07 | **Conditionally clean but presently too short for the declared historical-envelope purpose** | Only 11 completed exchange sessions as of the decision date; prospective accumulation remains the strongest future clearance route |
| 2020-01-01 through 2021-12-31 | **Recommended candidate; clearance pending** | Nearest two complete pre-v2 calendar years; first returned session 2020-01-02; 505 sessions; disjoint from retained v1/v2 date coverage; still requires bounded retrospective clearance, current SIP proof, and post-ingestion metadata qualification |
| 2019-01-01 through 2021-12-31 | **Broader reserve / unknown** | 757 sessions and six early closes, but no outcome-blind reason has yet approved the extra year; it cannot become an automatic fallback after a calibration failure |
| Earlier history or another provider/feed | **Unknown and outside this declaration** | Requires a new metadata, access, temporal-shift, and provenance review |

“Ruled out” is about use as the new separate terminal source, not permission to
delete historical evidence. “Clearance pending” means exactly that: the known
v1/v2 date ranges do not overlap, but opaque retained sources and manual/external
access still need bounded resolution or attestation. A new file hash, an older
date, or a different storage path cannot establish freshness.

## Draft bounded retrospective-clearance standard

The proposed `bounded_retrospective_clearance_v1` is deliberately stronger than
date non-overlap and weaker than an impossible claim of omniscience. It would
permit a historical envelope only when all of the following are recorded:

1. **Semantic non-overlap:** every full `[start-60, start+200)` symbol/UTC window
   lies outside every retained training, validation, final-test, proposer,
   tuning, calibration, checkpoint-selection, diagnostic, plot, and manually
   inspected value-bearing interval.
2. **All known sources:** the scan covers dataset-v2, legacy dataset-v1, every
   retained experiment/config root, recovered and unresolved exposure records,
   repository history, local analysis artifacts, and declared external tools or
   copies. Legacy-v1 is not omitted merely because v2 superseded it.
3. **Uncertainty is exclusionary:** an unresolved source that might overlap the
   proposed dates blocks clearance. Start-unbound randomness cannot be narrowed
   from a convenient seed reconstruction, while an exact dataset identity may
   still bound the dates that randomness could have touched.
4. **Consumer completeness:** the ledger names training, progress monitoring,
   proposer/curriculum, tuning/PBT, calibration, checkpoint choice, diagnostics,
   plots, human inspection, agents, and later online/capability consumers.
5. **Human boundary statement:** the owner records whether any unlogged chart,
   notebook, broker UI, external drive, or prior copy exposed 2020-2021 values.
   Filesystem absence cannot answer this on its own.
6. **Append-only continuation:** declaration, acquisition, metadata selection,
   Q26 calibration, terminal opening, and every rejected/failed attempt enter one
   access ledger. Hashes establish identity and drift; they do not prove truth.

The present evidence supports proposing 2020-2021 for clearance review because
it is disjoint from known exact-hash v1/v2 file dates. In particular, the 42
unresolved v2 inventory records are source-bound to dataset-v2; their uncertainty
does not reach backward into 2020-2021. Clearance still has not passed: 120 older
run configs bind only a canonical path without a historical hash, three retained
configs name a now-absent external checkout, twelve task pools and six candidate
ledgers lack dataset/time coordinates, and the human boundary statement cannot
be supplied by filesystem evidence. No report can prove that an unretained
external copy never existed.

## Draft acquisition declaration for review

Declaration ID: `q23_terminal_screen_only_20200101_20211231_v0`

| Field | Frozen proposed value |
|---|---|
| Purpose | Separate Q23 terminal-screen-only evaluation source; never adaptive validation or final-test replacement |
| Provider / feed | Alpaca historical stock bars / explicit `sip`; no fallback feed |
| Exact inclusive request bounds | 2020-01-01 through 2021-12-31; first calendar session expected 2020-01-02 |
| Universe | The twelve symbols above, in that fixed order |
| Time / adjustment | `1Min`, split-adjusted, `America/New_York`; dividends not applied |
| Sessions | Retain the builder's supported `pre_market`, `regular`, and `after_hours` rows as immutable raw evidence; qualify and expose only `regular` rows to the terminal role |
| Missing data | Preserve provider omissions; no forward fill, synthetic bars, symbol substitution, or date extension |
| Storage identity | New immutable snapshot root; usage classification `terminal_screen_only_q23_evaluation_only`; raw calendar/pages and request metadata retained and hashed |
| Role identity | First-class `terminal_screen_only`, separate training and terminal dataset identities, certified eligible-interval hash; never relabel as `validation` |
| Permitted pre-opening projection | Symbol, aware UTC timestamp, session, opaque session ID, calendar open/close, row existence/count, duplicate/gap flags, and provenance hashes |
| Forbidden pre-opening fields | OHLCV values, trade count/VWAP, returns, directions, volatility/regime labels, task difficulty, actions, logits, rewards, policy/control outcomes |
| Panel rule | Accept 32 distinct symbol/session anchors across the twelve symbols, at least two per symbol and the remaining eight by a predeclared recorded seed; only the accepted 60/120-minute decision offsets; every anchor must have a continuous same-session 60-bar lookback and full 200-bar future support before crossing 96/128/160/200 |
| Outcome gate | No outcome access until declaration, snapshot, exposure, layout, Q26 method, all-16 completion, checkpoint, control, and producer identities are sealed |

This declaration is **not yet approved** and must not be passed to the snapshot
builder as collection authority.

### Stop rules

Stop without fallback or reinterpretation if:

- the retrospective-clearance review or human boundary statement is incomplete;
- the DEV role/certificate seam has not landed and passed independent review;
- the first sealed explicit-SIP request fails authentication/entitlement,
  returns a mismatched feed, defaults to another feed, or would require a paid
  upgrade;
- any provider, feed, timeframe, adjustment, universe, date, or session policy
  differs from the declaration;
- raw request/response retention, retrieval time, usage classification, or hashes
  are missing;
- metadata finds duplicate/reversed timestamps, cross-session windows, or
  insufficient eligible anchors to satisfy the frozen 32-anchor/twelve-symbol
  rule or the later predeclared Q26 calendar-support gate;
- a prohibited consumer or human opens value-bearing data before authorization;
- no predeclared candidate Q26 procedure qualifies on the locked candidate
  layout under the approved synthetic calibration contract; or
- anyone proposes redrawing dates, dropping a difficult symbol/null family, or
  widening the envelope in place after seeing calibration or market results.

A failed declaration is never edited in place to make a result pass. If its
predeclared metadata support fails, or no candidate procedure qualifies on its
locked layout, GRILL may authorize a genuinely new declaration only with an
independent support rationale, preserved failed receipt, new hash/layout lock,
and complete recalibration. A method-specific failure is preserved against that
method, not mislabeled as proof that the layout itself is inadequate. This is
not permission to iterate envelopes until a preferred result appears.

## Q26 synthetic-calibration boundary

The approved sequence permits candidate-method work on synthetic outcomes after
the envelope and metadata layout are frozen. It does not approve a method or
numbers. The
[Q26 calibration contract](evaluation-metadata-first-envelope-2026-10-07-evidence/q26-synthetic-calibration-boundary.md)
requires two distinct studies:

1. **Synthetic null calibration:** empirical Type-I error, interval coverage,
   numerical/support failure, and stability under predeclared serial,
   cross-symbol, within-anchor/horizon, heavy-tail/asymmetric, heterogeneous-
   scale, and combined dependence families.
2. **Separately named simulated-effect power:** rejection probability under
   predeclared nonzero synthetic Q24 effects, reported with Monte Carlo
   uncertainty and never described as solver profitability or expected return.

Every null family, method attempt, invalid replicate, and failure remains in the
ledger. Power cannot rescue failed null calibration. The procedure may not tune
panel dates, remove a failing null, change block settings until Type-I/coverage
gates pass, later change them until terminal significance appears, or select the
narrowest interval. If no predeclared candidate procedure qualifies on the
locked layout, the neutral result is `Q26_UNRESOLVED_FOR_LAYOUT`; GRILL decides
what happens next without opening market outcomes.

Still-open numerical decisions include the confirmatory contrast family,
calendar unit, candidate methods, synthetic generators/parameters, block or
bandwidth rule, interval construction, alpha, coverage/error tolerances,
replication count, RNG streams/seeds, multiplicity family, support/failure
thresholds, and the limited simulated-effect grid. Two training seeds remain a
descriptive mechanics/attribution screen, not population-level inference.

## Ordered gates after the next decision

1. Human approves or rejects the retrospective-clearance standard and exact
   draft declaration below; no acquisition yet if either changes.
2. DEV completes and independently reviews the first-class role, dual-identity,
   certificate, qualification, evidence, and terminal-reporting path.
3. The approved declaration is serialized and hashed; storage/access policy and
   append-only ledger are active before the first provider call.
4. The existing adapter makes the first explicit-SIP request into sealed storage.
   An entitlement/feed mismatch stops the run; there is no IEX or paid fallback.
5. Acquisition completes under the exact request. A metadata-only independent
   audit either qualifies the snapshot or fails it without reading values.
6. The frozen outcome-blind algorithm/seed selects one **candidate layout** and
   locks its coordinates, windows, weights, calendar, exposure, dataset, and
   code hashes solely for synthetic calibration.
7. Candidate Q26 procedures run on predeclared synthetic nulls and separate
   simulated effects over that exact candidate layout. Failed batteries are
   retained.
8. GRILL selects Q26 or leaves the screen descriptive. The final panel and Q26
   contract are jointly sealed before any of the sixteen target-screen training
   runs; this final seal cannot inspect market outcomes.
9. Only after all sixteen target cells complete and checkpoint, control,
   producer, dataset, panel, and method identities pass preflight may the
   terminal executor open sealed values. The old final test remains sealed.

Autoresearch is not active in these gates. A later bounded loop may optimize
runtime or memory on fixed synthetic fixtures while requiring byte/semantic
equivalence. It may never optimize dates, methods, block choices, alpha,
thresholds, scenario membership, or conclusions.

## One next GRILL decision

**Approve `bounded_retrospective_clearance_v1` and the exact
`q23_terminal_screen_only_20200101_20211231_v0` declaration, subject to the human
external/manual-access statement and every stop rule above; or require
prospective-only accumulation instead?**

Recommendation: approve the declaration conditionally, not retrospective
clearance itself. The proposed interval is
the nearest reproducibly defined two-year block outside both retained datasets,
has substantially more calendar breadth than the eleven currently available
post-v2 sessions, and can be rejected without market inspection if clearance,
entitlement, or metadata support fails. This recommendation is not a freshness
certificate and does not pre-answer Q26.

Only after this one decision should GRILL ask which Q24 contrast family is the
sole confirmatory Q26 target.

## Limitations

- No bar endpoint was called, no bar value was read, and no acquisition occurred.
- Current SIP entitlement remains unknown; historical old-SIP success is not a
  plan or pricing claim.
- Venue sessions do not prove per-symbol bar completeness or eligible anchors.
- The retained exposure inventory is bounded and contains unresolved records; it
  cannot certify deleted, remote, or manually viewed material.
- Because dataset-v2 begins in 2022, this pre-v2 block is a backward-time,
  cross-period retrospective representation screen. It is not prospective
  forward-time validation, a live-economic promotion gate, or evidence that the
  selected solver will generalize to current markets. Its temporal/provider
  shift must remain explicit in every interpretation.
- Synthetic calibration can reject bad inference procedures but cannot prove its
  generators match markets, create training-seed replication, or establish a
  profitable solver.
- The recommended interval, role implementation, snapshot, panel, Q26 method,
  terminal execution, and any operational capability inference all remain
  uncompleted.

## Evidence and provenance

- [Alpaca access and calendar evidence](evaluation-metadata-first-envelope-2026-10-07-evidence/alpaca-access-and-calendar.md)
- [Calendar and value-blind metadata receipt](evaluation-metadata-first-envelope-2026-10-07-evidence/calendar-metadata-receipt.json)
- [Retained identities and exposure evidence](evaluation-metadata-first-envelope-2026-10-07-evidence/retained-identities-and-exposure.md)
- [Q26 synthetic-calibration boundary](evaluation-metadata-first-envelope-2026-10-07-evidence/q26-synthetic-calibration-boundary.md)
- [Q23 role and metadata-feasibility contract](evaluation-experimental-pilots-2026-10-07-evidence/q23-evaluation-only-metadata-feasibility.md)
- [Historical exposure recovery](evaluation-code-audit-2026-10-06-evidence/exposure-recovery.md)
- [Statistical protocol primary sources](evaluation-quant-methodology-2026-10-06-evidence/statistical-protocol-primary-sources.md)
- [Current architecture](../../architecture/itrader_system_architecture.md)
- [Canonical masterplan](../../plans/itrader_masterplan.md)
