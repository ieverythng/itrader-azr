# Alpaca access and calendar evidence

Date: 2026-10-07, Europe/Madrid

Scope: metadata-first Q23 acquisition-envelope planning only

Status: **calendar/access feasibility confirmed within a narrow boundary;
current SIP bar entitlement remains untested by design**

## Outcome

The repository can reuse its existing read-only Alpaca adapter for the approved
metadata-first route. On 2026-10-07 it successfully fetched Alpaca's US market
calendar through the configured calendar endpoint without requesting bars or
printing credentials. A fresh request for the retained snapshot's exact
2022-01-03 through 2026-09-23 calendar interval produced the same canonical
payload SHA-256 as the retained raw calendar.

This does **not** certify the account's current SIP bar entitlement. The
calendar endpoint does not name a market-data plan or feed, the adapter has no
entitlement method, and no bar, quote, trade, snapshot, or websocket request was
made in this pass. The retained 2026-09-24 snapshot proves that explicit old
`feed=sip` requests worked for this account at that time; it does not prove the
current plan, current recent-SIP access, price, or licensing terms. In
particular, this note does not claim that SIP is free.

No provider switch, paid upgrade, bar acquisition, market-value inspection, or
panel selection is recommended or authorized here.

## Repository seam inspected

[`AlpacaMarketDataSource`](../../../../itrader/inputs/market_data/alpaca.py)
has exactly two read paths:

- `fetch_calendar(start, end)` calls
  `https://paper-api.alpaca.markets/v2/calendar`; and
- `iter_bar_pages(request)` calls
  `https://data.alpaca.markets/v2/stocks/bars` with an explicit feed,
  adjustment, timeframe, and date interval.

Only the first path was exercised. This preserves the architectural boundary:
the adapter is a read-only historical-data source, not a broker, executor,
risk engine, proposer, or solver component.

The current
[`build_market_snapshot`](../../../../itrader/inputs/market_data/snapshot.py)
would persist each raw response with `retrieved_at_utc` and place hashes for all
raw files in the final manifest. The retained dataset-v2 snapshot was created
with an older manifest shape: its raw files have no per-file retrieval time,
its request and manifest omit `usage_classification`, and its manifest has no
`raw_files` hash ledger. A future separately declared `terminal_screen_only`
snapshot must use the current builder contract; copying the old manifest is not
sufficient.

## Credential and request boundary

The following checks exposed no secret values:

| Check | Result |
|---|---|
| Repository `.env` | Ignored by Git; non-empty assignments exist for both `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` |
| Process environment before the check | Both Alpaca variables absent |
| Credential loading | Values read locally only to instantiate the existing adapter; no value, prefix, length, or digest was printed or retained |
| Provider calls | Calendar endpoint only; no market-data or account endpoint |
| Market outcomes | None read or printed |

The successful calendar response shows that this configured request path is
usable. It should not be overstated as a credential or SIP entitlement
certificate because the calendar response itself carries no plan/feed claim.

## Fresh calendar receipt

The adapter fetched the inclusive interval 2016-01-01 through 2026-10-07 at
`2026-10-07T21:43:43.304596+00:00`.

Hashes use UTF-8 canonical JSON with sorted object keys and compact separators.
They are hashes of the request record and response payload, respectively; they
are not hashes of OHLCV data.

| Field | Value |
|---|---|
| Canonical request | `{"end":"2026-10-07","endpoint":"calendar","start":"2016-01-01"}` |
| Request SHA-256 | `9b7ad05b36c01f8e80ede7b621f6390c98b35d54e7c902a60f3aa455ced64f9b` |
| Calendar payload SHA-256 | `1b64dc77a9dada951d4a5fa3d3d0460b2224c3836c48d7e42ce00a5be7c3ad1f` |
| Returned sessions | 2,706 |
| First / last returned trading date | 2016-01-04 / 2026-10-07 |
| Standard `09:30`-`16:00` sessions | 2,685 |
| `09:30`-`13:00` early closes | 21 |
| Non-`09:30` opens | 0 |

The 21 early-close dates in this response were:

```text
2016-11-25
2017-07-03, 2017-11-24
2018-07-03, 2018-11-23, 2018-12-24
2019-07-03, 2019-11-29, 2019-12-24
2020-11-27, 2020-12-24
2021-11-26
2022-11-25
2023-07-03, 2023-11-24
2024-07-03, 2024-11-29, 2024-12-24
2025-07-03, 2025-11-28, 2025-12-24
```

Alpaca's first-party calendar reference says this endpoint returns market days
and their actual open/close times, including early closes, and supports bounded
inclusive start/end queries ([Alpaca US Market Calendar API](https://docs.alpaca.markets/us/reference/legacycalendar)).
The calendar establishes session topology only. It cannot establish that every
symbol has a bar in every minute, that a specific historical feed is accessible,
or that any date is unexposed to training or analysis.

## Exact retained-calendar comparison

At `2026-10-07T21:44:08.100375+00:00`, a second calendar-only request reproduced
the retained snapshot interval exactly:

| Field | Value |
|---|---|
| Request | `{"end":"2026-09-23","endpoint":"calendar","start":"2022-01-03"}` |
| Current response rows | 1,185 |
| Current canonical payload SHA-256 | `82b35d5b218cb02dc364b7de9a680eab89955c8d4aa1154de807005d37e4d080` |
| Retained canonical payload SHA-256 | `82b35d5b218cb02dc364b7de9a680eab89955c8d4aa1154de807005d37e4d080` |
| Exact canonical payload match | yes |
| Retained gzip-file SHA-256 | `0bb92722b805a05af29f11db1922081b279e6b5142265f4ad371e3266ac44a72` |
| Retained first / last date | 2022-01-03 / 2026-09-23 |
| Retained early closes | 9 |

This is evidence that the endpoint returned the same calendar payload for that
interval on this check. It is not a guarantee against a later provider revision
and does not transform the old snapshot into a fresh terminal dataset.

The ad hoc current response was intentionally not written as a new data
snapshot. A reviewed acquisition run must retain its own raw calendar response,
retrieval timestamp, request, and file hash through the snapshot builder.

## Retained SIP source evidence

The authoritative retained dataset path is
`itrader/data/datasets/dataset_v2_20220103_20260922_alpaca_sip_split`.
No market values were printed or used in this inspection.

| Identity | Receipt |
|---|---|
| Build request file SHA-256 | `e2fd032d9af90dc74a2ae2b1762a96ef8b7d47c49bd957706f9e646ceee4df62` |
| Manifest file SHA-256 | `4c6202883a46cbe5bfbf4986003dcd1eb05b3738ac2d48eeb42d9a57fe974ca5` |
| Manifest content SHA-256 | `fac41f6d42f8c5b1eac739755667bb38407461f82c168d7955de26bb2e5fb79c` |
| Declared source | Alpaca, explicit `sip`, `1Min`, split adjustment, `America/New_York` |
| Declared interval | 2022-01-03 through 2026-09-22 |
| Declared universe | AAPL, MSFT, AMZN, NVDA, AMD, AVGO, GOOGL, META, SPY, QQQ, IWM, TLT |
| Retained source pages | 1,067 bar pages plus one calendar response |
| Snapshot rows | 10,370,064 total; 5,515,061 regular-session rows |

Two value-blind sentinel checks confirm that retained, non-empty provider pages
carry explicit SIP requests at opposite ends of the snapshot:

| Raw receipt | Safe request metadata | Record count | File SHA-256 |
|---|---|---:|---|
| `000001_AAPL_2022_bars.json.gz` | AAPL; 2022-01-03T05:00:00Z to 2023-01-01T05:00:00Z; `1Min`; `split`; `sip`; ascending; limit 10,000 | 10,000 | `d4c7d5af0b815a4786ea983551545d42426a4734c8c5ee02cfe1b5db307a1bc1` |
| `001067_TLT_2026_bars.json.gz` | TLT; 2026-01-01T05:00:00Z to 2026-09-23T00:00:00Z; `1Min`; `split`; `sip`; ascending; limit 10,000 | 3,658 | `a3cf189862cc6ce84e9d875f738ac5e7ba54ec135a80b7d4ff001697ed5ed63d` |

These retained responses are strong evidence that the account could obtain old
explicit SIP bars during the 2026-09-24 acquisition. Because the old raw record
shape omitted `retrieved_at_utc`, the date is supported by the snapshot audit
and completed artifact chronology rather than an individual raw-page timestamp.
The evidence cannot certify current entitlement.

See the contemporaneous
[dataset-v2 audit](../dataset-v2-r0-audit-2026-09-24.md) and
[provider evaluation](../market-data-provider-evaluation-2026-09-23.md).

## What Alpaca currently documents—and why it is not an entitlement receipt

Alpaca's current first-party pages do not support a categorical account-specific
claim from documentation alone:

- The current plan table describes Basic as free, real-time IEX-only, with
  stock history since 2016 and a latest-15-minute historical limitation; Algo
  Trader Plus adds unrestricted current consolidated coverage
  ([About Market Data API](https://docs.alpaca.markets/us/docs/about-market-data-api)).
- The Market Data FAQ says a historical query whose `end` is at least fifteen
  minutes old can query SIP without a subscription, while recent SIP requests
  require the paid plan
  ([Market Data FAQ](https://docs.alpaca.markets/us/v1.1/docs/market-data-faq)).
- A separate historical-feed overview says IEX is the only feed usable without
  a subscription
  ([Historical Stock Data](https://docs.alpaca.markets/us/v1.1/docs/historical-stock-data-1)).

The last two statements are not fully consistent. Therefore the safe research
interpretation is operational and account-specific: the retained snapshot proves
past old-SIP access, while a current explicit SIP bars response would be needed
to certify current access for the declared interval. That request is deliberately
deferred because the present authorization is metadata-first and forbids bar
queries before the exact acquisition envelope is declared and reviewed.

Neither the existing adapter nor the official references reviewed exposes a
non-market-data entitlement check that can replace that eventual gated probe.
The generic account endpoint is not used here and, in any case, is not part of
the existing market-data adapter contract.

## Certification boundary for the acquisition declaration

The following claims are supported now:

1. the existing adapter and snapshot builder can retrieve and normalize
   calendar topology without a provider change;
2. its configured local request path returned a calendar successfully on
   2026-10-07;
3. the retained calendar is hash-stable against a same-range request on that
   date; and
4. the account historically returned non-empty explicit old-SIP bar pages for
   the current twelve-symbol dataset-v2 source.

The following claims are **not** supported now:

1. the account is presently subscribed to Algo Trader Plus or any named tier;
2. the account can presently retrieve SIP bars for an undeclared candidate
   interval;
3. SIP is free, licensed for every contemplated use, or available without a
   time-delay restriction;
4. calendar presence implies all-twelve-symbol minute completeness;
5. a calendar date is fresh or unexposed merely because it is later than the
   existing snapshot; or
6. the retained dataset may be reused as `terminal_screen_only` evidence.

After the exact outcome-blind envelope is declared and reviewed, the first
acquisition step should reuse the existing adapter and issue an explicit
`feed=sip` request inside that fixed envelope into sealed storage. It must stop
on an authentication/entitlement error, a returned/default feed mismatch,
missing raw-request provenance, or any need for a paid upgrade or provider
change. It must not silently retry as IEX. The acquisition receipt—not this
calendar note—will decide current SIP access.
