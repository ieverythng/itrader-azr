# Market-data provider evaluation for `dataset_v2`

**Status:** provisional acquisition decision record  
**Date:** 2026-09-23  
**Scope:** Tradier, Databento, Cboe DataShop/Cboe One, Alpaca Basic, and the existing Polygon/Massive path  
**Evidence policy:** official provider documentation, product pages, specifications, and first-party downloadable samples only

## Executive decision

Tradier cannot supply the required multi-year one-minute backfill. Its advertised two-month Pro trial is real, but it is a brokerage-subscription promotion; it does not extend the Time & Sales endpoint's one-minute history beyond 20 days for regular-hours-only data or 10 days when all sessions are included. Tradier remains useful as a forward collector and live/shadow feed, not as the source of record for `dataset_v2`.

The strongest potentially permanent $0 candidate is **Alpaca Basic**, whose current plan advertises one-minute aggregate bars, extended hours, corporate actions, 200 historical calls/minute, and history since 2016. Alpaca documentation is inconsistent about whether a free account may retrieve old consolidated SIP history or only IEX history: its plan/tutorial material says full-market SIP data older than 15 minutes is available, while its feed guide calls IEX the only unsubscribed feed. That entitlement must be tested with an authenticated, old `feed=sip` request before selection. IEX alone represents only about 2.5% of US equity volume and is not an acceptable canonical substitute.

The other strong zero-upfront-cost candidate is a **Databento historical pilot using the $125 signup credit**, followed by usage-based T+1 historical updates if the exact quoted cost and selected US-equities dataset are acceptable. The historical API has explicit schemas, metadata/cost endpoints, batch downloads, UTC timestamps, and separate corporate-action/adjustment-factor data. The source feed must be frozen explicitly: `EQUS.MINI` begins only on 2023-03-28 and is a proprietary multi-venue composite, so it cannot alone satisfy a 2022 start date.

The lowest-risk paid fallback remains **Polygon, now branded Massive, Stocks Starter at $29/month**. It provides five years of consolidated US-market minute aggregates, all market sessions, corporate actions, unlimited API calls, and flat-file downloads. This already matches the repo's provider choice; the current `dataset_v1` defect was introduced by local ingestion, not by the provider's aggregate payload.

**Cboe DataShop Equity & ETF Quotes** is the closest packaged match to the certified regular-hours bar contract: selectable intervals from one minute, OHLC, volume, VWAP, bid, and ask, with history from 2010. It is a paid, RTH-only CSV/SFTP product whose exact price is calculated from the selected symbols and dates. It is best positioned as an independent validation source or a premium primary source if a quote is acceptable, not as the default free path.

Recommended order:

1. Implement one provider-neutral raw-bar contract and snapshot manifest before downloading the full dataset.
2. Test Alpaca Basic with an explicit old `feed=sip` one-minute request; reject it as canonical if the account is limited to IEX.
3. Run a Databento cost/coverage pilot under signup credit, with the exact dataset ID recorded.
4. Use Polygon/Massive Starter for a one-month, reproducible five-year backfill if neither free pilot provides a consistent 2022-present composite.
5. Use Tradier only for forward live collection/cross-checking; never splice its bars into the canonical historical series without a named provider-regime boundary.
6. Validate a small overlap panel against the official Cboe sample or a purchased Cboe slice before certifying `dataset_v2`.

## Required iTrader contract

The provider must support, directly or through a reproducible derivation:

- US stocks and ETFs, initially the proposed 12-symbol universe.
- True one-minute OHLCV bars, not close-only synthetic bars.
- At least 2022 through the latest completed trading day.
- Explicit UTC timestamps and session classification.
- Regular-hours certification while retaining pre-market and after-hours data where available.
- Deterministic bar rules, missing-bar semantics, and corporate-action provenance.
- Repeatable downloads, immutable raw retention, hashes, and manifestable request parameters.
- Personal/internal research use and local persistence consistent with the accepted provider agreement.
- A path to forward T+1 updates; real-time streaming is useful but is not required for the offline Phase A gate.

Provider mixing is not a free substitute for coverage. Vendors use different venue sets, sale-condition filters, aggregation clocks, and adjustment conventions. A snapshot may include a secondary source for validation, but one canonical dataset version should not silently switch providers mid-series.

## Comparison matrix

| Criterion | Tradier | Databento | Cboe DataShop | Alpaca Basic | Polygon/Massive |
|---|---|---|---|---|---|
| Multi-year 1-minute OHLCV | **No.** 20 days RTH / 10 days all sessions | **Yes**, dataset-dependent; `EQUS.MINI` only from 2023-03-28 | **Yes**, 2010-present | **Advertised yes**, since 2016; free SIP entitlement must be tested | **Yes:** free 2 years; Starter 5 years |
| Intraday schema | JSON/XML Time & Sales OHLCV+VWAP | `ohlcv-1m` DBN/CSV/JSON | CSV OHLCV+VWAP+bid/ask | REST OHLCV+VWAP+trade count | REST/flat-file OHLCV+VWAP+trade count |
| Market source | Consolidated Level 1 feed | Selected exchange feed or explicit composite | Consolidated US equity/ETF interval product; RTH | IEX real-time free; historical SIP availability is the gate | SIPs, exchanges, FINRA/off-exchange and other sources |
| Sessions | `open` or `all`; live event session field | Feed-dependent; UTC event timestamps; pre-market supported by equities examples | **RTH only** for Equity & ETF Quotes | Extended-hours bars; separate BOATS/overnight feeds also exist | Pre, regular, and after-hours |
| Corporate actions | Adjustment status is not sufficiently explicit for this use; historical data may not be dividend-adjusted | Separate point-in-time corporate actions and adjustment factors | Not advertised as adjusted; treat as raw and supply external action metadata | Request `raw`, `split`, `dividend`, or `all`; corporate actions advertised | Split-adjusted by default or raw; not dividend-adjusted; split/dividend endpoints |
| Free access | Brokerage account API access; Pro trial has deposit/renewal conditions | $125 historical credit, expires after six months | Sample downloads / API trial messaging only; historical product is paid | $0 Basic, 200 calls/minute, 30 real-time symbols | Basic $0, five calls/minute, two years |
| Relevant paid path | Pro $10/month does not extend history | Historical pay-per-GB; Standard live plan $199/month | Selection-priced historical order/subscription; academic discount has $500 minimum | Algo Trader Plus $99/month for unrestricted current SIP | Starter $29/month, five years, unlimited API calls/files |
| Rate/download model | 120 market-data requests/minute production; one market stream | Historical streaming or batch; exact preflight cost API; pricing page advertises unlimited API calls/downloads | CSV files delivered by download/SFTP; exact cart quote | Paginated REST, 10,000 rows/page, 200 requests/minute | Basic five calls/minute; paid unlimited; REST pagination and flat files |
| Real-time suitability | Good for a personal forward collector; one stream session | Strong but US-equities live requires subscription | Direct Cboe One is institutional-price territory; DataShop is mainly file delivery | Good IEX live subset; complete current SIP requires $99 plan | Strong delayed/real-time websocket by plan |
| Canonical-v2 verdict | Forward/cross-check only | **No-upfront-cost pilot** | Independent RTH validation / premium option | **First $0 entitlement pilot** | **Lowest-risk paid primary fallback** |

## Tradier

### What the promotion actually provides

Tradier advertises two months of Pro or Pro Plus at no subscription charge for eligible new customers. The official terms require a **$500 minimum deposit within 30 days**; after the promotion, Pro renews at **$10/month** and Pro Plus at **$35/month**. The pricing page lists API access, but this promotion is about brokerage-plan fees, not expanded historical-data depth ([Tradier pricing](https://tradier.com/pricing), [developer offer terms](https://trade.tradier.com/developer-api/)).

Tradier's FAQ says it does not charge brokerage account holders separately for API access. A standard account already exposes developer API access, so Pro is not a reason to select Tradier as a historical data vendor ([Tradier API FAQ](https://docs.tradier.com/docs/faq), [Tradier standard/pro account comparison](https://join.tradier.com/partner)).

### Historical coverage is disqualifying

The Time & Sales endpoint supports `tick`, `1min`, `5min`, and `15min`, and a `session_filter` of `open` or `all`. Its official depth table is decisive:

- one-minute, `open`: 20 days;
- one-minute, `all`: 10 days;
- five-minute: 40 days RTH / 18 days all sessions;
- fifteen-minute: 40 days RTH / 18 days all sessions.

The separate historical-pricing endpoint can cover a security's lifetime, but only at daily, weekly, or monthly intervals. It cannot fill a multi-year one-minute snapshot ([Tradier Time & Sales](https://docs.tradier.com/reference/brokerage-api-markets-get-timesales), [historical endpoint](https://docs.tradier.com/reference/brokerage-api-markets-get-history), [historical-data guide](https://docs.tradier.com/docs/historical-data)).

Time & Sales interval records do contain `open`, `high`, `low`, `close`, `volume`, and `vwap`, so the data shape is suitable for recent validation. Tradier warns that historical data may not be dividend-adjusted ([Time & Sales schema](https://docs.tradier.com/docs/timesale), [historical-data guide](https://docs.tradier.com/docs/historical-data)).

### Automation and live use

Production market-data requests are limited to 120 requests per minute per access token, with explicit remaining/expiry headers. The production brokerage feed is real-time and consolidated across US exchanges; sandbox data is delayed 15 minutes ([rate limits](https://docs.tradier.com/docs/rate-limiting), [market-data feed](https://docs.tradier.com/docs/market-data)).

Tradier supports HTTP and WebSocket streaming for hundreds of symbols in one connection, but permits only one market-data streaming session at a time and explicitly requires reconnect logic. Streaming events expose exchange, sequence, cancellation/correction, and session fields; `tradex` is recommended over the simpler `trade` event during pre/post market ([streaming guide](https://docs.tradier.com/docs/streaming-data), [stream response schema](https://docs.tradier.com/docs/streaming)). API use by ordinary brokerage users is personal-only unless they become a Tradier Partner ([Tradier FAQ](https://docs.tradier.com/docs/faq)).

**Verdict:** useful free/low-cost live shadow collector and 10–20 day overlap validator; categorically unsuitable for the 2022-present one-minute backfill.

## Databento

### Historical data and bar semantics

Databento exposes explicit `ohlcv-1m` records. `ts_event` marks the inclusive start of the interval in UTC, and no record is emitted when no qualifying trade occurs. That makes missing bars observable instead of silently forward-filled ([schemas and OHLCV conventions](https://databento.com/docs/knowledge-base)). Historical delivery supports HTTP streaming and batch flat files through official Python, C++, Rust, or raw HTTP clients ([historical API](https://databento.com/docs/api-reference-historical)). DBN is the native format, with server-side CSV and JSON transcoding available ([DBN format](https://databento.com/docs/knowledge-base/new-users/dbn-encoding)).

The precise dataset identity matters:

- `EQUS.MINI` is a proprietary blend of multiple NMS exchanges and ATS feeds. Its OHLCV bars aggregate prices and volume across those component venues, but venue identities are anonymized.
- Its history begins **2023-03-28**, so it cannot alone fulfill a 2022 start.
- `EQUS.ALL` and individual venue datasets have different market coverage and economics. The exact candidate must be selected and frozen before judging parity with SIP-based Polygon bars.

Sources: [EQUS.MINI specification](https://databento.com/docs/venues-and-datasets/equs-mini), [EQUS.MINI availability announcement](https://databento.com/blog/databento-us-equities-mini-now-available), [US equities dataset introduction](https://databento.com/docs/examples/equities/equities-introduction).

Databento also exposes point-in-time corporate-action records from 2018-05-01, including splits, reverse splits, dividends, mergers, and listing events. Separate adjustment-factor data can be applied explicitly rather than accepting an undocumented adjusted series ([corporate actions](https://databento.com/docs/venues-and-datasets/corporate-actions), [adjustment-factor example](https://databento.com/docs/examples/adjustment-factors/applying-adjustment-factors)).

### Cost and access

Every new account receives **$125 of historical-data credit**, usable across historical datasets and expiring after six months. Historical data is metered by uncompressed DBN bytes; streaming and batch are available, and a batch request can be downloaded repeatedly for 30 days without being billed again. Metadata and symbology calls are free. Exact price can be queried before download through the metadata cost endpoint or batch quote ([pricing and credit](https://databento.com/pricing), [usage FAQ](https://databento.com/docs/faqs/usage-pricing-and-data-credits)).

Historical access can remain usage-based with no monthly subscription. Current Standard live service is advertised at **$199/month**; US-equities usage-based live service is not available. Historical T+1 data generally avoids live exchange licensing, while live, non-display, professional, or redistributed use must follow dataset-specific licensing ([Databento quickstart comparison](https://databento.com/docs/quickstart), [licensing overview](https://databento.com/docs/portal)). Download-center files expire after 30 days, so iTrader must retain its licensed local raw copy and manifest; the public docs do not state that locally downloaded historical data must be deleted at that point, but the accepted dataset license remains authoritative.

**Verdict:** best first no-upfront-cost pilot and best API design for reproducibility. Do not approve it as canonical until the exact 2022-present dataset/feed, quote, session coverage, and local-retention terms are captured in the snapshot manifest.

## Cboe

### DataShop Equity & ETF Quotes

Cboe DataShop's Equity & ETF Quotes product supports selectable intervals from **one minute through end-of-day**, with OHLC, trade volume, VWAP, bid, and ask. It covers nationally listed US equities and ETFs, excludes OTC securities, is restricted to **regular trading hours**, and is available from **2010 to present**. Historical files can be grouped by day or month; subscriptions support intraday or nightly delivery ([Cboe Equity & ETF Quotes](https://datashop.cboe.com/equity-etf-quotes), [DataShop FAQ](https://datashop.cboe.com/faqs)).

The public product page offers a downloadable sample. This review downloaded the first-party sample and verified a CSV schema of:

```text
underlying_symbol,quote_datetime,open,high,low,close,
trade_volume,vwap,bid,ask
```

The current sample is a 15-minute, one-day file rather than a one-minute acquisition sample, but it validates the advertised delivery schema. The product page's interval control is the source for one-minute availability.

Historical price is selection-dependent and calculated in the DataShop cart; no fixed public figure for the proposed 12-symbol 2022-present order was found. Cboe offers qualifying academic users 50% of standard price with a **$500 minimum charge** for Equity, ETF & Index Quotes ([academic discount](https://datashop.cboe.com/academic-discount)). Historical purchase/subscription and any external distribution are governed by Cboe's selected-use agreements; the product form defaults external distribution to `No`, so iTrader should treat a personal order as internal research data and should not publish raw files.

### Cboe One live feed is a different product

Cboe One combines quotes and trades from BZX, BYX, EDGX, and EDGA, not the entire SIP. Its official internal-distribution fees are **$1,500/month for Summary** and **$15,000/month for Premium**, before applicable user/consolidation fees. A one-day test file is available, but this is not a practical personal-project live feed ([Cboe One](https://www.cboe.com/market_data_services/us/equities/cboe_one/), [US equities products](https://www.cboe.com/data/market-data-services/us/equities)).

**Verdict:** excellent RTH reference product and potentially the cleanest purchased training source, but not the free primary route and not a source for pre/post-market bars.

## Alpaca Basic

Alpaca's current Basic market-data plan is **$0/month** and advertises US stock/ETF aggregate bars, corporate actions, extended-hours data, 200 historical API calls per minute, and history since 2016. Historical bars support timeframes from one minute upward and include timestamp, OHLC, volume, trade count, and VWAP. Requests are paginated with up to 10,000 rows and a continuation token. The adjustment parameter supports `raw`, `split`, `dividend`, or `all` ([Alpaca data plans](https://alpaca.markets/data), [market-data plan documentation](https://docs.alpaca.markets/us/v1.1/docs/about-market-data-api), [historical request guide](https://alpaca.markets/learn/fetch-historical-data)).

The feed entitlement needs an authenticated test before promotion. Alpaca consistently documents that Basic real-time equities are IEX-only and that IEX is roughly 2.5% of US market volume. Its plan table and official historical tutorial also indicate that consolidated SIP history older than the most recent 15 minutes is available on Basic. Another official feed overview says IEX is the only feed usable without a subscription. The safe interpretation is conditional: request an old date explicitly with `feed=sip`, record the response entitlement and volume, and reject Basic as the canonical source if the account returns only IEX ([historical stock feeds](https://docs.alpaca.markets/us/v1.1/docs/historical-stock-data-1), [Basic versus Plus](https://docs.alpaca.markets/us/v1.1/docs/about-market-data-api), [historical tutorial](https://alpaca.markets/learn/fetch-historical-data)).

Live minute bars include pre-market and after-hours trades. Alpaca can emit a corrected `updatedBars` record when a late trade changes the preceding minute, which requires upsert semantics in a forward collector ([real-time stock bars](https://docs.alpaca.markets/us/docs/real-time-stock-pricing-data)). Basic websocket access is limited to 30 symbols, sufficient for the proposed 12-symbol universe. The personal plan must be treated as personal/internal research data; raw market data must not be republished.

**Verdict:** highest-priority permanent-free pilot. It becomes the preferred canonical source only if an authenticated old `feed=sip` request confirms consolidated coverage and the accepted agreement permits the immutable local research snapshot. IEX-only bars are an independent venue diagnostic, not a replacement for consolidated OHLCV.

## Polygon / Massive

Polygon's official product pages now redirect to the Massive brand. The existing repo integration and environment variable names can remain provider-compatible, but new documentation should record the rebrand and endpoint contract.

The current individual stock plans advertise:

- Basic: **$0**, two years of history, five API calls/minute, minute aggregates;
- Starter: **$29/month**, five years of history, unlimited API calls and file downloads, minute aggregates, 15-minute-delayed websocket/snapshots;
- Developer: **$79/month**, ten years;
- Advanced: **$199/month**, 20+ years and real-time trades/quotes.

The five-year Starter window covers the proposed 2022-present snapshot as of this review. The provider describes its equity coverage as consolidated input from SIPs, exchanges, FINRA facilities, dark/off-exchange venues, ATSs, and other sources. It supplies pre-market, regular, and after-hours data and UTC Unix timestamps ([stock plans and coverage](https://massive.com/stocks), [stock-session and source overview](https://polygon.io/docs/rest/stocks/overview?source=post_page---------------------------)). Daily downloadable stock minute-aggregate files contain OHLCV across US equities ([stock minute flat files](https://polygon.io/docs/flat-files/stocks/minute-aggregates/2013/12)).

Aggregate bars are built from qualifying trades according to sale-condition rules, so missing bars should be retained as missing rather than automatically fabricated. Historical aggregates are split-adjusted by default and can be requested unadjusted; they are not dividend-adjusted. Splits and dividends are available through separate corporate-action endpoints ([aggregate rules](https://polygon.io/knowledge-base/article/how-does-polygon-create-the-open-high-low-close-volume-aggregate-bars), [adjustment policy](https://polygon.io/knowledge-base/article/is-polygons-stock-data-adjusted-for-splits-or-dividends), [split API](https://polygon.io/docs/rest/stocks/corporate-actions/splits)).

**Verdict:** simplest and cheapest deterministic fallback for the complete requested window. The existing Phase A data defect should be repaired in iTrader's adapter by persisting `o/h/l/c/v/vw/n/t` directly; changing providers is not required to recover true OHLCV.

## Acquisition experiment before committing a provider

The provider bake-off should be bounded and should not create a mixed training dataset:

1. Freeze one symbol (`AAPL`), five completed trading days containing normal, half-day if available, and corporate-action-adjacent cases, plus one low-volume stress symbol.
2. Request one-minute data in both regular-hours and all-session modes where supported.
3. Persist each provider unchanged in a provider-specific raw namespace.
4. Normalize only after recording provider, dataset/feed ID, request, adjustment flag, retrieval time, timezone, session policy, response hash, and license/use classification.
5. Compare:
   - row counts and minute-grid coverage;
   - OHLC invariants and percentage of non-flat candles;
   - bar-boundary convention;
   - gaps/no-trade treatment;
   - volume, VWAP, and close deviations;
   - split/dividend behavior;
   - RTH versus pre/post classification;
   - retry/replay determinism and cost quote.
6. Select one source-of-record and keep the others as validation evidence only.

### Provider-specific pilot

| Pilot | Credential/state change | Pass condition |
|---|---|---|
| Alpaca Basic | Create a free account/API key; no paid subscription | Old explicit `feed=sip` one-minute request returns consolidated OHLCV; 2022-present pagination, adjustments, and retention terms pass; IEX-only fails |
| Databento | Create free account; no charge beyond signup credit; call dataset-range and cost metadata first | Exact 2022-present feed identified; quoted 12-symbol OHLCV cost within credit or agreed small budget; local retention terms acceptable |
| Polygon/Massive | Existing key may exercise Basic; Starter only if explicitly purchased | Direct OHLCV fields persist unchanged; five-year request reproducible; pagination/rate-limit behavior tested |
| Tradier | Brokerage account/token required; Pro not required for the data-depth test | Recent 10/20-day bars and live stream serve as secondary cross-check; never treated as historical source |
| Cboe | Public sample is immediately testable; historical slice requires cart purchase | CSV parser and RTH semantics verified; optional overlap quote/purchase is economically justified |

## Acceptance recommendation

Do not let “free” weaken the dataset contract. The provisional preference is:

```text
Canonical offline history:
  Alpaca Basic authenticated old feed=sip pilot
      └─ if consolidated 2022-present history passes → canonical primary
      └─ if IEX-only → fail canonical gate
  Databento $125 credit + Polygon Basic + Cboe + Tradier
      └─ validation and cross-provider evidence
  Polygon/Massive Starter
      └─ paid fallback if Alpaca SIP entitlement fails

Forward/shadow collection:
  same canonical provider when affordable
      └─ Tradier may run as an explicitly secondary live validator

Independent RTH audit:
  Cboe DataShop sample or small purchased slice
```

The final provider decision remains gated on these explicit results:

- whether creating a free Alpaca account/API key is acceptable;
- whether that account can retrieve old consolidated `feed=sip` bars, rather than IEX-only bars;
- the Databento preflight quote and exact dataset-range response as secondary evidence;
- whether one paid Polygon/Massive Starter month is acceptable if the free canonical gate fails.

No data purchase, subscription activation, brokerage funding, or provider account creation was performed during this research pass.
