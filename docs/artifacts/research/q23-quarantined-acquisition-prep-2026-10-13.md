# Q23 ACQ-PREP35: future quarantined acquisition and metadata scope

2026-10-13 Europe/Madrid. Preparation only, branch `agent/Watson-dev`, HEAD
`8f9fcf9` plus inherited dirty work. No request, credentials, raw bars/outcomes,
loader, model or evaluation was opened. [Human decision](../../project-state/decisions.md)
authorizes this plan, not its execution. Metadata-first ordering is already
accepted; [SUPPORT30](q26-support-ordering-2026-10-12.md) is not reasked.
GRILL owns subsequent review/dispatch. Current heavy GPU scope belongs to TRAINING.

## Outcome: prepared, not executable-as-approved

Use the existing Alpaca adapter/snapshot builder, in a separately reviewed
quarantine. The present collector lacks enforced total limits and terminal-role
isolation; **do not run the command below standalone**. A later narrow control/
isolation implementation and independent review are prerequisites, not work
authorized here. No new collector framework, data-role adoption or code is added.

## Current evidence and missing authority

**Source facts:** [Alpaca adapter](../../../itrader/inputs/market_data/alpaca.py)
requests stock bars and the paper-API calendar, uses pagination, timeout60s and
up to6 retries by default. Page limit is10000, but pages/retry delays/total time
are unbounded at the job level. [CLI](../../../itrader/inputs/market_data/cli.py)
supports output/symbol/date/feed/usage/env-file flags, not RTH-only, metadata-only,
page/byte/time budgets or role guards. It reads an env file and existing process
variables win via setdefault; its credential-source identity needs later review.
No secret file, variable value or account was inspected now.

[Builder](../../../itrader/inputs/market_data/snapshot.py) retains raw gzip pages,
request/manifest hashes, split-adjusted partitioned Parquet and completion markers;
omissions are not filled. It processes an entire symbol/year's pages in memory,
classifies pre/regular/after-hours, and emits all three session partitions.
Therefore existing acquisition is not RTH-only even though later Q23 metadata
selection must be RTH-only. It can delete incomplete unit files during resume;
resume/deletion must stay inside the resolved exclusive acquisition root.
Manifest existence prevents a new build, not arbitrary reading or modification.

**Primary provider facts:** [official historical bars](https://docs.alpaca.markets/us/reference/stockbars)
defines inclusive bounds, pagination, split adjustment and feed. Symbol mapping
uses an `asof` default of the current day; the collector does not pass/freeze it.
Treat META/FB and other entity mappings as a reproducibility decision, not an
assumed historical symbol identity. Freeze a supported asof strategy or explicitly
review the limitation before fetch; no new argument is claimed implemented.
[Official feed documentation](https://docs.alpaca.markets/us/docs/about-market-data-api)
describes subscriptions, not this account's entitlement. Existing human feed
access is a planning premise, not verified credentials/2020 coverage or permission
to substitute IEX for SIP. No endpoint entitlement probe was made.
[Versioned official calendar](https://docs.alpaca.markets/us/v1.4.2/reference/getcalendar-1)
documents the adapter's v2 endpoint and early-close times. Bind a complete reviewed
2020-2021 response; public documentation alone is not that certificate.

**Exposure/layout absent:** the documented existing dataset starts2022; it cannot
supply2020-2021. No candidate source bytes/calendar/layout/exposure certificate
exist in this receipt. Retrospective retained histories, source/entity equivalence
and human/external/manual usage remain unresolved. New acquisition/hash/date does
not prove non-exposure; unknown histories stay unknown. Quarantine may be acquired
after separate approval even while scientific clearance is unresolved, but cannot
be called fresh/qualified or released for solver use.

**Isolation gap:** `usage_classification` is arbitrary nonempty manifest text,
not `terminal_screen_only` enforcement. Env accepts existing paths; training
builds train/validation from the supplied dataset. A same-user directory can be
passed manually. Terminal preflight still requires common dataset identity and
returns evaluation_authorized=false. Approved terminal-role/dual identities are
not end-to-end isolation in current source.
[Env](../../../itrader/envs/itrader_env.py),
[trainer](../../../itrader/rl_lib/train/train_solo_V2.py),
[preflight](../../../itrader/rl_lib/evaluation/run_terminal_screen.py).

## Proposed immutable future packet and destinations

These paths are proposals only; none is created or inspected for access now.
Root `C:/Users/Admin/PROJECTS/iTRADER/q23-quarantine/acq35_20200101_20211231/`
is outside the repository and normal training snapshot tree. Reject if occupied,
aliased, junction/symlink redirected or overlapping any training/output root.
Snapshot destination: `snapshot/`; its only collector writes are build_request,
raw/alpaca, bars, completed, temporary siblings and manifest under that root.
Separate `metadata/` holds outcome-free projection/calendar identities; `receipts/`
holds reviewed scope, argv/source/runtime/entitlement/resource/ownership hashes,
status/request-count log and final acquisition seal. Never include secrets,
response bodies, sample OHLCV, return summaries or diagnostic charts in receipts.
Do not register this root in training/proposer/catalog/evaluation configs.

Requested role is quarantine-reserved-for-terminal-screen, not training,
validation or an implemented role enum. `terminal_screen_only` release remains a
separate later implementation/clearance decision. Writer process may ingest
values solely to serialize/normalize them; neither agent nor human analysis may
inspect values. This machine processing also needs explicit future authority.

Freeze twelve existing candidate symbols in order:
`AAPL,MSFT,AMZN,NVDA,AMD,AVGO,GOOGL,META,SPY,QQQ,IWM,TLT`;
date envelope2020-01-01 through2021-12-31, America/New_York,1Min,SIP,split,
schema2, no dividends or forward fills. Preserve raw timestamps and reviewed
entity mapping. Exact request endpoints/timezone/year boundaries must match
`_year_ranges`: year-transition local midnight and final-date20:00local, both
converted to UTC. Provider inclusive endpoints require duplicate/boundary checks;
do not silently modify semantics or substitute daily/IEX data on failure.

**Proposed future argv, inert documentation only:**

```text
<reviewed-native-python> -m itrader.inputs.market_data.cli build
  --output C:/Users/Admin/PROJECTS/iTRADER/q23-quarantine/acq35_20200101_20211231/snapshot
  --symbols AAPL,MSFT,AMZN,NVDA,AMD,AVGO,GOOGL,META,SPY,QQQ,IWM,TLT
  --start 2020-01-01 --end 2021-12-31 --feed sip
  --usage-classification quarantined_reserved_terminal_unqualified
  --env-file <separately-approved-private-credential-path>
```

Interpreter and secret path are intentionally unresolved; no new venv/install
or credential read is authorized. Freeze the actual typed argv vector and source
closure before approval, not this placeholder. The existing CLI's free-form usage
label grants no control. Do not reuse training GPU launcher/lease for collection.

## Frozen limits and independent isolation gate before execution

**Proposed operational caps, not adopted:** one sequential acquisition attempt;
one calendar request and24 symbol/year units; at most50 bar pages per unit,
10000 rows/page and1201 total HTTP attempts inclusive of calendar; retries0;
request timeout60s; wall cap90minutes; total acquisition-root bytes10GiB;
start freeRAM4GiB/freeDisk20GiB, abort below2GiB/10GiB. These are resource bounds,
not eligibility, sample-size, process or Q26 numerical thresholds. They may
truncate acquisition and produce failure, not a partial scientific success.
No load/data was used to estimate sufficiency. GRILL must approve or replace
and freeze caps/resource lease; no collection while TRAINING owns heavy work.

The command cannot enforce those caps today. Require a reviewed bounded adapter/
supervisor at existing collector seams to enforce requests including retries,
page loops/repeated tokens, bytes, wall/memory/disk and child containment; log
actual usage without payloads. Review memory before all-pages materialization.
Enforce approved credential precedence and safe status-only error capture because
current HTTP error strings include response text. No unreviewed wrapper/transport
after freeze, no unlimited Retry-After, automatic resume or alternate feed.

Independent isolation review must cover resolved paths/aliases/globs/catalogs,
OS read/write permissions plus application deny rules, every proposer/train/env/
replay entrypoint, and the exact metadata allowlist. If the same Windows account
can read everything, record that limitation; folder names are not OS isolation.
Synthetic/no-network review scenarios must prove quarantine rejection BEFORE
opening outcome columns, unexpected alias refusal, caps/expiry/nonzero propagation,
secret/payload non-disclosure and clean no-survivor cleanup. These scenarios are
proposals, not tests run here. Follow REVIEW.md's two-reviewer requirement for
high-risk isolation/control changes; today's child audit is not acceptance.

## Later metadata and exposure qualification, still no outcome release

Only a separately approved projection may export
`symbol,ts,session,session_id,trading_date` plus deterministic row/window identities;
reject configurable added columns, OHLCV/volume/vwap/trade_count or outcome-derived
labels. Freeze ordering equivalent to later loader indexing, dataset/raw/calendar
hashes and byte-hash permission. Existing projection machinery is reusable but
not called here; ordinary ITraderEnv/catalog construction can open outcomes.

Review complete calendar ordering and selected/unselected open sessions;
duplicates/missing minutes, full-minute bounds/early closes and maximal
`[start-60,start+200)` eligibility; offsets60/120, 32anchors/12symbols and crossed
96/128/160/200 horizons remain existing design terms. Keep exact anchor selection
and seed recipe separately frozen; no anchor chosen from bars/returns. Retain
date incidence/gaps, same-date coupling, fixed weights and overlaps, not effective
sample-size claims. Acquisition counts alone certify neither windows nor support.

Keep retained source-bound historical exposure proofs and external attestation
separate from metadata. Existing
[panel qualification](../../../itrader/rl_lib/evaluation/panel_qualification.py)
checks supplied histories/selected dates under validation/single-dataset rules;
do not route a quarantine through it as ordinary validation to bypass missing
terminal-role integration. A raw-page seal, metadata-valid receipt, exposure-
clearance receipt and scientifically qualified panel are distinct statuses.
Partial/failed artifacts stay quarantined with reasons; no cleanup of historical
evidence, eligibility renormalization, freshness claim or automatic release.

## Handoff and next authorization

Prepared scope is **blocked for fetch** on enforceable caps/isolation, exact
credential/runtime/path identities, endpoint/entity/feed entitlement review,
independent approvals and a serialized acquisition lease. Source/exposure/
metadata qualification is unavailable, not assumed passing. GRILL may dispatch
the narrow prerequisite control/review work, then separately authorize one frozen
fetch; this note itself grants neither. No centering/kernel/bandwidth/process/MC
settings,95/94/3 changes, simulation, proposer/training/evaluation, packages/WSL
or Git operations are part of this scope. Receipt/hashes remain on the owned board.
