# Retained data identities and exposure boundary

Date: 2026-10-07, Europe/Madrid; 2020-2021 clearance follow-up:
2026-10-08

Status: **outcome-blind metadata inventory; not a freshness certificate, data
acquisition declaration, panel selection, Q26 choice, or terminal-screen
authorization**

## Finding

The retained history contains two different market-data authorities that must
not be collapsed into one exposure ledger:

1. the legacy flat `itrader/data/ohlcv.parquet` plus its DuckDB source; and
2. the immutable twelve-symbol Alpaca/SIP dataset-v2 snapshot.

The existing dataset-v2 exposure inventory is useful but is scoped to the
dataset-v2 bars-tree digest. It does **not** clear semantic market windows that
were already used through the legacy Parquet, proposer/curriculum summaries,
unhashed manual analysis, or consumers outside its three declared roots. A new
file hash, a new retrieval time, or an older calendar range does not establish
freshness when the same symbol/time outcomes may already have been exposed.

The conservative clearance rule is therefore a union: dataset-v2 adaptive
exposure, legacy-data possible exposure, retained proposer/curriculum exposure,
and any unresolved manual inspection must all be disjoint from a future full
`[start - 60, start + 200)` terminal-screen window. This is a possible-exposure
bound, not a claim that every bar in each bound was actually viewed.

The proposed 2020-2021 calendar interval is chronologically disjoint from both
**exact hash-bound** market-data authorities. That fact includes all 42
dataset-v2 configurations whose evaluation starts remain unresolved: the
uncertainty is their start coordinate, not their dataset identity. It does not,
however, clear 2020-2021 against source-unbound task/candidate summaries,
path-only historical configs, the absent external-checkout source, or unlogged
manual/external inspection.

## Outcome-blind inspection boundary

This pass read only:

- file names, sizes and SHA-256 identities;
- JSON configuration/provenance fields;
- task start/horizon identities and opaque market-summary references;
- Parquet/DuckDB schema plus `symbol`, timestamp and session metadata; and
- source code that defines data loading, partitioning and consumer paths.

It did not read or print any OHLCV value, return, reward, regime label, policy
output, checkpoint output, terminal result, or sealed final-test content. It did
not run a model, select a date/window, acquire bar data, or call a bar endpoint.
The separate calendar-only receipt described below was produced through the
existing read-only adapter and records `bar_endpoint_called=false`.

## Retained source identities

### Legacy flat dataset (provisional name: `legacy_dataset_v1`)

| Item | Retained identity |
|---|---|
| Parquet | `itrader/data/ohlcv.parquet`; SHA-256 `5f2e449548532ff71f419431df86962a2aedf7275d7bff5592729bbb8a900f73`; 719,855 rows |
| DuckDB source | `itrader/data/itrader.duckdb`; SHA-256 `16c46b20b4a7e7460e0d748318a77c16ad79dcc5961e65a61567866a9c132748` |
| Producer code | Polygon one-minute aggregate request with `adjusted=true`, written to `ticks`, then resampled by [`etl_job.py`](../../../../itrader/inputs/etl_job.py) |
| Missing authority | No retained request manifest, content-addressed snapshot manifest, raw-page hash ledger, retrieval timestamp, usage role, session label, exchange calendar, or acquisition-host timezone |

[`backfill.py`](../../../../itrader/inputs/backfill.py) converts provider epoch
milliseconds with naive `datetime.fromtimestamp`, and the Parquet schema stores
naive timestamps. Because the host timezone was not retained, the following are
**stored coordinate spans, not certified UTC spans**:

| Symbol | Rows | Stored first timestamp | Stored last timestamp | UTC status |
|---|---:|---|---|---|
| AAPL | 378,521 | `2023-07-05 09:00` | `2025-07-03 21:59` | unresolved |
| AMZN | 188,431 | `2024-07-05 09:00` | `2025-07-03 21:59` | unresolved |
| MSFT | 152,903 | `2024-07-05 09:00` | `2025-07-03 21:55` | unresolved |

The DuckDB `ticks` table has the same per-symbol stored endpoints; its retained
backfill cursor ends at 2025-07-05. Neither the cursor nor filesystem mtime is an
acquisition receipt. The file has no `session` or `session_id` column, so an RTH
window, early close, or exchange-calendar interpretation cannot be certified
from this identity alone. Retrospective matching should conservatively widen a
legacy overlap to the entire retained three-symbol domain unless the original
host timezone is recovered. Any later date-level mapping needs an explicit
timezone-uncertainty buffer; stored dates must not simply be relabelled UTC.

### Immutable dataset-v2 snapshot

| Item | Retained identity |
|---|---|
| Snapshot ID | `dataset_v2_20220103_20260922_alpaca_sip_split` |
| Bars tree | SHA-256 `20aac96de6da77e20beeaff214d380ef06184faf301ff0dd53bed0ffd4c04ce9` |
| Manifest bytes | [`manifest.json`](../../../../itrader/data/datasets/dataset_v2_20220103_20260922_alpaca_sip_split/manifest.json); SHA-256 `4c6202883a46cbe5bfbf4986003dcd1eb05b3738ac2d48eeb42d9a57fe974ca5` |
| Manifest content identity | `fac41f6d42f8c5b1eac739755667bb38407461f82c168d7955de26bb2e5fb79c` |
| Build request | [`build_request.json`](../../../../itrader/data/datasets/dataset_v2_20220103_20260922_alpaca_sip_split/build_request.json); SHA-256 `e2fd032d9af90dc74a2ae2b1762a96ef8b7d47c49bd957706f9e646ceee4df62` |
| Source contract | Alpaca, SIP, split adjustment, one-minute bars, `America/New_York`, 2022-01-03 through 2026-09-22 |
| Symbols | AAPL, MSFT, AMZN, NVDA, AMD, AVGO, GOOGL, META, SPY, QQQ, IWM, TLT |
| Physical layout | `session/symbol/year`; 10,370,064 rows: 5,515,061 regular, 2,829,443 pre-market, 2,025,560 after-hours |
| Retained raw calendar | `raw/alpaca/000000_calendar.json.gz`; SHA-256 `0bb92722b805a05af29f11db1922081b279e6b5142265f4ad371e3266ac44a72` |

The metadata-only regular projection contains all twelve symbols on all 1,184
retained bar dates from 2022-01-03 through 2026-09-22, with zero duplicate
`(symbol, timestamp)` pairs. Every symbol begins at
`2022-01-03T14:30:00Z`; every symbol's last regular bar is
`2026-09-22T19:59:00Z`. Those endpoints do not imply a fixed daily UTC open:
New York daylight-saving offsets apply.

The current `0.7/0.15/0.15`, 200-row-purge contract recomputed from metadata
only places possible training outcomes from 2022-01-03 through 2025-04-28
(the latest symbol-specific boundary), and possible validation outcomes from
2025-04-24 through 2026-01-09. This describes current partition semantics, not
proof that every possible row was sampled. The final-test partition was not
projected or opened in this pass and remains sealed.

The old manifest predates the current per-raw-record retrieval timestamp,
usage-classification, missing-bar declaration and raw-file hash ledger. Its
hashes prove the retained bytes and request fields; they do not retroactively
supply those missing provenance facts.

### Current calendar-only receipt

The sibling [`calendar-metadata-receipt.json`](calendar-metadata-receipt.json)
records an existing-adapter calendar request for 2019-01-01 through 2026-10-07:

- canonical request SHA-256
  `f1978c5153d32c5504f8ce59689dde7ff7dae4a2ae3d30e0c569f055ae3ff5c7`;
- canonical response-payload SHA-256
  `360d219baff281f84b1185b9febb2590d176b8c2e4447bda540daf034ece7b9a`;
- 1,952 calendar sessions from 2019-01-02 through 2026-10-07; and
- `bar_endpoint_called=false`.

This establishes a current calendar metadata surface, not bar entitlement,
OHLCV completeness, a normalized calendar certificate, or acquisition
authority. The retained dataset-v2 raw calendar and this newer response remain
separate identities rather than silently replacing one another.

## Exposure inventory by consumer

| Consumer | Retained evidence | Conservative interpretation |
|---|---|---|
| Solver training | The bounded accessible-root scan found 123 retained `result.json` files whose projected config names an `ohlcv.parquet`: 104 under `OLD_RESULTS`, 16 under `Phase_A_PBT_overnight_20260216`, and 3 under `Phase_A_PBT_followup_20260216`. Of these, 19 name the current checkout's absolute legacy path, 101 use the repository-relative legacy path, and 3 old results name a separate, now-absent OneDrive checkout path. None retains a dataset hash or train/evaluation split in the projected config. Nineteen record `stream_worker`; 104 retain no start-mode field. | The 120 canonical-location records establish use of a named legacy path, not historical byte identity. The 3 absent-checkout records are source-opaque. Exact episode windows are unavailable, so none of these path-only records independently certifies 2020-2021 as clean. |
| Later legacy train/validation runs | Sixteen retained stable `resolved_config.json` files bind the exact legacy SHA-256. Each declares `train`/`validation`, training `stream_worker`, evaluation `random_episode`, and one fixed evaluation task. | The split roles are known, but random evaluation starts and complete step-level windows are not retained here. Do not convert the config into exact exposure coordinates. |
| PBT, tuning and checkpoint selection | The retained PBT roots above used the legacy path. `best_config.json` (SHA-256 `9b5641eaebc2906f459fe79cada67d2e494d2eceac480c44863e6d22d1c62849`) also names it. Thirty-seven checked-in experiment loadouts, covering 85 declared variants, still target the legacy file. | Result/config existence establishes use or planned use, not exact sampled rows. Hyperparameter/checkpoint selection makes those windows adaptive, never terminal-screen-only. A checked-in loadout alone is not proof it ran. |
| Evaluation and baseline tools | [`run_policy_baselines.py`](../../../../itrader/rl_lib/evaluation/run_policy_baselines.py) and [`train_solo_V2.py`](../../../../itrader/rl_lib/train/train_solo_V2.py) still default to the legacy path; [`ITraderEnv`](../../../../itrader/envs/itrader_env.py) also falls back to it when no path is supplied. | A default is a possible consumer, not an access receipt. Missing run provenance must fail closed for freshness. |
| Proposer/task curriculum | `_tmp` retains 12 task-pool JSONL files (119 task records), including 48 explicit-start records over 25 unique indices. Six candidate-group ledgers contain 76 groups and 18 opaque market-summary references. Their schema can retain market summaries and solver-feedback fields; those values were deliberately not inspected. The checked-in Phase-B comparison references one four-task pool, but no matching executed result was found in the bounded roots. | The retained files prove proposer/curriculum artifacts exist. They do not bind their market summaries to a dataset hash, symbol, timestamp, or source request, so their exact value-bearing exposure cannot be certified. Planned consumption is not execution proof. |
| Market-summary and regime tooling | [`gen_task_pool.py`](../../../../itrader/scripts/gen_task_pool.py) reads a Parquet only when `market_summary_mode=ohlcv_window`, whose default path is the legacy file; static mode does not read it. [`tag_pool_regimes.py`](../../../../itrader/scripts/tag_pool_regimes.py) defaults to the legacy file and, if run, reads dataset-wide and task-local value windows. | Source capability is not execution proof. With no source-bound command/access receipt, it remains an unresolved possible proposer/manual exposure rather than a known clean region. |
| Calibration/capability | No implemented calibration job that reads market Parquet was found. The future online/shadow `CapabilityProfile` record stores a calibration field but does not establish a data-access pipeline; offline diagnostics remain evidence rather than an operational profile. | Absence of a code path does not prove absence of notebook, ad-hoc, or human calibration. Terminal-screen data must remain excluded from offline calibration and future online capability feedback. |
| Human/manual inspection | Retained research reports, plots, result JSON, candidate-group summaries and generated analyses are value-bearing artifacts. The dataset-v2 inventory records file/config identities, not file-open events or who saw them. | Filesystem retention can prove an artifact exists and bind its bytes; it cannot certify non-inspection. Unlogged access stays unknown. |
| Dataset-v2 run history | [`exposure_inventory.json`](../r1-prerequisites-2026-10-05-evidence/exposure_inventory.json), SHA-256 `74b4fff809a85ebb22c055b3fd0a33e9c2c3e145a044e366d296622ded81162f`, binds 163 configurations to the dataset-v2 bars hash: 121 explicit-evaluation-start, 26 random-evaluation, and 16 initially unresolved controls. | This is an evaluation-start inventory, explicitly not a complete training, proposer, tuning, manual-access, or exposure certificate. |
| Dataset-v2 start recovery | [`exposure-recovery-starts.json`](../evaluation-code-audit-2026-10-06-evidence/exposure-recovery-starts.json), SHA-256 `4a5127facc102d8fd6a61890df518a553ee5f7e69f1cb6692f89ca8a4706b6d7`, maps 512 recorded control rows to 48 unique starts. The 26 random configurations remain start-unbound within the exact dataset-v2 source. | Recorded control windows are conditional evidence. Random-run uncertainty still expands possible exposure to the full configured validation-start domain; zero of the 2,052 provisional anchors is certified fresh. |

For the retained `_tmp` proposer material, aggregate inventory identities were
computed as SHA-256 over sorted lines of
`relative_path<TAB>file_sha256<LF>`: twelve task pools
`4e98d9ca7eb2984a6f078e6b1cd6e811314aa83c6e36a9bf00a440d51f0cf032`,
six candidate-group ledgers
`b0aa098bd697f49735ce800e58be6410572860c7dbf00ddac88d63d22d502ee6`,
and five QA sidecars
`f120a22b160688586623a29ba34b1df24351b2ff99533f5c17523b57f6a5f095`.
These aggregate digests bind the bounded inventory without opening its
value-bearing summary or feedback fields.

## Candidate 2020-2021 clearance classification

| Evidence class | Retained material | Exact conclusion for 2020-2021 |
|---|---|---|
| Known-disjoint | The dataset-v2 exposure inventory binds all 163 records, including the 42 `requires_historical_start_qualification` records, to bars-tree SHA-256 `20aac96de6da77e20beeaff214d380ef06184faf301ff0dd53bed0ffd4c04ce9`. That snapshot begins on 2022-01-03. | The 42 unresolved records cannot include 2020-2021 values through that exact snapshot. “Unresolved” means the evaluation start/source-of-start is unavailable; it does not unbind the market-data file. |
| Known-disjoint | Sixteen later stable configs bind exact legacy SHA-256 `5f2e449548532ff71f419431df86962a2aedf7275d7bff5592729bbb8a900f73`. Its earliest stored coordinate is 2023-07-05. | The exact retained legacy bytes are disjoint from 2020-2021. The legacy timezone uncertainty can move session interpretation by hours or a date boundary, not by years. |
| Source-unbound but bounded to a known local path | The 120 older result configs name either the current absolute legacy location or `itrader/data/ohlcv.parquet`, but retain no hash. | The current file at that path is disjoint, but the historical bytes used by those runs are not provably identical. This tier is path-bounded, not byte-identity-bounded, and cannot by itself certify 2020-2021 clean. |
| Genuinely opaque/external | Three `OLD_RESULTS` configs name `ohlcv.parquet` in a separate OneDrive checkout. The path is absent on 2026-10-08, the configs retain no dataset hash, and no source copy was found in the bounded roots. | No semantic date bound survives. Those records can plausibly have used 2020-2021 values and must be resolved or conservatively treated as exposure. |
| Genuinely opaque proposer/manual provenance | The 12 task pools and 6 candidate-group ledgers contain no dataset hash/path, source request, symbol, timestamp, or UTC-window binding. The candidate schema permits opaque market-summary references. No literal `2020-`/`2021-` string was found in these files, but summaries may omit dates and their value-bearing fields were not opened. Manual/notebook/prompt/export access has no retained access log. | Retained metadata cannot exclude 2020-2021 values. These artifacts and unlogged human/external consumers remain uncleared even though the checked-in Phase-B comparison config later points at the current legacy path. |

The three declared run roots also contain config shells outside the 163-record
dataset-v2 inventory. A config-only scan found 210 `resolved_config.json`
files: 163 exact dataset-v2 identities, 16 exact legacy identities, and 31
pytest-fixture records with placeholder/blank identities. Across 678 Ray
`params.json` files, 335 are empty, 240 serialize only a Python object
representation, and 103 are parseable objects (98 name dataset-v2 and 5 name
the current legacy path). None of the 888 config files contains a literal
`2020-` or `2021-`. The fixture and shell records are not affirmative evidence
of another market dataset, but neither empty metadata nor the absence of a year
literal is a freshness certificate.

Therefore 2020-2021 is **conditionally clear only against exact hash-bound
local datasets**. It is not retrospectively clear against the full exposure
union until the opaque proposer/manual/external tier is affirmatively bound or
the interval is conservatively rejected. This conclusion does not authorize a
bar request or select the interval.

## What the retained ledgers can and cannot certify

They **can** certify, within their declared scan roots and hashes:

- exact retained file/config bytes;
- which dataset digest a sufficiently complete config names;
- declared split/start modes, fixed starts and panel identities when retained;
- recorded control-sidecar coordinates under the stated canonical row-order
  assumption; and
- that the old dataset-v2 exposure inventory covered its three named roots at
  the time it was produced.

They **cannot** certify:

- that no other checkout, notebook, prompt, export, chart, copied file or human
  inspection used the same market outcomes;
- exact randomized starts when the executing dirty source and RNG/loader
  semantics were not retained;
- UTC/session identity for the legacy naive timestamps;
- truth merely because an assertion or manifest is hashed;
- freshness merely because a snapshot has new bytes, a later retrieval time,
  or a date absent from one indexed panel; or
- current Alpaca bar entitlement, licensed use, or future terminal-screen
  isolation.

`C:/Users/Admin/ray_results` resolves to the retained offload root already
scanned by the dataset-v2 inventory; it is not an independent clean source.
This pass also inspected `C:/itrader-ray` and repository experiment outputs.
That bounded filesystem inventory is still not an operating-system access log.

## Required carry-forward into the acquisition declaration

Before any price-data request, the reviewed declaration should:

1. name separate training and `terminal_screen_only` dataset identities and
   forbid semantic window overlap, not just byte/hash equality;
2. attach a union exposure certificate covering legacy and dataset-v2 training,
   proposer/curriculum, PBT/tuning, calibration, checkpoint selection,
   diagnostics/plots and manual inspection;
3. treat the legacy three-symbol stored date spans as uncleared unless their
   naive timestamps and every consumer are independently resolved;
4. preserve the old dataset-v2 inventory/recovery files rather than overwriting
   them with a new empty ledger;
5. start an append-only access ledger before acquisition, including automated
   jobs and human-readable exports; and
6. stop without opening outcomes if any window has unknown exposure, incomplete
   source/calendar binding, or a provenance mismatch.

No Q26 method, block rule, alpha, seed, draw count, support threshold, panel
date, anchor, or volatility estimator is selected here.

## Commands and primary evidence used

Read-only command families used in this pass:

```text
git status --short --branch
tools/knowledge/status.sh
Get-FileHash -Algorithm SHA256 <explicit retained files>
Get-ChildItem <declared roots> -Filter resolved_config.json/result.json
rg -n/-l <identity-only patterns> <source/config/docs roots>
Test-Path -LiteralPath <historical external dataset path>
pyarrow.parquet metadata plus symbol/timestamp-only projections
DuckDB SELECT symbol, session, COUNT(*), MIN(ts), MAX(ts)
JSON projections restricted to provenance/config/start/reference fields
```

Primary local sources were the retained files named above plus
[`snapshot.py`](../../../../itrader/inputs/market_data/snapshot.py),
[`data_partition.py`](../../../../itrader/rl_lib/evaluation/data_partition.py),
[`audit_panel_exposure.py`](../../../../itrader/scripts/experiments/audit_panel_exposure.py),
and [`panel_qualification.py`](../../../../itrader/rl_lib/evaluation/panel_qualification.py).
GitNexus was stale (indexed commit `1d5ba82`, current commit `3a3583d`), so it
was advisory only; current source and retained bytes control this note.
