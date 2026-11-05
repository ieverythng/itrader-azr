# Historical exposure recovery — 2026-10-06

## Finding

The retained control sidecars recover 512 recorded episode rows across 16
configs. Those rows reduce to 48 unique starts: 32 unpanelled starts, eight
starts in the September 24 crossed panel, and eight starts in the January 2026
panel. Each maps to a symbol and timestamp in the canonical 12-symbol,
regular-session dataset projection. The recorded sidecar symbol agrees with
the mapped symbol for all 512 rows.

The six- and eight-symbol counts previously described as different frame
projections are task-symbol coverage. All 16 control configs list the same 12
`data_symbols`; the 8x4 panels contain eight start anchors, of which the
September panel maps to six distinct symbols and the January panel to eight.
No six- or eight-symbol input-frame config was found in this bounded scope.

The timestamp mapping is conditional on the canonical row ordering described
below. Symbol identity is directly cross-checked by the sidecars, but an exact
historical source snapshot proving the within-symbol row order was not found.
For the 26 randomized configs, no exact source recovery or source-bound RNG
reconstruction is available. Their possible exposure therefore remains the
full validation-start domain: the earlier metadata audit's conservative
exclusion still covers all 2,052 provisional anchors, leaving zero certified
fresh anchors.

## Scope and method

The audited inventory is
[`exposure_inventory.json`](../r1-prerequisites-2026-10-05-evidence/exposure_inventory.json),
SHA256 `74b4fff809a85ebb22c055b3fd0a33e9c2c3e145a044e366d296622ded81162f`.
Its dataset SHA256 is
`20aac96de6da77e20beeaff214d380ef06184faf301ff0dd53bed0ffd4c04ce9`.
The inventory classifies 16 controls without top-level start-mode metadata and
26 `random_episode` configs.

For the controls, I read each resolved config and at most 32 lines from its
`episodes.jsonl` sidecar, projecting only task seed, horizon, symbol, start/end
indices, and steps. I checked the two referenced panel JSON files using only
task seed, horizon, and start index. The data mapping read only `symbol`, `ts`,
and `session_id` from regular-session Parquet rows, one symbol at a time. No
OHLCV, return, reward, regime, or other outcome field was read for selection.

All 16 control configs specify the same twelve-symbol list, in this order:
`AAPL, MSFT, AMZN, NVDA, AMD, AVGO, GOOGL, META, SPY, QQQ, IWM, TLT`.
They specify `data_sessions=regular`, `expected_bar_interval=1min`, and
`episode_boundary_cols=[session_id]`. The regular-session metadata projection
contains 5,515,061 rows. The canonical projection used for coordinate lookup is
ordered by symbol alphabetically and then by UTC timestamp. Cumulative
per-symbol ranges are:

| Symbol | Canonical row range (end exclusive) | Rows |
| --- | ---: | ---: |
| AAPL | `[0, 460140)` | 460,140 |
| AMD | `[460140, 920280)` | 460,140 |
| AMZN | `[920280, 1380289)` | 460,009 |
| AVGO | `[1380289, 1835974)` | 455,685 |
| GOOGL | `[1835974, 2294096)` | 458,122 |
| IWM | `[2294096, 2754232)` | 460,136 |
| META | `[2754232, 3214371)` | 460,139 |
| MSFT | `[3214371, 3674511)` | 460,140 |
| NVDA | `[3674511, 4134651)` | 460,140 |
| QQQ | `[4134651, 4594791)` | 460,140 |
| SPY | `[4594791, 5054927)` | 460,136 |
| TLT | `[5054927, 5515061)` | 460,134 |

For a recorded index, local symbol row is `start_idx - symbol_block_start`; that
row's UTC timestamp and `session_id` give the coordinate. The sidecar symbol
matches this mapping in 512/512 records. Using the configs' listed symbol order
as frame order matches only 138/512 records; treating all sessions as the
frame matches 15/512. This identifies the regular-only, alphabetically grouped
projection as the consistent mapping. It does not independently prove that
every historical process used the same within-symbol sort implementation.

Example: recorded `start_idx=786049` is local row 325,909 in AMD's canonical
block, mapping to `2025-05-07T16:19:00Z`,
`AMD:2025-05-07:regular`.

## Recorded control starts

| Control group | Configs | Sidecar rows | Unique starts | Symbols observed | Panel identity check |
| --- | ---: | ---: | ---: | ---: | --- |
| Unpanelled September 24 controls | 5 | 160 | 32 | 12 | No panel file referenced |
| September 24 crossed 8x4 panel | 5 | 160 | 8 | 6 | All 32 task identities match |
| January 2026 crossed 8x4 panel | 6 | 192 | 8 | 8 | All 32 task identities match |

All 16 sidecars contain 32 rows, and every row satisfies `steps == horizon`
and `end_idx == start_idx + steps`. Sidecar task-identity rows are identical
within each group. For both explicit panels, every `(seed, horizon, start_idx)`
in the 32-row sidecar matches the referenced panel. Across all 512 rows, the
mapped start and final episode row remain within the same symbol and
`session_id`. The full list of 48 mapped coordinates and the config/sidecar
hashes are in
[`exposure-recovery-starts.json`](exposure-recovery-starts.json).

The panel labels are not frame-size declarations. The older crossed panel has
eight unique indexed anchors but six observed symbols (AMD, AMZN, AVGO, IWM,
NVDA, QQQ); the January panel has eight anchors and eight observed symbols
(AMD, AMZN, AVGO, IWM, MSFT, NVDA, QQQ, SPY). The unpanelled controls cover all
twelve configured symbols.

## Randomized configs and source recovery

The 26 randomized configs identify Git commit
`891fcaa2ba2200d0affe2c97a6d9ada592dbcfaa`, but each also records an aggregate
fingerprint over 82 Python files. The Git tree at that commit has 72 tracked
Python files and its reconstructed aggregate fingerprint is
`f284c7381d9cb0a5023b69d7231be63c85bbc2946c1947c1dce35a710879f482`. None of
the six run fingerprints equals the Git-tree fingerprint:

| Run fingerprint | Configs |
| --- | ---: |
| `0368366b50363546f8ddeefda188cb09ea522c48faa1d80779564f603f239df5` | 2 |
| `5eb39fb7516dfe3e2a5481ae82f6106ed6c50308db8719424f8017f2ca6520af` | 8 |
| `17ce9ff1fa5248a2940f201aa78ac07782f4cc0e76ad784ebcadd36e6d4cabcc` | 4 |
| `35e103d40e3c53bc853f6f721afe409db743759d3aa9575e153a9158f93366d9` | 4 |
| `adea033f1c7b85b27ff0f9ca444b1f9a18a5f23d18459293907dd69034f860b9` | 4 |
| `9374f7710665a4c557de9784a33909265592ea70ee25b21b9094caf30a66e7de` | 4 |

The bounded search examined metadata names in the 26 exact `stable_trial_dir`
directories and immediate sibling directories whose names indicated source,
snapshot, or manifest material. It also checked those candidate directory names
at the eight corresponding stable-metrics and trial parents. No source,
snapshot, or manifest candidate file or sibling directory was found. The
resolved configs retain only aggregate 82-file fingerprints, not per-file
hashes or source contents. The search did not enter other trial directories,
read `result.json` or metrics, or inspect checkpoints.

The inventory reports 594 randomized evaluation reports without retained
episode-start payloads. Their fixed-task seeds can generate candidate starts
under an assumed implementation, but no such reconstruction is bound to the
six dirty source fingerprints. I therefore admit zero randomized starts as
historically recovered and do not use seed-derived candidates to reduce the
possible-exposure domain. The control sidecars recover their starts as
recorded evidence; they do not supply the missing random-run source identity.

## Exposure consequence

The prior qualification's conservative domain remains controlling: exclude
every possible validation start for each configured symbol, including the
60-bar input lookback and up to 200 future bars. That domain covers all 2,052
provisional eligible anchors. Under the current evidence, zero anchors can be
certified fresh. This is a possible-exposure bound, not a claim that every
excluded bar was observed.

No panel was selected or frozen. This audit performed no training, checkpoint
replay, R1 work, data collection, WSL operation, staging, commit, push, or
deletion. It wrote only this report and its small start-coordinate/provenance
JSON.
