# Independent Q27 review

Base: `3a3583d`. Scope: the Q27 files named in the assignment, including the
dated mechanical-repairs report. This review is limited to synthetic and
persisted public evidence; it does not use Ray, WSL, training, replay against
real market data, or real outcomes.

Reviewer assignment: `q27_mechanical_review_luna`. `REVIEW.md` requests
`gpt-6.1-sol` at `max`; runtime model/effort telemetry is not available here,
so this report records the request and assignment label without claiming the
actual runtime pair.

## Predeclared probe inputs

Recorded before opening implementation source:

1. Metadata eligibility and strict starts: use a small synthetic chronological
   session with regular timestamps, a session break, and declared split
   boundaries. Probe an eligible first/interior/last start; negative and
   length-valued starts; a missing timestamp; an off-cadence timestamp; a
   session-edge start; a start whose requested window crosses a split; and
   source/diagnostic start pairs in both orders. Verify metadata-only
   preflight and strict runtime task starts against the same eligibility rules.
2. EOF and reset identity: on 100 deterministic rows, probe `(start,
   horizon)` pairs `(99,1)`, `(98,2)`, `(96,4)`, plus each start one row earlier
   and later where representable. Check exact start/end, terminal observation,
   completion, liquidation accounting, reward components, and reset state.
   Repeat the same reset twice and compare initial observation and state.
3. Replay selection and completion: probe empty input; one row; a unique highest
   update with and without completion metadata; duplicate highest updates in
   both JSONL orders; same update with different sampled steps; and rows where
   completed status, sampled steps, and RLlib iteration agree or each disagree
   independently. Check that lower-update completion cannot mask a higher
   incomplete row.
4. Explicit R4 state tail: probe the declared two-value tail with valid and
   invalid modes, missing/wrong base or agent widths, reordered features,
   zero/one/two tail values, and actual observation-shape mismatch. Compare
   interventions with identical prefixes and changed tails, then identical
   tails and changed prefixes. Probe invalid CLI declarations before Ray
   startup.
5. For each probe, compare the current public consumer with base `3a3583d`
   using synthetic or already-persisted fixtures. Do not open market outcomes.

## Findings

### 1 of 3 found so far — BLOCKING: strict diagnostic starts still coerce malformed indices

Reproduction: on a 6,000-row synthetic validation Parquet file, use source start
`1000`, horizon `4`, window `60`, and a diagnostic task with
`start_idx: 2000.75`. Expected: reject the non-integer explicit index. Actual:
`load_validation_context()` accepts and hashes `2000.75`; strict `env.reset()`
reports start `2000`. The supported `start_index` alias also accepts
`2000.75`, and the numeric string `"2000"` passes preflight. Base
`_choose_start_idx()` likewise converted `2000.75` to `2000`; the new strict
gate still converts before checking. Thus the panel receipt can describe a
different start from the one actually evaluated. Relevant code:
`run_checkpoint_replay.py:49-51`, `data_partition.py:176-178`, and
`itrader_env.py:491-493`.

### 2 of 3 found so far — BLOCKING: metadata loading assumes lowercase, present `symbol` and `ts` columns

Reproductions: (a) a synthetic Parquet schema with `SYMBOL` and `TIMESTAMP`,
plus `data_symbols: ["AAPL"]`, makes metadata preflight raise ArrowInvalid
because `_load_market_frame()` filters on lowercase `symbol`; (b) OHLCV plus
`session_id` and `episode_boundary_cols: ["session_id"]`, without symbol or
timestamp columns, yields valid metadata ranges from `load_market_partition()`
but makes `ITraderEnv` initialization raise ArrowInvalid because
`_projected_market_columns()` requests lowercase `symbol` and `ts` anyway.
Expected: resolve optional metadata fields from the actual schema and allow
the configured boundary-only partition. Actual: both consumers reject these
layouts. Base `DataPartition` accepted an uppercase `SYMBOL`/`TIMESTAMP` frame
when passed its actual column names. The separate OHLCV-only metadata fallback
is now count-backed and passed its new public regression test. Relevant code:
`itrader_env.py:98-110`, `itrader_env.py:123-131`, and
`itrader_env.py:151-168`.

### 3 of 3 found so far — NIT: the dated report overstates when actual observations are checked

Reproduction: an R4 fake environment declares `observation_space.shape ==
(4,)` but returns a three-value observation from `reset()`. Expected: describe
the declared space check before reset and the returned-observation check after
reset, before policy scoring. Actual: the helper calls `reset()` once, then
rejects the observation before calling the policy. The guard works before any
counterfactual policy invocation, but line 62 of the dated report says the
actual environment shape is checked before reset. At
`policy_baselines.py:277`, the pre-reset check receives the declared
`observation_space.shape`; the returned array is checked after reset at line
134. Clarify that distinction in the report.

## Design principles

- Separation of concerns: OK — `DataPartition` owns eligibility and the
  environment shares its partition builder with metadata preflight.
- Programming by intention: VIOLATION — malformed explicit starts are
  silently coerced before strict eligibility checks (`run_checkpoint_replay.py:49-51`,
  `data_partition.py:176-178`, `itrader_env.py:491-493`).
- Encapsulation: OK — environment start selection uses the partition API
  rather than duplicating range calculations.
- High cohesion: OK — the in-scope helpers each handle panel validation,
  checkpoint selection, environment transitions, or state-tail checks.
- Low coupling: OK — replay preflight reuses the environment's partition
  construction; state diagnostics validate a declared schema before scoring.

## Executed checks

- Current focused suites before the follow-up metadata fallback change:
  `.venv\Scripts\python.exe -m pytest -q itrader/tests/test_env_core.py
  itrader/tests/test_policy_baselines.py itrader/tests/test_data_partition_sessions.py`
  — 62 passed.
- After the fallback change: the targeted Q27 selection passed 18 tests, with
  45 deselected; `test_metadata_preflight_supports_ohlcv_only_parquet_without_reading_features`
  passed separately. Synthetic OHLCV-only metadata preflight accepted starts
  `99`/horizon `1` and `96`/horizon `4`.
- A supplemental probe disabled `pandas.read_parquet` while calling that
  metadata-only preflight; row-count eligibility still succeeded. Synthetic
  validation starts were `69` rejected, `70` accepted, and `82` rejected for
  window `5`/horizon `4`; a session-edge start was rejected. Strict runtime
  starts accepted last eligible `96` and rejected `97`, `100`, and `-1` on the
  100-row fixture.
- Current synthetic 100-row EOF probes for horizons 1, 2, and 4 passed through
  `test_env_core.py`; base `3a3583d` clamped start `99` to `98` and reported
  done at `t=3`, `idx=99`, horizon `4`. Current terminal selection chose the
  unique highest update with matching completed status; duplicate-highest
  ordering and completion mismatches were covered by the focused tests. Base
  had no terminal selector.
- Current synthetic panel probes reproduced finding 1 for fractional, alias,
  and numeric-string forms. Current uppercase-filter and boundary-only runtime
  probes reproduced finding 2. The state-diagnostic CLI rejected an invalid
  R0 contract before importing Ray. A mismatched returned observation was
  rejected after reset and before any policy call.
- Terminal probes rejected an empty index, selected one unique highest row
  without status metadata, and rejected overlapping panel tasks in both input
  orders. The R4 validator accepted all four supported state modes, rejected
  absent/wrong dimensions and modes, and accepted only the two-value tail
  shape.
- No Ray startup, WSL, training, checkpoint replay, real market data, or real
  outcomes were used. The detached base comparison checkout was at
  `3a3583d`.

VERDICT: CHANGES
