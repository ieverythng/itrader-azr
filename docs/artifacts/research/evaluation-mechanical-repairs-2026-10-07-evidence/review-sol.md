# Q27 independent mechanical review

Review base: `3a3583d109af7a4c094124e73ad84008f371d8e3`.

Requested invocation pair: `gpt-6.1-sol` / `max`. This records the requested
pair; I have not independently verified the runtime model or reasoning setting.

Scope: metadata partition preflight, strict explicit starts and EOF/reset in
`itrader_env.py`; diagnostic eligibility and terminal selection in
`run_checkpoint_replay.py`; R4 state-tail validation in `policy_baselines.py`
and `run_checkpoint_state_diagnostics.py`; their focused tests; and the dated
Q27 artifact. Other dirty changes are outside this review.

## Inputs declared before reading the implementation

The repository status, documentation index and `REVIEW.md` were read before
declaring these inputs. No implementation, diff, focused test, or writer report
was read before this declaration.

1. Synthetic persisted CSV/JSON fixtures with no rows, one row, a minimally
   usable window, exactly one full horizon, and one additional row; metadata
   with missing, duplicate, or reordered timestamps; multiple assets in both
   input orders; and partition ranges with a middle gap or two disjoint pieces.
2. Explicit start inputs `0`, `1`, the last legal start, one beyond it, and a
   negative index; Python and NumPy integers; boolean, numeric string, integral
   float, fractional float, NaN, and infinity. Probe both environment config
   and reset options, including conflicting values and train/evaluation modes.
3. Repeated resets at the file boundary: prove no silent wrap, shortened
   requested episode, or use of a row outside the allowed partition. Probe
   `episode_length=1`, exactly the data boundary, and a horizon past it, as well
   as warmup/window requirements and the terminal/truncated signal.
4. Partition timestamps around 2024-02-29, 2025-12-31/2026-01-01, and equivalent
   UTC/offset representations when the public contract admits them; probe
   adjacent versus separated allowed spans and inconsistent metadata widths.
5. Synthetic checkpoint metadata for no checkpoints, one checkpoint, two
   checkpoints in both orders, explicitly pinned terminal checkpoints, legacy
   selected checkpoints, diagnostic-only eligible records, missing terminal
   records/paths, duplicate records, and conflicting terminal/evaluation
   metadata. Verify the public rejection text as well as the selection result.
6. R4 observations with missing state, an empty state, a short state, exactly
   the documented tail width, and a longer valid state; wrong policy family,
   absent/contradictory declared state width, and a flat observation whose
   market prefix could be mistaken for state. Probe all state ablation modes
   admitted by the public interface and inspect both baseline and diagnostic
   consumers for the same rejection boundary.

For executable probes I will use temporary synthetic fixtures without Ray,
WSL, training, checkpoint replay, or real market outcomes. I will compare the
available public behavior with the base commit, and explicitly record APIs
that do not exist there rather than invent a baseline result.

## Execution evidence

The review ran on `agent/Watson-dev`, HEAD `3a3583d109af7a4c094124e73ad84008f371d8e3`,
against the uncommitted Q27 changes. I did not write production code or tests,
stage, commit, launch Ray/WSL/training/checkpoint replay, or inspect real market
data/outcomes. Synthetic state policies replaced checkpoint execution.

Native runtime: repository `.venv/Scripts/python.exe`, Python 3.11.13,
NumPy 1.26.4, pandas 2.1.4. The default `C:/Users/Admin/itrader/python.exe`
failed importing pandas: NumPy 2.3.5 was incompatible with its pandas/pyarrow
binaries (`_ARRAY_API` missing; dtype size mismatch). No installation changed.

Executed commands and public probes:

- `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
  itrader/tests/test_env_core.py itrader/tests/test_policy_baselines.py`:
  **60 passed** on the original reviewed snapshot.
- After the finding below was fixed,
  `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
  itrader/tests/test_env_core.py itrader/tests/test_policy_baselines.py
  itrader/tests/test_data_partition_sessions.py`: **63 passed**.
- Scoped `git diff --check`: exit 0. Git emitted only its Windows LF/CRLF
  notices.
- `.venv/Scripts/python.exe itrader/tools/healthcheck.py`: exit 0; required
  dependencies passed. Optional llama_cpp/transformers were absent, CUDA was
  unavailable. This check did not start Ray.
- Explicit Git Bash `tools/knowledge/status.sh`: stale index at `1d5ba82`,
  current commit `3a3583d`; exit 1 because `python3` is absent in Git Bash.
  I used direct source instead of graph findings and did not refresh the graph.
- Six inline Python probes, piped through the native interpreter, exercised
  `ITraderEnv.reset/step`, `load_market_partition`, `load_validation_context`,
  `load_validation_env`, `load_checkpoint_selection`,
  `counterfactual_state_details`, `evaluate_checkpoint_state_diagnostics`, and
  the state diagnostic CLI's invalid-input preflight.

For base comparisons, a read-only import loader obtained every `itrader` module
from `git show 3a3583d:<path>`, including namespace packages. It did not fall back
to dirty checkout modules. An initial loader attempt failed because it omitted
namespace packages; that attempt produced no baseline result. The corrected
loader executed the same environment fixtures successfully.

Observed behavior, including rejection text:

| Public seam and synthetic input | Current result | Base comparison |
|---|---|---|
| 20-row constant OHLCV, window 5, starts 19/18/16, horizons 1/2/4, repeated task reset | Exactly 1/2/4 actions, end 20, done on final action, truncated false; reset preserves requested start | Base ended at 19; horizons 2/4 had only 1/3 actions. Horizon 1's requested start 19 was also reset-clamped internally |
| Strict task starts 0/1/17/-1 with horizon 4 | `Explicit task start ... is not eligible for horizon 4` | Base clamped these starts to 5/16 |
| Strict starts 5, 16, NumPy integer 16 | Exact legal start/horizon | Base accepts but has the EOF defect at 16 |
| Missing strict start; nonboolean strict flag; strict flag with random mode | Explicit missing-start, boolean-type, or task-spec-mode ValueError | New guard absent from base |
| Ordinary start 999999 with strict flag false | Clamps to 16 | Preserved generic clamp |
| 100-row validation metadata, source start 20, diagnostic 50 | Diagnostic-only context accepted; strict task mode enabled | Diagnostic-context API absent from base |
| Same metadata, diagnostic 95 or 999999 | `Explicit diagnostic/source panel start is not eligible` | Diagnostic-context API absent from base |
| Two symbols, two 40-row sessions, either raw symbol order or all rows reversed, window 5/horizon 4 | Metadata/runtime ranges both equal AAPL 8–36/45–68, MSFT 88–116/125–148 | Session-aware metadata helper absent from base |
| Leap-day minute data with a 10-minute gap, or reversed rows containing that gap | Both paths yield ranges 10–48/55–88; 49/50/54/89 rejected; 48/55/88 retain exact identity and two actions | New metadata helper absent from base |
| Duplicate timestamp near the gap | Both paths yield 10–48/56–88; seam-invalid starts rejected | New metadata helper absent from base |
| Equivalent UTC and +01:00 stored timestamp representations; Dec-31/Jan-1 session boundaries | Same eligible row identity; no new date-dependent failure | No new temporal rules introduced |
| Empty terminal index | `Terminal checkpoint selection requires checkpoint-index rows` | Selection API absent from base |
| Two highest-update rows at 57600/58000 steps, in either order; duplicate highest rows | `Ambiguous terminal checkpoint: highest update is not unique` | Selection API absent from base; I do not claim independent reproduction of the writer's pre-Q27 order-dependent selector |
| Update True, 12.0, string 12, null, or -1 | `pbt_progress_updates must be a nonnegative JSON integer` | New selector API absent from base |
| Completion status running, wrong steps, or wrong iteration | Completed-status/completion-mismatch ValueError | New selector API absent from base |
| Matching completion; best selector; unique explicit progress 57600/58000 in either order | Correct corresponding record accepted | Best-checkpoint JSON remains the historical selection mechanism |
| R0 state-free contract or observation width 0/1/4/6 against declared agent width 5 | R4-v1 contract/shape ValueError; policy calls 0 | Counterfactual/validation APIs absent from base |
| R4 contract, float dimension, boolean state dimension, unknown schema, reversed features | R4-v1 contract/feature-order ValueError; policy calls 0 | APIs absent from base |
| Valid R4 contract and exact width 5 | Seven named interventions; original observation preserved | API absent from base |
| Actual synthetic env modes masked/position/time/both, four-step episode | Four rows, 28 policy calls, correct terminal row; each intervention preserves the market/task prefix | R4 diagnostic APIs absent from base |
| CLI off/missing-dimension/reversed-feature/conflicting-env fixtures with matching synthetic checkpoint/config/dataset hashes | R4 ValueError before Ray import; an import blocker verified Ray was never reached | CLI file absent from base |

The interior protected controls used synthetic closes `100 + arange(100)`, start
20, horizon 4. Base/current matched exactly for short-only, flat-only, long-only,
and action sequence `[2, 0, 1, 2]`: reward sums
`-33.397878799263914`, `0`, `32.997878799263916`, `7.666749322679895`;
turnover `2`, `0`, `2`, `6`; identical gross/net returns and start/end/step
evidence. Base/current `load_validation_env` also returned the same unchanged
source config.

Projection instrumentation wrapped `pandas.read_parquet` and rejected any call
without an explicit metadata-only projection. The final price-only fallback
made **zero pandas data reads**; the timestamp/symbol fixture projected exactly
`['symbol', 'ts']`. Both admitted start 96/horizon 4.

Synthetic persisted fixture directories (all outside the repo; not deleted):

- Original current probes: `C:/Users/Admin/AppData/Local/Temp/q27-sol-6i9co9jx`.
- Original/fixed extended probes:
  `C:/Users/Admin/AppData/Local/Temp/q27-sol-more-07j4cfe5` and
  `C:/Users/Admin/AppData/Local/Temp/q27-sol-more-8xj5q32t`.
- CLI: `C:/Users/Admin/AppData/Local/Temp/q27-sol-cli-ytafstj5`.
- Gaps/orders: `C:/Users/Admin/AppData/Local/Temp/q27-sol-boundaries-ljrjgl7j`.
- Protected controls current/base:
  `C:/Users/Admin/AppData/Local/Temp/q27-sol-controls-dl_jwdtv` and
  `C:/Users/Admin/AppData/Local/Temp/q27-sol-controls-3blfok0v`.
- Projection: `C:/Users/Admin/AppData/Local/Temp/q27-sol-projection-mjjag4r6`.

The coordinator owns full-suite/pre-commit execution; I did not duplicate it
and do not claim its results here.

## Findings

**1 of the 1 found so far. BLOCKING — resolved and independently re-executed.**
Original location: `itrader/envs/itrader_env.py:168`, in
`load_market_partition`'s no-metadata fallback.

Reproduction: persist 100 rows with only OPEN/HIGH/LOW/CLOSE/VOLUME, each equal
to 100. Use window 5, validation fractions `[0.1, 0.8, 0.1]`, source task start
20/horizon 2 and diagnostic task start 50/horizon 2. `ITraderEnv` can reset and
run those starts, but `load_validation_context(...,
evaluation_task_panel_path=panel)` raised `ValueError: Market dataframe is empty`.
The same disagreement occurred on 6 rows with window 5/horizon 1: runtime reset
at 5 succeeded while `load_market_partition` failed.

Expected: preflight admits the same row starts as runtime without reading
prices. Actual: the old fallback constructed a DataFrame with an index but zero
columns. pandas reports such a frame as empty, and DataPartition rejected it.
The new API is absent from base; base's environment accepts the underlying
OHLCV-only format.

The writer replaced the columnless frame with a count-backed int8 metadata row
marker at current `itrader/envs/itrader_env.py:169`. Fresh-process verification
of the same inputs now returns range `[5, 5]` for 6 rows and `[5, 99]` for
100 rows; the validation diagnostic context is accepted. Empty input still
rejects. I reran the independent gap/order/state probes and 63 focused tests
against this fix. The marker performs no feature/outcome read and leaves
DataPartition as the only eligibility owner. No remaining BLOCKING or NIT
finding was identified in the assigned mechanical scope.

## Five design principles

1. **Separation of concerns: OK.** DataPartition owns eligibility; ITraderEnv
   supplies the market layout and transition boundary; replay owns checkpoint
   selection and declared completion comparison; optimization reward and
   economic evidence remain separate.
2. **Programming by intention: OK.** `load_market_partition`, strict task
   starts, and `validate_state_diagnostic_contract` expose the purpose of the
   checks, and failures describe the rejected public contract.
3. **Encapsulation: OK.** Source/diagnostic tasks use public partition membership;
   state consumers use a shared guard. The discovered zero-column adapter bug
   was fixed without copying partition rules into replay.
4. **High cohesion: OK.** Metadata/runtime frame construction shares one local
   partition constructor, and tail validation/mutation stay in the state
   diagnostic helper rather than scattered through scoring.
5. **Low coupling: OK.** Replay preflight lazily uses environment metadata loading;
   the R4 guard works without Ray or residual-initialization assumptions.
   No second owner of eligibility, duplicated business logic, or business rules
   in a template/UI were found in this scope.

## Verdict

The assigned Q27 mechanical scope is approved after the metadata fallback fix.
This verdict does not review the concurrently added opt-in diagnostic pilot,
other inherited dirty code, or certify fresh panels, all-sixteen completion,
economic promotion, or launch authority.

Explicit limits: numeric task starts such as `16.5` still canonicalize to 16,
and reset options do not override task-owned starts; these are preserved base
TaskSpec/reset behaviors rather than new type guarantees. A terminal row without
a path is returned by the selector but rejected by `load_replay_inputs`; the
selector alone does not validate checkpoint bytes. Completion's current schema
owns steps/iteration, not an added `pbt_progress_updates` field. Historical
indices with no completion receipt retain the documented bounded highest-update
behavior. I did not infer broader certification from these cases.

Snapshot hashes at 2026-10-07T23:24:09+02:00 identify whole files; this review
covers only the declared Q27 mechanical blocks:

| File | SHA-256 |
|---|---|
| itrader/envs/itrader_env.py | b9678ffa226e5673b9e4f50ad5b76e571d9457fa592d59fdfec9c8c30c7fc004 |
| itrader/rl_lib/evaluation/run_checkpoint_replay.py | f6058b677c6af23c6e205bb344b1d16608f431a65f75af800d2dbbad4c1efb60 |
| itrader/rl_lib/evaluation/policy_baselines.py | 9a13e18a049d59edcfecd0d3491659c8ae1ded5cdb89b340bbfe78150765ca52 |
| itrader/rl_lib/evaluation/run_checkpoint_state_diagnostics.py | 6ee19f5ec4086529c95a7d6af8c3e1e2a853a16e34ce21fc46e7de55ad968c9f |
| itrader/tests/test_env_core.py | 06f4a448cd6455835056e2bab116a35d69de65c4b82693093a87ac84c52d8cef |
| itrader/tests/test_policy_baselines.py | 087fd7dad233834f66393d0ac1c26d598aca2a83097f756d8cd8cbf8dd0a16c9 |
| docs/artifacts/research/evaluation-mechanical-repairs-2026-10-07.md | fe0759c9ee64988054d7354f8ad3e1c1d79600c75133bd6a27929720756eb077 |

VERDICT: APPROVE
