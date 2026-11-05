# Independent evaluation-mechanics review (Sol)

Date: 2026-10-06. Base: `d3eb4c517075b1b85506f5500a9a5552d6186e3f`.
Scope: dirty evaluation modules, their tests, and environment/episode-evidence producers and consumers. The reviewer did not write the implementation. The initial assignment supplied the repository, dirty-diff scope, base, and review contract. Requested reviewer invocation: `gpt-6.1-sol` / `max`, `fork_turns=none`; trustworthy actual-runtime model/effort telemetry was not exposed, so this records the requested pair. This review covers evaluation correctness, not scientific approval of an uncertainty method or authorization to run the terminal screen.

## Inputs declared before implementation reads

The expectations below come from `REVIEW.md`, `AGENTS.md`, active architecture/masterplan, and `docs/project-state/r1-development-readiness.md` Q21/Q23/Q24/Q25 and replay acceptance requirements. At declaration time, no scoped implementation or test file had been read.

1. **Complete fixed panel and arithmetic.** Twelve literal symbols `[AAPL, MSFT, AMZN, NVDA, AMD, AVGO, GOOGL, META, SPY, QQQ, IWM, TLT]`; 32 distinct symbol/session anchors (three anchors on each of the first eight symbols, two on each of the last four); exactly horizons `[96,128,160,200]` on each anchor (128 tasks). For symbol ordinal `s` (zero-based) and horizon ordinal `h`, literal episode net return is `0.001*s + 0.0001*h`, gross return is net plus `0.0002`, total turnover is `2`, and cost is `1 bp` per turnover. Expected equal-symbol primary is `0.00565`, raw mean `0.00515`, and strongest single whole-panel constant direction must be selected once, not per anchor. Permute task/row order without changing these results; an identical control has exact zero paired contrasts. Actual decision steps must be positive integral and endpoint-consistent, and liquidation adds no decision bar. For unequal duration return pairs `(net=0.01, steps=100)` and `(net=0.02, steps=200)`, each rate is exactly `1 bp/bar`; macro rate is `1`, never the result of pooled-return division for unequal episode rates.
2. **Panel/row integrity.** Starting from the complete fixed panel, separately remove a task/episode, duplicate one, change a horizon to 97, replace one symbol, collide task identities, mismatch a row's symbol/start/end/horizon against its task, omit anchor/session/calendar metadata, use missing/zero/negative/nonintegral/bool/NaN steps, use an endpoint one bar off, and supply negative/nonfinite cost/turnover, net inconsistent with gross-minus-cost, or short/flat/long counts that do not sum to actual decisions. Expected: fail closed or explicit unavailable; no available-case renormalization or false economic labels.
3. **Sixteen-cell completion and binding.** Four representations `[R0,R0+R4-v1,R1,R1+R4-v1]` crossed with two curricula and literal paired seeds `[11,22]`; terminal selection predetermined. Construct synthetic manifests/checkpoint indices/configs and process receipts in temporary directories. Expected: the gate opens only after all 16 declared cells finish and their actual resolved native configs, dataset, monitoring panel, source, runtime, seed, progress, terminal checkpoint digests and source bindings agree with the declaration except for declared factors. Attack with 15 cells, duplicate/mislabelled cells, string/number truth flags, wrong progress, missing/best transient checkpoint, stale/mutated checkpoint, mismatched/wrong native optimizer recipe, missing/extra declaration fields, mutated resource declarations, and a copied/relabelled config. Expected: rejection with actionable reason; qualification alone must not launch evaluation.
4. **Qualification and freshness.** Calendar/session metadata includes aware RTH open/close timestamps, an opaque session ID `AAPL:2024-02-29:regular`, leap day, year transition, DST boundaries, an early-close session, exact 60/120-minute offsets, 60 past and 200 future contiguous eligible minute bars; all records remain validation-only. Test an unreviewed/unhashed calendar, weekday-only inference, gap, early close with insufficient bars, development overlap, two different entities in either order, and incomplete/unbound historical possible-exposure proof. Expected: no false certified-fresh claim, and incomplete metadata remains provisional/unknown. Sealed final test must stay excluded.
5. **Replay provenance and lifecycle.** Synthetic small checkpoint trees, literal JSON/JSONL artifacts, fresh UUIDs and two independently launched lightweight Python subprocess identities. Expected: digest verified before restore, immutable output invocation, atomic completed receipt only after final byte hashes/row counts, payload parity distinct from score parity. Attack absent/wrong checkpoint digest, corrupted/mutated artifact, copied producer identity, same process under two invocation UUIDs, string `"false"` parity/completion, missing trace/metric vector, changed effective config/panel/data/source/runtime/seed, reordered JSON records, interrupted/incomplete receipt, and duplicate artifact filename. No Ray/train/checkpoint replay or full market data will run.
6. **Episode producer and controls.** Synthetic OHLCV frame of contiguous aware one-minute RTH rows across two symbols/sessions, canonical literal task/reset metadata and deterministic action sequences. Expected: terminal task/family/regime metadata is truthful, fallback/generated task identities are distinguishable from explicit task panels, task/window/reset location matches evidence, economic gross/cost/net and action/step accounting are consistent including final liquidation. Literal flat, long-only and short-only controls preserve matched model/observation/state/task/source identities; unavailable metadata is not guessed.

Predeclared boundary set: empty, one, exactly expected cardinality, cardinality + 1; duplicate/missing/reordered alternatives; numeric value versus same value as string; booleans versus numbers; missing unit versus explicit bp/minute/sqrt-bar units; leap-day/year/DST/early-close dates. Probe adapters may use public signatures and existing focused fixtures after this declaration, but expected values remain the independent literals above.

## Initial repository checks

- Branch: `agent/Watson-dev`; HEAD equals review base; tracked staged/unstaged and untracked work are all in scope where relevant. Existing work is preserved.
- `tools/knowledge/status.sh` via Git Bash reported stale index `1d5ba82` (2026-06-17), then failed with `python3: command not found`. Direct source is authoritative; no graph refresh or WSL use in this review.
- Required source-doc reads completed before source inspection: `REVIEW.md`, `AGENTS.md`, documentation index, both architecture documents, masterplan, and current Q21/Q23/Q24/Q25/replay acceptance sections.

## Execution evidence and findings

### Executed checks

All Python calls used `C:/Users/Admin/PROJECTS/iTRADER/itrader-azr/.venv/Scripts/python.exe`. No Ray runtime, training, real checkpoint restore/replay, WSL, full market-data read, production/test edit, staging, commit or deletion occurred. Synthetic fixtures, receipts and scripts are in Windows temporary directories. Imports in focused existing tests load training/Ray modules, but no training or Ray cluster was started.

- Before scoped implementation reads: 37 tests across `test_terminal_panel_report.py`, `test_screen_completion.py`, and `test_terminal_screen.py` passed; then `probe_before_source.py` executed the independent arithmetic/integrity matrix and recorded base-file availability. Test fixtures supplied input schemas, not the numerical expected results.
- Before the qualification/receipt source reads: 15 focused tests passed (5 process/PBT recovery cases deselected), including a synthetic environment/control panel and two lightweight Python producer processes; `probe_metadata_receipt_before_source.py` exercised public metadata/receipt/parity APIs.
- Before the producer source reads: 46 tests across `test_policy_baselines.py`, `test_data_partition_sessions.py`, and `test_env_core.py` passed.
- After source tracing: one focused `test_checkpoint_evidence.py` test passed. Total head focused tests: **99 passed**, with no broad suite or healthcheck repeated concurrently with the parent.
- `git archive d3eb4c5` exported the tracked `itrader` package and `pyproject.toml` to a temporary base directory. The base's policy-baseline/environment tests passed **19/19**. The identical independent `probe_boundaries.py` ran against both head and that isolated base.
- Scoped tracked `git diff --check d3eb4c5 -- ...` passed. Git emitted only LF-to-CRLF notices. This command does not lint the untracked added modules; their behavior was exercised by the focused tests/public probes.
- `probe_state_schema.py` concretized the predeclared state/observation invariant after source inspection. `probe_initial_recipe.py` created initially wrong native recipes before any gate invocation; it did not rewrite an executed experiment's authority.

Probe scripts are retained at `C:/Users/Admin/AppData/Local/Temp/itrader-evaluation-review-sol-3c2fe02bc54a44ea8c4e4caf9c641fce/`. Exact commands for the most consequential head/base reproductions:

```powershell
& '.venv/Scripts/python.exe' 'C:/Users/Admin/AppData/Local/Temp/itrader-evaluation-review-sol-3c2fe02bc54a44ea8c4e4caf9c641fce/probe_boundaries.py' 'C:/Users/Admin/PROJECTS/iTRADER/itrader-azr'
& '.venv/Scripts/python.exe' 'C:/Users/Admin/AppData/Local/Temp/itrader-evaluation-review-sol-3c2fe02bc54a44ea8c4e4caf9c641fce/probe_boundaries.py' 'C:/Users/Admin/AppData/Local/Temp/itrader-evaluation-review-sol-3c2fe02bc54a44ea8c4e4caf9c641fce/base'
& '.venv/Scripts/python.exe' 'C:/Users/Admin/AppData/Local/Temp/itrader-evaluation-review-sol-3c2fe02bc54a44ea8c4e4caf9c641fce/probe_state_schema.py'
& '.venv/Scripts/python.exe' 'C:/Users/Admin/AppData/Local/Temp/itrader-evaluation-review-sol-3c2fe02bc54a44ea8c4e4caf9c641fce/probe_initial_recipe.py'
```

The following compact reproduction is retained here so the decisive checks do not depend on temporary scripts surviving. Run with the repo venv, pointing `PYTHONPATH` at either the dirty checkout or an isolated archive of the base. It creates only a temporary synthetic dataset/config/index and never starts Ray or restores a checkpoint.

```python
import copy, hashlib, json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from itrader.envs.itrader_env import ITraderEnv
from itrader.rl_lib.evaluation.policy_baselines import ControlPolicy, evaluate_policy_control
from itrader.rl_lib.evaluation import run_checkpoint_replay as replay

root = Path(tempfile.mkdtemp(prefix="itrader-independent-eval-repro-"))
data = root / "synthetic.parquet"
frame = pd.DataFrame({k: np.full(2000, 100.) for k in
                      ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME")})
frame["symbol"] = "AAPL"
frame["ts"] = pd.date_range("2024-02-29T14:30:00+00:00", periods=2000, freq="1min")
frame.to_parquet(data, index=False)
task = dict(family_id="00", price_id="SIM", horizon=96, reward_id="net_return",
            reward_params=dict(cost_bps=1, reward_scale=1000), seed=11,
            obs_dim=332, action_dim=3, bar_resolution="1m", start_idx=1400)
cfg = dict(df_path=str(data), window_size=60, data_split="validation",
           start_idx_mode="task_spec", evidence_cost_bps=1,
           task_sampler=dict(fixed_tasks=[task]))

def evaluate(config):
    env = ITraderEnv(config)
    try:
        rows = evaluate_policy_control(env, ControlPolicy("flat", 11), episode_count=1)
        return {k: rows[0][k] for k in ("start_idx", "end_idx", "steps", "horizon")}
    finally:
        env.close()

def diagnostic(source_start, diagnostic_start):
    config = copy.deepcopy(cfg)
    config["task_sampler"]["fixed_tasks"][0]["start_idx"] = source_start
    manifest = dict(config=dict(evaluation_config=dict(env_config=config)),
                    dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest())
    resolved, panel = root / "config.json", root / "panel.json"
    resolved.write_text(json.dumps(manifest))
    target = dict(task, start_idx=diagnostic_start)
    panel.write_text(json.dumps([target]))
    return evaluate(replay.load_validation_env(resolved, evaluation_task_panel_path=panel))

for label, inputs in (("clamped diagnostic", (1604, 999999)),
                      ("adjacent windows", (1400, 1556))):
    try:
        print(label, diagnostic(*inputs))
    except (TypeError, ValueError) as exc:
        print(label, type(exc).__name__, str(exc))

eof = copy.deepcopy(cfg)
eof["data_split"] = "all"
eof["task_sampler"]["fixed_tasks"][0]["start_idx"] = 1904
print("EOF", evaluate(eof))

if hasattr(replay, "load_checkpoint_selection"):
    records = [dict(pbt_progress_updates=12, pbt_progress_steps=steps,
                    selection_metric="evaluation/net_return_mean",
                    selection_score=score, stable_checkpoint=str(root / str(steps)))
               for steps, score in ((57600, 1.), (58000, 2.))]
    for order in (records, list(reversed(records))):
        (root / "checkpoint_index.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in order))
        print("terminal", replay.load_checkpoint_selection(root, "terminal"))
else:
    print("terminal selector absent on base")

from itrader.rl_lib.evaluation import policy_baselines as baselines
if hasattr(baselines, "evaluate_checkpoint_state_diagnostics"):
    class TailPolicy:
        def action_details(self, obs):
            tail = [float(v) for v in obs[-2:]]
            return dict(action=1, action_logits=[0., *tail],
                        action_probabilities=[.2, .6, .2], value_prediction=sum(tail),
                        policy_entropy=1.)
    env = ITraderEnv(dict(cfg, solver_state_mode="off"))
    try:
        rows = baselines.evaluate_checkpoint_state_diagnostics(env, TailPolicy(), episode_count=1)
        first = rows[0]["counterfactuals"]
        print("state-free schema", env.observation_schema_id, env.solver_state_dim,
              first["observed"]["action_logits"], first["masked"]["action_logits"])
    finally:
        env.close()
else:
    print("state diagnostics absent on base")
```

### Confirmed findings

**1 of the 5 found so far. BLOCKING — a declared disjoint diagnostic panel can actually replay its source window.** `itrader/rl_lib/evaluation/run_checkpoint_replay.py:102` checks raw declared starts, then passes them to `ITraderEnv`, whose reset clamps them at `itrader/envs/itrader_env.py:454`. The diagnostic loader never confirms that an explicit start is eligible or will remain unchanged. This invalidates the new non-overlap claim; diagnostic mode also suppresses source-panel parity, so that later check cannot contain this defect.

Reproduction: persist a synthetic single-symbol 2,000-row constant-OHLCV Parquet frame, a validation-only config with `window_size=60`, `start_idx_mode='task_spec'`, source task start `1604`, horizon `96`, Family `00`/`net_return`, and a cardinality-matched diagnostic task start `999999`. Call `load_validation_env(config_path, evaluation_task_panel_path=diagnostic_path)` and run public `evaluate_policy_control` on a synthetic flat policy. Expected: reject the ineligible explicit diagnostic start before evaluation; a certified disjoint identity cannot silently move. Actual head: accepts and emits `(start_idx=1604, end_idx=1700, steps=96)`, exactly the source outcome window. Base: diagnostic override API is absent (`TypeError`), so this is introduced with the new diagnostic path. This reproduction uses only a synthetic environment and deterministic control, not checkpoint/Ray replay. The qualified Q21 loader is a separate path with eligibility validation; this finding does not claim it opened a terminal panel.

**2 of the 5 found so far. BLOCKING — the new terminal checkpoint selector silently resolves ambiguous latest-update evidence by file order.** `itrader/rl_lib/evaluation/run_checkpoint_replay.py:185` uses `max(... pbt_progress_updates ...)` without requiring a unique terminal identity or matching completion status. Two rows at the highest update produce an arbitrary selection that is labelled `terminal_checkpoint`.

Reproduction: `checkpoint_index.jsonl` contains two records with `pbt_progress_updates=12`, steps `57600` and `58000`, distinct checkpoint paths, and literal selection scores `1.0` and `2.0`. Call `load_checkpoint_selection(stable_dir, 'terminal')`, then reverse the two rows and call it again. Expected: reject ambiguous terminal evidence consistently (or resolve through an independently predetermined completion identity); order cannot decide a terminal label. Actual head: first order returns steps `57600`/score `1.0`, reversed order returns steps `58000`/score `2.0`. Base: selector API absent. The dedicated Q21 `verify_completed_screen` does reject ambiguous matching terminals; it does not use this legacy replay selector. Both episode replay and state-diagnostic entrypoints consume the ambiguous selector, so the defect is not confined to a helper that has no callers.

**3 of the 5 found so far. BLOCKING — state diagnostics accept a state-free schema and alter the task embedding while claiming state interventions.** `itrader/rl_lib/evaluation/policy_baselines.py:129` unconditionally assigns to the last two observation values, and `evaluate_checkpoint_state_diagnostics` at `:268` never checks the environment's declared state block. Its caller in `run_checkpoint_state_diagnostics.py` also accepts an R0 checkpoint/config without a state-mode guard.

Reproduction: use the same synthetic 2,000-row validation environment with `solver_state_mode='off'`. It declares `observation_schema_id='r0_base'`, `solver_state_dim=0`, and width `332 = 60*5 + 32`. A lightweight policy returns logits `[0, observation[-2], observation[-1]]` and deterministic flat actions. Expected: state interventions fail or remain explicitly unavailable because no solver-state block exists; task inputs must stay fixed. Actual head: emits 96 state-diagnostic rows, changing the last two task-embedding logits from `[-0.05592424049973488,-0.40175363421440125]` to `[0,0]` under the named `masked` intervention. Base: this diagnostic API is absent. No learned-policy conclusion is drawn from the synthetic policy; the reproduction demonstrates the mutated slice directly.

**4 of the 5 found so far. BLOCKING, inherited — the final eligible dataset start yields a shortened episode.** `itrader/rl_lib/evaluation/data_partition.py:156` allows `last = outcome_end - horizon`, while `itrader/envs/itrader_env.py:492` terminates at `idx >= len(df)-1`. These endpoint semantics disagree on the final dataset row. The new terminal aggregator correctly rejects shortened fixed-horizon evidence; the producer/eligibility defect still exists.

Reproduction: synthetic 2,000-row frame, `data_split='all'`, window `60`, task start `1904`, horizon `96`, deterministic flat policy. Expected from eligibility/evidence contract: 96 decisions, end `2000`, final decision uses row `1999`. Actual in both head and base: 95 decisions, end `1999`, last row never consumed. This is **not introduced by the dirty evaluation diff**, and it is not evidence that retained controls were affected. Parent independently checked retained-control endpoints against the current dataset footer; that check is separate from this synthetic reproduction. Root owns any correction.

**5 of the 5 found so far. NIT — eligible adjacent diagnostic windows are conservatively rejected as overlapping.** `itrader/rl_lib/evaluation/run_checkpoint_replay.py:73` uses inclusive `<=` comparisons for intervals whose upper bound is `start+horizon`, which is exclusive in actual environment/frame evidence.

Reproduction: same 2,000-row validation config, source start `1400`/horizon `96`/window `60`, diagnostic start `1556`/horizon `96`. Full source rows are `[1340,1496)` and diagnostic rows `[1496,1652)`; both starts are eligible. Expected: non-overlap check accepts these disjoint row sets, subject to the other gates. Actual head: `ValueError: Diagnostic evaluation panel overlaps the source selection panel: start_idx=1556, horizon=96`. Base: override API absent. This is a false rejection and not a freshness bypass; normalize the half-open interval rule when fixing finding 1.

### Successful and bounded results

- Independent Q24 literals produced equal-symbol net `0.0056500000000000005` versus expected `0.00565`, raw episode mean `0.00515`, and equal-symbol Q25 rate `0.4141666666666667 bp/bar`. The rate expectation is the mean of literal horizon rates `0.5729166666666667`, `0.4375`, `0.35625`, and `0.29`. Reversed rows give the same values; identical controls give exact zero contrasts.
- The reporter rejects missing/duplicate/wrong horizon coverage, mismatched symbol/seed/family, nonfinite economics, wrong gross-minus-cost arithmetic, invalid liquidation/turnover, wrong 1-bp cost contract, invalid action fractions/exposure bounds, and missing/invalid/censored step/end evidence. Gross drawdown is labelled gross, and the output explicitly says descriptive rather than promotion/provenance acceptance. Uncertainty remains `not_scored_requires_equal_symbol_calendar_block_method`; no method/default receives approval here.
- Fifteen cells fail completion. The gate checks frozen asserted config hashes, seed/representation/state/curriculum, dataset/source/progress-panel identities, indexed terminal progress/generation, owned checkpoint path and actual checkpoint bytes. It rejects the tested stale/corrupt/partial/duplicate inputs. The official entrypoint rejects before panel access when completion or qualification is missing and returns `evaluation_authorized=False` on successful preflight.
- Qualification rejects malformed bounded intervals, lookback exposure, naive/off-minute timestamps, missing calendar membership, invalid RTH full-minute endpoints, sealed-test split, wrong task semantics and incomplete historical inventory coverage. `load_qualified_panel` binds retained proof/calendar/dataset/panel identities. It deliberately trusts the externally reviewed content of its certificate; hash binding alone does not certify historical starts or exchange-calendar truth.
- Receipt publication reserves an output without overwriting it and atomically publishes after artifact hashes. Incomplete receipts fail. The low-level byte verifier accepts optional contradictory claim/count fields, but the tested downstream two-process comparison validates one complete episode artifact, trace/payload hash, count, effective contract and witnessed process identity; the economic consumer additionally requires strict true parity flags and indexed metric/payload identities. These layered behaviors are not reported as bypasses based solely on the low-level API accepting an arbitrary report.
- A literal flat control at start `1400`/horizon `96` emits 96 decisions, end `1496`, zero net and action fractions `[0,1,0]` in both head and base. The full synthetic 128-task environment/control integration test confirms actual terminal evidence fits Q24/Q25 for ordinary eligible validation starts.

### Remaining gates, not implementation defects inferred from scope

`probe_initial_recipe.py` constructs all synthetic completion artifacts with native gamma `0.5`, epochs `1`, minibatch `4`, and LR `0.1` **on their first write**. The hash-binding completion function accepts all 16 because these are the declaration's initial config identities. Its documented contract relies on an external reviewed pretraining semantic seal; it does not certify Q22 optimizer equivalence or original entropy/cost/objective settings by itself. A wrong recipe declared initially therefore demonstrates why the pending semantic seal is required. It is not an attack that rewrites a reviewed predeclared authority after execution, and it is not a claim of an existing sealed-run bypass.

Similarly, arbitrary extra resource/source-proof keys and optional `action_counts` are not interpreted by the current completion/report schemas. The report requires `action_fractions` instead, and its real producers derive them from counted decisions. Ignoring an out-of-contract field is not classified as a defect without a relying consumer. Resource approval, source-bound historical exposure reconstruction, complete model-matched control receipts, Q26's method and the real 128-task executor remain required. Their absence prevents readiness/promotion claims, but this review does not label the read-only entrypoint as an implemented executor.

## Five design principles

1. **Separation of concerns — OK.** Provenance hashing, receipt publication, completion, qualification, descriptive aggregation, environment accounting and model inference have distinct roles. New summaries do not alter reward/PBT selection or invent broker/risk semantics; no business logic was found in templates/UI in this scope.
2. **Programming by intention — violation at `itrader/rl_lib/evaluation/run_checkpoint_replay.py:185`.** `terminal_checkpoint` is asserted for a file-order-dependent choice in finding 2. The intended deterministic terminal identity must be explicit.
3. **Encapsulation — violation at `itrader/rl_lib/evaluation/policy_baselines.py:129`.** A diagnostic assumes a semantic observation block from an unvalidated tail slice and can mutate another block's owned values (finding 3). The declared observation schema must control whether that block exists.
4. **High cohesion — OK.** The new report handles Q24/Q25 arithmetic, qualification handles metadata/history bounds, completion handles retained-cell evidence, and receipt modules handle provenance/lifecycle. Scientific inference and future execution are explicitly outside these helpers.
5. **Low coupling — violation at `itrader/rl_lib/evaluation/run_checkpoint_replay.py:106`.** Diagnostic non-overlap relies on the undocumented assumption that the environment preserves explicit declared starts, while reset may relocate them. Finding 1 is an unchecked dependency between the declaration gate and actual reset semantics; make that contract explicit and fail closed.

Review result: three newly introduced blocking correctness findings, one inherited blocking endpoint defect, and one optional conservative-boundary nit. No production/tests/active docs were changed by this reviewer, and no scientific or execution readiness follows from the green focused tests.

VERDICT: CHANGES
