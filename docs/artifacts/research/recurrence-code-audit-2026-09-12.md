# iTrader recurrence / FiLM / PPO code audit

Date: 2026-09-12
Scope: Windows repository `C:\Users\Admin\PROJECTS\iTRADER\itrader-azr`, branch `agent/Watson-dev`, HEAD `bec5c775d3adaafe758202c7175791f6e8b323a`. This is a read-only source/config/test audit. The pre-existing staged `itrader/rl_lib/tasks/rewards.py` change was not touched. No recurrent training result is reported here.

## Findings first

### F1 — High: the active PPO path hard-codes R0; no LSTM/FiLM variant selector exists

**Evidence**

- `itrader/rl_lib/train/train_solo_V2.py:112-123` always returns `"custom_model": "solver_mlp_film"` and configures FiLM.
- `itrader/rl_lib/train/train_solo_V2.py:536-539` registers three names, but registration is not selection: `solver_lstm_v2`, `solver_mlp`, and `solver_mlp_film` are registered while the base config still selects only `solver_mlp_film`.
- The CLI at `train_solo_V2.py:665-700` has no `--solver_variant`, `--model`, `--conditioning`, `--max_seq_len`, or recurrent-state option.
- `itrader/scripts/experiments/run_ablation.py:174-196` only merges generic JSON args and invokes the same training module. Its configs vary start-index/normalization/horizon/reward settings, not R0–R4 heads.

**Impact**

The current executable training path is R0 (`SolverMLPFiLM`) only. R1, R2a, R2b, R3, and R4 are design cells, not implemented ablation variants. A run from the existing ablation runner cannot be evidence for recurrent integration or a FiLM-placement comparison.

**Disposition**

The active architecture/masterplan wording is accurate when it calls recurrent heads unimplemented. Keep this explicit until a model selector and end-to-end tests land.

---

### F2 — High: the legacy LSTM is neither a valid task-embedded window encoder nor a stateful RLlib recurrent model

**Evidence**

- `itrader/models/solver_lstm.py:19-24` infers `self.n_feat = flat_dim // self.win` from the entire observation width.
- `itrader/models/solver_lstm.py:31-38` reshapes all of `obs` into `(B, win, n_feat)`, runs an LSTM, discards the returned hidden state (`seq, _`), and selects the last within-window output.
- `itrader/models/solver_lstm.py:43` returns the incoming `state` unchanged. There is no `get_initial_state`, `forward_rnn`, sequence-length handling, or terminal masking.
- The active environment layout is `window_size * 5 + 32`; `itrader/envs/itrader_env.py:119-123` creates that shape, and `train_solo_V2.py:68-75` configures the same dimensions. At the default window 30 this is 182 values. The legacy LSTM computes `182 // 30 == 6`, then asks PyTorch to reshape 182 values into 180 values, which is invalid.
- The corrected observation split required by the research contract is not present in this file. The research design correctly describes the required split at `docs/research/recurrent-solver-ablation.md:71-86`.

**Impact**

The file name and model registration must not be used as evidence that iTrader has an LSTM solver. Selecting `solver_lstm_v2` on the current observation contract should fail at the reshape; even with a divisible input, it would only be a reset-every-forward window encoder and would not carry state across environment steps.

**Suggested acceptance checks**

1. Instantiate the candidate with `[T*F + task_dim]` and assert the exact price/task split.
2. Assert chronological order (`newest_bar == price_window[..., -1, :]`).
3. For a stateful model, assert `state_out != state_in` after a nontrivial step, carry state into the next step, and reset it at an episode boundary.

---

### F3 — High: no R-series runner/config can currently enact the documented experiment matrix

The research matrix is documented at `docs/research/recurrent-solver-ablation.md:88-101`, but the code/config surface does not implement it.

- `itrader/scripts/experiments/schemas/ablation.schema.json:43-83` defines PPO/data knobs but no solver-head, recurrent-cell, conditioning-placement, state-feature, or sequence-length contract. `trainArgs.additionalProperties` is permissive, but that does not make unknown arguments executable.
- `_to_cli_args` in `run_ablation.py:25-36` forwards arbitrary JSON keys as CLI flags. A future key such as `solver_variant` would reach the parser, which currently rejects it.
- `ablation_smoke_quick.json:25-43` compares `random_episode` with `stream_worker` and return-normalization choices; both use the default R0 model.
- `episode_regime_alignment_ablation.json:21-64` contains five alignment/horizon cells, not R0–R4 cells.

**Impact**

The existing alignment configs are valid controls for start-index/normalization questions, but they cannot support claims such as “R2b beats R2a” or “recurrence improves memory.” A model selector must be added before the runner can be called a recurrent ablation runner.

---

### F4 — High: default task embeddings alias task semantics and expose a seed shortcut

**Evidence**

- `itrader/rl_lib/tasks/spec.py:39-69` defines task fields including `reward_params`, `bar_resolution`, `regime_tags`, `difficulty_target`, `metadata`, and `start_idx`.
- `itrader/rl_lib/tasks/encoder.py:25-55` derives the default embedding only from `family_id`, `price_id`, `reward_id`, `horizon`, `obs_dim`, `action_dim`, and `seed % 997`. It does not encode reward parameter values, bar resolution, regime tags, difficulty targets, metadata, or start index. A supplied `t_embed` bypasses the derivation at `encoder.py:25-35`, but there is no embedding-version contract.
- `itrader/rl_lib/tasks/samplers.py:85-99` samples/normalizes reward parameters separately from task identity. `samplers.py:101-118` creates family-00 tasks with a bar resolution and reward parameters, but those fields are not represented in the default embedding.
- `itrader/envs/itrader_env.py:262-268` uses `current_task.seed` to choose a random episode start. The same task seed is then visible through `seed % 997` in the task embedding. This is not a direct start-index feature, but it is a low-cardinality deterministic proxy for the seed that drives the window-selection path and can become an identity shortcut.
- `itrader/envs/itrader_env.py:362-364` computes the embedding before `_merge_task_sampler_overrides`; reward overrides applied at reset are therefore definitely absent from the embedding.

**Impact**

Two tasks can produce the same default embedding while differing in reward parameters, bar resolution, or regime metadata. Conversely, the seed feature can let a policy memorize pseudo-task/window identity instead of learning the intended semantics. This is especially dangerous in task-conditioned recurrent comparisons: apparent “memory” or “task conditioning” gains may be encoding an RNG shortcut.

**Suggested control**

Version the task-embedding schema. For the first matched study, either freeze omitted fields and remove `seed` from the model-visible embedding, or explicitly encode normalized fields that are intended to be observable. Generate start indices from an independent RNG/identity path when testing task generalization. Report whether task identity, reward parameters, regime tags, and bar resolution are observable in each cell.

---

### F5 — High: the documented R1→R2 comparison changes input information and reset warm-up, so it does not isolate persistent memory

The design says R1 consumes a full observed window while R2/R3 consume the newest bar (`docs/research/recurrent-solver-ablation.md:88-101`). The environment confirms that its window is chronological and fully assembled at reset: `itrader/envs/itrader_env.py:196-201` fills rows `[idx-T, ..., idx-1]`, and `_get_obs` concatenates the deque in order at `envs/itrader_env.py:144-148`.

Therefore, a future R1/R2 result would change at least two factors:

1. R1 sees `T` market bars immediately at step zero; a newest-bar R2 starts with a cold hidden state and sees one bar per recurrent step.
2. R1 has no persistent state but has a full local history; R2 has persistent state but initially lacks the prior `T-1` bars unless an explicit causal warm-up is added.

This is a useful engineering progression, but it is not a causal “memory versus no memory” comparison. Add a same-input carry/reset pair before attributing a gain to recurrence, or label the existing R1→R2 path as a deliberately unequal information comparison. Keep the three time scales explicit: observation window, recurrent unroll/truncated-BPTT length, and episode/reward horizon.

---

### F6 — High: R2a versus R2b currently conflates conditioning placement with the claim “task-conditioned memory”

The design names R2a “task concat” and R2b “per-step FiLM” at `docs/research/recurrent-solver-ablation.md:103-115`. If R2b applies FiLM only after the LSTM, task information changes the policy/value readout but not the recurrent transition. It cannot make the hidden-state update task-dependent unless task information reaches the LSTM input or cell gates.

The existing FiLM implementation also has two concrete semantics that the architecture docs must preserve:

- `itrader/models/solver_mlp_film.py:53-59` returns `tanh(gamma(task)) * hidden + beta(task)`, not the documented conservative identity-centered form `(1 + scale(task)) * hidden + shift(task)` at `recurrent-solver-ablation.md:117-124`.
- `solver_mlp_film.py:53-55` returns the unconditioned hidden representation whenever `conditioning != "film"`; it does not implement concatenation. A config value of `"concat"` would silently ignore task features.

**Suggested comparison vocabulary**

Name and test these separately: input concat, pre-cell/input FiLM, recurrent-state FiLM, and readout FiLM. Do not call readout-only FiLM task-conditioned memory. If the project changes the FiLM parameterization to identity-centered scale, treat that as a separate architecture factor and retain the current R0 control.

---

### F7 — High: R4 solver-state observability is documented but absent, and must be matched across controls

`ITraderEnv._build_obs` at `itrader/envs/itrader_env.py:150-153` returns only the flattened market window plus task embedding. The environment tracks position internally (`envs/itrader_env.py:191-205`) and exposes it in reward context (`envs/itrader_env.py:311-328`), but position, previous action, PnL, and episode progress are not appended to policy observations.

The corrected research contract explicitly requires the same fields for feed-forward and recurrent cells at `recurrent-solver-ablation.md:170-174`. No implementation, config, or test currently does this. A recurrent-only addition would give recurrence extra information and invalidate R4 attribution. Define typed fields, shape/order, normalization, and terminal semantics first, then add them identically to R0 and the winning recurrent control.

---

### F8 — High: no recurrent PPO state lifecycle, rollout, or evaluation evidence exists

**Training configuration**

- `train_solo_V2.py:208-237` configures Torch PPO, `batch_mode="complete_episodes"`, and PPO minibatches, but does not configure recurrent sequence chopping, `max_seq_len`, sequence-preserving minibatches, or a recurrent model contract.
- `batch_mode="complete_episodes"` is a collection boundary, not proof of recurrent state handling or full-episode BPTT.

**Checkpoint path**

- `training_fn` restores an algorithm checkpoint at `train_solo_V2.py:451-462` and saves checkpoints at `train_solo_V2.py:485-500`. This is weight/algorithm checkpointing; no rollout state or hidden-state reset/carry assertion is performed.
- `itrader/scripts/inspect_ckpt.py:169-279` inspects configs/metrics and prints an RLlib restore hint. It does not execute a rollout or validate recurrent state.
- `itrader/scripts/pbt_output.py:256-264` likewise emits `Algorithm.from_checkpoint(...)` and `algo.evaluate()` as a hint, not as a repository evaluation implementation.

**Metrics/evaluation path**

- The active config has no `evaluation_interval`/evaluation worker setup in `build_base_config`.
- Search of the source tree found no recurrent `get_initial_state`, `forward_rnn`, `state_in`, `state_out`, `compute_single_action`, or rollout implementation.

**Impact**

There is currently no evidence for hidden-state shape, `(h,c)` padding, valid sequence lengths, terminal reset, truncated BPTT, state carry after checkpoint restore, or evaluation warm-up. These are mandatory acceptance gates for R2/R3.

---

### F9 — High: family/regime evidence claims are not currently supported by emitted metrics

The source registry is conservative:

- `itrader/rl_lib/tasks/families.py:26-69` defines roadmap families, but only family `00` is enabled at `families.py:73-79`.
- `TaskSampler00._sample_random_task` hard-codes `family_id="00"` at `itrader/rl_lib/tasks/samplers.py:101-118`.
- The family tests assert the active set is only `("00",)` at `itrader/tests/test_task_families.py:17-20`.

`PSMetrics` records task horizon at `itrader/rl_lib/callbacks/ps_metrics.py:119-141`, but does not emit family ID, regime tags, reward parameters, bar resolution, or task metadata into episode custom metrics. The capability report only builds family/regime/horizon groups when `task_metadata` exists (`itrader/scripts/generate_capability_report.py:139-172`). The current stable report path therefore cannot substantiate family/regime capability claims by itself.

Do not describe families `01`/`02` as active learned semantics, and do not describe family `03` as trend following: the source calls it trade management (`families.py:55-60`). Add per-episode metadata with a documented aggregation key before using family/regime slices as evidence.

---

### F10 — Medium: `train_AZR.py` has a real unbounded backend-exception retry path

This is adjacent to the recurrence scope but affects evidence-generation reliability. The loop is bounded by `n_generated < max_attempts` at `itrader/rl_lib/train/train_AZR.py:83-95`, and exceptions are caught at `train_AZR.py:102-109` with `continue`. On repeated proposer exceptions, neither `n_generated` nor `empty_batches` increments, so the loop can continue indefinitely and never reach the nominal attempt/empty-batch bound.

The newer generator has the intended bounded counter at `itrader/scripts/gen_task_pool.py:268-289` (`backend_errors`). This report does not change `train_AZR.py`; the issue should be tracked separately and should not be hidden behind a successful task-pool claim.

---

### F11 — Medium: dependency/version authority is inconsistent for future RLlib recurrence work

`docs/research/recurrent-solver-ablation.md:126-149` describes a pinned RLlib 2.9 legacy ModelV2 contract. `docs/phase_b_plan.md:81-90` names Ray RLlib 2.47.1 and Gymnasium >=0.29 as the Phase B target, while `docs/environment_strategy.md:39-52` records a current workstation with Ray 2.9.0 and Gymnasium 1.0.0 plus a compatibility mismatch.

Before implementing R2/R3, make one version matrix authoritative for the actual runner, lock, and recurrent API. Otherwise a model can be “correct” against one RLlib API and fail collection or state handling against another.

## Focused verification

### Repository/status checks

- `tools/knowledge/status.sh --advisory` ran successfully and reported the GitNexus index stale: indexed commit `1d5ba82`, current commit `bec5c77`. Direct source reads were used; generated knowledge was not treated as authoritative.
- Initial Git status was:

  ```text
  ## agent/Watson-dev...origin/agent/Watson-dev
  M  itrader/rl_lib/tasks/rewards.py
  ```

  The staged rewards file was not edited.

### Focused test command attempted

Command:

```text
./itrader/tools/run_repo_python.sh -m pytest -q \
  itrader/tests/test_no_nan_forward.py \
  itrader/tests/test_env_core.py \
  itrader/tests/test_train_config.py
```

Actual result:

```text
./itrader/tools/run_repo_python.sh: line 42: /mnt/c/Users/Admin/PROJECTS/iTRADER/itrader-azr/.venv/Scripts/python.exe: cannot execute binary file: Exec format error
./itrader/tools/run_repo_python.sh: line 42: /mnt/c/Users/Admin/PROJECTS/iTRADER/itrader-azr/.venv/Scripts/python.exe: Success
```

Exit code: `1`. No package installation, source edit, branch switch, or training run was performed. The runner selected the Windows virtual-environment executable from WSL, so these tests did not execute in this audit. The closest completed checks are the direct source/config/test inspection above. No test pass or recurrent smoke result should be claimed from this session.

### Native Windows bridge follow-up (2026-09-12)

The audit provenance was rechecked against the clone after the initial report:

- `git branch --show-current` returned `agent/Watson-dev`.
- `git rev-parse HEAD` returned `bec5c775d3adaafe758202c7175791f6e8b323a8`.
- The HEAD recorded in the report header is the complete 40-character value; the earlier truncated `...323a` form is not used as provenance.

Before bridge execution, pure-stdlib AST parsing completed for the staged reward module, healthcheck, and the three focused test files:

```text
AST_OK itrader/rl_lib/tasks/rewards.py
AST_OK itrader/tools/healthcheck.py
AST_OK itrader/tests/test_no_nan_forward.py
AST_OK itrader/tests/test_env_core.py
AST_OK itrader/tests/test_train_config.py
```

Bridge discovery returned:

```text
/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe
/mnt/c/WINDOWS/system32/cmd.exe
```

One read-only subprocess attempt was made for each bridge:

```text
powershell.exe -NoProfile -NonInteractive -Command "$PSVersionTable.PSVersion.ToString()"
cmd.exe /c ver
```

Both failed before process launch with the same exact WSL error:

```text
OSError: [Errno 8] Exec format error: 'powershell.exe'
OSError: [Errno 8] Exec format error: 'cmd.exe'
```

Because neither bridge can execute from this WSL session, the native focused test command and healthcheck were not run. The blocked native commands were:

```text
cd /d C:\Users\Admin\PROJECTS\iTRADER\itrader-azr && .venv\Scripts\python.exe -m pytest -q itrader/tests/test_no_nan_forward.py itrader/tests/test_env_core.py itrader/tests/test_train_config.py
cd /d C:\Users\Admin\PROJECTS\iTRADER\itrader-azr && .venv\Scripts\python.exe itrader/tools/healthcheck.py
```

No install, system repair, source edit, branch change, config change, or training run was attempted. The AST results are syntax-only validation and do not establish test or healthcheck success.

The pre-existing staged reward diff remained unchanged during this follow-up:

```text
git diff --cached --numstat -- itrader/rl_lib/tasks/rewards.py
23\t96\titrader/rl_lib/tasks/rewards.py

git diff --cached --binary -- itrader/rl_lib/tasks/rewards.py | sha256sum
956f7624aea90a19d8276216f9cae6a5ca2a4b821cd7629beb21871418c0a341  -
```

There was no unstaged diff for `itrader/rl_lib/tasks/rewards.py`; the only report change in this follow-up is this audit append.

## Architecture/documentation recommendations

1. **Add an implementation matrix** to the active architecture entrypoint: R0 implemented/default; R1–R4 unimplemented; legacy `SolverLSTMModel` incompatible and non-stateful. Include the exact selector/config key once implemented.
2. **Define one observation schema/version**: `[price_window_flat | task_embedding | optional matched solver-state fields]`, feature count, chronological order, padding, and which TaskSpec fields are intentionally observable. Explicitly ban seed-derived shortcuts in generalization studies unless the study is about identity conditioning.
3. **Separate the three time scales** in the recurrent design: market input window, RLlib unroll/max sequence length, and episode/reward horizon. State whether R1→R2 is an unequal-information engineering comparison or a matched memory comparison.
4. **Replace the loose FiLM-placement wording** with a table for input concat, pre-cell FiLM, state/readout FiLM, and initial-state conditioning. State which placements affect recurrent transitions and which only affect logits/value.
5. **Specify the RLlib lifecycle contract**: `get_initial_state`, state tensor ordering/shapes, valid sequence lengths, padding masks, reset at `terminated`/`truncated`, truncated BPTT, minibatch sequence preservation, and evaluation warm-up. Add a checkpoint-restore rollout test.
6. **Specify matched R4 semantics** for both feed-forward and recurrent controls, including position/progress/PnL normalization and action/observation timing.
7. **Define an evidence schema** for per-episode family, regime tags, horizon, reward ID/parameters, start index, seed policy, checkpoint, git commit, dataset hash, and evaluation split. Do not rely on aggregate `episode_return_mean` for family/regime claims.
8. **Make the dependency matrix authoritative** before using RLlib-specific recurrent APIs; reconcile the RLlib 2.9 current environment, the 2.47.1 target plan, and Gymnasium pins.
9. Keep the AZR comparison bounded: `train_AZR.py` currently creates a task-pool/bootstrap handoff to PPO, not AZR's jointly optimized proposer/solver loop. The existing AZR stack review correctly frames this as “AZR-inspired adaptive curriculum learning,” not literal zero-data AZR reproduction (`docs/artifacts/research/azr-stack-review-2026-09-12.md:20-43`). Preserve the small numerical PPO solver, typed task contracts, and deterministic risk/execution shell; recurrence should be a measured opt-in, not a reason to replace PPO with an LLM policy or to add MoE/JEPA prematurely.

## Evidence boundary / conclusion

The source supports the following claims:

- R0 `SolverMLPFiLM` is the active PPO solver.
- The checked-in LSTM is an incompatible reset-every-forward window encoder, not a stateful recurrent policy.
- R1–R4 are a documented research design, not implemented solver variants.
- Current metrics/configs do not establish recurrent gains, family/regime capability, or evaluation generalization.
- The active architecture and masterplan are broadly aligned with that boundary, but the project should add the implementation matrix, task-observability contract, and RLlib lifecycle/evidence schema before any recurrent result is promoted.

No claim is made here about a trained recurrent model, profitable strategy, successful PPO smoke run, full-suite pass, or reproduced AZR result.
