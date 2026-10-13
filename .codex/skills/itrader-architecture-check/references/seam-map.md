# Source Seam Map

Discovery aid, not a second specification or a run authorization. Read only the
route relevant to the current scope. Confirm filenames, source behavior, nearest
tests and receipts on the current branch; source and AGENTS override this map.
Active decisions live in [masterplan](../../../../docs/plans/itrader_masterplan.md),
[next actions](../../../../docs/project-state/next-actions.md) and
[evaluation](../../../../docs/evaluation/README.md).

## Runtime And Experiment Entry

- [run_ablation.py](../../../../itrader/scripts/experiments/run_ablation.py)
  owns loadout variants and artifact collection. It launches native Python argv
  lists; its configured `python_exe` must select the intended environment, not an
  assumed PATH interpreter. Do not replace a lifecycle smoke with a market run.
- [run_repo_python.py](../../../../itrader/tools/run_repo_python.py) owns native
  repository interpreter selection and argv dispatch. [CUDA identity probe](../../../../itrader/tools/cuda_identity_probe.py)
  and [PowerShell compatibility launcher](../../../../itrader/tools/invoke_cuda_tested.ps1)
  are its runtime-profile neighbors; inspect their current supported interfaces.
  Use installed profiles without provisioning/sync unless separately authorized.
  Distinguish CPU default, CUDA selection, tensor execution and PPO qualification.
  Nearest tests: `test_run_repo_python.py`, `test_cuda_test_launcher.py`.
- [train_solo_V2.py](../../../../itrader/rl_lib/train/train_solo_V2.py) remains the
  main PPO entry. A launcher/binding failure is not a CUDA-kernel failure. DEV
  owns a production repair; TRAINING supplies a minimal reproducible command.
  Do not grow experiment-local argument converters or hide either output stream.
- [Process scope](../../../../itrader/rl_lib/evaluation/process_scope.py) owns
  replay process containment. Respect the actual reviewed run supervisor and
  live lease; inventory native helpers as well as Python/Ray. Serialize plain
  text and explicit scalar records, not live/provider objects. Kill only proven
  owned descendants under the scope, not apps/processes selected by name.

## Environment, Task And Solver

- [ITraderEnv](../../../../itrader/envs/itrader_env.py) owns reset, chronological
  observation assembly, position/horizon state and reward context. [TaskSpec](../../../../itrader/rl_lib/tasks/spec.py)
  owns task semantics; [RewardEngine](../../../../itrader/rl_lib/tasks/rewards.py)
  owns reward normalization/calculation, not economic promotion. `pnl_linear`
  remains default. Tests: `test_env_core.py`, `test_task_contracts.py`.
- [Observation contract](../../../../itrader/models/observation_contract.py)
  pins ordered 60x5 OHLCV, task embedding and optional two-value state. [Registry](../../../../itrader/models/registry.py)
  owns model selector/restore identity; [MLP](../../../../itrader/models/solver_mlp_film.py)
  remains default; [window LSTM](../../../../itrader/models/solver_window_lstm_film.py)
  resets hidden/cell state at each decision, not just episode end. Shared
  [task FiLM](../../../../itrader/models/task_film.py) conditions the market
  readout; state residuals remain outside the market window and task embedding.
  Stream continuity is not recurrent-state continuity. Tests:
  `test_solver_selection.py`, `test_semantic_observation_contract.py`,
  `test_representation_checkpoint_integration.py`.
- [PBT scheduler](../../../../itrader/rl_lib/train/pbt_scheduler.py) owns population
  selection/recovery; [restore adapter](../../../../itrader/rl_lib/train/ppo_restore.py)
  reconciles learner settings; [provenance](../../../../itrader/rl_lib/evaluation/provenance.py)
  verifies checkpoint bytes. Matching labels/scores cannot replace hashes.

## Proposer And Evaluation

- [Proposer factory](../../../../itrader/proposers/factory.py) selects backends;
  [OpenAI-compatible backend](../../../../itrader/proposers/openai_compat.py) handles
  inference. [Market summary](../../../../itrader/rl_lib/tasks/market_summary.py)
  describes supplied windows; [validation](../../../../itrader/rl_lib/tasks/validation.py)
  contains candidate task failures. Verify actual call-site window/end-index
  plumbing; a summary helper alone does not prove worker-specific causal input.
- [Economic diagnostics](../../../../itrader/rl_lib/evaluation/economic_diagnostics.py)
  separates accounting from optimization reward. [Disposition](../../../../itrader/rl_lib/evaluation/evaluation_disposition.py)
  validates declared evidence, metric, inference and promotion statuses; it does
  not derive them or authorize promotion. [Terminal preflight](../../../../itrader/rl_lib/evaluation/run_terminal_screen.py)
  checks completion and panel identities but explicitly does not authorize
  outcome evaluation. [Panel qualification](../../../../itrader/rl_lib/evaluation/panel_qualification.py)
  checks exposure/metadata evidence. Envelope approval is not exposure clearance;
  Q26 needs independently qualified uncertainty, not a green preflight or Sharpe.
- [Coordination](../../../../docs/coordination/README.md) owns chat handoffs and
  scoped leases. Only named human/GRILL-approved DEV--TRAINING repair pairs may
  communicate directly; boards still own evidence and GRILL owns advancement.
  Future UAH, family heads, MarketFramer/risk/execution are not inferred as live
  from contract classes. Keep deterministic online authority outside TaskSpec.

## Bounded Wording Check

Before a major skill edit, freeze the target, one routing objective, representative
train requests and independently chosen holdouts. Record the existing contract
coverage, make at most three logical mutations, then check both sets against
current source. Accept only improved explicit routing with unchanged ownership,
access and run gates; reject regressions and preserve the baseline. Keep one
compact accept/reject receipt. Static checks do not prove agent success, token
savings or runtime qualification; execution needs its own approved scope.
