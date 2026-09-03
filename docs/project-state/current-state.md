# Current State

Last updated: 2026-05-02

## Product Goal

`itrader-azr` is an RLVR-style trading research stack with a solver trained on explicit, decomposed rewards and a proposer that generates validated task curricula. The near-term goal is to make the solver reliable enough that proposer quality, future QLoRA/LoRA work, and online task/candidate selection can be judged against a stable baseline.

## Current Phase

Phase B is active on branch `Phase-B-Training`.

Recent commits show the project has moved through:

- task contract hardening and reward plumbing,
- future online contracts for `CandidateTask`, `ExecutionIntent`, `CapabilityProfile`, and `DecisionRecord`,
- GitNexus-backed repo knowledge docs,
- OpenAI-compatible proposer serving through local llama.cpp/vLLM-style endpoints,
- dependency lock and main-PC operations improvements.

## Repository Shape

- Main environment: `itrader/envs/itrader_env.py`
- Main PPO training entrypoint: `itrader/rl_lib/train/train_solo_V2.py`
- AZR bootstrap runner: `itrader/rl_lib/train/train_AZR.py`
- Task contracts and validation: `itrader/rl_lib/tasks/`
- Proposer backends: `itrader/proposers/`
- Knowledge layer: `docs/knowledge/` and `tools/knowledge/`
- Project memory: `docs/project-state/`
- Solver recurrence assessment: `docs/project-state/solver-recurrence-assessment.md`

## Architecture Boundaries

- `TaskSpec` remains the canonical offline training task contract.
- `CandidateTask`, `ExecutionIntent`, `CapabilityProfile`, and `DecisionRecord` are future online/shadow-mode contracts and should stay explicit and validated.
- The current primary solver path is PPO with `SolverMLPFiLM` task conditioning, even though `SolverLSTMModel` is still registered and available.
- Reward computation stays explicit through `RewardEngine` with finite outputs and component metrics.
- Proposer outputs must pass validation/clamping before entering a task pool or online decision path.

## Solver And Episode Facts

- PPO config currently uses `batch_mode = "complete_episodes"`, so rollouts are collected as complete episodes before training batches are built.
- `train_batch_size` defaults to `4800`, `sgd_minibatch_size` defaults to `128`, and `num_sgd_iter` defaults to `30`.
- Default task horizons are `[96, 128, 160, 200]` bars at `1m` resolution.
- `ITraderEnv` supports `start_idx_mode` values `random_episode`, `stream_worker`, and `task_spec`.
- `ret_norm_reset` can reset return normalization per `episode` or per `stream`.
- The existing `SolverLSTMModel` is a rolling-window sequence encoder, not a stateful recurrent RLlib policy.
- The existing `SolverLSTMModel` is likely stale for the current `[price_window_flat, task_embed]` observation layout because it infers feature shape from the whole vector.

## Known Gaps

- The current PPO optimization batch can mix full episodes from different market windows, regimes, rewards, and horizons.
- There is no dedicated experiment yet comparing baseline mixed PPO batches against more regime-contiguous or horizon-aligned training schedules.
- Capability tracking contracts exist, but the repo still needs a concrete updater that evaluates solver performance by family/regime/horizon on held-out windows.
- Online/shadow-mode candidate evaluation is specified in docs but not yet fully implemented.
- The solver-head frontier needs an explicit comparison between `SolverMLPFiLM`, a corrected time-series window encoder, and a true stateful recurrent policy.

## Validation Baseline

Recent inspected tests cover:

- PPO config assembly,
- env reset and finite step outputs,
- task validation and contract coercion,
- dynamic metrics extraction,
- proposer backend/task generation surfaces.

Validation not rerun in this handoff pass:

- full pytest suite,
- actual PPO training,
- proposer-backed training comparison,
- online/shadow-mode replay.

## External Source Documents

- `README.md`
- `docs/architecture.md`
- `docs/phase_b_plan.md`
- `docs/iTRADER_Codex_Implementation_Spec.md`
- `docs/phase_b_handoff_2026-03-04.md`
- `docs/windows_llama_cpp_qwen35.md`
- `docs/knowledge/wiki/training-pipeline.md`
- `docs/knowledge/wiki/trading-environment.md`
- `docs/project-state/solver-recurrence-assessment.md`
