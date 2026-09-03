# Handoff

Last updated: 2026-05-02

## Start Here

Repo: `/Users/juanbendek/repos/itrader-azr`

Branch: `Phase-B-Training`

Current focus: assess whether PPO solver learning should synchronize gradient updates more tightly with episode/market-window boundaries, and preserve that decision for future proposer/online work.

## Current State

- The latest inspected commits add llama.cpp serving improvements, env locks, GitNexus slice contracts, future online contracts, and proposer infrastructure.
- The solver training baseline is PPO in `itrader/rl_lib/train/train_solo_V2.py`.
- The default model config currently uses `SolverMLPFiLM`, not the registered `SolverLSTMModel`.
- The existing `SolverLSTMModel` is a rolling-window sequence encoder, not a stateful recurrent RLlib policy.
- The existing `SolverLSTMModel` is probably stale for the current observation shape because it does not split off the task embedding before reshaping the price window.
- PPO uses `batch_mode = "complete_episodes"`, `train_batch_size = 4800`, `sgd_minibatch_size = 128`, and `num_sgd_iter = 30`.
- Task horizons default to `[96, 128, 160, 200]` bars at `1m`, though generated task pools may narrow that range.

## Latest Assessment

The strongest current recommendation is to keep PPO's complete-episode baseline and test stricter episode/regime alignment as an ablation.

Reasoning:

- Rewards, episode termination, and GAE are already episode-scoped by `batch_mode = "complete_episodes"`.
- PPO still aggregates many complete episodes into one optimization batch and shuffles minibatches. That is normal and often stabilizes gradients.
- Forcing every gradient update to stay inside a single episode would likely increase variance and overfit to local market noise unless the goal is a deliberately online/adaptive solver.
- The more promising structure is regime-aware evaluation and capability tracking by family/regime/horizon, especially for future online gating.

For solver-head work, the recommended order is:

- keep `SolverMLPFiLM` as the baseline,
- add a corrected window LSTM/GRU/attention encoder that explicitly splits price window and task embedding,
- compare on identical task pools and held-out rolling windows,
- only then implement a fully stateful recurrent policy if longer memory appears necessary.

## Working Tree

At the time of this handoff, the repo already had user/generated modifications before this docs pass:

- `.DS_Store`
- `itrader/proposers/openai_compat.py`

This handoff added docs under `docs/project-state/` only.

## Commands Run

- `git status --short --branch`
- `git log --oneline -n 12 --decorate`
- `git show --stat --oneline --decorate --no-renames -n 5`
- `git show --name-only --format=fuller bd6453e5`
- `git show --name-only --format=fuller 78d4d4fc`
- `git show --name-only --format=fuller cd25ade2`
- `rg` searches across docs, tests, task contracts, rewards, and training code
- Read `train_solo_V2.py`, `itrader_env.py`, task contracts, validation, rewards, solver models, and relevant docs/tests

No full test suite was run in this pass.

## Caveats And Blockers

- Solver-quality conclusions require real training/evaluation runs on the target workstation.
- Current `CapabilityProfile` is a contract, not a fully wired updater.
- Current LSTM code should not be treated as proof of a stateful recurrent PPO setup.
- `sgd_minibatch_size` alignment to horizon length may help diagnostics, but it does not by itself guarantee one episode per gradient update in PPO.
- More detail: `docs/project-state/solver-recurrence-assessment.md`.

## Next Action

Create a small ablation that compares:

- current baseline: `random_episode`, `ret_norm_reset=episode`,
- stream-contiguous variant: `start_idx_mode=stream_worker`, `ret_norm_reset=stream`,
- horizon-focused variant: one narrow horizon bucket at a time,
- optional low-mix variant: lower `train_batch_size` and `sgd_minibatch_size` near the episode horizon for diagnostics only.

Judge by held-out rolling-window capability by family/regime/horizon, not training reward alone.

## Continuation Prompt

Read `/Users/juanbendek/repos/itrader-azr/docs/project-state/handoff.md` and continue Phase B solver-quality work. First task: design and implement a controlled PPO ablation for episode/regime alignment without changing the default baseline. Preserve `TaskSpec` and future online contracts, and report results by family/regime/horizon so `CapabilityProfile` can later drive online candidate gating.
