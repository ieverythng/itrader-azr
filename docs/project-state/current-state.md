# Current State

Last updated: 2026-09-03

## Product Goal

`itrader-azr` is an RLVR-style trading stack with a solver trained on explicit, decomposed rewards and a proposer that generates validated task curricula. The near-term goal is to make the solver reliable enough that proposer quality, future QLoRA/LoRA work, and online task/candidate selection can be judged against a stable baseline.

## Current Phase

Phase B is complete through `feat: itrader main architecture E2E` (`3b3abc8`). Development has moved to the `feat/codex-adversarialproposer` branch, which is strictly ahead of `Phase-B-Training` by 3 commits (task contract scaffolding, gitignore for llama.cpp artifacts, merge) with zero divergence — a clean fast-forward target if merged back.

Recent commits show the project has moved through:

- task contract hardening and reward plumbing,
- future online contracts for `CandidateTask`, `ExecutionIntent`, `CapabilityProfile`, and `DecisionRecord`,
- GitNexus-backed repo knowledge docs,
- OpenAI-compatible proposer serving through local llama.cpp/vLLM-style endpoints,
- dependency lock and main-PC operations improvements.

## What Was Done (2026-09-03)

- **Episode/regime alignment ablation scaffolded** (next-actions item 1):
  - Config: `itrader/scripts/experiments/configs/episode_regime_alignment_ablation.json` — 4 variants (v0 random baseline, v1 stream contiguous, v2 stream+ret_norm_reset=stream, v3 horizon-aligned).
  - Regime tagger: `itrader/scripts/tag_pool_regimes.py` — labels task pools with vol/trend/liquidity regime tags from OHLCV.
  - Capability reporter: `itrader/scripts/generate_capability_report.py` — parses Ray stable_metrics.jsonl into p_solve / expected utility / calibration error by family/regime/horizon.
  - Lazarus launcher: `itrader/scripts/experiments/run_ablation_lazarus.sh` — probes llama.cpp at 172.24.16.1:8080 before launching; aborts if server is UP (OOM guard). Override with `--force`.
- **p_solve recovery**: `on_train_result` hook restored in `itrader/rl_lib/callbacks/ps_metrics.py` (was in stash@{1}, never committed). Exposes `p_solve` at top level of Tune results for stable TB/CSV consumption.
- **Stash audit**: All 3 stashes inspected. `stash@{0}` content ~70% already committed; unique artifact is a superseded Transformers+bitsandbytes `proposer_llm.py`. `stash@{1}` p_solve hook recovered. `stash@{2}` (Phase-A GitHub Desktop) fully superseded.
- **SolverLSTMModel obs-shape issue confirmed**: `flat_dim // self.win` breaks when task embeddings or regime tags are prepended to the flat observation vector. Decision: keep all 4 ablation variants on `SolverMLPFiLM`; defer LSTM head fix to the solver-head ablation (next-actions item 4).

## What is currently true in code

- Main environment: `itrader/envs/itrader_env.py`
- Main PPO training entrypoint: `itrader/rl_lib/train/train_solo_V2.py`
- AZR bootstrap runner: `itrader/rl_lib/train/train_AZR.py`
- Task contracts and validation: `itrader/rl_lib/tasks/`
- Proposer backends: `itrader/proposers/` (ollama, openai_compat, rule_based — the old Transformers-based `proposer_llm.py` from stash@{0} is NOT in any branch)
- Knowledge layer: `docs/knowledge/`
- Ablation/experiment harness: `itrader/scripts/experiments/` (run_ablation.py, configs/, run_ablation_lazarus.sh)

## What is intentionally not true yet

- The default solver is still the FiLM MLP baseline. The existing LSTM model is a window encoder over the flat rolling observation, not a stateful recurrent policy, and has an obs-shape limitation that must be fixed before it can consume task-embedded observations.
- Episode/regime alignment ablation is scaffolded but **not yet run**. Training requires the llama.cpp server to be offline (16GB RAM constraint). Use the Lazarus launcher or manually stop the server first.
- Capability profiles are not yet calibrated on held-out evaluation slices. The reporter script exists but needs real training output to produce meaningful numbers.
- Proposer work is still scaffolding for later QLoRA/LoRA and online candidate selection, not a production loop.

## Highest-value next actions

1. **Run the ablation**: Stop llama.cpp, invoke `bash itrader/scripts/experiments/run_ablation_lazarus.sh`, let all 4 variants complete.
2. Generate capability reports from the ablation runs using `generate_capability_report.py`.
3. Compare v0-v3 on held-out rolling windows; decide whether stream/horizon alignment is worth promoting to default.
4. If recurrence shows value, fix `SolverLSTMModel` obs-shape handling and run the solver-head ablation (item 4).

## Open risks / watch items

- **OOM**: Training + Qwen 3.8 UD-Q3_K_XL (13GB) cannot coexist on 16GB RAM. The Lazarus script guards this, but manual training invocations must still respect it.
- **Stash@{0} proposer_llm.py**: The 199-line Transformers+bitsandbytes LLM proposer exists only in the stash. It was superseded by ollama/openai_compat proposers. If you want it back, `git show "stash@{0}:itrader/llm/proposer_llm.py"`. Otherwise the stash can be dropped after verification.
- **Solver head confusion**: Do not attribute alignment-ablation results to recurrence. The FiLM baseline has no hidden state across steps; any recurrent claim needs the corrected window encoder or a true RLlib recurrent policy first.
- **Capability calibration**: Until evaluation-slice capability reports exist, online gating thresholds should stay advisory, not production-blocking.
