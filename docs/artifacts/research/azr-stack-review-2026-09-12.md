# AZR inspiration and iTrader stack review

Date: 2026-09-12
Scope: Windows project `C:\Users\Admin\PROJECTS\iTRADER\itrader-azr`, audited from `agent/Watson-dev` at `bec5c77` with a pre-existing staged `itrader/rl_lib/tasks/rewards.py` change. This is analysis, not a training result or model implementation.

## User clarification after the initial audit

Phase B includes both offline/curriculum and online/replay work. The shared recurrent trunk selected through future ablations, family-specific policy/value heads and reward-specific components are under discussion in the ongoing `grill-me` workflow, not settled by this report. Families `00`–`04` are the target decomposition; `00`–`02` share `Discrete(3)` but can differ in inputs and semantics. The priority ordering below is an analyst proposal, not approval to defer all online work to Phase C or to implement a topology. Preserve uncommitted work and require user agreement before implementing these design decisions.

## Primary source and retrieval limits

Primary reference: Andrew Zhao et al., *Absolute Zero: Reinforced Self-play Reasoning with Zero Data*, local `C:\Users\Admin\PROJECTS\iTRADER\AZR-paradigm.pdf` (sibling of the clone; PDF cover identifies arXiv:2505.03335v2). Citations below use printed PDF pages and sections. The PDF contains 50 pages. `read_file` extraction failed because pages 20, 45–46 require OCR; PyMuPDF successfully exposed the text-layer sections used here. The audit read the method, equations, main ablations and Appendix D, not every image/example page. A public v3 was initially consulted before the user pointed to this local source; conclusions below are grounded in the local v2, not silently substituted from v3.

## What AZR actually contributes

- **Joint proposal and solving:** a single LLM fills both roles and is updated from both signals. Tasks are conditioned on past generated examples, validated with a code executor, then solved. See §§2.2–3.1, pp. 3–5, and Algorithm 1, p. 7.
- **Learnability proxy, not measured causal uplift:** Eq. 4, p. 5, defines proposer reward as zero for observed success rate `p=0` or `p=1`, otherwise `1-p`. It is asymmetric and favors hard-but-observed-solvable tasks, not a symmetric reward peaking at 50%. A finite sample with zero successes does not establish intrinsic impossibility. The reward is a proxy for learning potential, not a pre/post learning-progress estimate.
- **Validation and replay buffer:** valid programs/inputs/outputs enter task-type buffers; reference examples are sampled uniformly, and valid historical tasks fill short batches. Validity and difficulty reward are distinct. See §§3.3.2–3.3.3, pp. 7–8.
- **Verifier scope:** program integrity, restricted-package safety checks, and approximate determinism checks ground feedback. Solver correctness is executable value equivalence, including held-out examples for induction. See §§3.2–3.3.4, pp. 6–8. This is not a guarantee about an unknown future market.
- **Task-role normalization:** TRR++ uses separate reward baselines/statistics for each task type and role, not PPO/GAE and not a requirement for episode-isolated trading minibatches. See §3.3.5, Eq. 8, p. 8.
- **Ablation evidence:** Table 2, p. 12 reports overall averages 43.8 without generation references, 45.4 training only the solver, and 46.8 for the full setup. These are coding/math results on the studied AZR model, not predictions of iTrader returns. The solver-only row still uses the current learner to propose; it is not a frozen independent-proposer experiment.
- **Restraint matters:** Appendix D, pp. 49–50 reports no clear gains from extra complexity/diversity rewards and no significant benefit from the tested composite-function curriculum. Recency-based reference sampling showed potential collapse. Use these as warnings against untested complexity, not universal impossibility claims.

## Adaptation to iTrader

The useful analogue is:

```text
training-only market data -> structured task proposal
-> deterministic contract/feasibility validation -> validated task pool
-> solver rollout -> fixed, inspectable outcome metrics
-> curriculum feedback / proposed-task utility records
-> another proposal/training cycle
```

Current `itrader/rl_lib/train/train_AZR.py` generates a task pool and hands it to `train_solo_V2`; this is a bootstrap handoff, not the jointly optimized self-play loop in AZR. Backend availability is not proposer learning. iTrader deliberately separates a language-model proposer from a numerical PPO policy. Historical OHLCV is external data. Describe the project as **AZR-inspired adaptive curriculum learning for market control**, not literal zero-data AZR reproduction.

The market verifier must separate reproducible simulator accounting from evidence of generalizable economic edge. Freeze return/cost/risk metrics independently of proposer-controlled reward parameters. Split chronologically; purge overlapping information/label windows; fit preprocessing only on training data. Development feedback used to adapt the proposer is validation data, not an untouched final test. Keep a final time-separated test unavailable to proposal, tuning, and head selection. Report costs, turnover and drawdown alongside return and Sharpe.

Recommended proposer study (separate from R-series solver tests):

1. Fixed rule/random validated curriculum control.
2. Frozen LLM proposer under the same validation and solver budget.
3. Frozen LLM plus solver-feedback/reference-conditioned task selection.
4. Learned proposer only after repeatable utility/uplift measurement exists.

Match accepted solver timesteps and data eligibility; separately report generation calls, rejected tasks, tokens/latency and total compute. Do not call different-sized accepted task pools a fair proposer ablation. Novelty is not usefulness, and harder-to-profit market windows may simply be noisy rather than educational.

## Solver and conditioning findings

The source baseline is `SolverMLPFiLM`, not a recurrent policy. Its FiLM is `tanh(gamma(task))*hidden + beta(task)` (`solver_mlp_film.py`, `_condition_hidden`), with task dropout during training. Identity-centered `(1+scale)*hidden+shift` would be a changed parameterization requiring its own control; zero task input does not imply identity because projection layers have biases. Non-`film` conditioning currently ignores task features rather than implementing concatenation.

The default `TaskEncoder.embed` represents family, price field, reward ID, horizon, dimensions and a bounded seed feature. It does not directly encode reward parameters or bar resolution. Thus FiLM alone does not prove complete task semantics are observable. Audit task aliasing and seed-as-window-identity shortcuts; either freeze omitted fields in initial ablations or add explicit normalized task fields in a separate matched encoder version. Custom `t_embed` can provide a different contract and must be versioned.

True recurrence requires carried `(h,c)` at inference, matching training sequence states, correct valid lengths, padding masks, truncated BPTT, and explicit reset semantics. Stateful rollout is not full-episode gradient flow. Keep three time scales distinct: input window, recurrent learning-unroll length, and episode/reward horizon.

A full-window R1 versus newest-bar R2 comparison changes both accessible input and state lifetime. Retain it as an engineering comparison, but add a same-input, same-conditioning carry/reset pair before attributing gains to memory. Document the newest-bar variant's cold start and optional causal warmup separately. Do not count repeated overlapping windows as new chronological steps.

Post-LSTM FiLM modulates the readout but, without a task path into the cell, cannot make the hidden-state transition task-dependent. Distinguish readout conditioning from pre-cell/input conditioning. Keep concat, readout FiLM and pre-cell FiLM as named placements; use conditioning-matched controls rather than claiming a pure FiLM effect from multiple simultaneous changes. Task-conditioned initial state is deferred, not implied by `get_initial_state()`.

No market-only RNN can reconstruct the agent's realized sampled actions, fills, position or reward exactly from OHLCV alone. Add execution/position state to both feed-forward and recurrent controls when studying trade management. Online multi-candidate scoring must not mutate a shared recurrent state repeatedly: define task-independent shared market state or task-specific state/replay ownership before deployment.

## Opinion and priority

The strongest choices are a small task-conditioned numerical solver, typed contracts, independent environment/reward ownership, and deterministic risk/execution with abstention. Keep these. A compact recurrent trunk is a sensible research candidate, not a reason to replace PPO with an LLM policy or expand immediately to MoE/JEPA.

The main risk is experimental attribution: changing reward, observation information, task encoding, head capacity and curriculum together can produce an impressive curve with no interpretable cause. An RNN might help long-horizon tasks, fail to help, or lose to a well-observed feed-forward control. None of those outcomes should be pre-decided by family names.

Recommended priority: truthful data/metric contracts and matched R0 baseline; corrected window encoding as a useful control; a minimal truly recurrent vertical slice with lifecycle tests; conditioning/carry-state ablations; reward interaction; adaptive curriculum; only then family-routed heads and calibrated shadow gating. R1 need not win a benchmark before the true-recurrence hypothesis is tested—it must be correct and measurable. Preserve R0 as the default until held-out, multi-seed evidence warrants promotion.

## Evidence boundary

This document provides source/paper analysis and proposed experiment contracts. It does not report a trained recurrent solver, reproduced AZR results, profitable strategy, successful PPO smoke run, or fresh full-suite pass. Code audit and validation results belong in `recurrence-code-audit-2026-09-12.md` and the final documentation revision audit.
