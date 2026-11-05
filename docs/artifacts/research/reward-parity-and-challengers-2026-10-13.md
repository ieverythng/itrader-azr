# REWARD38: reward parity first, declared challengers second

2026-10-13 Europe/Madrid. Source/primary-literature decision packet; no code,
training, replay, synthetic run, outcome access or reward adoption. Branch
`agent/Watson-dev`, HEAD `8f9fcf9` plus inherited dirty work. Hashes and bounded
child receipts: [owned board](../../coordination/research.md).

## Outcome and immediate decision

**Recommendation:** corrected PPO/PBT and implemented R1 justify a fresh reward
audit, not assuming a particular reward will escape the flat basin. Preserve the
priority matched R0/R1 study on ONE explicitly approved optimization recipe;
prepare component/accounting stress contracts in parallel. Mixtures are secondary.

**Unresolved approval:** documented CLI default remains `pnl_linear`, whereas the
[4x2x2 draft](../../../itrader/scripts/experiments/configs/phase_a_r1_corrected_compat_fixed_4x2x2_20261006.json)
explicitly selects `net_return`,1bp,scale1000. Its fixed evaluation panel also
requests net-return tasks. Human approval of a representation/optimizer recipe
does not silently approve this objective switch. Keep the default until GRILL
records a reward decision; if net-return is selected for the matched study, label
it explicitly and retain the legacy control/provenance. No draft is edited here.

Commit88361f0 contains the explicit net-return implementation; f045fde contains
the Sep14 separate `pnl_linear_reference`/`net_return_1bp_pbt` bridge configuration.
Those are committed experiment/implementation provenance, not current human
approval or an isolated causal reward comparison: that historical config also
changes iterations/PBT/population. No historical run outcomes were reopened.
CUDA37 qualifies a bounded mechanical lifecycle, not reward quality or market edge.

## Actual route and current mechanics

Public route: [train_solo_V2](../../../itrader/rl_lib/train/train_solo_V2.py)
`_task_sampler_config/build_base_config` ->
[TaskSampler00](../../../itrader/rl_lib/tasks/samplers.py) ->
[TaskSpec](../../../itrader/rl_lib/tasks/spec.py) ->
[env reset/context](../../../itrader/envs/itrader_env.py) ->
[RewardEngine](../../../itrader/rl_lib/tasks/rewards.py) -> legacy PPO/GAE;
[EpisodeEvidence](../../../itrader/rl_lib/evaluation/episode_evidence.py) calculates
economic evidence independently. Reuse these owners, not a new reward wrapper.

**Source facts:** the training sampler samples turnover weights
`[0,.0005,.001,.002]`, DD weights in`[0,.04]`, delta-DD, turnover cap`.02`,
DD cap`.10`. Reward RNG is separate from task RNG. Sampling then configured
ranges/global/per-ID overrides then normalization determines random-task params;
fixed/pool tasks bypass that sampling path. Freeze resolved TaskSpecs, not just
the reward name. Registry defaults are not necessarily training defaults.

Reset clears position, PnL/DD and strategy-return history. Return EMA may persist
across same-stream episodes under stream reset, otherwise resets. Cost schedules
are net-return-only sampled-step schedules and affect subsequent resets, not an
already running task. Evaluation resets EMA per episode. The draft fixed panel
has1bp tasks; without a fixed panel, inherited sampled reward params may retain
initial schedule cost even though independent evidence still scores1bp. Do not
claim universal evaluation reward-cost parity from evidence_cost_bps alone.

For literal target positions, gross step return is `p_t*r_t`, turnover is
`abs(p_t-p_previous)`, and terminal liquidation is`abs(p_H)` with no extra bar.
Economic evidence is `G-c*1e-4*(sum(turnover)+abs(p_H))`. Reversal costs two units.

`pnl_linear` is a SHAPED objective: PnL multiplied by pnl_w and adaptive unit
in norm_mode all; abstract turnover_w times turnover, capped but NOT multiplied
by that unit; positive DD increments or terminal DD multiplied by weights/unit
and capped; final reward_scale multiplies the components. It has no liquidation
cost. Sum of positive DD increases is not maximum DD. Same-step asset return
updates EMA before reward scaling; this is post-transition reward processing,
not evidence of anticipatory policy input. Hidden EMA/reset history still matters.

**Inference:** adaptive PnL scaling versus unscaled turnover changes their
effective tradeoff across time; `.0005` cannot generally be called5bp execution
cost. Only under explicit raw/unit1,pnl_w1,uncapped arithmetic would that
coefficient correspond to5bp per transition, still omitting liquidation.
Cap/normalization/DD changes alter policy incentives, not merely readability.

`net_return` is `k*(pnl_delta-c*1e-4*(turnover+terminal*abs(position)))`, without
adaptive normalization/DD/caps. Constant positive k preserves undiscounted
fixed-task ordering. At fixed cost, summed reward/k equals terminal economic net;
at gamma`.99`, PPO still discounts later profit/cost/liquidation. Accounting
parity is not equality of the discounted training and undiscounted evidence
objectives. Gamma1 is a separately approved factor, not an automatic fix.

## What to fix/refactor, versus what to name as a challenger

**Minimal behavior-preserving work proposed:** reconcile reward names/units and
resolved params in receipts; retain whole-episode literal arithmetic, component
signs/sums, terminal-once accounting, schedule/reset identities and optimizer
scale diagnostics. Separate stale prose from behavior. RewardEngine converts
nonfinite output/components to zero: audit explicit fault evidence so sanitation
does not masquerade as a legitimate flat/zero reward; changing failure policy
requires a reviewed contract, not a silent refactor.

**Named behavioral challengers, not fixes to historical pnl_linear:** remove
adaptive normalization; remove/alter caps; disable DD; change delta versus terminal
DD; add liquidation cost; express turnover in economic bp. Preserve legacy ID/
recipe. For an exact raw-net control use existing net_return rather than claiming
an approximately configured pnl_linear is identical. Tune one declared factor
at a time; sampled risk preferences versus fixed params are themselves a factor.

**Do not reuse composed_linear unchecked:** current sampled`pnl_w` reads context
`pnl` (absent from the public context), not`pnl_delta`; positive dd_w/turnover_w
add positive DD/turnover terms. This is a source-level sign/key issue for that
candidate, not evidence that the active default used it. Exclude it from the
ladder until independent literal oracles settle intended semantics; do not remove
legacy registry behavior as collateral work.

## Scale, discount and task-conditioning gates

**Source facts:** the draft freezes gamma`.99`,value-loss coefficient2,
vf_clip20,net scale1000; these are declarations, not choices made here.
Installed default-env RLlib legacy PPO standardizes advantages over collected
policy batches, not each reward task. Its value loss clamps SQUARED error to
vf_clip_param, not an old-value prediction delta. The repository
[diagnostic PPO adapter](../../../itrader/rl_lib/train/ppo_diagnostics.py) follows
that loss. Saturation can suppress value gradients. Installed CPU-source findings
must be hash-checked against the selected CUDA profile before any future run.

**Local inference:** global advantage normalization does not normalize value
targets, balance each task, eliminate critic scale or ensure optimizer invariance.
Track target/error dispersion, loss saturation, actor/critic gradient interaction,
entropy-relative influence and restoration identity before attributing a plateau
to reward design. Changing reward scale and vf settings together tests a complete
optimization recipe, not an isolated scale effect.
[PPO](https://arxiv.org/abs/1707.06347) separates policy/value/entropy objectives;
[Engstrom et al.](https://arxiv.org/abs/2005.12729) show implementation choices
matter on their benchmarks, not that any chosen change improves iTrader.

[semantic_v2 encoder](../../../itrader/rl_lib/tasks/encoder.py) hashes canonical
reward ID/params and task fields into standardized bytes, excluding seed/start.
It discriminates many contracts, but is NOT structured numerical preference
conditioning or guaranteed smooth generalization. Every changed scalar can change
the whole hash. Legacy encoding omits reward params and includes seed; explicit
t_embed bypasses derivation. Reset overrides precede embedding. For a pnl-linear
training control, silently evaluating only net-return TaskSpecs also changes
conditioning and may be out-of-training-distribution: freeze the evaluation task
contract separately from the independent1bp scoring formula.

[Successor features](https://arxiv.org/abs/1606.05312) models reward changes with
features/weights under shared dynamics; it does not validate opaque hash transfer.
[PopArt](https://arxiv.org/abs/1602.07714) adapts value targets while compensating
outputs; [multitask PopArt](https://arxiv.org/abs/1809.04474) addresses task scales.
These motivate a later optimizer/conditioning factor, not a new env reward flag
or guaranteed FiLM-PPO gain. Defer normalization-state/head/restore changes until
simple fixed-scale diagnostics show a need.

## Sharpe: implemented signal, theoretical alternative, missing state

**Source facts:** current sharpe_only and pnl_sharpe use GROSS strategy mtm
history. They compute rolling sample mean/std, tanh(score/target), then weighted
score difference; warmup/near-zero std returns0. pnl_sharpe adds this after the
base's final scaling, so reward_scale also changes their relative weight.
Pure Sharpe excludes turnover/liquidation/DD/PnL. It is not net Sharpe evidence.

At constant weight and gamma1, differences telescope to the FINAL ROLLING-window
score, not full-episode Sharpe. For gamma<1, with initial scorePhi0:

```text
sum(t=0..H-1) gamma^t*(Phi_(t+1)-Phi_t)
 = gamma^(H-1)*Phi_H - Phi_0
   + (1-gamma)*sum(t=1..H-1) gamma^(t-1)*Phi_t
```

**Algebra/inference:** intermediate scores are rewarded. This is not automatically
policy-preserving shaping; [Ng/Harada/Russell](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)
use gamma*Phi_next-Phi_previous with appropriate state/terminal assumptions.
Even a corrected potential form is not a proof of financial utility improvement.

[Moody/Saffell differential Sharpe](https://proceedings.neurips.cc/paper_files/paper/1998/file/4e6cd95227cb0c280e99a195be5f6615-Paper.pdf)
is a first-order EMA-moment derivative, not this rolling tanh difference. Their
monthly-data RRL results do not establish minute-data PPO performance. A DSR
challenger needs its own cost-net return convention, moment initialization,
denominator regularization and reset/terminal contract; no alias or formula is
adopted here.

**State/incentive limitations:** R1 resets market-window LSTM state every decision;
R4 exposes only position/remaining horizon. Neither supplies past action-dependent
PnL/DD, EMA or rolling strategy returns. DD/Sharpe history is endogenous hidden
reward state. Exact rolling updates need outgoing-return information as well as
moments; EMA DSR needs previous moments. A reward-state addition is a separately
matched state-contract factor, not a reason to assume recurrence solves it.

Flat and constant-positive histories both score0 under current zero-SD handling;
tiny nonzero dispersion can saturate tanh. Warmup suppresses feedback; eviction
and final-window gaming can matter. Flat0 can beat negative scores without
positive net edge. These are incentive risks, not measured basin causes. Test
gross-versus-net, cold-start, near-zero variance, window eviction and liquidation
with independent oracles before any learning claim. Do not force activity or
annualize minute Sharpe to manufacture signal.

## Ranked separate stress/experiment ladder — authorization still required

1. **Parity freeze before the priority study.** Human names its optimization
   reward, exact sampled/fixed params, schedules, scale/gamma/value loss and
   evaluation-conditioning contract. All matched architectures retain the same
   approved recipe/runtime/data/steps/checkpoints. No PBT during representation
   attribution. Fixed-cost and staging stay separately named existing factors.
2. **Parallel deterministic stress preparation.** Reuse public reset/step/reward/
   terminal paths: AccountingLifecycle (flat/hold/reverse/liquidation),
   RewardStateLifecycle (EMA/DD/history/reset/schedule), and
   ObjectiveScaleConditioning (caps/scales/gamma/encoder/value saturation).
   Hand-derived expected values independent of helpers; nearest tests below
   establish partial mechanics, not all these scenarios. No fixtures run here.
3. **First declared economic challenger: existing net_return versus frozen
   legacy pnl_linear.** Match R0 and R1 plus fixed state choice, initializations,
   data/task starts/horizons, compute and independent1bp evaluator. This is a
   reward-by-architecture comparison, not R1 uplift inferred across objectives.
   Preserve all failed/flat/negative results; do not reuse reserved terminal data.
4. **Dismantle pnl_linear minimally.** One normalization, DD, cap or turnover/
   liquidation factor at a time with an unchanged-recipe control. Keep optimizer
   scale-only diagnostics separate; then compare a justified scale/critic recipe.
   No broad Cartesian grid or arbitrary new coefficients selected here.
5. **Cost-aware bounded Sharpe supplement before pure Sharpe/DSR.** Require
   gross/net basis and minimal reward-state/reset contract first, crossed with
   a state-matched control. Current pnl_sharpe is available but gross based;
   a net-aware supplement is NOT implemented. Pure gross sharpe_only is lower
   priority and cannot certify economic profitability. Declare every objective.
6. **Mixture last.** If later approved, reward choice/params remain episode-static;
   compare pure-recipe controls, fixed mix probabilities and full call/step costs.
   Global advantage scaling and hash-conditioning effects remain separate axes.
   No50/50 ratio or mixture curriculum is adopted.

For true risk requirements, consider a separately specified Pareto tradeoff or
[CMDP/CPO](https://arxiv.org/abs/1705.10528) expected-cost constraint only after
the human states the constraint. Expected constraints are not pathwise broker
safety; deterministic online RiskEngine/Executor remain authoritative. Defer
new algorithms/heads rather than convert arbitrary DD penalties into guarantees.

Nearest contracts read: [rewards](../../../itrader/tests/test_rewards.py),
[task encoding](../../../itrader/tests/test_task_contracts.py),
[train overrides/schedules](../../../itrader/tests/test_train_config.py),
[env reset/history](../../../itrader/tests/test_env_core.py),
[PPO diagnostics](../../../itrader/tests/test_ppo_diagnostics.py), and
[accounting](../../../itrader/tests/test_economic_diagnostics.py).
Source inspection/test existence is not a new execution receipt.

**Next human choice for GRILL:** freeze the priority R0/R1 optimization recipe
explicitly: retain legacy pnl_linear, or approve the draft's existing net_return
as a named objective. Recommendation: preserve the default until this choice is
recorded; favor net_return as the first economic-parity challenger, not silently
as the baseline. Then approve only the smallest separate stress/ablation scope.
No source changes, stress/training runs or promotion follow from this note.

Q26 remains separate: sole fixed1bp paired terminal-net primary/two frozen pairs,
Q24 weights/95equal-tailed/coverageLOWER94/directionalUPPER3, both mandatory
families under both DGP clocks/session inference candidate unchanged. Remaining
support/process/method/MC terms stay queued. Reserved outcomes and acquisition
remain closed; no reward ablation feeds the terminal-only evidence back to tuning.
