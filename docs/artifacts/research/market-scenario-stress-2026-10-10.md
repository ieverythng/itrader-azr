# Market-scenario stress: bounded shock/decay research wing

Date: 2026-10-10. Scope: RESEARCH25 / CONTINUATION25.
Status: research-only proposal, not a simulator, adopted branch identifier,
implemented TaskSpec/env feature, curriculum or market-performance result.
GRILL owns dashboard registration and HTML. This note complements the
[selection/causal-feature evidence](saluki-methodology-research-2026-10-10.md#8-research25-selection-exposure-and-a-small-causal-feature-pilot).

Branch `agent/Watson-dev`, HEAD `8f9fcf94e028b51d883a1a484c2c9466a3d6b547`;
inherited dirty work preserved. No prices/outcomes, acquisition, models,
simulations or tests are accessed/run. Source/input/output identities and
validation receipt are on the [owned RESEARCH board](../../coordination/research.md).
Labels distinguish cited/source **facts**, mathematical/local **inferences**,
and **open** decisions. No real OHLCV is currently perturbed by this research.

## 1. Purpose and primary-source limits

**Recommendation:** investigate frozen-policy robustness to explicitly altered
exogenous market paths first. Prefer a small external causal transform module
with identity and seeded-disturbance adapters, producing an immutable derived
view before environment construction. Do not start with a large simulator,
learned adversary, reward redesign or proposer curriculum.

| Primary source / cited fact | Transfer limit |
|---|---|
| [RARL, Pinto et al.](https://proceedings.mlr.press/v70/pinto17a.html) studies RL training against physical disturbances in control environments. | Motivates disturbance robustness; does not establish financial realism, local improvement or the need for adversarial training here. |
| [Huang et al.](https://arxiv.org/abs/1702.02284) perturb policy inputs and demonstrate vulnerabilities in studied RL policies. | Observation corruption is different from changing the market path used for both observation and PnL. Neither result proves trading edge. |
| [ABIDES, Byrd et al.](https://arxiv.org/abs/1904.12066) models trading agents, exchange messages and network latency. | A minimal OHLCV transform lacks order-book, execution and impact behavior; do not call it a high-fidelity exchange simulator. |

Keep three studies distinct: **path stress** alters observations and economic
returns consistently; **sensor corruption** changes inputs but not the economic
path; **Q26 calibration** imposes known synthetic endpoint/process truth to test
coverage and false-positive uplift. Q26 calendar metadata provides layout, not
a known real-market effect. This wing neither selects a Q26 numerical method nor
creates, filters or certifies architecture uplift. Existing nominal95 /
coverageLOWER94 / directionalUPPER3 terms and endpoint weights stay unchanged.

## 2. Price displacement is not return disturbance

**Mathematical inference, not an executed generator:** let a finite positive
factor multiply all four prices of bar `t`: `P'_t = P_t * exp(g_t)`.
Away from the environment's denominator floor, close arithmetic return becomes
`r'_t = (1 + r_t) * exp(g_t - g_(t-1)) - 1`.

| Scenario definition | Consequence |
|---|---|
| Permanent price-level displacement: `g` jumps, then stays constant. | Adds a transition jump; later baseline close returns are unchanged. |
| Decaying price-level displacement: `g` jumps, then tends back to zero. | Adds the jump and an opposing reversion component. Recovery is built into the assumption, not discovered market truth. |
| Decaying log-return disturbance: added `q_t` fades, while `g_t` accumulates `q_t`. | Return disturbance fades; the accumulated price offset need not disappear. |

No amplitude, decay law, duration or numerical admissible range is selected.
Volatility scaling is a different later factor, not another name for these
three constructions. Avoid confusing a mechanically predictable injected
reversion with real predictive capability.

Same-factor OHLC scaling preserves within-bar order and positivity if the source
bar is coherent. Require `0 < LOW <= min(OPEN,CLOSE) <= max(OPEN,CLOSE) <= HIGH`,
finite representable values and finite nonnegative volume. It does not establish
realistic intrabar evolution, spread or executable prices. Open need not equal
the previous close; gaps are allowed. Close-only shocks can violate high/low
coherence. Do not rely on sanitization to repair invalid scenarios silently.

**Source constraints:** the env uses a `1e-8` previous-close floor, float32
feature caches and bounded observation clipping. A merely positive mathematical
path can still overflow, underflow or hit those floors. A later implementation
must reject or explicitly disposition inadmissible paths before scoring.
Constant positive scaling across a complete visible window cancels under
market-relative price normalization, apart from numerical effects: useful as a
negative control, not a useful volatility shock.

## 3. Current owners and the narrow proposed seam

| Source fact | Design implication, not implemented behavior |
|---|---|
| [Snapshot builder](../../../itrader/inputs/market_data/snapshot.py), lines263-273, retains immutable raw-plus-normalized snapshots. | Never mutate provider/raw/snapshot originals; bind the original and derived identities separately. |
| [ITraderEnv](../../../itrader/envs/itrader_env.py), lines272-273, caches feature rows and economic closes separately. | Feed both caches from the same derived path before env construction. Observation-only perturbation is a separately named experiment. |
| Env step, lines674-692, selects a target position before consuming the next close. | The affected successor bar becomes visible only after that action. No future sign, onset or duration in the earlier observation. |
| Env observation transform, lines342-370, uses the visible window and last visible close, log-volume normalization and clipping. | Keep current transform in the first path-stress comparison; do not also change the feature representation. |
| [TaskEncoder](../../../itrader/rl_lib/tasks/encoder.py), lines42-57, hashes `price_id`, regimes, difficulty and all metadata under `semantic_v2`; legacy encoding also uses task seed. | A scenario reference in metadata is policy input, not provenance-only. Keep TaskSpec fields/embedding fixed for pure hidden-path stress and put scenario identity/seed in an external manifest. |
| [DataPartition](../../../itrader/rl_lib/evaluation/data_partition.py) owns eligible symbol/session/continuity starts. | Preserve timestamps, missing bars and current session eligibility; do not invent observations to make scenarios fit. |
| [RewardEngine](../../../itrader/rl_lib/tasks/rewards.py) and env terminal evidence own existing rewards and economic accounting. | Preserve these owners and all parameters; a path transform does not own new action, cost, risk or broker semantics. |

The proposed module interface is small: immutable source view plus externally
owned scenario definition/seed in; validated derived view and provenance receipt
out. Identity and disturbance adapters provide the actual variation. A reviewed
env input adapter might be needed later, but do not push scenario generation into
`step()` or design a new TaskSpec schema now. Any future task-visible scenario
contract needs its own visibility/encoding review. R4-v2 joint fusion, R-series
identities and solver-state contracts remain untouched.

## 4. Causality, sessions, reset and economic invariants

**Proposed controls:** onset follows a frozen seeded or past-only rule, never
future extrema, realized profitability or future volatility. Future schedule,
sign and duration may be retained for offline auditing but must not enter task
embedding, reset context, features or solver inputs. Prefix causality alone is
insufficient if public episode offsets reveal a repeated predictable schedule;
the later holdout/timing design must address that shortcut too.

Declare decay in eligible-bar time versus elapsed wall time; overnight/gap
persistence is open. Preserve timezone/session/missing-bar identities. Choose
whole-path versus episode-relative event scope explicitly: resetting an ongoing
displacement at episode boundaries can manufacture jumps or contradictory
versions of overlapping history. Whole-path identities are the simpler initial
candidate; no clock/scope is adopted here.

For joint symbol shocks, common calendar time—not concatenated row index—is
the alignment. A shared factor with declared symbol loadings and optional
idiosyncratic streams is a later candidate; it alters dependence, not preserves
the original covariance by fiat. Any randomized covariance construction needs
validity checks and a separately approved model. Keep volume unchanged initially
and label this **price-path-only stress**, not a liquidity crisis. Do not assume
inverse volume scaling, realistic order flow or price-dependent slippage.

Manifest proposal: original/derived digests, transform version, scenario seed,
event scope and clock, symbol coupling, admissible ranges, observation/reward/
cost identities and development/holdout role. Use scenario RNG distinct from
task seed; identical resets must reproduce the intended path without mutating
source data or carrying state across unrelated episodes. Env statistics, return
EMA, sampler and recurrent states retain their existing reset contracts.

Keep target-position `Discrete(3)`, fixed cost assumptions and terminal
liquidation unchanged. Independently reconcile arithmetic gross return
`sum(p_t * r'_t)` and cost
`(sum(abs(p_t-p_(t-1))) + abs(p_H)) * cost_bps * 1e-4`, starting flat.
Here `p_t` is the target position applied to `r'_t`, not the price `P_t`.
Liquidation is charged exactly once. Policy reactions may change turnover and
cost totals; unchanged accounting does not mean identical totals.
`pnl_linear` can include return-EMA normalization, drawdown and clamped penalties;
its shaped optimization sum is not terminal economic net. Stress can legitimately
change those values without changing the reward definition. No new liquidity,
fill, financing or impact mechanics are inferred from existing OHLCV accounting.

## 5. Stress-first evaluation, then possible curriculum

**Future ablation proposal:** freeze checkpoint, original paths, starts/horizons,
TaskSpec, costs, state initialization and preprocessing; compare identity and
bounded stress adapters. Fixed-action traces first verify arithmetic; frozen
policies then measure behavioral responses. Do not freeze actions in the policy
robustness comparison: reacting differently is the question.

Predeclare development versus held-out scenario seeds and definitions, retaining
original-path controls. Held-out seeds test variation within an assumed scenario
family; separately withheld families probe transfer to different assumptions.
Once held-out stress results guide tuning, record the exposure rather than keep
calling them untouched. Shock replicas share source history and imposed rules;
they are not new independent market samples or fresh historical confirmation.

Only after mechanics and useful frozen-policy diagnostics qualify should a
separately authorized stress-training/curriculum comparison be considered.
Match solver data/budgets, retain original-only training controls, predeclare
mixture/schedule and keep held-out real-market and held-out stress evidence
outside training and proposer reward. No learnable adversary or AZR-loop change
is required initially.
Flat can be the appropriate response to poor edge or adverse conditions: reject
force-trade/anti-flat rewards or injecting predictable profitable shocks just to
make an active policy appear successful. Existing promotion gates are unchanged.

## 6. Minimum public integration oracles and disposition

These are independent-oracle designs, not implemented/run tests:

- **IdentityControl:** identity adapter reproduces rows, eligibility, observations
  and fixed-action terminal evidence; originals remain unchanged.
- **PrefixNoLeak:** future suffix/event changes leave earlier rows, observations
  and task embedding unchanged; schedule identities are absent from policy input.
- **CoherenceEligibility:** literal positive finite OHLC ordering, declared volume,
  timestamps/gaps/session support; malformed or unrepresentable paths fail closed.
- **EconomicReconciliation:** hand-computable action/price traces match gross,
  turnover, fixed-rate costs and exactly-once liquidation, independently of shaped reward.
- **ResetLineage:** repeat source/scenario/reset identities reproduce; different
  streams do not inherit scenario RNG, EMA or recurrent state outside the contract.
- **MatchedPolicyControl:** policy/checkpoint/panel/task/cost/state identities agree
  across original and stress arms; only the external path adapter changes.

| Disposition | Recommendation |
|---|---|
| Retain now as research discipline | Separate path stress, sensor attacks and Q26 calibration; preserve immutable source, causal visibility and existing accounting. |
| Ablate after authorization | Identity versus bounded seeded disturbance on frozen policies, with matched controls and withheld scenario identities. |
| Defer | Stress curriculum, learned adversary, world models, impact/liquidity simulation and task-visible scenario contracts. |
| Reject | Synthetic replicas as independent market evidence; OHLC coherence as financial realism; anti-flat incentives; stress success as proof of profitability. |

**Open / next decision:** approve a minimal external path-stress design for later
implementation review? Recommended conditional yes, after specifying permitted
development evidence, scenario ranges/clocks/scope, hidden schedule, holdout
design, matched panel, budgets and reporting/acceptance criteria. No parameters,
method, execution authority or capability promotion follows this note.
