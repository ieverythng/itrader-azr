# Evaluation and quantitative methodology for the R-series screen

Date: 2026-10-06
Status: research synthesis and candidate protocol; no panel, threshold, metric,
Q26 method, code path, or promotion decision is adopted by this document

## Executive decision

The next R0/R4/R1 evaluation should remain a **paired economic attribution
screen**, not be converted into one universal score.

The accepted Q24 point estimand remains the authority: average the four crossed
horizons within each anchor, anchors within each symbol, and the twelve symbols
equally. Q25 net bps per actual bar remains a secondary duration diagnostic.
Neither should be reopened by this research pass. The active contracts and
their current implementation status are recorded in
[`r1-development-readiness.md`](../../project-state/r1-development-readiness.md),
[`r1-panel-qualification-2026-10-06.md`](r1-panel-qualification-2026-10-06.md),
and the [masterplan](../../plans/itrader_masterplan.md).

The strongest candidate for Q26 is a **paired, synchronized calendar-block
procedure that recomputes the exact Q24 hierarchy in every resample**. It must
keep all horizons of an anchor together and preserve cross-symbol market
dependence. This is a candidate, not a decision: block construction, support,
calibration, interval type, multiplicity family, and failure policy still need
outcome-blind analysis and human approval. Block-bootstrap theory makes block
choice part of the estimand's finite-sample behavior, not a harmless display
setting ([Kunsch 1989](https://doi.org/10.1214/aos/1176347265),
[Politis and Romano 1994](https://users.ssc.wisc.edu/~behansen/718/Politis%20Romano.pdf),
[Hall, Horowitz, and Jing 1995](https://academic.oup.com/biomet/article-abstract/82/3/561/260651)).

Economic interpretation should use a vector:

```text
gross return - modeled turnover cost = net return
directional loading + timing residual - cost = net return
```

That vector must sit beside exposure, turnover, liquidation, action shares,
gross drawdown, horizon, symbol, calendar, and named controls. A flat solver is
not skilled merely because an active policy loses; a profitable solver is not
necessarily learning state if a constant direction explains the result.

The proposed 60-past-bars plus one-next-bar analysis is useful as a **causal
local-action diagnostic**, but not as one-step Sharpe and not as a replacement
for terminal evaluation. The action and every normalizer must be frozen before
the outcome bar is read. Report immediate gross, transition cost, net, exposure,
and active return versus named controls; keep the hindsight-best action as an
oracle-only ceiling.

Finally, `CapabilityProfile` remains primarily a future online/shadow
eligibility and matching contract. Offline evaluation may build the evidence
needed to derive it later, but the boundary is:

```text
held-out offline capability evidence
  -> separately versioned calibration/reduction
  -> future online/shadow CapabilityProfile
```

The current dataclass is a scaffold, not an estimator. In the present R-series
screen it must not become a solver reward, training input, unapproved proposer
feedback signal, promotion gate, broker/risk rule, or substitute for
`DecisionGate`. This is not a permanent ban on curriculum feedback: a separately
approved Phase B path may later transform calibrated held-out validation or
rolling capability evidence into versioned `r_prop` or proposer/curriculum
feedback. Terminal-screen and sealed final-test evidence must remain isolated
from adaptive proposer updates. This preserves the phase boundaries in the
active [system architecture](../../architecture/itrader_system_architecture.md),
[implementation specification](../../iTRADER_Codex_Implementation_Spec.md), and
[masterplan](../../plans/itrader_masterplan.md).

## Scope and epistemic labels

This pass opened no market outcomes, selected no candidate dates, froze no
panel, ran no replay or training, and changed no source/config/data/active
architecture. The final test remains sealed. R1 remains unimplemented on this
branch, and the native migration remains a separate later concern.

Claims use these labels:

- **Verified local fact:** confirmed in current source, tests, or active docs.
- **Verified external fact:** supported by a linked primary paper or official
  first-party implementation/documentation.
- **Inference:** a proposed iTrader interpretation of those facts.
- **Hypothesis:** a design candidate requiring prospective validation.

The GitNexus index was not used as authority because its recorded index is
stale and WSL is intentionally off. Direct source and active docs take
precedence.

## 1. The evaluation object

### 1.1 What Q24 and Q25 estimate

For policy `p`, matched control `c`, symbol `s`, anchor `a`, and horizon `h`,
define the paired terminal-net difference:

```text
d[p,c,s,a,h] = net[p,s,a,h] - net[c,s,a,h]
```

The accepted Q24 contrast is:

```text
anchor_delta = mean over h in {96, 128, 160, 200}
symbol_delta = mean over frozen anchors for that symbol
theta        = equal mean over the 12 symbols
```

This describes a task design that chooses a declared symbol uniformly, then an
available anchor for that symbol, then a frozen horizon. It is **not** a
simultaneously traded twelve-asset portfolio. The raw 128-task mean answers a
different question because symbols with three anchors receive more weight than
symbols with two. Both remain visible, with Q24 primary and raw-task averaging
secondary. These are verified current contracts in
[`terminal_panel_report.py`](../../../itrader/rl_lib/evaluation/terminal_panel_report.py).

Q25 first computes, per complete episode:

```text
net_bps_per_bar = 10000 * terminal_net_return / actual_decision_steps
```

and only then applies the same hierarchy. It is an arithmetic duration
diagnostic, not an annualized, compounded, or pooled-step return. Missing,
censored, duplicate, or wrong-horizon cases fail rather than being reweighted.

### 1.2 What two seeds can support

The market panel and training randomness are different populations. Calendar
resampling may describe market-sample uncertainty **conditional on a frozen
checkpoint**. It cannot turn two optimizer seeds into a population-level
estimate of training-run variation. Deep-RL evaluation work documents how
unstable finite-run conclusions can be and recommends reporting uncertainty
over aggregate performance, but it does not make two runs inferentially
adequate ([Agarwal et al. 2021](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf),
[Henderson et al. 2018](https://ojs.aaai.org/index.php/AAAI/article/download/11694/11553)).

For this screen, report each seed, paired effects, sign agreement, and a
descriptive mean/range. Keep any claim about the algorithm across fresh
training runs for a later independently powered replication.

## 2. Economic attribution before statistical inference

### 2.1 Exact current accounting

Current `EpisodeEvidence` separates optimized reward from economic evidence.
For one matched episode:

```text
G = sum_t position_t * return_(t+1)
turnover = sum_t |position_t - position_(t-1)| + |terminal_position|
C = turnover * cost_bps * 1e-4
N = G - C
```

The terminal position contributes liquidation cost but no extra decision bar.
A short-to-long flip contributes two turnover units. The implementation and
edge behavior are in
[`episode_evidence.py`](../../../itrader/rl_lib/evaluation/episode_evidence.py)
and [`itrader_env.py`](../../../itrader/envs/itrader_env.py).

The exact 1 bp result is therefore a transparent linear cost scenario. It is
not a live fill model: it does not estimate spreads, queueing, partial fills,
latency, borrow/funding, capacity, or nonlinear impact. Empirical execution-cost
work separates commissions, spread, impact, and opportunity cost and finds
nonlinear size effects, illustrating why a proportional bp charge should not be
called execution truth
([Frazzini, Israel, and Moskowitz](https://pages.stern.nyu.edu/~afrazzin/pdf/Trading%20Cost%20of%20Asset%20Pricing%20Anomalies%20-%20Frazzini,%20Israel%20and%20Moskowitz.pdf)).

### 2.2 Direction, timing, and cost bridge

On an identical symbol/anchor/horizon price path, let `T` be episode length,
`p_bar` average signed exposure, and `L` the matched long-only gross return:

```text
p_bar      = long_action_fraction - short_action_fraction
G_direction = p_bar * L
G_timing    = G - G_direction
N           = G_direction + G_timing - C
```

This identity separates average directional loading from the gross residual
created by changing position within the episode. `G_timing` is an attribution
residual, not automatically causal alpha or forecasting skill. It can contain
noise, path dependence, and selection. The full derivation and limits are in
[the quantitative attribution evidence note](evaluation-quant-methodology-2026-10-06-evidence/quant-attribution-primary-sources.md).

For every paired representation/control comparison, preserve:

| Layer | Required outputs | Question |
|---|---|---|
| Economics | gross, modeled cost, net, net bps/bar | Was value earned after the declared cost scenario? |
| Gross source | directional component, timing residual | Was gross mostly static direction or state-dependent behavior? |
| Activity | absolute/signed exposure, short/flat/long fractions, changes | Was the solver active, directional, or in a flat basin? |
| Friction | total turnover, liquidation turnover/share, repriced cost scenarios | Did cost avoidance explain the result? |
| Path | gross maximum drawdown, per-bar net Sharpe, positive-net-bar fraction | What path and variability accompanied it? |
| Concentration | symbol, synchronized calendar block, decision offset, horizon, ex-post regime | Where did the result occur? |

Current drawdown is gross additive drawdown, not net drawdown. Current episode
Sharpe is mean divided by sample standard deviation of within-episode net bar
returns and simple square-root annualization, when requested, does not repair
serial dependence. Lo shows why Sharpe time scaling depends on the return
process and autocorrelation
([Lo 2002](https://alo.mit.edu/publications/page/18/)).

### 2.3 Benchmark ladder

Sharpe's differential-return formulation makes the benchmark part of the
meaning of a risk-adjusted result
([Sharpe 1994](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)). Use a ladder:

1. flat/no exposure;
2. long-only and short-only, shown separately;
3. seeded random;
4. architecture/state-matched seeded untrained;
5. exposure-matched static attribution; and
6. only later, a timestamp-aligned market/factor residual with lagged-only
   loadings if its data and support are adequate.

The whole-panel stronger of long and short may remain a descriptive hurdle.
Selecting it after outcomes are visible is a two-benchmark selection, so a
confirmatory superiority claim should either beat both with simultaneous
inference or include that selection in the predeclared multiplicity procedure.
Never take a per-anchor hindsight maximum.

A flat control answers whether exposure added value. It does not demonstrate
skill. A market-beta or factor model is model-dependent; daily Fama-French
factors must not be copied across minute bars as if they were independent
intraday factors ([official Kenneth French data library](https://mba.tuck.dartmouth.edu/pages/Faculty/ken.french/data_library.html)).

### 2.4 Risk normalization is secondary

Keep raw net as the economic base. Candidate companion views include paired
active return, net bps per actual bar, differential-return Sharpe from a genuine
aligned return stream, lagged-volatility-standardized active return, downside
risk, tail loss, and alpha/beta. None is a universal replacement.

Avoid promotion ratios such as net/turnover, net/exposure, or net/drawdown:
near-zero denominators can reward inactivity or create unstable values. Any
zero denominator must produce explicit unavailable status plus coverage, not an
epsilon-tuned winning value.

## 3. Candidate Q26 statistical protocol

### 3.1 Dependence that must survive

The 128 rows are not 128 independent market observations:

- four horizons at one anchor share lookback and most outcome bars;
- nearby anchors for one symbol can overlap or share serial regimes;
- symbols on the same sessions share market shocks; and
- paired R cells share panel construction while training seeds remain separate
  learning replications.

Long-horizon return research demonstrates how overlapping observations distort
conventional approximations
([Richardson and Stock 1989](https://www.nber.org/papers/w3335)). Resampling each
symbol independently would also erase cross-symbol covariance, while IID
resampling of four horizons would multiply apparent support.

### 3.2 Leading candidate: synchronized calendar-block resampling

The leading design candidate is:

1. join candidate and control by immutable task identity and subtract first;
2. keep all four horizon deltas inside their anchor;
3. derive, from metadata only, each anchor's maximal information/outcome window
   from the 60-bar lookback through the 200-bar future bound;
4. merge intersecting or predeclared-adjacent windows into ordered calendar
   components;
5. apply the same calendar block draw or multiplier to every affected symbol;
6. rebuild complete replicas and recompute horizon -> anchor -> symbol ->
   equal-symbol Q24 in every draw; and
7. report the point estimate, interval, effective support, block specification,
   and predeclared sensitivity results.

This construction is an iTrader inference. It requires layout-specific null
simulation before outcomes are opened. A sparse panel may not permit ordinary
resampling without dropping a symbol; silent eleven-symbol renormalization is
not Q24. Fixed-support block multipliers, moving/circular/stationary blocks,
fixed-b subsampling, and HAC are candidates to compare on the exact frozen
layout, not interchangeable implementations.

The [statistical evidence note](evaluation-quant-methodology-2026-10-06-evidence/statistical-protocol-primary-sources.md)
details assumptions, alternatives, primary sources, failure modes, and tests.

### 3.3 Candidate alternatives and their limits

| Method | Useful target | Proper role here | Main limit |
|---|---|---|---|
| Synchronized moving/stationary block bootstrap | Exact nonlinear Q24 paired contrast | Leading candidate pending layout calibration | Block/support sensitivity; cannot create complete symbol coverage |
| Calendar-block subsampling/fixed-b | Same contrast | Strong finite-sample sensitivity or fallback candidate | Needs enough complete chronological subsamples |
| HAC long-run variance | Ordered mean/linearized contribution series | Analytic sensitivity | Small-sample bandwidth and irregular-support sensitivity |
| PSR/DSR | A genuine Sharpe-series question and, for DSR, logged search adjustment | Later diagnostic only | Not Q24; overlapping episode Sharpes are not a portfolio return series |
| White Reality Check / Hansen SPA | Whether any member of a declared family beats a benchmark | Optional global multiplicity layer | Requires complete trial family and valid dependence resampler |
| Romano-Wolf StepM | Adjusted individual comparisons | Optional if naming cells is the goal | Same inventory/resampling requirements |

PSR and DSR do not provide uncertainty for Q24 terminal net return. DSR also
depends on an honest inventory of tried strategies and their dispersion
([Bailey and Lopez de Prado 2014](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)).
White's Reality Check and Hansen's SPA address data snooping over a declared
candidate family; neither repairs an exposed panel, missing runs, or only two
training seeds
([White 2000](https://onlinelibrary.wiley.com/doi/10.1111/1468-0262.00152),
[Hansen 2005](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=264569)).

### 3.4 Outcome-blind support audit and failure policy

Before panel freeze, report from metadata:

- unique sessions, dates, anchors, symbols, offsets, and merged calendar
  components;
- component size, calendar span/gaps, symbol coverage, and share of fixed Q24
  weight;
- number of candidate contiguous blocks across a predeclared block grid;
- whether every method can preserve twelve-symbol/four-horizon support;
- component influence and a fully defined descriptive effective-support value;
  and
- simulated coverage/type-I error under common-factor, serially dependent,
  heavy-tailed, and null paired-effect designs matched to the layout.

Fail closed rather than shorten blocks until an interval appears when support is
degenerate, a component dominates, the hierarchy cannot be preserved, plausible
block choices reverse the conclusion without a design reason, or null
calibration fails its later predeclared tolerance. This note deliberately sets
no numerical cutoff, confidence level, alpha, block length, or bootstrap count.

## 4. Causal local-action diagnostics

### 4.1 Replace one-step Sharpe with an action-value vector

A Sharpe ratio requires a series of differential returns and a dispersion
estimate; a single realized return has neither
([Sharpe 1994](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)). Current
`EpisodeEvidence` also returns zero by convention for fewer than two returns.
Therefore do not call `one_next_return / lagged_volatility` a one-step Sharpe.

For prior position `q_t`, target action `a` in `{-1,0,+1}`, next asset return
`r_(t+1)`, and cost rate `c`, define a continuation probe:

```text
gross_1(a)        = a * r_(t+1)
switch_cost_1(a)  = c * |a - q_t|
net_continue_1(a) = gross_1(a) - switch_cost_1(a)
```

A forced one-bar round trip is a different estimand:

```text
net_roundtrip_1(a) = net_continue_1(a) - c * |a|
```

The latter includes terminal liquidation; the continuation probe intentionally
does not. Mixing them would reward flat for an accounting choice rather than
answer the intended action question.

### 4.2 Causality and controls

At decision time, serialize and freeze the 60 past bars, prior position, task
identity, policy action, and any scale/loading estimate. Only then read the next
bar as outcome. The historical simulator can mechanically score all three
actions on the same exogenous path, but:

```text
oracle_value = max_a net_continue_1(a)
oracle_gap   = oracle_value - net_continue_1(policy_action)
```

is a hindsight diagnostic. It must never be reported as policy PnL, fed into
the action, or used as deployment confidence. This is consistent with offline
policy evaluation: a target policy is fixed from context before the realized
reward is used to score it
([Li et al. 2011](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Published-3.pdf)).

Report the policy action versus:

- hold prior position;
- target flat;
- target long and target short separately;
- a seeded random action policy; and
- optionally a lagged-loading residual benchmark, only if supported.

Hold and flat differ whenever the prior position is nonzero. Flat then incurs a
close cost even though next-bar exposure is zero.

### 4.3 Lagged volatility and edge cases

A candidate normalizer may use trailing RMS, sample standard deviation, EWMA,
or a time-of-day-conditioned estimate, but it must use only bars through `t`,
be identical across compared actions/policies at the anchor, and remain named
and versioned. Research on volatility-managed portfolios supports lagging the
scale; it does not validate a particular 60-minute estimator for iTrader
([Moreira and Muir 2017](https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf)).

Always retain raw gross/cost/net. Zero or non-finite scale makes the normalized
field unavailable and increments a coverage counter. Zero exposure similarly
makes exposure-normalized ratios unavailable. Neither condition is evidence of
zero risk or skill.

One-step results diagnose local direction and switch-cost sensitivity. Full
terminal replay remains necessary for continuation behavior, repeated
switching, state use, drawdown, horizon, and liquidation. If a future evaluator
branches one action then follows the policy, every branch needs independently
cloned environment, inventory, model/recurrent state, and randomness.

The complete construction and source ledger are in
[the capability diagnostics note](evaluation-quant-methodology-2026-10-06-evidence/capability-diagnostics-primary-sources.md).

## 5. Offline evidence and the future online CapabilityProfile

The current
[`CapabilityProfile`](../../../itrader/rl_lib/tasks/capability_profile.py) stores
identity, horizon/regime descriptors, `estimated_p_solve`, expected net utility,
calibration error, sample count, and metadata. **Verified local fact:** it does
not specify how those values are estimated, calibrated, refreshed, or made safe
for online use.

The research boundary is therefore two-stage:

### Stage A: offline held-out capability evidence

Preserve an inspectable evidence bundle containing:

- solver/checkpoint/source/data/panel/cost identities;
- task family, symbol/resolution, horizon, and causal context definition;
- raw economic and active-return vectors;
- one-step local diagnostics and terminal metrics, explicitly distinguished;
- named controls and oracle-only fields;
- action/exposure/turnover and coverage slices;
- uncertainty object and method identity; and
- probability/calibration evidence only when a real probabilistic prediction
  and held-out outcome definition exist.

Metric construction, uncertainty, and calibration are separate concerns.
Proper scoring-rule theory distinguishes whether probabilities match observed
events from how sharp or useful those predictions are
([Gneiting and Raftery 2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)).
If no probability model exists, `calibration_error` is unavailable, not zero.

### Stage B: future online/shadow operational projection

A separately versioned reducer may later transform qualified held-out evidence
into `CapabilityProfile` for solver eligibility and candidate-task matching.
That transform requires its own source-panel identity, calibration method,
freshness/staleness rules, support requirements, abstention behavior, and tests.

The same underlying held-out evidence may later serve an explicitly approved
offline Phase B curriculum consumer through a separate, versioned adapter. That
adapter must use a declared adaptive validation or rolling-evaluation source,
must preserve the evidence/provenance record, and must never train on the
terminal screen or sealed final test. This is the planned capability-to-`r_prop`
feedback seam, not a behavior enabled by the current dataclass or this research
artifact.

The profile remains evidence consumed by a future capability-matching layer
after `MarketFramer` produces `CandidateTask` options, and by `DecisionGate`
eligibility logic only after held-out calibration. This matches the existing
[project decision](../../project-state/decisions.md) that profiles are updated
from held-out or rolling evaluation slices for later online gating. The profile
does not place orders, override deterministic risk, or make `WAIT` a fourth
solver action. `WAIT` remains a first-class online-shell outcome; the solver's
action space remains short/flat/long.

## 6. Lessons from open-source trading environments

The survey found transferable **contract patterns**, not validated trading
edge:

| Platform | Verified useful pattern | iTrader mapping | Do not import blindly |
|---|---|---|---|
| [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta) | Data/environment/agent separation; cost and run memories | Preserve explicit environment and evidence ledgers | Continuous share actions or turbulence liquidation |
| [Microsoft Qlib](https://github.com/microsoft/qlib) | Explicit train/valid/test roles, fit bounds, benchmark, costs, recorder artifacts | Bind data role, preprocessing, run, position, and benchmark provenance | Example dates, fees, or naming `test` as proof of final-test status |
| [TensorTrade](https://github.com/tensortrade-org/tensortrade) | Composable action/reward/observer/feed/exchange/portfolio seams | Keep ownership explicit | Treating arbitrary swappability as scientific validity |
| [Gym-AnyTrading](https://github.com/AminHP/gym-anytrading) | Compact windowed environment | Negative comparison for implicit initial short and binary actions | Fee-free oracle or binary action contract |
| [Gym-Trading-Env](https://github.com/ClementPerroud/Gym-Trading-Env) | Target positions `[-1,0,1]`, current/last position, fees and borrow cost | Closest action/inventory vocabulary | Its default reward/leverage/termination assumptions |
| [TradeMaster](https://papers.neurips.cc/paper_files/paper/2023/file/b8f6f7f2ba4137124ac976286eacb611-Paper-Datasets_and_Benchmarks.pdf) | Chronological rolling periods, multiple metrics and seeds | Separate time-period from seed variation | Its particular windows, ranks, or algorithms as sufficient evidence |

The durable lesson is explicit data roles, position/cost semantics,
benchmark-relative evidence, and run provenance. No surveyed platform proves
that its default split, cost, reward, action space, or headline result is right
for iTrader. See the
[benchmark-environment evidence note](evaluation-quant-methodology-2026-10-06-evidence/benchmark-environment-primary-sources.md).

## 7. Q23: recovery versus a separately declared evaluation dataset

Q23 is upstream of Q26. A sophisticated interval cannot repair an exposed or
unprovable panel. Repeated adaptive access to a holdout changes its validity;
formal reusable-holdout work treats information released from held-out data as
a first-class resource rather than assuming a date label remains untouched
([Dwork et al. 2015](https://pubmed.ncbi.nlm.nih.gov/26250683/)).

### Option A: source-bound historical recovery

The source/code audit owns this route. Certification requires, at minimum:

- exact dirty-source binding for historical producer hashes, not merely the
  current Git tree;
- RNG/reset/cursor semantics and worker identity;
- reconstruction of the actual historical projected frame for each producer,
  mapped by symbol and UTC timestamp; for the recovered controls the frame is
  the same canonical twelve-symbol configuration, while the observed six- and
  eight-symbol counts are task-symbol coverage of two panels, not different
  market-frame layouts;
- reconstruction of per-evaluation starts for every relevant report;
- dataset, task, config, checkpoint, and producer identities; and
- proof that candidate anchors/windows were not previously inspected.

DEV's corrected metadata audit establishes that all sixteen legacy-control
configs declare the same twelve `data_symbols`. The canonical regular-frame
mapping matches all 512 recorded sidecar `(index, symbol)` pairs and identifies
48 unique start coordinates: 32 unpanelled, eight September crossed-panel, and
eight January crossed-panel starts. That mapping evidence does not prove the
exact historical within-
symbol ordering without the dirty source snapshot, and recorded locations are
not source-bound RNG reconstruction. The bounded search of the 26 exact
randomized stable-trial directories plus source/snapshot/manifest-named
immediate siblings at eight parent locations found zero candidate source
snapshots. The randomized starts therefore remain unrecovered, and the
conservative 2,052-anchor bound still yields zero certified-fresh candidates.
Both independent evaluation-code reviews return `CHANGES`; Q27's four bounded
mechanical repairs remain a separate pending human decision, not a methodology
adoption. DEV's
[evaluation-code audit](evaluation-code-audit-2026-10-06.md) owns the dated
counts and reconstructability correction. Unresolved cases stay excluded;
"likely" starts are not non-exposure proof.

### Option B: separately declared evaluation-only data

Older or newer alone does not prove non-exposure. Before reading returns,
declare:

- provider/feed, adjustment policy, bar resolution, RTH/calendar source, and
  continuity requirements;
- acquisition universe and the fixed twelve-symbol role or an explicitly
  human-approved change;
- whether the slice is adaptive validation, terminal screening, or a new sealed
  final test;
- preprocessing-fit boundaries, purge/overlap rules, and inspected-anchor
  exclusions; and
- metadata coverage/diversity and Q26 calendar-support feasibility.

Do not apply a generic 70/15/15 split to a short evaluation-only slice. Do not
repartition the existing sealed final test. If the new source differs from
training data, provider/acquisition shift becomes an explicit factor, not an
invisible freshness benefit.

### Decision rule

Choose one primary route before further terminal-panel work. A hybrid is
possible, but it answers two different questions and adds a provider/source
factor; it should not become an indefinite attempt to find whichever panel
looks better. Panel identity, eligibility, and exclusion must be fixed from
metadata before returns or model outcomes are read.

## 8. Candidate practical protocol

This sequence is a research recommendation, not execution authorization:

1. Decide Q23 recovery versus separate evaluation-only data, including the
   data role and strict non-exposure proof.
2. Run only metadata-level eligibility and calendar-support diagnostics.
3. Generate candidate Q26 layouts and calibrate them on synthetic nulls that
   reproduce the exact sparse/dependent structure.
4. Human-select and seal the Q26 family, block rule/grid, interval construction,
   multiplicity family, and fail policy without seeing terminal outcomes.
5. Complete R1 implementation/review and freeze all sixteen effective configs,
   source/runtime identities, training seeds, checkpoints, controls, and the
   terminal panel.
6. Train all cells; open the terminal panel only after completion, using the
   same deterministic 1 bp task set and exact candidate/control joins.
7. Report per-seed Q24/Q25, the attribution vector, named controls,
   concentration slices, and checkpoint-conditional calendar uncertainty.
8. Report two-seed agreement descriptively; do not claim the training-run
   sampling distribution.
9. Run any predeclared familywise procedure only with the complete registered
   family and the same dependence-preserving resampler.
10. Treat the result as attribution/feasibility evidence, not automatic
    economic promotion or deployment permission.

## 9. Minimum tests for a later implementation pass

### Economic and aggregation contracts

- Property-test `net = gross - cost` and
  `net = direction + timing - cost`, including all position transitions and
  exactly one true terminal liquidation.
- Prove long-only has zero timing residual, short-only mirrors it, and flat has
  zero gross/turnover from zero prior exposure.
- Reject mismatched, missing, duplicate, censored, wrong-horizon, or wrong-cost
  candidate/control joins.
- Hand-check Q24 hierarchy under unequal anchor counts and row permutations;
  never silently renormalize incomplete symbols/horizons.
- Verify Q25 normalizes each episode before aggregation and uses actual positive
  decision steps.

### Dependence and multiplicity contracts

- Keep all horizons of an anchor on one resampling weight.
- Apply one synchronized draw/multiplier to overlapping calendar windows across
  symbols; include a test that independent-per-symbol resampling fails.
- Preserve twelve-symbol/four-horizon support in every valid replicate.
- Simulate common-factor, serially dependent, heavy-tailed nulls and record
  coverage/type-I error before outcome access.
- Refuse population-seed inference from two seeds.
- Require the complete predeclared cell/metric/control family for multiplicity
  procedures; treat selected stronger direction as simultaneous or descriptive.

### Causal diagnostic contracts

- Persist observation, policy action, lagged volatility, and any lagged loading
  before outcome access; future-row perturbation must not alter them.
- Exhaustively test all nine prior-position/target-action transition costs.
- Separate continuation from forced-roundtrip liquidation semantics.
- Serialize hindsight maxima only under `oracle`/`diagnostic_only` names.
- Keep zero/non-finite scale and zero-exposure ratios unavailable while retaining
  raw values and coverage.
- Ensure no diagnostic field enters solver observations, `RewardEngine`, task
  encoding, unapproved proposer feedback, or current R-series promotion.
- For any later Phase B capability-to-curriculum adapter, require a distinct
  version and a declared validation/rolling source; assert that terminal-screen
  and sealed final-test records cannot enter adaptive proposer updates.

### Capability/provenance contracts

- Require a named outcome and estimator before populating `estimated_p_solve`.
- Require actual held-out probabilistic forecasts before calibration fields are
  populated; otherwise record unavailable.
- Version and test any offline-evidence-to-profile reducer separately.
- Bind every result to source, dataset, config, checkpoint, panel, task, cost,
  seed, producer, and method identities.
- Ensure a future profile cannot override deterministic `DecisionGate`,
  `RiskEngine`, `Executor`, or WAIT behavior.

## 10. Assumptions and known failure modes

- Historical prices are treated as exogenous to these unit-position actions;
  live impact/capacity is outside the simulator.
- Additive episode returns are not compounded capital returns.
- Equal-symbol results describe the declared universe, not a deployable
  capital-weighted portfolio.
- Correlated equities and ETFs mean symbol count is not independence count.
- Ex-post regime and best-direction summaries are descriptive selectors.
- Episode Sharpes over 96--200 bars are short and serially structured.
- A 1 bp proportional cost scenario does not establish live execution.
- Sparse calendar components may make confirmatory Q26 inference impossible;
  honest descriptive reporting is preferable to manufactured precision.
- Two seeds cannot establish robustness to training randomness.
- One-step action diagnostics cannot establish terminal policy value.
- `CapabilityProfile` fields do not create calibration, uncertainty, or online
  safety semantics by existing.

## 11. Ordered one-question-at-a-time Q23/Q26 grilling agenda

Ask one question, record the answer, then ask the next; later questions depend
on earlier choices.

1. **Q23 source authority:** Will the terminal panel use source-bound recovered
   historical starts, or a separately declared evaluation-only dataset?
2. **If separate data:** Is its role adaptive validation/terminal screening or
   a newly sealed final test, while the existing final test remains untouched?
3. **Metadata contract:** Which provider/feed, calendar, adjustments,
   continuity, twelve-symbol universe, offsets, and purge/exclusion rules prove
   eligibility without outcomes?
4. **Recovery/freshness stop rule:** What unresolved provenance or overlap
   condition makes a candidate unavailable rather than "probably fresh"?
5. **Q26 confirmatory target:** Is Q24 terminal net the sole confirmatory
   estimand, with Q25/horizon/Sharpe diagnostic, or are more metrics entering the
   multiplicity family?
6. **Contrast family:** Which R1/R0, R4, interaction, curriculum, and control
   contrasts are confirmatory before the panel opens?
7. **Calendar unit:** Based on metadata-only layout diagnostics, should Q26 use
   session dates, merged maximal-window components, or another predeclared
   synchronized unit?
8. **Method family:** Which synchronized block bootstrap, fixed-support
   multiplier, fixed-b subsampling, and HAC candidates should enter null
   calibration?
9. **Calibration/failure rule:** What prospective coverage/error and instability
   policy makes Q26 fail closed, and what support evidence is required?
10. **Multiplicity output:** Is the goal a global RC/SPA-style family test,
    adjusted individual StepM comparisons, or descriptive attribution only?
11. **Seed scope:** Do we explicitly retain two seeds as a mechanics/attribution
    screen and require a later seed-powered replication before promotion?
12. **Capability handoff:** Which offline evidence fields may later feed a
    separately validated online/shadow `CapabilityProfile`, and what event is
    any `p_solve` probability calibrated against?

## Evidence package

- [Quantitative attribution](evaluation-quant-methodology-2026-10-06-evidence/quant-attribution-primary-sources.md)
- [Statistical protocol and primary sources](evaluation-quant-methodology-2026-10-06-evidence/statistical-protocol-primary-sources.md)
- [Causal capability diagnostics](evaluation-quant-methodology-2026-10-06-evidence/capability-diagnostics-primary-sources.md)
- [Open-source benchmark/environment survey](evaluation-quant-methodology-2026-10-06-evidence/benchmark-environment-primary-sources.md)

## Bottom line

The evaluation problem is not solved by adding more metrics. It is solved by
making each claim's estimand, comparison, dependence unit, data role, and
consumer explicit. For the immediate R-series work, retain Q24/Q25, deepen the
paired economic attribution, settle Q23, and calibrate Q26 before opening a
panel. Build offline capability evidence in parallel. Transfer it into the
future online `CapabilityProfile`, or into a later approved Phase B curriculum
feedback adapter, only through separate versioned and validated contracts that
keep terminal/final evidence isolated from adaptive training.
