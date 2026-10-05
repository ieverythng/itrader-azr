# CLM and Jev for iTrader: System One integration

Date: 2026-09-24

Status: research artifact; design-only; not an active architecture or promotion decision

Scope: primary-source review of Contrastive-LM/CLM and TypeSafe Jev, plus a
bounded solver, offline task-ranking, and future online capability-matching
ablation proposal. Market-JEPA is mentioned only to keep its separate future
research boundary explicit.

## Executive finding

There is a worthwhile ablation here. The concrete first experiment is to use the
released CLM-8B state/action scorer against iTrader's three target-position
actions, first zero-shot and then with only its small projection heads
fine-tuned. This does **not** mean full-training an 8B model, and it does not
replace the R-series.

The released Contrastive Language Model, `Contrastive-LM/CLM-v0.1-8B`, is a
non-generative decision model built from a frozen Qwen3-8B text encoder plus
separate trainable state and action projection heads. It scores a supplied set
of actions by scaled cosine similarity to the supplied state. This makes a
three-candidate decision mechanically compatible with iTrader's
`short / flat / long` target-position space. It does **not** predict future
states, learn from OHLCV tensors natively, or produce absolute action values.
Its "zero-shot" behavior is text state-to-text action retrieval learned from a
large general corpus, not zero-shot financial policy learning. [CLM model
card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B), [CLM README and
method](https://github.com/Contrastive-LM/CLM#how-it-works)

The immediate CLM-8B solver probe is:

```text
causal market/task/agent state -> deterministic bounded text
    -> frozen Qwen3-8B pooling encoder -> CLM state head

three explicit target-position descriptions
    -> cached Qwen3-8B embeddings -> CLM action head

scaled cosine similarities -> softmax -> short / flat / long
```

The zero-shot form is evaluated directly as a policy in the existing environment.
The heads-only form uses the released per-question `choice` objective or an
equivalent multi-positive objective over the same three actions, then freezes the
selected head for exact replay. Both forms preserve the environment's
`short / flat / long` target-position semantics. CLM-8B is still an LLM encoder
even though it generates no text; an all-numerical cosine-action head is a later
distillation/follow-on ablation, not a substitute for testing the released model
first. Market-JEPA is outside this experiment round.

The proposed order is:

1. retain `SolverMLPFiLM` R0 as the control;
2. run frozen CLM-8B zero-shot over the three explicit action descriptions;
3. if the interface and latency gates pass, fine-tune only CLM's projection heads
   on a versioned three-choice trading corpus with explicit label provenance;
4. use the same CLM/Jev-style state/candidate interface to rank an already
   validated offline `TaskSpec` pool at equal solver compute;
5. only after offline evidence, evaluate a future MarketFramer that ranks frozen
   validated `CandidateTask` candidates plus `WAIT` in replay/shadow mode;
6. with those candidates frozen, separately evaluate a capability matcher over
   versioned solver/checkpoint profiles;
7. distill a successful scorer into a numerical cosine-action head only if the
   exact CLM-8B path establishes useful signal but misses deployment latency or
   resource gates.

## Three names that must remain separate

| Name | What primary sources actually describe | What it is not |
|---|---|---|
| TypeSafe **Jev** | A hosted System One model that receives state plus typed `Choice`, `Score`, or `Noul` questions and returns structured decisions and probabilities in parallel. TypeSafe says it uses a new architecture, a parallel sampler, and Reinforcement Learning for Calibrated Decisions (RLCD), but does not publish the model architecture, parameter count, weights, or training code in the reviewed first-party material. | Not an open 8B checkpoint; not JEPA; not presently self-hostable from official weights. |
| Contrastive-LM **CLM-v0.1-8B** | A frozen Qwen3-8B last-token-pooled text encoder plus separate state/action MLP projection heads, trained with bidirectional InfoNCE. It exposes a TypeSafe-compatible typed-decision API and a direct ranking API. | Not Jev weights; not a transition/world model; not a numerical market encoder; not LLM-free. |
| **JEPA** / proposed Market-JEPA | A separate future joint-embedding predictive-representation hypothesis in the local research archive. | Not part of this CLM/Jev integration, not an action classifier by itself, and not evidence of price direction or tradable edge. |

TypeSafe's own documentation says Jev evaluates typed questions against a state,
returns probability distributions without text generation, and expects complex
judgments to be decomposed into atomic questions and recombined in code.
[TypeSafe introduction](https://docs.typesafe.ai/introduction), [TypeSafe AI
primer](https://docs.typesafe.ai/introduction/machine-learning-primer),
[TypeSafe launch note](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

The launch note is the strongest first-party architectural disclosure found for
Jev: it names a "new model architecture," a parallel sampler, and RLCD, but gives
no layers, parameter count, checkpoint, or reproducible training recipe. It also
states that Jev returns all probabilities in parallel and that higher-cardinality
choices can use a two-stage scoring-then-choice system. Claims beyond that would
be inference, not source-grounded fact. [TypeSafe launch
note](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

Market-JEPA is retained here only as a boundary: it would be a future predictive
representation study, whereas CLM is a present state/action retrieval study.
Nothing in the CLM experiment depends on Market-JEPA.

## Exact released CLM contract

### Architecture and score

The released checkpoint is locked to Qwen3-8B last-token pooling. Qwen3-8B has a
4,096-wide hidden state, 36 transformer layers, 32 attention heads, and 8 KV
heads. [Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json)

CLM applies one MLP head to the state embedding and another to each action
embedding. The public head implementation accepts checkpoint-configured width,
depth, projection dimension, activation, layer normalization, and residual
settings. Its default projection width is 512. Both outputs are L2-normalized,
then scored as:

```text
score(state, action) = exp(logit_scale)
                       * cosine(state_head(state), action_head(action))
```

The checkpoint contains `state_head`, `action_head`, `logit_scale`, and `cfg`.
The released head file is reported as approximately 75 MB; the README describes
the state/action head pair as approximately 20M trainable parameters over the
frozen 8B encoder. [projection-head
source](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/heads.py), [CLM
README](https://github.com/Contrastive-LM/CLM#architecture-data-recipe-and-scaling-laws)

The model card names one base release, `CLM_v0.1-8B.pt`, under Apache-2.0. The
code is Apache-2.0, and the Qwen3-8B base is also Apache-2.0. The checkpoint does
not include a portable 8B encoder; inference still needs the compatible Qwen3-8B
pooling server. [CLM model
card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B), [CLM
license](https://github.com/Contrastive-LM/CLM/blob/main/LICENSE), [Qwen3
model card](https://huggingface.co/Qwen/Qwen3-8B)

### Training objective and released data recipe

For a batch of matched state/action pairs, CLM forms a full state-by-action
similarity matrix and applies cross-entropy in both retrieval directions. The
public trainer normalizes each head output, learns the logit scale, masks records
sharing the same `(task_id, step_idx)`, and averages state-to-action and
action-to-state cross-entropies. [CLM method](https://github.com/Contrastive-LM/CLM#training-algorithm),
[fine-tuning loss](https://github.com/Contrastive-LM/CLM/blob/main/train/finetune.py#L161-L170)

The authors report three stages:

- pre-training on about 60M Nemotron question/answer pairs;
- mid-training on about 30M synthetic hard negatives;
- post-training on about 1M agent trajectory pairs, with 40% question/answer
  replay and 60% agentic trajectories.

These are author-reported results and dataset counts, not independently verified
in this artifact. [CLM data recipe](https://github.com/Contrastive-LM/CLM#data-recipe)

For downstream work, `train/finetune.py` freezes the encoder embeddings and
trains the projection heads. Its `clm` path accepts transition records containing
`state`, `action`, `task_id`, and `step_idx`; it can also consume precomputed
embedding datasets and task-disjoint holdout folds. Its `choice` path builds a
closed candidate set for every question and supports bidirectional InfoNCE or
per-question softmax cross-entropy with hard or soft targets. [transition
adapter](https://github.com/Contrastive-LM/CLM/blob/main/train/adapters.py),
[trainer CLI](https://github.com/Contrastive-LM/CLM/blob/main/train/finetune.py#L598-L646)

The embedding recipe is important: states keep the **tail** of the tokenized
context, actions keep the **head**, and Qwen3-8B is used as a last-token pooling
model. The reference API converts objects and arrays into prose-like text; it
does not preserve an arbitrary numerical tensor or JSON encoding verbatim.
[embedding recipe](https://github.com/Contrastive-LM/CLM/blob/main/train/embed_utils.py),
[wire schema](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/schema.py)

### Inference API and the meaning of zero-shot

There are two useful interfaces:

```python
engine.rank(state_text, [candidate_a, candidate_b, candidate_c])
```

and the TypeSafe-compatible `POST /v1/systemone`, where a `choice` question has
named criteria. The state head sees state text plus question instructions, the
action head sees each candidate description, and a softmax over scaled cosine
scores produces a distribution relative to that candidate set. Candidate
embeddings can be cached independently. [in-process engine](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/engine.py),
[API reference](https://github.com/Contrastive-LM/CLM#api-reference)

"Zero-shot" therefore means that the released state/action geometry is used to
rank a new caller-supplied candidate set without task-specific head training.
It does **not** mean:

- that the model has learned market dynamics;
- that it has seen iTrader's normalized OHLCV representation;
- that its relative softmax probabilities are calibrated expected returns;
- that it estimates counterfactual value for all three trading actions; or
- that it predicts a next state as JEPA does.

The model card explicitly warns that CLM does not generate, only scores supplied
candidates, that probabilities are relative to the candidate set, and that
reported verifier results require fine-tuned heads. [CLM model-card
limitations](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B#limitations)

The authors report zero-shot performance on par with Jev for computer-use,
gaming, and tool-calling, with up to 9x lower latency; that comparison is not a
financial benchmark. For only three fixed action candidates, independent action
caching saves little compared with the cost of re-encoding every fresh market
state through Qwen3-8B. This must be measured locally rather than inferred from
their high-cardinality examples. [CLM zero-shot
results](https://github.com/Contrastive-LM/CLM#zero-shot-evaluation)

## Current iTrader evidence and the documentation drift

No Jev- or Contrastive-LM-specific iTrader research exists in the current
working tree or reachable document history found by text and Git-history search.
The nearest prior artifact, [`iTrader Market-JEPA.html`](./iTrader%20Market-JEPA.html),
is separate provenance rather than the contract for this study. It describes a
future MarketFramer representation and explicitly denies that such a model should
execute directly or prove price direction. It also labels Market-JEPA "Phase C,"
while the active masterplan now defers the optional idea to Phase D, behind
stable market-window and candidate-task contracts. The active documents take
precedence. See [`itrader_masterplan.md`](../../plans/itrader_masterplan.md),
[`itrader_system_architecture.md`](../../architecture/itrader_system_architecture.md),
and [`azr-stack-review-2026-09-12.md`](./azr-stack-review-2026-09-12.md).

The present runtime is materially different from the CLM input contract:

- observation is `[price_window_flat | task_embedding]`;
- action `0/1/2` means target position `short/flat/long`, not a buy/sell/hold
  order;
- R0 does not observe its current position, previous action, cumulative PnL,
  turnover, drawdown, episode progress, or recent strategy returns;
- `TaskSpec` remains the canonical offline problem contract;
- `CandidateTask`, `CapabilityProfile`, and `DecisionRecord` are future online
  seams, not fields to fold back into `TaskSpec`.

Those points are source-defined in `ITraderEnv`, the active architecture, and
the typed task contracts. [`itrader_env.py`](../../../itrader/envs/itrader_env.py),
[`candidate_task.py`](../../../itrader/rl_lib/tasks/candidate_task.py),
[`capability_profile.py`](../../../itrader/rl_lib/tasks/capability_profile.py)

### Latest Dev-panel evidence: why collapse gates matter

The most recent read-only Dev status at the time of this review showed no active
iTrader training process. The latest completed non-flat-search review stopped
at its approximately 96k-step continuation gate. All four members had reached
97,656--102,248 trial-owned steps, yet all four reported exactly the static
short-only control's development-panel metrics: **+0.0578545% 1 bp mean net
return and -0.01558 Sharpe per square-root bar**. The gate also retained one
lineage event still awaiting its first post-restore evidence. This is a stopped
development result, not a promoted solver.
[`gate_decision.json`](./dataset-v2-r0-audit-2026-09-24-evidence/p0_nonflat_search_gate_20260924/gate_decision.json)

Two nearby checkpoint replays reinforce the same warning:

- no-cost member `00002` produced +0.061768% 1 bp net return and -0.00976
  Sharpe with 75% short, 0% flat, and 25% long actions. Its paired block interval
  versus short-only was `[-0.18645%, +0.19428%]`, crossing zero;
- early-staged-cost member `00001` produced +0.066663% 1 bp net return and
  -0.01483 Sharpe while taking the short action 99.967% of the time.

Both analyses explicitly label themselves exploratory, frozen-selection-panel,
and not selection-adjusted. They are evidence of directional collapse and panel
dependence, not evidence of tradable edge or promotion.
[`no-cost member02 evidence`](./dataset-v2-r0-audit-2026-09-24-evidence/p0_no_cost_gate_member02_analysis/checkpoint_evidence.json),
[`early-cost member01 evidence`](./dataset-v2-r0-audit-2026-09-24-evidence/p0_early_cost_member01_analysis/checkpoint_evidence.json)

This changes the emphasis of the System One study. A contrastive action geometry
must not be rewarded merely for reproducing the panel's short prior. Every
challenger needs action-distribution attribution, direct comparison with the
matching static-direction control, and a selective-utility/reliability report
for its score margin. A cosine softmax is not calibrated confidence by default.

## What should count as the state?

The action must be matched against a **causal decision state**, not against an
isolated OHLCV row and not against terminal performance statistics.

Recommended versioned input groups are:

| Group | Candidate fields | Reason |
|---|---|---|
| `market_state_v1` | chronological normalized OHLCV window; causal returns/volatility/time features | Describes the tradable context without future leakage. |
| `task_state_v1` | `TaskSpec` family, horizon, reward semantics, bar resolution, regime tags, difficulty target | Preserves task conditioning and avoids silently turning the model into one unconditional policy. |
| `agent_state_v1` | current target position, previous action, position age, realized cumulative net return, unrealized PnL, drawdown, turnover, remaining horizon/episode progress, recent causal strategy-return statistics | Makes transaction costs and trade management observable. It must be added identically to feed-forward and recurrent controls under the existing `R4-v1` rule. |

OHLCV positions alone are insufficient because the same market window can
require a different target action when the current position, cost already paid,
remaining horizon, or task differs. Conversely, overall PnL or Sharpe alone is
not a market state and cannot replace the window. They are path-dependent agent
statistics. A cumulative or rolling Sharpe feature is especially noisy early in
an episode and can leak information if computed from future or terminal returns;
if used at all, it must be computed only from strategy returns available through
time `t`, carry an observation-count/warmup indicator, and be matched on R0 and
the challenger.

For the exact CLM-8B probe, serialize a small, deterministic, precision-bounded
summary of these fields and use semantically explicit action candidates:

```text
short: set the next target position to -1
flat:  set the next target position to 0
long:  set the next target position to +1
```

Do not call them `sell / hold / buy`; that would change the environment contract
and make `flat` ambiguous when the current position is nonzero.

## The critical three-action training problem

Naively placing iTrader transitions into CLM's generic `clm` fine-tuning path is
incorrect. In a minibatch with thousands of rows but only three repeated action
texts, the published loss treats every diagonal `(state_i, action_i)` as the
unique positive and most off-diagonal entries as negatives. Repeated copies of
the same `short`, `flat`, or `long` action would therefore become false negatives
for one another. The public trainer masks records sharing a `(task_id, step_idx)`,
not records sharing an action label. [published loss
implementation](https://github.com/Contrastive-LM/CLM/blob/main/train/finetune.py#L161-L170)

Three technically valid alternatives are:

1. use the released `choice` path, where every state is scored against its own
   three candidates with per-question softmax cross-entropy;
2. replace single-positive InfoNCE with a supervised multi-positive contrastive
   loss that treats all matching action labels as positives;
3. avoid supervised state/action imitation and place the cosine-action head
   directly inside the PPO policy, letting existing advantage estimates train
   the three logits.

Alternative 1 is the direct trained test of the released CLM and therefore the
primary follow-up to the zero-shot probe, provided labels come from an explicit,
versioned teacher. Alternative 3 is a later distillation study that cleanly
isolates policy-logit geometry inside PPO and avoids inventing an oracle label,
but it no longer tests the released CLM-8B model. Alternative 2 is useful only
after the labeling question is settled.

Possible offline labels have different meanings and must never be mixed:

- frozen solver distillation copies an R0/R-series teacher and can only test
  compression/generalization of that teacher;
- behavior actions copy historical or policy behavior, including its mistakes;
- hindsight best-action labels use future information in the target, which is
  acceptable for supervised training only if input features remain causal, but
  require a declared horizon, costs, and tie/abstain rule;
- counterfactual utility distributions require an explicit simulator and are
  not present in current `EpisodeEvidence` for all unchosen actions.

## Proposed CLM/Jev-focused stack

```text
CLM-8B Discrete(3) solver probe
causal state summary + {short target, flat target, long target}
    -> frozen CLM-8B zero-shot or heads-only fine-tuned choice scorer
    -> target position action -> unchanged ITraderEnv/reward/evidence

offline AZR-inspired curriculum selector
validated TaskSpec catalog + causal market summary + solver capability gaps
    -> CLM-8B state/candidate ranker
    -> selected validated TaskSpec (or keep current curriculum)
    -> solver held-out success/utility
    -> capability profile + proposer/curriculum feedback

future replay/shadow online shell
market window -> MarketFramer -> CandidateTask set + WAIT
    -> CLM/Jev-style CapabilityMatcher(CandidateTask, CapabilityProfile)
    -> eligible solver scores
    -> DecisionGate -> RiskEngine -> Executor -> DecisionRecord
```

The offline selector ranks only candidates that have already passed `TaskSpec`
validation; it does not generate contract fields and it does not move execution
semantics into `TaskSpec`. Its candidate text should contain canonical task
semantics, while its state contains only causal market context and versioned
held-out solver capability gaps. Curriculum uplift is measured at equal solver
steps against rule/random selection.

The `CapabilityMatcher` is shown explicitly to prevent role drift. MarketFramer
describes which tasks the current market supports. The matcher determines which
validated solver, if any, has relevant held-out capability. The solver scores an
eligible candidate. `DecisionGate`, deterministic risk, and execution retain
authority over whether and how an entry occurs. `WAIT` is a real candidate, not
an afterthought. No broker/risk semantics enter `TaskSpec` or the solver.

The same state/candidate interface can serve the future framer without making
Market-JEPA a dependency:

```text
state encoder:  causal market summary + regime + liquidity + horizon context
candidate encoder: CandidateTask semantics + required capability
score: CLM scaled cosine, or a Jev `Choice` result as an external benchmark
candidates: task candidates plus explicit WAIT
```

Unlike the action head, this matcher should be trained from chronological
framing outcomes such as cost-aware solver success, expected net utility, and
abstention correctness. Raw PPO logits are not capability estimates. The
`CapabilityProfile` input must come from held-out or rolling evaluation slices,
and candidate-specific recurrent state must not be mutated while comparing
candidates.

Jev is useful here as a hosted typed-decision comparator because the choice
contract naturally accepts `CandidateTask + WAIT`. It is not the trainable local
model: its weights and architecture are unpublished. CLM-8B is the reproducible,
open-weight implementation path for local zero-shot and heads-only experiments.

## Bounded baseline-to-challenger program

This is a predeclared autoresearch loop, not authorization to launch training.
Every challenger changes one factor, runs against the same data/panel/budget,
and ends in `keep`, `reject`, or `needs-repeat`.

### Frozen experiment contract

- **Primary scalar search metric:** mean held-out
  `evaluation/net_return_mean` at exactly 1 bp on the fixed chronological
  validation panel. This is the only rank-order metric.
- **Promotion gates, not optimization terms:** positive net Sharpe per
  square-root bar; beat flat and strongest matched directional control; flat
  action fraction below 98%; finite evidence; no missing action; exact
  fresh-process replay; nonnegative paired block-aware uplift; positive net at
  2 bp before independent replication.
- **Data:** immutable dataset-v2 RTH authority, identical train/validation/test
  separation and purge, identical `TaskSpec` identities, and a sealed final
  chronological test.
- **Optimization:** the exact CLM-8B path freezes Qwen3-8B. `S1-P0` trains
  nothing; `S1-P1/P2` train only the released projection heads with fixed data,
  optimizer, and epoch budget. PPO is used only by the later numerical
  distillation challenger, with fixed hyperparameters and no PBT mutation.
- **Budgets:** heads-only studies use exactly 250,000 train-split examples,
  three seeds, at most 20 epochs, and patience 5; the checkpoint is chosen by
  label-space validation loss before economic replay. Policy evaluation uses
  the same episode count and task identities as R0. The optional numerical PPO
  distillation uses three seeds, a 20-update/~96k-step correctness gate, and the
  same 60-update cap per healthy seed.
- **Repeat rule:** a nominal win is `needs-repeat` until it improves the primary
  scalar in at least two of three seeds and passes every gate. Re-run suspicious
  wins with a fresh seed and exact replay.
- **Holdout rule:** choose on the declared validation panel, replicate once on a
  new non-overlapping chronological validation block, and open the sealed final
  test only after configuration freeze.

The scalar follows the current ablation contract and avoids optimizing directly
for a noisy or mis-annualized Sharpe. Sharpe, turnover, drawdown, action balance,
latency, memory, and calibration remain mandatory diagnostic/gating outputs.
The current dataset-v2 audit already establishes the cost, replay, block, and
independent-lineage standard that this study should inherit.
[`dataset-v2-r0-audit-2026-09-24.md`](./dataset-v2-r0-audit-2026-09-24.md)

### Challenger ladder

| ID | Single factor | Training budget | Decision |
|---|---|---|---|
| `S1-P0` | Frozen CLM-v0.1-8B ranks the three explicit target-position texts from a deterministic serialization of the same market-window and `TaskSpec` information available to R0. No fine-tuning, PPO, or production claim. | Fixed 10,000 train-split diagnostic states plus the fixed validation panel; one pass; record cold/warm latency and 3-way distributions. | Diagnostic only. Reject if actions collapse, serialization is unstable, or latency is unsuitable. |
| `S1-P1` | Fine-tune only CLM's state/action projection heads on a versioned three-choice trading corpus; Qwen3-8B stays frozen and the generic duplicate-action InfoNCE path is forbidden. | Exactly 250k examples, 3 seeds, max 20 epochs, patience 5; equal fixed economic replay per seed. | Keep only if label-space validation, the frozen economic scalar, collapse gates, and exact replay all pass. |
| `S1-P2` | Add `agent_state_v1` to CLM and to a matched R0 control; no other factor changes. | Same heads-only budget plus matched R0 budget/seeds. | This is an `R4-v1` observability study, not CLM evidence by itself. |
| `S1-P3` | CLM-8B offline task ranker scores only validated `TaskSpec` candidates from causal market summaries and versioned capability gaps; the evaluated solver is frozen. | Fixed chronological curriculum corpus, 3 seeds, identical candidate sets and equal downstream solver steps. | Keep only if downstream held-out solver improvement beats rule/random task selection without opening the final test. |
| `S1-P4a` | Future CLM-style `MarketFramer` ranks validated `CandidateTask` objects plus `WAIT`; a deterministic capability-eligibility mask and every solver are frozen. | Replay/shadow-only chronological corpus, 3 seeds, fixed candidate sets. | Search on cost-aware realized utility; gate calibration, coverage, abstention, latency, and no-lookahead. No live entry authority. |
| `S1-P4b` | With MarketFramer candidates frozen, a separate `CapabilityMatcher` ranks eligible versioned solver/checkpoint profiles for each candidate or returns no match. | Same replay corpus, fixed candidate order, frozen solver outputs and capability snapshots. | Keep only if held-out utility and calibration beat rule matching without converting solver logits into capability estimates. |
| `S1-P5` | If exact CLM-8B has useful signal but fails resource gates, distill its geometry into a numerical state projection plus three learned action prototypes; compare against the unchanged R0 linear head. | 3 PPO seeds, 20-update gate, then 60 updates. | Keep only on the same economic scalar and gates; this is the no-LLM follow-on, not evidence about released CLM-8B by itself. |

Do not start with a joint `CLM head + explicit state + adaptive proposer` run. A
win would be uninterpretable and a loss would not identify the bad seam.

### Correctness gates

Before any economic comparison:

- action indices round-trip exactly as `0 -> -1`, `1 -> 0`, `2 -> +1`;
- every state sees exactly three finite logits and all three actions can be
  sampled;
- probabilities are finite and sum to one; learned temperature is finite and
  bounded;
- value-head and task-conditioning paths remain behaviorally present;
- no feature or label from after decision time reaches the policy input;
- terminal liquidation, cost, turnover, and episode evidence remain unchanged;
- same observation/state schema is used in training, evaluation, checkpoint
  restore, and online replay;
- CLM supervised probes do not use duplicate actions as single-positive
  negatives;
- candidate scoring does not mutate shared recurrent state;
- capability and confidence values are not treated as calibrated until measured
  on held-out chronological data;
- a near-constant long or short policy is not accepted from aggregate net return:
  it must beat the corresponding static-direction control with paired
  block-aware evidence rather than merely reproduce its action prior;
- supervised probes report class balance, macro accuracy, log loss, Brier score,
  and calibration error, while PPO policies report score-margin buckets against
  held-out cost-aware utility and make no probability-calibration claim.

### Required artifacts per attempt

```text
results.tsv                         # one primary-metric row per attempt
experiment_config.json             # exact baseline/challenger and fixed budget
dataset_manifest.json              # snapshot and split hashes
task_panel.json                    # exact task/start/horizon identities and hash
state_schema.json                  # market/task/agent fields, timing, normalization
action_contract.json               # index, target-position semantics, descriptions
checkpoint_manifest.json           # model/config/data hashes
evaluation_summary.json            # net, Sharpe, cost, drawdown, turnover, actions
block_and_regime_report.json        # paired chronological attribution
replay_receipt.json                 # fresh-process checkpoint equality
latency_memory.json                 # train and inference resource cost
leakage_audit.md                    # feature/label timing and causal-input audit
```

Each ledger row should end with:

```text
Decision: keep | reject | needs-repeat
Baseline:
Challenger:
Primary metric delta:
Correctness gates:
Economic gates:
Compute/latency:
Artifacts:
Next experiment:
```

## Feasibility judgment

| Proposal | Technical feasibility | Research value | Recommendation |
|---|---|---|---|
| Frozen CLM-8B zero-shot on serialized trading state | High to prototype; requires Linux, NVIDIA GPU, vLLM, Qwen3-8B, and the 75 MB head | Moderate diagnostic value; low prior for economic edge; poor fit to dense numerical OHLCV and only three cached actions | Run first as `S1-P0`; treat it as an interface/signal diagnostic, not solver promotion. |
| Fine-tune released CLM heads on three trading actions | Feasible, but the generic transition InfoNCE path is invalid without fixing duplicate-action positives | The primary trained CLM ablation if target labels and their provenance are explicit | Run `S1-P1` with the per-question `choice` objective; do not use naive pairwise InfoNCE. |
| CLM-8B offline validated-TaskSpec ranker | Feasible after a versioned capability-gap state and fixed candidate catalog exist | High value for testing System One selection in the AZR-inspired curriculum loop without giving it generation authority | Run `S1-P3` only after the solver probe; compare downstream capability gain at equal solver steps against rule/random selection. |
| Future CLM-style `MarketFramer` over `CandidateTask + WAIT` | Interface-compatible, but current contracts are future seams | Potentially high for candidate framing if calibration, abstention, and chronological replay survive | Keep replay/shadow-only as `S1-P4a`; it emits candidates and WAIT but does not select a solver or execute. |
| Future `CapabilityMatcher` over versioned `CapabilityProfile` rows | Feasible only after profile provenance, uncertainty, and solver identity become typed evidence | Potentially high for choosing which validated solver may score a framed candidate | Test separately as `S1-P4b`; it never bypasses `DecisionGate`, risk, or execution. |
| Numerical CLM-style three-action PPO head | Straightforward and cheap relative to R-series; no text encoder | High for isolating and deploying the state/action geometry, but it is not a test of released CLM-8B | Use only as `S1-P5` distillation if exact CLM shows useful signal but misses resource gates. |
| Hosted Jev as the trading solver | API-compatible with a three-way choice, but closed, remote, text-only, and not reproducibly trainable from published first-party material | Low as a core solver; potentially useful as an external framing benchmark | Do not make it the canonical solver. If access exists, use only as a logged offline benchmark. |
| Separate future Market-JEPA research | The local concept is not concrete enough for this integration round | Potentially useful later as a MarketFramer representation study | Defer; do not couple it to the CLM/Jev experiments or use it as a shortcut around R0/R-series evidence gates. |

## Bottom line

The released 8B checkpoint is worth testing because it makes one useful
factorization concrete:

```text
encode state once -> compare against a small reusable action/candidate set
```

For iTrader, the defensible sequence is to evaluate that exact CLM-8B geometry
zero-shot, train only its heads with a valid three-choice objective, then reuse
the typed state/candidate contract for an offline validated-`TaskSpec` ranker.
Only after those results should the CLM geometry reach replay/shadow evaluation:
first as a `CandidateTask + WAIT` MarketFramer, then as a separately measured
capability matcher over versioned solver profiles. A native numerical
three-prototype PPO head is a valuable later distillation route if CLM supplies
signal but fails latency or resource gates; it must not replace the direct CLM
test or be mislabeled as CLM-8B evidence. This preserves the R-series as the
control, keeps Jev as an external typed-decision benchmark, and leaves the
non-concrete Market-JEPA hypothesis outside the current ladder.

## Companion investigation files

- [`state_action_contract_v1.json`](./system-one-clm-jev-trading-ablation-2026-09-24-evidence/state_action_contract_v1.json)
  records the exact `Discrete(3)` target-position meanings, causal state groups,
  admissible label choices, duplicate-action loss guardrail, and calibration
  boundary.
- [`experiment_matrix.json`](./system-one-clm-jev-trading-ablation-2026-09-24-evidence/experiment_matrix.json)
  records the frozen invariants, `S1-P0` through `S1-P5` budgets (including
  separate `S1-P4a` framer and `S1-P4b` matcher probes), promotion
  gates, and stop conditions in machine-readable form.
- [`system-one-clm-jev-trading-ablation-2026-09-24.html`](./system-one-clm-jev-trading-ablation-2026-09-24.html)
  is the synchronized review companion; this Markdown file remains the source
  of truth.

## Primary sources reviewed

- Contrastive-LM: [model card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B),
  [repository/README](https://github.com/Contrastive-LM/CLM),
  [projection heads](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/heads.py),
  [inference engine](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/engine.py),
  [wire schema](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/schema.py),
  [fine-tuning trainer](https://github.com/Contrastive-LM/CLM/blob/main/train/finetune.py),
  [data adapters](https://github.com/Contrastive-LM/CLM/blob/main/train/adapters.py),
  [embedding recipe](https://github.com/Contrastive-LM/CLM/blob/main/train/embed_utils.py),
  and [fine-tuning protocol](https://github.com/Contrastive-LM/CLM/blob/main/docs/FINETUNING.md).
- Qwen: [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) and
  [configuration](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json).
- TypeSafe: [TypeSafe home/research framing](https://typesafe.ai/),
  [Jev introduction](https://docs.typesafe.ai/introduction),
  [AI primer/RLCD](https://docs.typesafe.ai/introduction/machine-learning-primer),
  [documentation index](https://docs.typesafe.ai/llms.txt), and
  [Jev launch note](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
Research was performed against sources available on 2026-09-24. Vendor benchmark,
latency, calibration, and scale claims are identified as author-reported and
should be independently reproduced before they influence promotion.
