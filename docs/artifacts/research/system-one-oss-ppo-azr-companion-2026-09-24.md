# Open System-One models, PPO coupling, and the AZR boundary

Date: 2026-09-24

Status: research companion; design-only; no model is promoted and no active
architecture, masterplan, or training code is changed

Companion to:
[`system-one-clm-jev-trading-ablation-2026-09-24.md`](./system-one-clm-jev-trading-ablation-2026-09-24.md)

## Executive verdict

There are now several genuinely open Jev-style models, but none is TypeSafe
Jev. They are independent implementations of the typed-decision interface or
of a similar one-forward-pass scoring pattern. The strongest immediate iTrader
research plan remains:

1. test the released `Contrastive-LM/CLM-v0.1-8B` exactly, first frozen and then
   with its projection heads only;
2. add a PPO-trained CLM-head variant without using pairwise InfoNCE;
3. run a matched, cheap numerical ablation that retains the existing
   `SolverMLPFiLM` or R-series trunk and swaps only the linear policy head for a
   normalized three-action scorer;
4. compare one compact open text decision model, preferably Laya for the
   encoder-class alternative and Kev-0.8B or decider-0.8B/2B for a fully
   trainable pointer/logit alternative;
5. only after solver evidence, test a System-One scorer as an **offline selector
   over already validated `TaskSpec` candidates** in the AZR-inspired
   curriculum loop.

The important correction to the earlier report is that PPO can train the exact
CLM state/action heads directly. Repeated `short`, `flat`, and `long` labels do
not create false negatives in PPO: the clipped objective uses the probability
of the action actually sampled in each transition. The false-negative problem
belongs to the generic single-positive batch InfoNCE loss, not to categorical
PPO.

The exact CLM-PPO path and the native numerical cosine-head path answer different
questions and must not be conflated:

- **Exact CLM-PPO:** frozen Qwen3-8B embeddings plus the released CLM projection
  heads produce policy logits. This tests the released text representation and
  scorer, at high inference cost.
- **Native System-One PPO head:** the existing iTrader numerical trunk produces
  the state vector and a cosine/prototype head produces three logits. This tests
  the scoring geometry cheaply, but is not CLM-8B evidence.
- **CLM-distilled numerical head:** the numerical trunk is aligned to cached CLM
  state embeddings and fixed CLM action prototypes before PPO. This is the only
  proposed native path that actually transfers CLM representation information;
  merely initializing three prototypes from CLM does not.

On the local RTX 5070 Ti (16 GB VRAM), CLM-8B BF16 is an inference-sidecar or
precomputed-embedding experiment, not a comfortable in-process RLlib model. A
421M Laya, 596M Tiny-Jev, 0.8B Kev/decider, or 1.9B decider is operationally much
easier. Hardware feasibility is not evidence that any text model can extract
tradable signal from serialized OHLCV.

## Downloaded exact-release assets

The exact released pair is present locally and pinned to immutable Hugging Face
revisions:

- `D:\MODELS\Contrastive-LM\CLM-v0.1-8B` at
  `87655cb835bd76fd66c2da78e1e3709f7fa11a94`;
- `D:\MODELS\Qwen\Qwen3-8B` at
  `b968826d9c46dd6066d109eabc6255188de91218`.

The 75,557,149-byte CLM head and all five Qwen safetensor shards were verified
with SHA-256. The machine-readable sizes and hashes are in
[`model_assets_receipt.json`](./system-one-clm-jev-trading-ablation-2026-09-24-evidence/model_assets_receipt.json).
This is the matching full-precision Hugging Face backbone expected by the
released head; unrelated GGUFs or differently sized Qwen checkpoints are not
valid substitutes for the exact-model ablation.

## Source and naming boundary

TypeSafe describes Jev as its hosted System One model: callers provide a state
and typed `Choice`, `Score`, or `Noul` questions, and receive structured
probabilities rather than generated prose. Its public materials do not provide
weights, parameter count, or a reproducible training implementation. Therefore:

- `Jev` below means the closed TypeSafe service;
- `Jev-compatible` means wire/interface compatibility only;
- `Jev-style` means an independent architecture with similar typed outputs;
- none of CLM, Laya, Kev, Tiny-Jev, decider, Reflex, Needle, or `system-one` is a
  release of TypeSafe's Jev weights.

Primary TypeSafe sources:
[introduction](https://docs.typesafe.ai/introduction),
[machine-learning primer](https://docs.typesafe.ai/introduction/machine-learning-primer),
and [Jev launch note](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

## Open-model landscape

All benchmark and latency numbers in this section are author-reported unless a
source says otherwise. They are not financial results and are not comparable
across different datasets.

| Model | Actual open artifact | Architecture/objective | Size and license | iTrader judgment |
|---|---|---|---|---|
| **CLM-v0.1-8B** | Apache-2.0 code and approximately 75 MB projection-head checkpoint; requires Qwen3-8B | Frozen last-token-pooled Qwen3-8B embeddings; separate state/action MLPs; normalized scaled cosine; bidirectional InfoNCE pretraining and a downstream per-question `choice` path | 8B encoder plus approximately 20M trainable head parameters; Apache-2.0 | Primary exact experiment. The action candidates map naturally to target positions, but every new market state still needs an 8B text embedding. Use the `choice` objective or PPO, never naive repeated-label InfoNCE. [Repository](https://github.com/Contrastive-LM/CLM), [model card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B), [fine-tuner](https://github.com/Contrastive-LM/CLM/blob/main/train/finetune.py) |
| **Laya** | Weights, Python runtime, Jev-shaped server, training/evaluation code, ONNX export | Bidirectional encoder plus decision head. English and typed-decision checkpoints use ModernBERT-large; multilingual uses mmBERT-base. The project describes full-model fine-tuning and RLCD/proper-scoring-rule training. | 421M English/typed-decisions; 322M multilingual; Apache-2.0 | Best compact encoder-class comparator. Three choices are trivial for its option budget. Treat zero-shot probabilities as uncalibrated on trading until local fit; the authors explicitly say fine-tuning supplies most of the value. [Model family](https://huggingface.co/convaiinnovations/laya), [source/runtime](https://github.com/NandhaKishorM/laya), [ONNX runtime](https://github.com/receptron/laya) |
| **Kev** | Weights for 0.8B/4B/9B, training code, evaluation data, Jev-shaped server | Qwen3.5 base with rank-16 LoRA and pointer head. Question isolation is explicit; the pointer scores option states and is trained with cross-entropy while base weights remain frozen. | 0.8B, 4B, 9B; Apache-2.0 | Most transparent small trainable replica for a PPO modification study. Start at 0.8B; 4B is feasible but less comfortable inside a 16 GB rollout/training process; 9B BF16 is not. The authors warn that returned confidence is not measured accuracy. [Repository](https://github.com/jaredpalmer/kev) |
| **decider** | Weights, inference package, data registry, training scripts, evaluation history, Jev-shaped API | Qwen3.5 base; one-pass letter-logit readout. Supervised stages use cross-entropy; decider-2B v10 adds outcome-only calibration-aware RL with a KL constraint. | 0.8B (1.4 GB BF16), 1.9B/2B (3.5 GB BF16), 4.2B (8.4 GB BF16); Apache-2.0 | Strong second trainable alternative because it demonstrates outcome-RL on an open scorer. The 0.8B or 2B model fits the local GPU. Its existing RL recipe is not iTrader PPO and does not establish market calibration. [Repository](https://github.com/Mapika/decider), [2B model card](https://github.com/Mapika/decider/blob/main/MODEL_CARD.md) |
| **Tiny-Jev** | One merged safetensors checkpoint and approximately 150 lines of custom inference code on the Hub | Qwen3-0.6B decoder stack without the LM head; each option marker is mapped by a scalar head; softmax; LoRA training with soft cross-entropy and post-hoc temperature fit | 596M BF16; Apache-2.0 | Very cheap inference/smoke comparator. The card provides a clear architecture and weights, but it is less attractive than Kev/decider for reproducible training-loop work because the complete original data-generation/training pipeline is not the main released artifact. It also requires `trust_remote_code=True`, which should be reviewed before loading. [Model card and code](https://huggingface.co/lostargon/Tiny-Jev) |
| **mpuig/system-one** | MIT research workbench, LoRA adapters, model cards, local HTTP service | Small Qwen/SmolLM/MiniCPM backbones with versioned candidate readouts and temperature fitting; some training targets include a pinned Jev service | Released 0.6B Qwen tier and 2B MiniCPM q8 tier; code MIT, bases retain their licenses | Useful methodology reference, especially its distribution-shift calibration findings. The implementation is Apple-MLX-first, so it is not the first Windows/NVIDIA choice, and Jev-derived targets weaken its value as an independent Jev comparison. [Repository](https://github.com/mpuig/system-one) |
| **system-one-270m** | Full Gemma 3 270M checkpoint and a direct candidate-logit example | Renders options as answer letters and softmaxes the next-token logits over allowed letters; trained on soft targets using log loss | 270M; Gemma license | Smallest credible smoke model here, but its letter scheme caps candidate count at 26 and its synthetic teacher defines the ceiling. Appropriate only as a cheap interface/control probe. [Model card](https://huggingface.co/kaivoss/system-one-270m) |
| **Needle 3** | 121M safetensors for tuning, 8--29 MB CQ2 runtime artifact, platform engines and LoRA tooling | Laddered generative tool/extraction model with grammar-constrained decoding and a learned confidence head; it can express classification through enum extraction | 121M, with 2--20 deployable layers; Apache-2.0 | Interesting edge deployment model, **not an apples-to-apples Jev/CLM scorer**. Its public contract returns a chosen structured result and confidence, not the full arbitrary-candidate distribution required by the three-action PPO policy. Keep it out of the first System-One solver ablation. [Model card](https://huggingface.co/Cactus-Compute/needle3) |
| **Reflex** | Open runtime around frozen Qwen3.5 checkpoints, Jev-shaped server and evaluation ledger | Prompt/readout method over candidate-letter logits; the current stable configuration uses frozen Qwen3.5-4B and two option orders after its authors found their fine-tunes regressed general judgment | Default 4B, roughly 8 GB; repository license controls the code and Qwen controls weights | Valuable zero-training/logit-readout control, not a projection-head model. It shows that prompt/readout controls and option-order sensitivity belong in evaluation. [Repository](https://github.com/kshetrajna12/reflex) |

### Shortlist for iTrader

Use three lanes rather than downloading every new replica:

1. **Released-representation lane:** CLM-v0.1-8B.
2. **Compact encoder lane:** Laya typed-decisions (421M).
3. **Open trainable pointer lane:** Kev-0.8B first; decider-0.8B/2B is the
   stronger alternative if its outcome-RL implementation is the research target.

Tiny-Jev and system-one-270m are useful smoke controls. Needle 3 is a separate
edge structured-generation family, and mpuig/system-one is primarily an
Apple-MLX research reference. This shortlist prevents model-shopping after
seeing validation results.

## Exact CLM heads under PPO

### Policy construction

The exact released scorer can be placed behind the existing RLlib PPO algorithm
without pairwise contrastive training:

```text
causal state at t
  -> deterministic schema-versioned text
  -> frozen Qwen3-8B pooling embedding e_s(t)
  -> trainable released CLM state_head -> unit vector z_s(t)

three fixed target-position descriptions
  -> cached frozen Qwen3-8B embeddings e_a(0..2)
  -> released CLM action_head -> unit vectors z_a(0..2)

policy logits_j = exp(logit_scale) * dot(z_s, z_a[j])
value_t = separate value head(e_s or z_s)
```

The three descriptions remain:

```text
0: set the next target position to -1 (short)
1: set the next target position to  0 (flat)
2: set the next target position to +1 (long)
```

They are not `sell`, `hold`, and `buy`; that would change the environment's
target-position semantics.

For transition `t`, PPO uses the categorical log-probability of sampled action
`a_t`:

```text
r_t(theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t)
L_clip = E[min(r_t A_t, clip(r_t, 1-epsilon, 1+epsilon) A_t)]
```

Repeated actions in other rows are neither positives nor negatives in this
objective, so the three-label false-negative problem disappears. The existing
entropy term, value loss, GAE, reward accounting, and environment action mapping
remain unchanged.

### Bounded unfreezing order

Do not train the whole scorer at once. Use the same rollouts and PPO settings:

| Step | Trainable policy parameters | Purpose |
|---|---|---|
| `CLM-PPO-0` | none; released policy frozen, value head only | Interface/action-collapse and value-path diagnostic; not a policy challenger |
| `CLM-PPO-1` | state projection head; action head and logit scale frozen | Cheapest test of whether PPO can align trading state to the released action geometry |
| `CLM-PPO-2` | state and action projection heads; scale frozen | Tests whether three prototypes need domain movement without adding temperature instability |
| `CLM-PPO-3` | both heads plus bounded logit scale | Only after finite-logit, entropy, and collapse gates pass |

Qwen3-8B stays frozen throughout. Encoder forward passes should run under
`no_grad`; embeddings can be stored in the rollout batch together with the
state-schema hash. Cached embeddings may be reused only when the exact serialized
state matches. Adding current position, PnL, or episode progress makes most state
keys unique; that is expected and is not a reason to omit causal agent state in
the later matched observability experiment.

The value function is a separate learned scalar head. Cosine action similarities
are not a value estimate, expected PnL, or calibrated action utility.

### Calibration boundary

PPO optimizes expected return under a clipped policy objective, not a proper
scoring rule against a known correct label. Therefore a PPO probability such as
`P(long)=0.8` must not be described as an 80% chance that long is correct. Report
entropy, margin, action mix, and realized cost-aware utility by margin bucket.
Fit a post-hoc temperature only against a separately defined held-out event if a
future gate actually needs a reliability probability.

## Correct coupling to the current numerical PPO trunk

The current implementation is:

```text
price_window_flat -> MLP trunk -> hidden
task embedding -> FiLM -> conditioned hidden h_t
h_t -> linear policy head (3 logits)
h_t -> value head
```

See [`SolverMLPFiLM`](../../../itrader/models/solver_mlp_film.py). A minimal
System-One geometry ablation changes only the policy readout:

```text
h_t -> state projection W_s -> normalize -> q_t
three action prototypes P[0:3] -> normalize
logits = temperature * q_t @ P.T
value = unchanged vf(h_t)
```

This preserves `TaskSpec`, `ITraderEnv`, `RewardEngine`, PPO/GAE, transaction
costs, and the value path. The same head can sit on an R-series hidden state
later, but R0 must remain the first matched control.

### The semantic-transfer trap

Initializing `P` from CLM's three action embeddings is not, by itself, CLM
knowledge transfer. With a freely trained `W_s`, any three non-degenerate fixed
vectors are merely a coordinate basis that the state projection can learn to
hit. A random orthogonal-prototype control can be equivalent up to rotation.

Use these controls:

| Variant | State path | Prototypes | Claim allowed |
|---|---|---|---|
| `R0-LINEAR` | existing numerical trunk | learned linear rows | Current baseline |
| `R0-COS-RAND` | same trunk + normalized projection | fixed seeded orthogonal vectors | Pure normalized/prototype-head effect |
| `R0-COS-LEARNED` | same trunk + normalized projection | learned from seed | Learnable cosine geometry effect |
| `R0-COS-CLM` | same trunk + normalized projection | fixed CLM action-head vectors | CLM-coordinate initialization only, not semantic transfer |
| `R0-COS-CLM-DISTILL` | trunk projection first aligned to cached CLM state vectors | fixed CLM action-head vectors | Actual CLM representation distillation |

`R0-COS-CLM-DISTILL` should pretrain only the numerical state projection against
the frozen CLM state vector for the same causal serialized observation, then run
PPO. A categorical teacher-policy KL over all three actions is also valid and
has no repeated-label false-negative problem. Keep the distillation phase,
teacher hash, and PPO phase separately logged. If PPO is allowed to move the
entire trunk freely, measure whether teacher alignment survives; otherwise the
run can become an ordinary cosine head while retaining a misleading CLM name.

Do not add `agent_state_v1` only to the System-One challenger. Under the active
`R4-v1` rule, position, previous action, causal PnL/drawdown/turnover, and episode
progress must be added to the feed-forward/recurrent control and the challenger
identically.

## AZR: where a System-One scorer actually helps

The original AZR paper trains one language model in both proposer and solver
roles. Proposed induction, deduction, and abduction tasks are executor-validated;
the solver is rewarded by executable correctness; and the proposer receives a
learnability proxy. In the v2 paper reviewed locally, proposer reward is zero at
observed success rate `p=0` or `p=1`, and otherwise `1-p`. It favors hard tasks
that the sampled solver sometimes solves. It is not a direct measurement of
pre/post learning progress. The official repository summarizes the repeated
`PROPOSE -> validate/learnability reward -> SOLVE -> executable accuracy reward`
loop and uses TRR++, not PPO/GAE. [Original paper](https://arxiv.org/abs/2505.03335),
[official source](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner),
[`azr-stack-review-2026-09-12.md`](./azr-stack-review-2026-09-12.md)

iTrader deliberately differs: it uses historical market data, a typed task
contract, and a numerical PPO solver. The current `train_AZR.py` creates a pool
and hands it to PPO once; it is not a co-adaptive loop. Keep the name
**AZR-inspired adaptive curriculum learning**, not zero-data AZR.

### Useful role: validated-task selector

A Jev/CLM-style scorer is a natural non-generative selector inside the offline
curriculum:

```text
rule/proposer creates candidate payloads
  -> validate_and_clamp_task
  -> canonical validated TaskSpec candidate set
  -> System-One ranker chooses one candidate or KEEP_BASELINE
  -> unchanged PPO solver training
  -> EpisodeEvidence on frozen chronological panels
  -> versioned CapabilityProfile / empirical success record
  -> ranker feedback
```

The ranker's state may include only training/development information: current
capability gaps, task-family coverage, recent valid-task outcomes, compute budget,
and reference examples. Each candidate action is a canonical serialization of a
validated `TaskSpec`. The scorer does not invent missing fields, bypass task
validation, rewrite reward formulas, or see the sealed test.

This can reduce wasted solver steps by preferring useful candidates, but it does
not generate new tasks. Open-ended novelty still comes from a rule generator or
proposer. That is the largest difference from the original AZR proposer.

### Reward choices and their risks

Three selector rewards should be separated:

1. **AZR-like empirical learnability:** compute the solver's observed `p` on a
   fixed panel and apply the documented proxy. It is cheap but may prefer noise,
   and finite `p=0` does not prove impossibility.
2. **Immediate cost-aware solver outcome:** net utility or success after
   training on the selected task. This is closer to the objective but is noisy
   and encourages panel specialization.
3. **Capability uplift:** change on a held-out capability panel before versus
   after equal solver compute. This best matches curriculum value, but is the
   most expensive and delayed reward.

Start with a frozen ranker comparison and logged outcomes. Only train a ranker
with contextual-bandit/PPO updates after uplift is reproducible. A variable
candidate set is part of the action space: every rollout receipt must retain the
ordered candidate list, validity mask, candidate hashes, chosen index, old
log-probability, and ranker checkpoint. Probabilities are conditional on that
specific set and cannot be compared across different sets as absolute utility.

### What must not be coupled yet

- Do not jointly change solver head, task selector, task encoder, and reward in
  one run.
- Do not let the System-One scorer alter `TaskSpec` or `RewardEngine` semantics.
- Do not use future `CandidateTask`/`ExecutionIntent` fields as offline
  `TaskSpec` fields.
- Do not turn policy logits into `CapabilityProfile.estimated_p_solve`.
- Do not let a learned ranker bypass chronological validation or select on the
  sealed test.
- Do not give a solver or selector broker, risk-limit, or live execution
  authority.
- In the future online shell, preserve explicit `WAIT`, followed by
  `DecisionGate -> RiskEngine -> Executor -> DecisionRecord`.

## Bounded research loops

These are predeclared research loops, not authorization to launch training.
Each loop changes one factor and uses the existing 1 bp chronological panel,
static-direction controls, action-collapse gates, replay receipts, and R0/R-series
controls from the primary report.

| Loop | Baseline -> challenger | Budget/stop | Evidence question |
|---|---|---|---|
| `O1` interface screen | R0 inference -> frozen CLM, Laya, Kev-0.8B, Tiny-Jev on the same 10k causal states | One fixed pass; stop any model with invalid probabilities, >98% one-action collapse, schema/order instability, or unusable latency | Does any released representation produce nontrivial, reproducible three-way signal before training? |
| `P1` exact heads PPO | `CLM-PPO-0 -> 1 -> 2`; Qwen always frozen | Three seeds; 20-update/~96k-step correctness gate; 60-update cap; stop on nonfinite scale, entropy collapse, replay mismatch, or no uplift against static-direction control | Can PPO align the released CLM heads without supervised labels? |
| `P2` native head | `R0-LINEAR -> R0-COS-RAND -> R0-COS-LEARNED` | Same seeds, steps, hyperparameters, observation schema, hidden width, and value head | Does normalized prototype geometry help independently of CLM semantics? |
| `P3` semantic transfer | `R0-COS-RAND -> R0-COS-CLM -> R0-COS-CLM-DISTILL` | Fixed cached teacher corpus and hash; same PPO budget; stop if teacher alignment disappears without economic gain | Does CLM representation transfer add anything beyond a rotated three-class head? |
| `A1` offline AZR selector | rule/random task selection -> frozen scorer ranking of the same validated candidate sets | Frozen solver first; three chronological folds; equal accepted tasks and solver steps; no final test | Does selection improve downstream capability or only re-order familiar tasks? |
| `A2` learned selector | frozen selector -> contextual-bandit/PPO selector | Begin only after `A1`; cap at three seeds and one selector update per frozen solver-evidence batch | Can delayed empirical capability reward train a selector without panel overfit? |

Promotion still requires positive held-out net return at 1 bp, positive net Sharpe,
paired block-aware uplift, action-distribution attribution, fresh-process replay,
2 bp survival, and independent chronological replication. Text-model benchmark
accuracy, ECE, or low latency cannot substitute for these gates.

## Recommended next decision

The best near-term branch is not “Jev instead of the R-series.” It is a parallel,
strictly matched head study:

```text
R0/R-series trunk + PPO
  ├─ current linear 3-logit head
  ├─ normalized learned three-prototype head
  └─ CLM-distilled three-prototype head

released CLM-8B
  ├─ frozen zero-shot scorer
  ├─ supervised choice-head fine-tune
  └─ PPO-trained projection heads, Qwen frozen
```

This design answers four distinct questions: whether released CLM already has
signal, whether its heads can learn from market labels, whether PPO can train the
same scorer without an oracle label, and whether any useful representation can
be compressed into the numerical trunk. It also allows the R-series to proceed
unchanged and supplies a later, evidence-gated ranker for the offline AZR-inspired
curriculum.

## Architecture invariants retained

- `TaskSpec` remains the canonical offline task/problem contract.
- `ITraderEnv` owns causal market-window assembly and task-conditioned reset.
- `RewardEngine` owns solver rewards and finite reward components.
- R0 remains the implemented/default control; R-series identities remain intact.
- Offline task candidates are validated before any learned ranker sees them.
- `CandidateTask`, `ExecutionIntent`, `CapabilityProfile`, and `DecisionRecord`
  remain separate future online/shadow seams.
- `WAIT` remains a first-class online outcome and is not the same as the solver's
  target-position action `flat`.
- `DecisionGate`, `RiskEngine`, and `Executor` remain deterministic authorities;
  no System-One score directly executes a trade.
