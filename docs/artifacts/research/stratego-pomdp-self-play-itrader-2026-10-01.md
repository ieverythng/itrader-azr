# Imperfect-information self-play and learning-progress curricula for iTrader

Date: 2026-10-01

Status: research artifact; design-only; no implementation, architecture promotion,
training launch, or final-test access is authorized by this document

Scope: primary-source review of Ataraxos and *Self-Play Pretraining with Zero
Data*, followed by a seam-preserving map into the registered iTrader research
wings. The two source mechanisms are intentionally kept separate.

## Executive decision

There are useful ideas here, but the headline is narrower than “turn trading into
Stratego self-play.” The existing R-series sequence remains first. Around that
fixed sequence, the strongest transfers are:

1. **Zero-data self-play -> proposer curriculum:** once one solver recipe is
   frozen and repeatable, test whether a validated `TaskSpec` is useful by the
   *learning progress it induces in that solver*, then add fresh/replay/mutation
   and quality-diversity controls. This is the cleanest route toward a real
   co-adaptive AZR-inspired loop.
2. **Ataraxos -> bounded PPO/evaluation controls:** first audit the resolved
   entropy, KL, clipping, critic, and advantage contracts; then compare raw versus
   EMA checkpoints. Coordinated schedules and absolute-advantage filtering are
   candidates, not presumed cures. They remain optimizer/evaluation factors, not
   new solver architectures.
3. **Ataraxos belief/search -> new PF, WM, and TTS research axes:** a causal,
   calibrated conditional distribution over future market outcomes may be useful,
   but it is not the same object as Stratego's posterior over currently existing
   hidden pieces. Test-time search is deferred until an independently validated
   generative market scenario model exists.

The following are **not** justified:

- literal opponent self-play against historical OHLCV;
- treating the market as a two-player zero-sum game;
- adding a raw action tape to Family `00` merely because Stratego uses move
  history;
- conditioning decisions on realized future bars or future actions;
- calling a forward-return classifier a POMDP belief state;
- feeding future-derived regime tags into the solver;
- redesigning R4-v2 around Ataraxos.

R4-v2's identity is frozen for this review: it is the separately named nonlinear
joint-fusion factor already defined by the project. The current workspace still
records its implementation/evidence status as deferred, while the user reports it
as staged in the active parallel work. That is status drift to reconcile, not
permission to reinterpret the design. This report neither changes that status nor
proposes a different R4-v2. R4-v1 remains the endogenous portfolio-state factor;
it does not make the external market fully observed.
[Active architecture](../../architecture.md), [system architecture](../../architecture/itrader_system_architecture.md),
[masterplan](../../plans/itrader_masterplan.md)

## Source basis and reproducibility boundary

### Ataraxos

- Samuel Sokota et al., “Scalable decision-making for games of imperfect
  information,” *Nature* 658, 55–59, published 30 September 2026.
  [Version of record](https://www.nature.com/articles/s41586-026-11036-y)
- [Official supplementary information](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf)
- Official source reviewed at commit
  [`92db29e8ffc323b1b8a2804b5c3f84695d036b05`](https://github.com/AtaraxosAI/stratego/tree/92db29e8ffc323b1b8a2804b5c3f84695d036b05).
  The repository contains the CUDA Stratego simulator, policy training, belief
  training, search code, and final policy checkpoints. Its README requires Linux
  and CUDA 11.8; the demo is reported at 10–20 seconds per iteration with 128
  environments on an RTX A6000. [Official repository](https://github.com/AtaraxosAI/stratego)

### Self-Play Pretraining with Zero Data

- Aditya Cowsik et al., “Self-Play Pretraining with Zero Data,” arXiv:2609.30063v1,
  submitted 24 September 2026. It is explicitly presented as an initial proof of
  concept. [Paper](https://arxiv.org/abs/2609.30063),
  [full text](https://arxiv.org/html/2609.30063)
- Official release reviewed at commit
  [`2c25ed621e959020e1cc9ca1b1032115e85c3798`](https://github.com/nourya-aliz/self_play_pretraining/tree/2c25ed621e959020e1cc9ca1b1032115e85c3798).
  The release includes figures, their data, scoring code, and learner checkpoints.
  Its own scoring README says the training code is not part of the release.
  Generator weights, generator optimizer state, and an end-to-end self-play
  training path are therefore unavailable from the official artifacts reviewed.
  [Repository](https://github.com/nourya-aliz/self_play_pretraining),
  [reproduction guide](https://github.com/nourya-aliz/self_play_pretraining/blob/2c25ed621e959020e1cc9ca1b1032115e85c3798/REPRODUCING.md),
  [released learners](https://huggingface.co/nourya-cohen/solomonoff-paper)

All external technical claims below are grounded in those primary sources. Local
iTrader status is grounded in current source, tests, active architecture, and
project-state documents. The GitNexus index was stale during this review, so it
was not used as authority.

Machine-readable companions: [source snapshot](stratego-pomdp-self-play-itrader-2026-10-01-evidence/source_snapshot.json)
and [experiment matrix](stratego-pomdp-self-play-itrader-2026-10-01-evidence/experiment_matrix.json).

## The two mechanisms are not the same kind of self-play

| Dimension | Ataraxos | Self-Play Pretraining with Zero Data |
|---|---|---|
| Learner's problem | Act in an imperfect-information sequential game | Predict the next byte in synthetic sequences |
| “Other side” | Another copy of the game policy, plus hidden opponent pieces | A generator model that proposes executable programs |
| Learner update | Policy/value RL from self-play trajectories | Ordinary next-byte cross-entropy |
| Generator/proposer update | Stratego setup policy receives game outcomes | Program generator receives a learning-progress RL reward |
| Hidden-state treatment | Separate supervised posterior model over true hidden piece identities | No POMDP belief model |
| Test-time planning | Belief samples + forced-action policy rollouts + one tabular policy update | None |
| Transferable iTrader seam | PPO controls; later predictive-distribution/search research | Offline `TaskSpec` curriculum and proposer feedback |
| Non-transferable shortcut | Market-as-opponent self-play | Universal-machine-generated “market data” replacing real OHLCV |

Calling both systems “self-play” is terminologically valid but architecturally
misleading. Ataraxos improves a policy through games against itself. The zero-data
paper implements a co-evolving teacher–student curriculum in which one model
generates training examples for another.

## Formal causality correction

Let the causal history available at time $t$ be

\[
h_t=(o_{0:t}, a_{0:t-1}, r_{0:t-1}, e_{0:t}),
\]

where $e$ contains any causally available endogenous portfolio/accounting state.
A deployable policy is

\[
a_t \sim \pi(\cdot\mid h_t, \tau),
\]

with task contract $\tau$. The current action can depend on causal history and an
internal state computed from it. It cannot condition on the *realized* future
bars, rewards, or actions $(o_{t+1:},r_{t:},a_{t+1:})$.

Its value nevertheless integrates possible futures:

\[
Q^{\pi}(h_t,a_t)=
\mathbb{E}_{\substack{o_{t+1:},a_{t+1:}\sim P,\pi}}
\left[\sum_{k\geq 0}\gamma^k r_{t+k}\mid h_t,a_t\right].
\]

This is the precise sense in which later policy behaviour affects a present
decision. The present action depends on a *distribution over possible future
trajectories*, not on the future trajectory that will later be observed.

Three state objects must remain distinct:

1. **POMDP belief:** $b_t(z)=P(z_t=z\mid h_t)$, a posterior over a currently
   latent state under an explicit latent-state model.
2. **World model / predictive distribution:**
   $P(o_{t+1:t+H}\mid h_t,a_{t:t+H-1})$, a distribution over future
   observations. For the present price-taking simulator, market bars do not
   become action-conditioned merely because portfolio PnL does.
3. **Endogenous portfolio state:** known causal variables such as current
   position and remaining horizon. R4-v1 exposes two such variables. They are not
   beliefs and not forecasts.

An exact Bayesian belief can be a Markov information state for a specified
POMDP, but Ataraxos does not demonstrate that its learned approximation makes an
arbitrary domain exactly Markov. It uses a large engineered information state for
the policy and a separate approximate posterior only when performing search.

## Ataraxos: what the system actually does

### 1. Two policy/value processes trained in tandem

Stratego has a private setup phase and an alternating-move phase. Ataraxos uses a
decoder-only setup transformer and an encoder-only move transformer. It samples
setups and moves directly from the corresponding networks during self-play. Setup
choices determine the initial boards; completed-game outcomes update both setup
and move processes. The authors chose the decomposition because the two phases are
substantially different tasks, not because the setup network is a generic
open-ended task proposer. [Nature Methods](https://www.nature.com/articles/s41586-026-11036-y#Sec9)

The released move network is an eight-layer, 384-wide transformer. It produces
move logits from a query–key product over board-cell embeddings and a categorical
win/draw/loss value distribution. It is not recurrent.
[Move-network source](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/networks/move_transformer.py#L18-L144)

### 2. Engineered information state, not learned temporal recurrence

The paper representation contains 455 current-board channels plus 32 recent-move
channels. The current-board block includes own piece types, heuristic opponent
type probabilities under uniform setup/move assumptions, hidden/empty/moved
flags, progress toward draw limits, and accumulated threat, evade, adjacency,
capture, death, protection, and starting-location information. Each recent move
is a board plane with (-1) at its source and (+1) at its destination.
[Supplement, Section S2](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf)

In source, 355 board-state planes and the separate piece-ID grid are combined by
the feature orchestrator; the move policy selects the latest 32 movement planes.
[Channel registry](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/src/env/boardstate_channels.h),
[feature orchestration](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/networks/feature_orchestration.py#L20-L118)

The article is explicit that Ataraxos accesses history through features rather
than learning across time directly. It reports that temporal attention or
recurrence helped belief modelling under compute normalization, but did not
produce an analogous out-of-the-box compute-normalized policy improvement due to
runtime/memory cost and interaction with advantage filtering. This is evidence
against using Ataraxos as an argument that iTrader should replace its R-series
definitions with a recurrent Stratego clone. [Opportunities for improvement](https://www.nature.com/articles/s41586-026-11036-y#Sec24)

### 3. Dynamically damped PPO-like learning

The move objective combines:

- clipped policy improvement with ratio range ([0.8,1.2]);
- reverse KL from current policy to the data-collection policy, coefficient
  (0.1);
- reverse KL to a “magnet” policy;
- categorical value cross-entropy;
- gradient norm clipping and an annealed Adam learning rate.

The Nature paper and supplement report a magnet that first selects a movable
piece uniformly and then a legal move for that piece uniformly. It is not a
no-op or safety policy. The pinned public checkout differs: its default selects
uniformly over legal actions, while piece-first sampling is a non-default branch,
and the reviewed release does not evidence a final-run override. Paper-reported
and released-code magnets must therefore remain distinct provenance claims. The
regularization coefficient and learning rate decay by power laws. The released defaults include
(lambda_{return}=0.8), (lambda_{advantage}=0.5), PPO clip (0.2),
EMA (0.999), advantage-filter quantile (0.75), magnitude floor (0.01), and
one training epoch for moves.
[Supplement, Tables S6–S7](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf),
[RL configuration and loss](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/core/rl.py#L47-L70),
[loss implementation](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/core/rl.py#L543-L585),
[magnet helper](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/utils/helper.py#L20-L36)

The absolute-advantage filter retains only positions at or above both the batch's
75th percentile and the fixed 0.01 floor. With non-tied advantages it retains
roughly the upper quarter, and the fixed floor can reduce that further; ties can
retain more. It is not merely a low-confidence-label filter.
[Buffer source](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/core/buffer.py#L194-L247)

For iTrader's current three always-legal actions, a *uniform* Ataraxos-style
magnet is not a new loss:

\[
\mathrm{KL}(\pi\,\|\,U_3)=\log 3-H(\pi).
\]

Minimizing it is entropy maximization up to an additive constant. iTrader already
anneals entropy, clips PPO ratios and gradients, and searched multiple entropy
schedules. The source-derived novelty worth testing is the *coordination* of
update size, behavior-policy KL, and regularization strength, not relabelling the
existing entropy term as a magnet.

### 4. Separate belief network trained after the final policy

Ataraxos trains the belief model on every position in trajectories generated by
the final self-play setup and move policies. Given the acting player's information
state, it teacher-forces an autoregressive prediction of the true opponent hidden
piece types. Ground truth exists because the simulator knows the complete board.
Dropout is used to improve generalization to policies unlike self-play, including
humans. [Nature belief modelling](https://www.nature.com/articles/s41586-026-11036-y#Sec14)

The reported Stratego belief network is a transformer encoder plus autoregressive
transformer decoder, not the policy's recurrent memory. Its reported size is
57.1M parameters. The repository also contains temporal and LSTM belief variants,
but their presence in source must not be mistaken for the reported final
configuration. [Belief source](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/networks/belief_transformer.py#L25-L152),
[belief loss](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/belief/belief.py#L210-L251),
[Supplement, Tables S8 and S11](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf)

### 5. Test-time search refines only the current decision

For each move, the reported Stratego search:

1. samples hidden-piece configurations from the learned posterior;
2. forces each legal root action under sampled configurations;
3. performs 1,000 depth-40 rollouts, with the blueprint move policy controlling
   both players after the forced move;
4. averages the leaf value predictions into action-value estimates;
5. takes one tabular magnetic-mirror-descent-like update of the current action
   distribution, regularized to the magnet and blueprint; and
6. samples the played move from that updated distribution.

The reported reverse-KL coefficients are 0.002 to the magnet and 0.02 to the
blueprint policy. The update changes only the current tabular decision; it does
not fine-tune the network. [Supplement, Section S3.7](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf),
[search source](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/core/search.py#L120-L178),
[rollouts](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/core/search.py#L180-L279),
[policy update](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/core/search.py#L431-L454)

### 6. Opponent modelling is limited

Ataraxos does **not** learn or adapt an explicit model of the current human
opponent's policy. It uses its self-play blueprint for both players in rollouts.
The belief approximates the hidden-state posterior induced by self-play policy
trajectories; dropout is the reported defence against out-of-distribution play.
The 20-game human opponent was explicitly told the bot would not adapt. Any
proposal for online market-regime adaptation is therefore a new iTrader
hypothesis, not a feature reproduced from Ataraxos.
[Nature search method and evaluation](https://www.nature.com/articles/s41586-026-11036-y)

### 7. Scale and evidence

The Stratego RL run used 16 H100s for one week; belief training used 4 H100s for
four days. The paper reports 163M completed games, 208B environment steps, 8.56M
move-network gradient steps, and 99.5K setup-network gradient steps. Each GPU ran
1,536 environments and generated 202 moves between updates.
[Training information](https://www.nature.com/articles/s41586-026-11036-y#Sec16),
[Supplement, Table S3](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf)

The headline human result is 15 wins, 1 loss, and 4 draws against Pim Niemeijer;
the World Championship demo produced 38 wins and 2 losses. The paper properly
notes that human games across the series were not independent and identically
distributed, so its binomial calculation is conditional rather than a clean iid
test. Cross-game evidence covers Barrage Stratego, Hanabi, and dou dizhu, which is
strong evidence that the general design pattern is not confined to one two-player
zero-sum game. It is not evidence of trading profitability.

The ablations are informative but conditional on the final configuration. The
paper states that remaining hyperparameters were not retuned for each ablated
system. Three-seed comparisons find material effects from learning-rate annealing,
regularization annealing, and advantage filtering; EMA has similar or better mean
performance with lower cross-seed variation. Those results justify isolated
iTrader ablations, not wholesale hyperparameter copying.
[Ablation caveat and EMA result](https://www.nature.com/articles/s41586-026-11036-y#Sec22)

## Self-Play Pretraining with Zero Data: what it actually does

### 1. Co-evolving program generator and byte learner

Both networks are independently parameterized Llama-style decoder transformers
initialized from scratch. Each round:

1. the generator samples programs in an augmented Brainf*ck-like, Turing-complete
   language;
2. a bounded universal machine executes each program with a random input tape;
3. the learner takes a next-byte cross-entropy step on the outputs; and
4. the generator takes an RL step toward programs aligned with the learner's
   learning trajectory, plus reward-weighted supervised distillation.

This is not a game-state POMDP, belief-state learner, or future-action planner.
Its long horizon is a horizon over learner updates and accumulated curricula.
[Algorithm](https://arxiv.org/html/2609.30063#S2)

### 2. Exact learning-progress reward

At round $e$, the paper defines

\[
p(e)=\lfloor e/2\rfloor,\qquad
\Delta\theta_e=\theta_{p(e)}-\theta_e,
\]

and rewards candidate output $y_i$ by

\[
r_i=\left|
\left\langle
\nabla_\theta\mathcal L(y_i;\theta_e),
P_e\odot\Delta\theta_e
\right\rangle
\right|,
\qquad
P_e=\frac{\mathrm{lr}}{\sqrt{\hat v_e}+\epsilon}.
\]

It is an absolute, AdamW-preconditioned gradient-alignment score. Forward-mode
automatic differentiation avoids materializing a full gradient for every
candidate. The intent is to reject both already-mastered examples, which have
small gradients, and hard-but-unlearnable noise, whose gradients do not align
with demonstrated learner progress. [Reward and implementation description](https://arxiv.org/html/2609.30063#S2.SS2)

The growing half-history lookback outperformed one-step alignment and realized
one-step loss delta in the paper's 1M-parameter, four-seed ablation. Shuffling
scores across programs and negating the reward also degraded results. The signed
form was slightly better on some individual datasets and worse on most, so the
absolute value is supported within this study but is not a universal rule for a
PPO trading solver. [Reward ablations](https://arxiv.org/html/2609.30063#A6)

### 3. Generator optimization and curriculum memory

The generator uses KL-regularized expected reward against a fixed short-program
prior, batch-normalized GRPO-style advantages, and sequence-level importance
ratios for off-policy replay. It also uses reward-weighted supervised fine-tuning
of positively rewarded programs. [Generator objective](https://arxiv.org/html/2609.30063#S2.SS2)

Its pool combines:

- fresh generator samples;
- local single-token mutations of positively rewarded programs; and
- replayed programs from previous rounds, re-executed with fresh random tapes.

A MAP-Elites-style archive partitions programs by length and dynamic loop depth,
retaining multiple strong examples per niche rather than only one global elite.
Stored reward decays because usefulness changes as the learner changes.
[Pool construction](https://arxiv.org/html/2609.30063#A7)

### 4. Evidence and limits

The adaptive generator scales more favourably than a fixed prior over the same
program space. It discovers arithmetic, Fibonacci, geometric, quadratic, and
cubic sequence families much earlier than the reported uniform sampling control,
and its learners exhibit in-context behaviour on six held-out synthetic tasks.
For each endpoint $T$, the fixed-corpus control mixes data from 16 generator
snapshots spaced through training up to $T$, then trains a fresh learner. Later
endpoint-indexed snapshot mixtures train fresh learners more effectively, which
is important evidence that improvement is in the curriculum rather than only in
a co-adapted learner.
[Empirical results](https://arxiv.org/html/2609.30063#S3)

The paper's domain-specific PCFG baseline is stronger on text and code, while the
adaptive universal generator transfers more broadly across several non-language
modalities. That is direct evidence against replacing iTrader's typed market task
grammar with a universal program language merely because the latter is more
general.

The released model ladder spans about 98K to 24.4M parameters, uses 4,096-token
contexts, and runs to 8,192 self-play rounds. The paper's maximum budget is
34.36B learner tokens. Its published “effective compute” axis counts learner
parameters, 1,536 programs per round, context length, rounds, and ensemble seeds;
it does not fully price generator optimization, execution, reward JVPs, or
hyperparameter search. Hardware and wall-clock cost are not specified well enough
to reconstruct total end-to-end expense.
[Scaling recipe](https://arxiv.org/html/2609.30063#S3.SS1),
[official scaling code](https://github.com/nourya-aliz/self_play_pretraining/blob/2c25ed621e959020e1cc9ca1b1032115e85c3798/figures/table2_scaling_exponents/scaling_analysis.py)

Natural data supplies no gradient updates during self-play, but DCLM text and DNA
validation loss are used to select hyperparameters. The “zero data” experiment
therefore has limited evaluation-driven leakage at the research-procedure level.
In the later pre-pretraining experiment, the clearest demonstrated benefit is
faster natural-data learning; the terminal loss gap narrows and the comparison
does not charge the initial self-play compute.
[Scaling recipe and pre-pretraining](https://arxiv.org/html/2609.30063#S3.SS1)

### 5. It cannot rediscover all pretraining facts

The paper explicitly says universal pretraining cannot recover contingent
information: facts about a particular world must enter through interaction with
that world. A universal program space may expose copying, recursion, hierarchy,
arithmetic, and other reusable structure. It cannot identify which possible world
contains a particular earnings release, law, news event, or historical price
path. Longer self-play does not solve this identifiability problem.
[Discussion](https://arxiv.org/html/2609.30063#S6)

For iTrader, real, chronological OHLCV remains indispensable contingent evidence.
The proposer may change which validated tasks are sampled; it must not invent
market outcomes and present them as historical evidence.

## iTrader system facts that constrain the transfer

- `TaskSpec` is the canonical offline task/problem contract.
- `ITraderEnv` owns causal market-window assembly, task-conditioned reset,
  target-position transitions, and reward context.
- `RewardEngine` owns finite solver reward components.
- `Discrete(3)` means target positions short, flat, and long. It is not a
  Stratego move set and `flat` is not the online `WAIT` decision.
- Family `00` is the only enabled learned family. `01`/`02` are bootstrap
  contracts, `03` is trade management, and `04` is abstention/unsuitable regime.
- Historical replay is presently price-taking: solver actions change position,
  costs, and PnL, not the recorded market path.
- R4-v1 supplies known endogenous position and remaining-horizon state. It does
  not infer latent market causes.
- The online authority chain remains `MarketFramer -> CandidateTask(TaskSpec +
  optional ExecutionIntent) + WAIT -> solver -> DecisionGate -> RiskEngine ->
  Executor -> DecisionRecord`; a `CandidateTask` may carry a validated optional
  `ExecutionIntent`, while broker and execution semantics stay in the online
  contracts and never in `TaskSpec`.
- `train_AZR.py` currently creates a validated task pool and hands it to PPO once;
  it is not a co-adaptive proposer/solver loop.

The latest solver evidence also limits what an optimizer transplant may claim.
Exact 1 bp from update zero converged to all-flat, while causal cost staging
converged to the January panel's short-only control. At the informative 77,832-
step checkpoint, greedy replay was 99.358% short even though mean action
probabilities were broad (40.18% / 27.30% / 32.52%) and categorical entropy was
1.08234 versus the three-action maximum of 1.09861. Mean explained variance rose
only to 0.0376, and `clip_fraction` is missing/NaN in all 320 captured reports.
This is not evidence that “more entropy” is the missing ingredient. It is a
reason to repair the critic/clip/advantage diagnostics before advantage filtering
or a coordinated damping claim. [Current causal forensics](solver-state-r4-v1-causal-forensics-2026-09-29.md#ppo-and-critic-interpretation)

Current source adds six implementation prerequisites:

- `train_solo_V2.py` still selects `solver_mlp_film`; R1–R3 remain design cells,
  not executable recurrent challengers.
- the registered legacy LSTM consumes the unsplit full observation and returns
  its incoming recurrent state unchanged; `batch_mode="complete_episodes"` does
  not by itself supply correct BPTT, masking, or state lifecycle;
- frozen-checkpoint evaluation calls `compute_single_action` without owning and
  passing recurrent state, so R2 evidence would be invalid until replay becomes
  state-aware;
- each environment currently owns a process-local task pool, and weighted
  sampling with `batch_size=1` reduces to random because elite/recent counts floor
  to zero; and
- episode realization mutates seed/start before callback hashing, while feedback
  remains in-memory, so a stable proposal UID and centralized versioned evidence
  are prerequisites for curriculum learning; and
- the current proposer retry counter advances only after a returned task, so
  repeated backend exceptions require a bounded-failure contract before any
  co-adaptive loop.

These are seam blockers, not arguments for changing R4-v2.

These boundaries come from the [active architecture](../../architecture.md),
[system architecture](../../architecture/itrader_system_architecture.md), and
[masterplan](../../plans/itrader_masterplan.md).

## Why Stratego is not a direct market analogue

| Stratego property | Current iTrader property | Consequence |
|---|---|---|
| Two strategic agents alter one shared state | Historical prices are exogenous to the solver | No literal market-opponent self-play |
| Hidden piece identities exist now and are known to the simulator | “Regime” is a chosen latent abstraction; future returns are not a hidden label existing in the simulator | A return classifier is predictive, not the same supervised posterior |
| Actions reveal opponent information and change the board | Past target positions add no exogenous market information, but can determine omitted endogenous accounting variables | Prefer explicit sufficient accounting state; do not copy a 32-action history block by analogy |
| Exact rules provide counterfactual rollouts | Only one historical future is recorded; a learned scenario model would be approximate | Replaying the realized future is an offline counterfactual diagnostic, not online search |
| Zero-sum terminal win/draw/loss | Dense, cost-aware, continuous trading reward | Categorical game value and equilibrium claims do not transfer directly |
| Policy rollouts control both players | Solver cannot control future bars | Blueprint rollouts require an independently validated exogenous world model |
| Setup policy changes the initial game distribution | Proposer can select tasks/windows but cannot change history | Map only to curriculum selection, never synthetic historical truth |

The current Family `00` environment can evaluate all three target positions on
the same *past* future path for diagnostics or teacher construction because the
market path is exogenous. It cannot expose that path to the deployed policy or
call the resulting hindsight choice an online search result.

## All-wings placement matrix

Provisional labels below are experiment names, not additions to the canonical
roadmap. They prevent borrowed ideas from silently taking over existing IDs.

| iTrader wing / seam | Ataraxos placement | Zero-data placement | Decision |
|---|---|---|---|
| **R0 MLP-FiLM** | May serve as a fixed solver control for optimizer-only factors | Fixed learner recipe for curriculum attribution | Preserve unchanged |
| **R1 reset-every-step window LSTM** | No direct redesign; Ataraxos policy itself is feature-history transformer, not recurrent | None | Preserve identity; no insertion |
| **R2a/R2c recurrent LSTM and CARRY/RESET** | The existing `R2c-CARRY + R4-v1` versus `R2c-RESET + R4-v1` control is iTrader's clean carried-memory attribution; it does not establish a POMDP belief model, and Ataraxos does not establish recurrent-policy advantage | None | Preserve identities; complete native lifecycle/replay mechanics first |
| **R2b init-only, R2d value-only, R3 GRU** | No clean source-derived change | None | No insertion |
| **R4-v1 portfolio state** | Distinct from belief; already fixes known endogenous aliases | None | Preserve; no insertion |
| **R4-v2 joint fusion** | No redesign. Belief/search does not belong in joint fusion | None | Frozen identity; user reports it staged elsewhere while this checkout still marks implementation/evidence deferred |
| **R-series value head** | Optional future distributional-return factor inspired by categorical value, separately named `ATX-DIST-VALUE-v0` | None | Defer until scalar-critic diagnostics justify it |
| **PPO optimizer/training** | Formula/config audit, then `ATX-EMA-EVAL-v0`; `ATX-ADV-FILTER-v0` and `ATX-DAMP-SCHED-v0` only under isolated controls | Learner recipe must be frozen when scoring curriculum progress | Clean placement, but no presumed basin cure |
| **PBT** | Would confound the schedules being tested | Changing optimizer trajectory invalidates a learning-progress reward | Disable during attribution; only reopen after fixed-recipe evidence |
| **C0/C1/shuffled-C1** | No insertion; these remain the registered causal feature/summary controls | None | Preserve unchanged |
| **PF predictive-state axis (new)** | `PF-PRED-DIST-v0` estimates calibrated future distributions; it is not a supervised posterior over a true hidden market board | None | Evaluate after the existing R and C controls; cross later as `accepted solver × PF1` |
| **WM world-model axis (new)** | Required to generate conditional future paths before deployable search exists | None | Validate separately; historical prices remain action-invariant under the present price-taker contract |
| **TTS search axis (new)** | Belief/scenario samples, branch-safe rollouts, and a stateless current-root logit update | None | Defer until PF/WM and blueprint calibration pass; replay/shadow only |
| **S-series CLM/Jev** | No direct belief/search insertion | Learning-progress task selection may later choose S training examples, but only after native S controls | Keep independent; do not bundle |
| **Family 00 directional** | Raw action history is redundant as *exogenous market* evidence; it may proxy omitted holding age, cost basis, turnover, or drawdown | Curriculum can select validated tasks/windows | Prefer explicit sufficient accounting state and matched controls |
| **Families 01/02** | No claim before the bootstrap contracts become executable | Typed task mutation may later cover them | Defer |
| **Family 03 trade management** | Explicit position/accounting sufficient statistics are more defensible than a raw action tape | Curriculum only after state/reward contract exists | Defer; do not infer state from OHLCV |
| **Family 04 / WAIT** | Ataraxos magnet is not an abstention policy | A proposer may include validated no-trade tasks later | Preserve WAIT as distinct online outcome |
| **TaskSpec / validation** | No hidden-piece fields or broker semantics added | Typed proposal, mutation, validation, hashing, and replay belong here | Strong clean placement for curriculum substrate |
| **RewardEngine** | No Stratego win/loss transplant; PPO training controls stay outside reward semantics | Learning progress is proposer reward, not solver economic reward | Preserve unchanged |
| **Proposer / AZR loop** | Setup/move co-training motivates explicit two-process feedback, but market history stays external | `AZR-LP-DIRECT-v0`, then `AZR-LP-GRAD-v0`, plus `AZR-QD-REPLAY-v0` | Strongest zero-data transfer |
| **CapabilityProfile / evidence** | Calibrated predictive-distribution and search diagnostics may be recorded, never inferred from logits alone | Holds task/checkpoint-specific uplift evidence for curriculum | Clean supporting seam |
| **CandidateTask / ExecutionIntent / DecisionRecord** | Search may score a candidate but cannot mint broker semantics or bypass the gate | Proposer utility has no execution authority | Preserve explicit online contracts; keep them out of `TaskSpec` |
| **MarketFramer** | May eventually consume calibrated causal predictive distributions | No universal-program generator online | Defer; replay/shadow only after validation |
| **DecisionGate/RiskEngine/Executor** | Search output cannot bypass deterministic controls | No proposer authority | Preserve unchanged |
| **Market-JEPA** | A future world/representation model could one day enable scenario rollouts, but Ataraxos does not validate that model | Universal pretraining is separate | Explicitly out of this experiment round |
| **Simulator/performance engineering** | CUDA/vectorized reset/history design is relevant only after profiling identifies rollout bottlenecks | None | Defer; measure before porting |

## Recommended separately named ablations

### A. `ATX-EMA-EVAL-v0` — lowest-risk first transfer

Maintain an EMA shadow of solver parameters and evaluate raw and EMA policies at
identical checkpoint steps on the identical held-out panel. Training continues
from the raw actor; EMA is not silently restored into PBT or treated as another
training seed.

Question: does EMA reduce cross-seed/checkpoint variance or improve replay-valid
economic metrics without hiding action collapse?

Required evidence: exact raw/EMA checkpoint lineage, net return, Sharpe, turnover,
short/flat/long frequencies, logit margins, block uncertainty, and fresh-process
replay. This can often be added with no additional sampled environment steps,
only evaluation and storage cost.

### B. `ATX-ADV-FILTER-v0` — promising but high-risk

This must follow the critic/advantage telemetry repair. Mean explained variance
near 0.038 makes top-|advantage| selection capable of selecting critic error or
volatility outliers rather than informative transitions. Compare three arms: all
valid PPO samples, a top-|advantage| subset, and an equal-retained-count random
subset. Do not combine the filter with a new LR schedule, recurrence, S-series
head, or proposer.

Log at minimum:

- raw versus retained samples;
- threshold and retained fraction by action, position, horizon, symbol, and
  chronological block, volatility, reward scale, and cost stage;
- positive/negative advantage balance;
- clip fraction, KL, entropy, gradient norm, critic loss, and policy/value drift;
- net, Sharpe, turnover, and action collapse.

The central trading risk is selection bias: rare profitable or risk-reducing
events can be discarded if their estimated advantages are noisy or small. Report
both equal sampled-environment-step comparisons (sample efficiency) and equal
retained-gradient-example or optimizer-FLOP/wall-clock comparisons (compute
efficiency). Normalize or stratify thresholds across tasks rather than letting
reward scale decide which task contributes updates.

### C. `ATX-DAMP-SCHED-v0` — low-prior coordinated schedule test

First produce an exact formula/config receipt. iTrader already anneals entropy,
clips PPO ratios and gradients, and PBT searched multiple entropy schedules. Its
causal arm retained high stochastic entropy while greedy actions became
short-only, so dynamic damping is an unproven factor rather than the obvious cure.

Under a fixed solver and with PBT off, screen learning-rate schedule only and
regularization schedule only. Test a predeclared coupled schedule only if the
isolated results justify it. Record the direction/reference of every KL term.

Do not create a uniform-magnet arm: with three always-legal actions it is exactly
the entropy loss up to a constant. A non-uniform, state-dependent stay or flat
prior is a new iTrader objective. “Stay” depends on current position and is
distinct from “flat”; either prior would require its own separately justified
experiment and may change switching/turnover.

### D. `PF-PRED-DIST-v0` — predictive state, not a Stratego belief clone

Keep C0, C1, and shuffled-C1 unchanged. On a separate PF axis, train a causal
conditional distribution over exogenous future-bar quantities such as returns,
realized volatility, and liquidity/spread proxies. Inputs use only $h_t$; targets
may use later bars on the training partition because they are labels, never
observation features.

Action- or policy-dependent net PnL and drawdown belong to a separate outcome-
model contract conditioned on current accounting state, candidate action, and a
fixed subsequent policy/action sequence. They are not labels determined by
$h_t$ alone.

Evaluate continuous targets with CRPS/quantile coverage and categorical targets
with chronological NLL/Brier score, plus reliability and sharpness against
climatology and simple causal baselines. Freeze the solver for the first test.
Call this a predictive distribution unless an explicit latent state and
transition/observation model justify the term “belief.” Cross it later as
`accepted solver × PF1`; here `PF0` means the matched solver without predictive
features and `PF1` means that solver with the accepted frozen predictive
distribution. Do not rename C-series or R4-v2.

### E. `AZR-LP-DIRECT-v0` — repeated direct learning-progress reference

For each validated candidate `TaskSpec`:

1. clone a fixed solver checkpoint and optimizer state;
2. perform one small, equal-compute update on candidate-specific causal data;
3. evaluate pre/post change on an independent chronological capability panel;
4. record cost-aware uplift and adverse changes; and
5. discard the clone.

This expensive micro-update is a noisy local intervention/reference estimate of
whether the task helped *that solver checkpoint*, not oracle truth. Repeat it
across rollout seeds and report confidence intervals. The record key must include
task hash, stable proposal UID, solver and behavior-policy checkpoint digests,
optimizer profile, rollout-batch digest, code/data partition hashes, seed, and
panel identity because task utility is nonstationary.

### F. `AZR-LP-GRAD-v0` — validate, do not assume, gradient alignment

Only after the repeated direct reference is stable, test whether cheap task
scores predict it:

- signed and absolute optimizer-preconditioned gradient alignment;
- task loss/difficulty;
- current return or solve rate;
- gradient norm and gradient sketches;
- random and score-shuffled controls.

PPO makes the language-model formula non-portable: policy gradients depend on
stochastic rollouts and estimated advantages; clipping changes the local
objective; value, entropy, and auxiliary losses point in different directions;
and task-dependent reward scale can dominate norms. Pin the behavior policy and
rollout batch, then score policy-head, value-head, and shared-trunk gradients
separately before considering a combined score. Absolute alignment can reward
movement in an economically harmful direction. Use repeated held-out economic
uplift as the reference before closing the proposer loop.

### G. `AZR-QD-REPLAY-v0` — typed fresh/mutate/replay curriculum

Use a typed `TaskSpec` grammar, deterministic validation, and a quality-diversity
archive. Potential archive descriptors include family, horizon, cost band, causal
regime-summary bucket, symbol group, and difficulty/uplift band. Mutation changes
only permitted typed fields and must never invent an OHLCV path.

Run a `Fixed-QD` control first: the same fresh/replay/mutation machinery without
adaptive utility scores. This isolates diversity and replay from learning-progress
selection. For each proposer endpoint, build a fixed corpus by mixing pinned
snapshots spaced through proposer training up to that endpoint, then train fresh,
identically initialized solvers. This mirrors the paper's strongest
co-adaptation control more closely than sampling only the latest proposer.

### H. `ATX-TTS-DIAG-v0` — defer test-time search

Ataraxos search needs (a) a posterior or predictive distribution over unresolved
state, (b) a fast counterfactual simulator, and (c) a calibrated blueprint value.
iTrader already has exact deterministic position/cost/PnL transitions *given a
price path*. It lacks a validated generative future-market transition model, a
well-defined supervised belief target, and a blueprint/value calibrated on
generated histories. A realized historical future may be used only for an
explicit offline hindsight ceiling.

Before any learned-market search is admissible, a scenario model must pass:

- strict causal input and train-only fitting;
- proper-scoring-rule improvement over simple baselines;
- multi-horizon calibration, coverage, tail, and regime-shift checks;
- independence from the sealed final test;
- common random scenario seeds across all three root actions;
- uncertainty propagation through position/cost accounting; and
- evidence that solver/value rankings remain calibrated under model-generated
  rather than real histories.

With only three root actions, enumerate all three. Every branch must clone
position, remaining horizon, recurrent/PF state, reward normalizer, equity,
max-equity, drawdown, turnover, `TaskSpec`, and reward context rather than mutate a
shared state. The Ataraxos root operation is a stateless current-decision logit
update, not online neural-weight fine-tuning.

Even then, run search only in offline replay/shadow. The selected target position
still passes `DecisionGate -> RiskEngine -> Executor`; solver `flat` remains
distinct from online `WAIT`, and search has no execution authority.

## Bounded autoresearch ladder

This ladder estimates and gates research loops; it is not authorization to run
them. One **screening-run unit (SRU)** is one fixed 12-update, two-seed arm. Under
the recent 4,800-sample/update contract that is 115,200 sampled environment
steps. A two-seed screen can reject but cannot promote. If the accepted frozen
recipe uses a different batch size, recompute the ledger rather than preserving
the number.

### Global objective and invariants

Primary evidence remains held-out chronological exact-1-bp net return and
Sharpe, paired block-aware uplift, strongest static-direction control, turnover,
and the existing 98% near-flat veto. A policy that is at least 98% short or long
is not automatically rejected, but it must beat the matching static-direction
control and demonstrate conditional state dependence. Fresh replay, lineage,
finite metrics, and no sealed-test access are mandatory. Each loop changes one
factor. PBT is off until factor attribution closes.

| Gate | Experiment | Initial budget | Continue only if | Hard stop |
|---|---|---:|---|---|
| `G0` formula/config receipt | Resolve LR, entropy, every KL direction/reference, PPO clip, gradient clip, critic/advantage telemetry, and exact fresh-panel replay | No training | Receipt is source- and checkpoint-backed; clip fraction and advantage distributions are trustworthy | Any lineage, panel, cost, formula, or replay ambiguity |
| `G1` recurrent mechanics | Correct native recurrent selection/lifecycle, masks, last-valid state, episode reset, and state-aware replay | Tests and deterministic smoke | Exact train/eval/checkpoint state parity | Any state leak, invalid mask, reset error, or stateless recurrent replay |
| `G2a` declared R screen | matched R0, clean R1, and R1 + R4-v1 | 3 SRU = 345,600 steps | Mechanics pass and at least one challenger remains measurable | Do not reinterpret a screen as promotion |
| `G2b` memory screen | `R2c-RESET + R4-v1` vs `R2c-CARRY + R4-v1` | 2 SRU = 230,400 steps | CARRY has paired signal with valid state norms/masks | Stop for lifecycle failure or no memory-specific signal |
| `G2c` R extension | Extend a surviving matched R comparison to 40 updates and three seeds | 2-arm comparison = 1,152,000 steps; 3 arms = 1,728,000 | Survives held-out 1 bp gates, direction controls, replay, then 2 bp stress | Do not extend multiple losers or mix optimizer factors into R attribution |
| `G3` EMA | `ATX-EMA-EVAL-v0`: raw vs EMA at identical snapshots | ~0 extra solver steps; evaluation only | Lower variance or better paired economics without hidden collapse and with exact lineage | EMA only wins by changing evaluation mode, checkpoint, or exposure |
| `G4` schedule screen | baseline, LR-schedule-only, regularization-schedule-only | 3 SRU = 345,600 steps | An isolated factor has paired signal; only then predeclare a coupled schedule | Existing entropy term is relabelled, PBT confounds the test, or KL/clip becomes unstable |
| `G5` filtering screen | all samples, top-|advantage|, and equal-count random subset | 3 SRU = 345,600 steps | Top-|advantage| beats random at equal environment queries and remains efficient at equal retained compute | Weak-critic outlier selection, task/action/regime skew, or erased rare events |
| `G6a` direct uplift reference | 24 tasks × 2 fixed checkpoints × 2 rollout seeds × one 4,800-step micro-update | 460,800 steps + evaluation | Repeated pre/post ranking is more stable than difficulty and intervals are usable | Panel noise, reward scale, or optimizer/behavior-policy mismatch dominates |
| `G6b` gradient surrogate | Score the same pinned batches with policy/value/trunk signed and absolute alignment, difficulty, shuffle, and random | Gradient/evaluation compute | Rank association is positive per seed and pooled interval excludes zero | Any sealed-test use or score wins only through scale/norm |
| `G7a` fixed QD | fixed sampler vs `Fixed-QD` | 2 SRU = 230,400 steps | Coverage improves without economic loss at equal accepted solver steps/data | More proposals but unequal solver compute or eligible data |
| `G7b` adaptive curriculum | Add one direct or validated-surrogate selector plus shuffled-score control, reusing valid fixed controls | At most 2 additional SRU per screen | Beats shuffled and fixed on fresh solver seeds | Panel specialization, task collapse, or compute-ledger overrun |
| `G8` PF distribution | Train-only supervised PF baselines first; solver cross uses no-PF, PF, zero-PF, and shuffled-PF | Supervised budget reported separately; policy screen maximum 691,200 steps | Proper scores/calibration pass on new chronology, then PF adds economic value | Leakage, uncalibrated tails, or zero/shuffle ties the claimed gain |
| `G9` WM/TTS | Hindsight ceiling, then WM validity, then frozen search off/on in replay/shadow | No PPO spend until PF/WM and blueprint calibration pass | Learned scenarios beat simple baselines and search beats equal-compute controls | Current recommendation: defer deployable search |

Do not launch the full curriculum factorial. The staged ladder caps spend by
requiring a repeated direct-uplift reference and fixed-QD controls to work before
adaptive surrogates receive solver-scale training. Environment steps, retained
learner examples, optimizer FLOPs/wall-clock, proposal calls, and evaluation cost
remain separate ledgers.

### Autoresearch decision rule

Every completed loop must emit exactly one of:

- **accept for replication:** passed causal/mechanical and economic screen gates;
- **reject:** failed a predeclared gate with valid evidence; or
- **defer:** measurement or prerequisite is invalid, with the missing contract
  named explicitly.

Do not turn “promising training curve,” critic fit, predictive accuracy, Elo-like
internal score, or proposer novelty into an economic promotion claim.

## Transferable ideas, summarized

### Strong transfers

- keep task generation and task solving as distinct processes with explicit
  feedback and independently auditable state;
- use learning progress rather than raw difficulty as a curriculum target;
- compare adaptive curricula to matched fixed distributions over the same typed
  task space;
- retain fresh, replayed, and mutated tasks while explicitly protecting diversity;
- evaluate later proposer checkpoints by the value of frozen corpora to fresh
  learners;
- isolate EMA, advantage filtering, and dynamic damping as PPO factors;
- use a separate calibrated predictive model only when its target and causal
  meaning are explicit;
- refine only the present action distribution during eventual search, leaving
  network weights and deterministic online authorities untouched.

### Conditional transfers

- action/history features only when they carry causal information not already in
  the observation or explicit portfolio state;
- distributional value only after critic diagnostics justify a separate value
  experiment;
- GPU simulator redesign only after profiling shows simulator/reset/history
  materialization is the bottleneck;
- learned test-time search only after a causal scenario generator and calibrated
  blueprint exist.

## Non-transferable assumptions and explicit rejections

- **Reject market-as-player:** market data is not another copy of the solver.
- **Reject zero-sum equivalence:** trading reward is not a symmetric terminal game
  outcome.
- **Reject “future actions determine the present” literally:** only a policy-
  induced distribution over possible futures enters value.
- **Reject belief/world-model conflation:** posterior latent state, future
  prediction, and known portfolio state are different objects.
- **Reject future leakage:** realized future bars, labels, or future-derived
  `regime_tags` never enter the online observation.
- **Reject raw 32-action copy:** present Family `00` target actions do not reveal
  exogenous market state in historical price-taking replay.
- **Reject direct search today:** one realized future path is not a generative
  counterfactual market simulator.
- **Reject universal synthetic market replacement:** contingent market facts must
  come from real observations.
- **Reject one-shot multi-factor adoption:** damping, filtering, EMA, belief,
  proposer learning, S-series, recurrence, and R4-v2 stay separately attributable.
- **Reject logits as capability:** `CapabilityProfile` requires empirical held-out
  evidence.

## Open questions that must be answered before implementation

1. Which frozen solver recipe/checkpoint will serve as the learner for
   `AZR-LP-DIRECT-v0` after current R-series attribution closes?
2. What is the smallest independent capability panel that measures genuine
   transfer rather than improvement on the candidate task itself?
3. Should proposer utility be signed economic uplift, a constrained vector of
   net/Sharpe/turnover/drawdown, or a Pareto admission rule? Absolute gradient
   alignment must not decide this policy question implicitly.
4. Can task-gradient sketches predict direct micro-update uplift across solver
   checkpoints and seeds after normalizing reward scale?
5. Which typed mutation fields preserve task validity and chronological
   eligibility without turning start index or seed into an identity shortcut?
6. What QD descriptors promote meaningful curriculum coverage rather than merely
   filling bins?
7. Is the current critic demonstrably miscalibrated enough to justify a
   distributional value experiment?
8. What latent variable, if any, would make “market belief” a well-defined
   posterior rather than a renamed forecast?
9. Can any future market scenario model beat simple causal baselines under
   multi-horizon proper scores, tail coverage, and regime shift before being used
   for action search?
10. What total wall-clock, proposal, JVP, evaluation, and solver-step cost is
    acceptable per unit of curriculum uplift?

## Recommended order

1. Close `G0`: repair/capture critic, clip-fraction, advantage, KL, action-margin,
   and exact fresh-panel replay receipts.
2. Make recurrent lifecycle and evaluation state-correct; this is a mechanics
   prerequisite, not a new solver wing.
3. Finish the fixed R-series boundary—matched R0, clean R1, and R1 + R4-v1—then
   run the existing `R2c-RESET + R4-v1` versus `R2c-CARRY + R4-v1` memory
   attribution, without modifying R4-v2's fixed design.
4. Add `ATX-EMA-EVAL-v0` as an evaluation-only experiment.
5. Only with trustworthy critic/advantage telemetry, run the LR/regularization
   schedule arms and the all/random/top-|advantage| filter arms separately on the
   frozen accepted solver recipe, with PBT off.
6. Build `AZR-LP-DIRECT-v0` on a tiny validated `TaskSpec` set and test whether
   repeated local learning progress is measurable at all.
7. Validate gradient-alignment surrogates against that direct reference before
   training an adaptive proposer.
8. Add fixed quality-diversity replay before adaptive quality-diversity selection.
9. Keep C0/C1/shuffled-C1 intact; develop PF as a separate calibrated axis, cross
   an accepted PF model with a solver, and measure the hindsight-search ceiling
   before paying to build a world model.
10. Validate WM separately. Leave learned TTS and Market-JEPA deferred and
    independent until their own causal generative-model evidence exists;
    shadow-mode TTS is last.

The architectural opportunity is real: a fast numerical solver, a typed
non-LLM proposer, and a measured learning-progress curriculum could form a far
more complete AZR-inspired loop. The disciplined route is to use Ataraxos for
training-control and later search principles, use the zero-data paper for
curriculum credit assignment, and refuse the tempting but false equivalence
between a hidden-piece game and historical financial markets.
