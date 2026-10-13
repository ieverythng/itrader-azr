# Adversarial examiner: bounded offline research wing

Date: 2026-10-10. Scope: RESEARCH21 / SALUKI-ADVISORY-ASSESSMENT21.
Status: design-only ablation candidate, not a canonical branch ID, implemented
validator, reward source or online authority. GRILL owns synthesis HTML.

## Receipt and outcome

Branch `agent/Watson-dev`; HEAD `8f9fcf94e028b51d883a1a484c2c9466a3d6b547`;
inherited dirty work preserved. Completed read-only examiner child supplied
primary-source findings; parent owns this note. The associated advisory inputs,
hashes and methodology assessment are recorded in
[Saluki methodology research](saluki-methodology-research-2026-10-10.md).
No models were called and no pilot was run.

Recommendation: investigate a bounded examiner for **offline fault detection**,
not automatic certification of task quality or trading edge. An examiner asks
the proposer focused follow-ups, then assesses the original proposal against a
frozen rubric. A single proposer plus examiner is cross-examination/consultancy,
not the two-sided debate protocol often cited in support.

Below, **fact** means a cited source finding; **inference** means a proposed local
design implication; **open** means a choice still requiring authorization.

## 1. What the literature establishes, and what it does not

| Primary evidence | Source-backed fact | Local transfer boundary |
|---|---|---|
| [Cohen et al., LM vs LM](https://arxiv.org/abs/2305.13281) | Multi-turn cross-examination can expose factual inconsistencies; the study evaluates factual QA detection. | Factual QA results are not TaskSpec correctness or trading-edge evidence. Consistent false explanations can evade contradiction checks. |
| [Kenton et al., scalable oversight](https://arxiv.org/abs/2407.04622) | Debate outperformed assigned-answer consultancy in studied tasks; results against direct answering were mixed without information asymmetry. | Inference-time evidence does not establish a trained oversight equilibrium or universal dialogue benefit. |
| [Zheng et al., MT-Bench/Chatbot Arena](https://arxiv.org/abs/2306.05685) | LLM judging exhibits position, verbosity, self-enhancement and reasoning limitations. | Preference agreement is not executable-contract correctness. |
| [Panickssery et al., self-preference](https://arxiv.org/abs/2404.13076) | Evaluators can recognize and favor their own generations. | Removing author labels alone cannot certify independence. |
| [JudgeDeceiver](https://arxiv.org/abs/2403.17710) | Attacker-controlled evaluated text can manipulate LLM-as-judge decisions through prompt injection. | Prompt delimiters alone are not a security guarantee; local attack rates remain unknown. |
| [Raina et al., adversarial LLM evaluators](https://aclanthology.org/2024.emnlp-main.427/) | Short adversarial phrases can inflate absolute scores; comparative judgments were less susceptible in their setting. | Pairwise scoring is not thereby immune or qualified for our tasks. |

**Inference:** judge selection should depend on measured competence against the
local frozen rubric and oracles, robustness and cost, not a mandatory parameter
count advantage. Bigger may be useful, but is a hypothesis rather than a
qualification rule. Same-family or cross-provider models can share assumptions
and errors; agreement must not be counted as independent replication. Diversity
is an ablation, not a certificate. More conversational turns can entrench a wrong
premise, so dialogue-induced damage must be measured alongside detection gains.

**Open:** no literature result establishes which available examiner is competent
on iTrader contracts, how often it abstains, or whether its gain justifies cost.
No performance percentages are transferred from these papers.

## 2. Proposed comparison: design only

Freeze one candidate set, admissible evidence packets, rubric, and independent
oracle labels before model/prompt comparisons. Keep complete task/conversation
families in one development or confirmation split.

| Arm | Procedure | Authority |
|---|---|---|
| Deterministic-only | Existing schema, eligibility and accounting checks; unsupported semantic claims stay unresolved. | Executable checks retain their existing bounded authority. |
| Static judge | Same checks plus one fixed-rubric assessment of the original candidate/evidence packet. | Advisory fault assessment only. |
| Adaptive examiner | Same checks plus bounded follow-ups to proposer and final fixed-rubric assessment of the original candidate. | Advisory fault assessment only. |

**Inference:** every arm scores the same frozen original proposal. Additional
proposer answers are untrusted assertions, not new verified evidence. Dynamic
examination acquires additional information, so any improvement is initially a
procedure benefit, not proof of superior judging in isolation. If candidate
repairs are later allowed, score revisions separately; otherwise dialogue may
conceal an initially invalid proposal. A later budget-matched fixed-questionnaire
arm could isolate adaptivity from receiving more responses.

**Open:** maximum turns, input/output tokens, elapsed-time caps, model choices,
case counts and pass thresholds must be selected before a separately authorized
pilot. No default numbers or pass gates are adopted here. Stop on sufficient
rubric evidence, sourced contradiction, budget exhaustion, malformed output or
abstention; do not retry selectively until an approval appears.

## 3. Independent executable oracles and rubric

Proposed small named scenarios, not new tests or an unlimited trivia benchmark:

- **ContractContradiction:** a plausible proposal conflicts with an explicit
  TaskSpec constraint or qualified eligibility metadata. Existing validator and
  independently checked literal fixtures provide the oracle.
- **AccountingArgument:** a claim about gross/net/turnover conflicts with a
  hand-computable trace. Literal arithmetic is independent of model judgment.
- **EvidenceSufficiency:** a confident claim exceeds a frozen report's support.
  The rubric requires descriptive/conditional wording and rejects invented
  samples, confidence or qualification. Semantic labels require blinded
  adjudication where executable truth is unavailable; preserve disagreements.
- **AdversarialDialogue:** initial text or follow-ups inject instructions,
  fabricate references, switch definitions or contradict prior claims. Score
  trust-boundary adherence and consistency, not persuasive style.

**Inference:** independently verify the oracle itself. Do not call the static
judge's answer ground truth. Terminal rubric states should distinguish supported,
contradicted, insufficient evidence and malformed/injected output. These are
pilot design candidates, not accepted production schema values.

## 4. Trust boundaries, leakage and correlated errors

The trusted rubric, evidence permissions and oracle execution stay outside the
untrusted proposer/examiner dialogue. The examiner cannot execute arbitrary
proposer code, change TaskSpec, obtain hidden labels, approve data access,
authorize training, order trades or override deterministic risk/execution.
Instructions inside candidate text, replies and retrieved documents remain data.

Reset each conversation; prohibit held-out-answer memory/retrieval. Freeze
development/confirmation membership before prompt tuning. Hidden labels and test
secrets must not appear in proposer prompts, examiner context or retrieved
packets. Log full input identities and transcripts so dialogue contamination is
auditable. Keep actual confirmation cases inaccessible during development.

Blind model identity where possible; use order swaps when pairwise judging is
introduced, and concise-correct versus verbose-wrong fixtures. Test injection in
both original proposals and later replies. Measure shared failures explicitly;
do not treat several judges or turns as additional independent cases.

The human clarification remains preserved: `CapabilityProfile` is online/shadow
facing, and `terminal_screen_only` evidence must not feed operational capability
calibration, proposer reward, curriculum, PPO or tuning. This wing must use
separately authorized synthetic/development evidence; it cannot repurpose sealed
terminal evidence as an interrogation/training corpus.

## 5. Evaluation and cost accounting, not a new adopted metric

**Inference:** a later pilot should prespecify false acceptance of known-invalid
cases, false rejection of known-valid cases, abstentions, incremental detection
beyond deterministic checks, paired case outcomes, and latency/cost. Preserve
timeouts, malformed outputs and negative results in denominators. These are
design suggestions; exact estimands and statistical methods remain open.

Account for all examiner calls, proposer replies, tools and final judgments,
including failed conversations. Report actual consumption alongside caps.
Compare quality-cost frontiers rather than attributing a gain to dialogue when
one arm simply receives substantially more compute. Conversation turns are not
independent replications; the frozen candidate/family is the relevant grouping.

## 6. Decision-ready disposition

| Disposition | Recommendation |
|---|---|
| Adopt now as research-design controls only | Independent oracles, frozen rubric, trust separation, explicit abstention, held-out cases and complete transcript/cost accounting. |
| Ablate after authorization | Deterministic-only versus static judge versus bounded examiner; competent same-family versus cross-family examiner; later fixed-question control. |
| Defer | Learned examiner, examiner-generated proposer reward, large judge committees, production integration and training against examiner preferences. |
| Reject | Mandatory larger examiner; consensus as truth; contradiction-free dialogue as proof; direct certification of market edge; examiner authority over online WAIT, solver policy, risk/execution or evidence qualification. |

Smallest next decision: approve this only as a future bounded offline
fault-detection pilot with frozen original proposals and independent oracles?
Recommendation yes, conditional on separately selecting budgets, evidence access
and scoring criteria. Existing proposer rewards and online authority stay intact.

Output hashes and validation receipt are on the
[owned RESEARCH board](../../coordination/research.md). No market outcomes,
acquisition, model calls, simulation, installs/driver/pins, WSL, Git shipping,
code changes, heavy lease or new metric/method/semantics adoption occurred.

## 7. CONTINUATION22: the proposer as its own examiner

2026-10-10 addendum; design-only extension of the existing comparison, not a
second framework. The human requested self-examination research. No model calls,
tools executed by models, training, pilot or UAH integration occurred.

### 7.1 Empirical support is conditional, not a guarantee

| Primary empirical source | Finding | What remains unproven here |
|---|---|---|
| [Self-Refine, Madaan et al.](https://arxiv.org/abs/2303.17651) | The same model generated, critiqued and refined outputs, improving results across its studied tasks without additional training. | Feasibility of refinement does not establish independent validation or TaskSpec competence. |
| [Huang et al., ICLR 2024](https://arxiv.org/abs/2310.01798) | Intrinsic reasoning correction without external feedback struggled and sometimes damaged correct answers. | This is not a universal impossibility result for every model, prompt and task. |
| [Liu et al., revised preprint](https://arxiv.org/abs/2406.15673) | The authors report intrinsic correction with neutral prompts and zero-temperature settings, disputing blanket negative conclusions. | Prompt/decode sensitivity is a local ablation hypothesis, not an adopted setting or repeatability guarantee. |
| [Tyen et al., ACL Findings 2024](https://arxiv.org/abs/2311.08516) | Separating mistake localization from repair, supplied ground-truth error locations improved correction on studied reasoning tasks. | Assisted repair is not unaided error detection. |
| [CRITIC, Gou et al.](https://arxiv.org/abs/2305.11738) | Tool-interactive feedback improved correction in QA, program synthesis and toxicity tasks. | External feedback is a separate evidence condition; tools are not automatically correct or permitted here. |
| [SCoRe, Kumar et al.](https://arxiv.org/abs/2409.12917) | Multi-turn RL training improved self-correction in studied math/code models; offline SFT correction traces showed distribution/collapse problems. | Training a correction behavior is different from prompting a frozen proposer. No trained examiner or reward coupling follows. |

**Inference:** these results motivate a bounded local comparison, not a verdict
from paper titles. Keep both failure and improvement possible. Existing
self-preference and injection evidence in section 1 applies equally to a
same-model examiner; its agreement with its own proposal is not ground truth.

### 7.2 Four kinds of freshness or independence

| Change | What it can change | What it does not establish |
|---|---|---|
| Fresh conversation/context | Removes carried dialogue and accessible per-session state if genuinely reset. | Erasure of learned priors, pretraining exposure, persistent retrieval or shared mistakes. |
| New process loading the same checkpoint | Resets runtime state and can isolate a session. | Different weights, different training lineage or an independent judge. |
| Different checkpoint/weights | Offers a potential diversity control. | Independent errors: shared training data, evidence and rubric misunderstandings remain possible. |
| Independent executable oracle or blinded adjudication | Supplies a separately checked reference for a specified claim. | Truth outside its competence, correctness of an unchecked verifier or trading edge. |

These are design distinctions, not empirical proof that a process restart improves
judging. A new inference instance normally reloads learned weights; literal random
weight initialization creates an untrained model, not a fresh competent judge.
Model diversity, runtime isolation and evidence independence are separate axes.
Likewise, fresh solver weights remove checkpoint carryover, not historical
research exposure to a proposed chronological test interval.

### 7.3 Minimal extension at the existing examiner seam

Keep deterministic-only unchanged. Within the existing static/adaptive arms,
label a **same-checkpoint, fresh-context examiner** variant and compare it with
the already-planned competent different-checkpoint examiner, on the same frozen
cases. Do not create an uncontrolled full factorial or assume the latter is an
independent oracle. Both receive only the original candidate, allowed evidence
packet and rubric, not private proposer history, author labels or previous
approvals. An optional verified-feedback condition is a later assisted condition,
not quietly pooled with intrinsic self-examination.

The proposed module interface remains a frozen candidate/evidence/rubric in,
bounded advisory assessment/transcript out; model variants are adapters at that
seam. Oracle checks stay independently owned, outside dialogue. This is interface
design only, not a new implementation or permission to execute proposer code.
Future UAH porting remains deferred.

**Detection before repair:** freeze the assessment of the original proposal
before any revision or oracle-label disclosure. If repair is later authorized,
retain a separately identified revision and score invalid-to-valid repair,
invalid-to-invalid failure, valid-to-invalid damage and valid-to-valid preservation.
If true error locations or verified diagnostics are supplied, label repair
assisted; do not credit those labels to autonomous detection. Hidden confirmation
labels remain external to both stages. These are reporting proposals, not adopted
production metrics or statistical thresholds.

Count initial generation, static/adaptive judgments, proposer replies, any repairs
and rechecks, failures and model-loading overhead in the whole-call/token/time
budget. Shared weights may simplify loading but do not imply lower total cost.
Repeated turns or stochastic seeds are not independent candidate replications.

### 7.4 Controls and next decision

Use neutral wording rather than assuming every proposal is wrong. Fresh context
must exclude earlier gold labels, repair histories, held-out conversations and
answer-bearing memory/retrieval. Mask identity where possible, but section 1's
self-recognition finding means blinding alone cannot certify independence.
Track paired shared failures, abstentions and damage as well as successful
corrections. Treat every proposer answer as untrusted, even with shared weights;
injected text cannot modify the rubric, access permissions or hidden labels.
Terminal-screen-only evidence remains excluded from examiner corpora, reward,
curriculum, PPO, tuning and operational capability calibration.

| Disposition | CONTINUATION22 extension |
|---|---|
| Adopt as documentation distinction only | Context reset, checkpoint diversity and independent evidence are different; detection and assisted repair are different. |
| Ablate after authorization | Same-checkpoint fresh-context variant at the existing seam, with frozen originals and externally scored outcomes. |
| Defer | Verified-feedback repair, trained self-examiner, proposer rewards, curriculum updates and UAH port. |
| Reject | Fresh process or self-approval as independent validation; guaranteed monotonic correction; hiding initial faults behind repairs. |

Smallest decision: include the same-checkpoint fresh-context variant as an
advisory-only future ablation, with original detection frozen before any separately
authorized repair? Recommended yes. Checkpoints, prompts, decoding, cases,
feedback permissions, budgets and scoring/statistical criteria remain open.

## 8. RESEARCH23: minimal frozen examiner packet at the existing seam

2026-10-10 compact follow-up. This is a research manifest proposal, not a new
evaluator, TaskSpec field, production schema or execution permission. It refines
section 7's candidate/evidence/rubric-in, advisory-receipt-out interface. Existing
oracle ownership and proposer validation remain unchanged; UAH port is deferred.

| Manifest group | Minimum information to freeze before a later pilot |
|---|---|
| Original case | Case/family identity and development/confirmation membership; immutable original TaskSpec/text references and hashes; initial proposer receipt. Originals never overwritten by repaired candidates. |
| Admissible evidence | Packet/source references and hashes, usage classification, permitted content and access; explicitly exclude hidden labels, prior confirmation dialogue and terminal-screen-only evidence. Candidate assertions are untrusted, not evidence grants. |
| Rubric and independent oracle | Rubric/version and executable-oracle implementation/fixture identities; independent verification or blinded-adjudication provenance and unresolved-label handling. Hidden labels remain outside proposer/examiner context. No size or self-agreement substitutes for this receipt. |
| Examiner adapter identity | Same-checkpoint fresh-context or demonstrably competent different-checkpoint variant; checkpoint/tokenizer, prompt and decoding identities; context/memory reset and evidence permissions. Different weights do not imply independent errors. Exact choices remain open. |
| Whole-procedure budget | Predeclared call/token/time/resource caps and stop/failure/no-selective-retry rules; shared versus incremental accounting and actual consumption. No numerical caps selected here. |
| Detection receipt | Original-candidate hash, bounded assessment with cited admissible evidence, abstention/malformed/timeout disposition, transcript/usage identities and independent scoring reference. Freeze before repair or oracle-answer disclosure. |
| Optional repair receipt | Separate authorization, revision identity/hash and parent-original link, permitted diagnostic feedback, recheck/scoring and incremental usage. Label assisted repair separately; never replace original detection results. |

**Proposed comparison:** retain deterministic-only, static judge and bounded
examiner arms. Same-checkpoint fresh-context is a variant within the existing
model-assisted arms, compared with a competent different-checkpoint variant on
the same frozen cases and admissible evidence. Competence needs independently
scored rubric evidence, not larger parameter count. Case-family grouping and
external oracle controls remain unchanged; no full factorial or new framework
is proposed.

Count shared original-generation cost once per candidate, distinguish each
arm's incremental calls from reused common inputs, and disclose end-to-end
procedure cost. Include proposer replies, judgments, any separately authorized
repair/rechecks, failures and loading overhead. Caps are not measured usage;
more dialogue turns are not independent observations. Keep initial fault
detection frozen so later repair cannot hide false acceptance, damage or an
original invalid task. Trusted manifest/rubric/access rules cannot be rewritten
by injected candidate text or replies.

**Smallest later recommendation for GRILL:** approve this frozen-packet and
detection-before-repair discipline at the existing seam before selecting models,
prompts, budgets, oracle-feedback permissions or criteria. Those choices remain
open. No direct human question, actual manifest execution, model/tool call,
pilot, reward/curriculum coupling or operational authority is authorized.
