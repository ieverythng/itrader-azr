# R1 readiness independent review — 2026-10-03

Base: `c13af18ae60520dfc4e0a1b0992c2ef3db49634e` plus the current dirty tree,
including relevant untracked sources. Branch: `agent/Watson-dev`.
Reviewer contract: [REVIEW.md](../../../REVIEW.md).
Scope: R4-v1/G0, experiment browser, replay/provenance, PPO/PBT and R1 readiness.
Unrelated imported and CLM/Jev research was excluded. No training or live
checkpoint restore was launched and no files were staged, committed or deleted.

## Different-model Spec review: GPT-6 Astra, MAX

Verdict: **CHANGES**. Three blocking findings were independently reproduced.
These are open implementation work; this review pass updates the development
contract rather than silently changing the training stack.

### 1. Selection-score equality is not replay parity

Owner: `itrader/rl_lib/evaluation/run_checkpoint_replay.py`, report construction
around line 305; `pbt_contract.economic_search_score`.

- Input: original net -0.001 with flat fraction 0.99; replay net -0.025 with
  flat fraction 0.98.
- Expected: differing economics/actions cannot pass an exact replay gate.
- Actual: both search scores are zero and `checkpoint_parity=True`.
- Base comparison: the earlier raw-net comparison rejects that change.
- Required correction: retain selection-score agreement as one field; separately
  verify source identities, underlying metrics and action/reward payloads.
  Missing historical reference traces must remain unavailable.

### 2. R4 attribution does not gate mismatched provenance

Owner: `itrader/scripts/experiments/analyze_r4_v1_ablation.py`, provenance
collection around lines 324–334 and promotion around 386–390.

- Input: complete synthetic 4-arm/3-seed, 12-update, 32-episode positive matrix;
  change one combined-state cell's dataset and source hashes.
- Expected: the comparison becomes ineligible because protected identities differ.
- Actual: both provenance uniqueness counts become two, but `promotion_passes`
  remains true.
- Base comparison: this analyzer is new relative to the base revision.
- Required correction: validate an explicit matched comparison contract before
  relative/absolute promotion. Allow declared factors, reject missing/mismatched
  protected identities; do not merely display counts.

### 3. PBT recovery replaces the scored checkpoint's authority

Owner: `itrader/rl_lib/train/pbt_scheduler.py`, `_recover_stable_checkpoint`,
around lines 457–468.

- Input: index a checkpoint, then modify `policy_state.pkl`; request its exact
  progress with provenance enforcement enabled.
- Expected: reject the changed tree against the indexed/scored digest.
- Actual: recovery succeeds and records the changed tree's newly computed hash.
- Required correction: compare actual and indexed digests before selection;
  bind downstream restore receipts to the verified original digest. Test missing
  indexed hashes, altered files and exact matching trees.

The reviewer retained disposable reproductions under Windows temporary paths
`itrader-review-r4gate-efpimj2f` and `itrader-review-donor-a4e08kvm`; these are
diagnostic aids, not durable experiment authority. Implement permanent regression
tests at the owning public seams in the next development pass.

## Standards review: GPT-6.1 Sol, MAX

Final verdict: **CHANGES**. No model fallback was needed. The reviewer found no
additional blocking defect in `REVIEW.md` or the independent-review ladder.

Two further blocking cases were independently reproduced by the default reviewer:

### 4. Truthiness can turn a failed replay into success

Owner: `analyze_r4_v1_ablation.evaluate_absolute_economic_gate`, around line 111.

- Input: otherwise-positive cells with `checkpoint_parity` set to the JSON
  string `"false"` rather than a boolean.
- Expected: reject the malformed evidence, with no promotion.
- Actual: `bool("false")` is true, so all-terminal-replays-exact and the gate pass.
- Required correction: require strict boolean `True` from a validated receipt;
  test strings, numbers, null and missing values. This analyzer is new at base.

### 5. Baseline control identities are discarded

Owner: `analyze_r4_v1_ablation._load_controls`, around line 188.

- Input: a copied baseline result with dataset/panel/source hashes removed or
  replaced by all-zero 64-character strings.
- Expected: reject controls that cannot be matched to the candidate's data,
  panel and protected comparison contract.
- Actual: controls are loaded, their identities discarded, and their economics
  remain eligible inputs to the promotion baseline.
- Required correction: validate and preserve candidate/control identities before
  evaluating superiority. This analyzer is new at base.

## Executed verification and limits

The Sol reviewer executed 99 focused tests across browser, R4 analyzer, PBT
scheduler/inspector, block uncertainty, data-partition sessions and the repo
Python launcher; all passed. Its healthcheck and base diff check passed. The
two additional in-memory gate/control probes exposed findings 4 and 5. All five
design principles were examined; Sol reported structural separation, intention,
encapsulation, cohesion and coupling as OK, with the validation defects above.
Its initial PATH interpreter failed on PyArrow/pandas/NumPy binary compatibility;
the repository launcher resolved the validated Ray-2.58 environment. Additional
base/replay fixture attempts failed before execution and establish no result.

The Spec reviewer executed 85 focused tests across browser, PBT inspector and
scheduler, checkpoint evidence and R4 analyzer, plus 51 selected policy-baseline,
uncertainty and train-config tests. All passed; the three adversarial probes
exposed the gaps above. The main agent ran the full suite: 344 passed, with the
required-dependency healthcheck passing on Python 3.11.13 / Ray 2.58.0. Optional
LLM packages remain absent. GitNexus is stale; source was used directly.

The reviewer checked all five G0 paired artifact hashes and the fresh panel hash:
they agree. Historical producer identities are absent. The findings demonstrate
acceptance vulnerabilities, not evidence that these completed files were altered
or that the negative G0 result should be reversed. No browser UI interaction was
performed in this review pass.

Design principles from the Spec review: separation of concerns OK; programming
by intention violated by the replay parity label; encapsulation violated by
bypassing the indexed checkpoint authority; high cohesion OK; low coupling OK
within the inspected seams.

## Development handoff

The [readiness contract](../../project-state/r1-development-readiness.md)
specifies the three repairs, producer identity, loss-time clip fraction and R1
model/lifecycle acceptance. The proposed matched study has four representation
cells (R0, R0 + R4-v1, R1, R1 + R4-v1), under each named cost curriculum. It
is preparation for a run declaration, not a new training result or promotion.
