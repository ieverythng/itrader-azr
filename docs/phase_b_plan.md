# Phase B Plan (Concrete Execution)

Date: February 24, 2026
Branch baseline: `Phase-B-Training`

## Goals

1. Ship an executable proposer-solver loop (AZR-style) on top of current Phase A stack.
2. Preserve Phase A stability and reproducibility guarantees.
3. Establish dependency lock and runbook flow for main workstation execution.

## Scope

In-scope:

- Proposer integration into runtime task generation.
- Task contract hardening and validation.
- AZR executable training entrypoint.
- Metrics and evaluation for proposer impact.
- Environment/dependency stabilization for repeatable runs.

Out-of-scope (Phase B initial slice):

- Multi-node production orchestration.
- Full online/live trading deployment.
- Extensive model-family expansion beyond baseline proposer backend + fallback.

## Milestones

### M0: Stabilization and Cleanup (Prerequisite)
Deliverables:

- Remove hardcoded secrets.
- Fix stale/broken entrypoints.
- Align dependency manifests around a canonical stack.
- Remove tracked cache artifacts and reduce repo noise.
- Add architecture + runbook + Phase B plan docs.

Acceptance criteria:

- Repo has no hardcoded API keys.
- CLI entrypoint resolves to valid modules only.
- Dependency strategy documented with lock-generation steps.

### M1: Proposer Contract + Validation Layer
Deliverables:

- Define `BaseProposer` interface.
- Add task validation/clamping utility shared by proposer and training.
- Add deterministic fallback proposer (rule-based/random) for no-LLM environments.

Acceptance criteria:

- Training can request tasks through one interface regardless of backend.
- Invalid proposer outputs cannot crash env reset or training loop.

### M2: Executable AZR Training Loop
Deliverables:

- Replace pseudocode `train_AZR.py` with runnable loop integrated with current env/task/reward stack.
- Add CLI flags for proposer backend, sampling mix, and update cadence.
- Persist run manifests with proposer settings and backend identity.

Acceptance criteria:

- End-to-end run completes on CPU-safe smoke config.
- Checkpointing and metrics remain compatible with existing tooling.

### M3: Evaluation and Regression Guardrails
Deliverables:

- Add integration tests for proposer -> task -> env -> rollout path.
- Add regression checks for NaN/Inf, invalid task specs, and restore behavior.
- Add comparison script/report template for proposer-on vs proposer-off runs.

Acceptance criteria:

- CI-safe smoke tests pass without requiring GPU.
- Proposer-on runs produce stable metrics artifacts and summary output.

## Dependency Strategy

Canonical target stack for Phase B:

- Python 3.11
- Ray RLlib 2.47.1
- Gymnasium >= 0.29
- PyTorch 2.3.x
- DuckDB 0.10.x
- PyArrow 13.0.0 (baseline pin, revisit only if required)

Lock workflow (run on main PC):

1. Create fresh env from `itrader/environment.yml`.
2. Validate with:
   - `python itrader/tools/healthcheck.py`
   - `python -m pytest -q itrader/tests`
3. Export lock:
   - `conda env export --from-history > env.lock.yml`
4. If pip-only lock is needed, generate with `pip freeze > itrader/constraints.lock.txt` from the validated env.

## Work Breakdown (Execution Order)

1. Finish M0 (this branch pass).
2. Implement M1 interfaces and tests.
3. Implement M2 runnable loop behind a new entrypoint.
4. Implement M3 evaluations and guardrails.
5. Run extended training only on main PC after smoke validation.

## Risks and Mitigations

1. Risk: proposer outputs malformed tasks.
- Mitigation: strict validator + clamp + rejection logging.

2. Risk: dependency conflicts on workstation.
- Mitigation: single canonical manifest and lock generation after validated install.

3. Risk: increased training instability with proposer dynamics.
- Mitigation: start with low proposer influence, fixed fallback schedule, and baseline A/B comparisons.

## Definition Of Done (Phase B Initial)

Phase B initial is done when:

1. `train_AZR.py` (or equivalent new entrypoint) is executable.
2. Proposer backend can be switched via CLI without code edits.
3. Smoke run passes and emits stable metrics + manifests.
4. At least one proposer-vs-baseline comparison run is reproducible from docs.
