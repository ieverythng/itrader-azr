# CONTINUATION22 DEV receipt — accounting test ROI and CUDA profile

Date: 2026-10-10. Status: implementation preparation; no runtime or economic
qualification. Source of authority is the GRILL `CONTINUATION22` scope and its
49-entry R1 lifecycle identity recheck. GRILL reports zero mismatches and
confirms the accounting target files are excluded. Branch/HEAD are recorded by
GRILL as `agent/Watson-dev` / `8f9fcf9`; DEV did not run Git, so this receipt
does not certify the current branch, index, or full dirty-tree snapshot.

## Accounting lifecycle oracle

The existing environment/diagnostics integration test was strengthened in
place rather than adding another test function. Its fixed synthetic close path
is `[100, 100, 110, 121, 110]` with window 2, start 2, horizon 3, actions
`[2, 2, 0]`, `net_return`, reward and evidence cost 10 bp, and reward scale
1,000. It exercises public reset/evaluation/step/evidence/attribution and then
checks reset state. The environment remains the accounting owner; no duplicate
`AccountingLifecycle` production class was added.

Independent expected values:

| Quantity | Expected |
|---|---:|
| Positions | `+1, +1, -1` |
| Per-step turnover | `1, 0, 2` |
| Terminal liquidation turnover | `1` |
| Gross return | `0.2909090909…` |
| Total turnover including liquidation | `4` |
| Cost return at 10 bp | `0.004` |
| Net return | `0.2869090909…` |
| Summed optimization reward | `286.9090909…` |

The trace assertions also pin row/as-of boundaries, terminal marker, hold cost
of zero, reversal cost, actual-vs-continuation terminal accounting, session and
state schema, and reset position/time observation. This is a characterization
test of existing code, not a red-green claim.

## Test return / consolidation assessment

The file has 11 test functions and 28 expanded cases by static inspection of
the parameter lists, before and after this edit; pytest collection was not run.
The existing integration slot now covers a richer exact economic lifecycle,
so the change adds no test function or parameterized case. Independent static
Standards and Spec reviews both returned APPROVE with zero findings. Both
explicitly note that runtime/base-comparison qualification remains pending.

Keep the nine-case prior/target transition matrix, six volatility-unavailable
inputs, three directional controls, three declared/effective partition
disagreements, causal future-row perturbation, and malformed/reordered trace
checks. These are compact oracles for different mathematical and trust
boundaries; replacing them with one large integration test would couple
arithmetic correctness to environment/data plumbing and weaken diagnostics.
The focused step terminal-cost test is the only plausible later overlap review,
but it also checks hypothetical round-trip cost distinct from realized
continuation; do not delete it unless a future green integration test asserts
that distinction and an independent review confirms no lost contract. No test
was deleted or merged in this scope. The former two-step pilot's public-path
reconciliation was replaced in place by a richer lifecycle integration; no
unit-level boundary was absorbed. A reviewer noted shared synthetic-Parquet
fixture-writing as a possible helper extraction, but keeping each scenario's
price path/window/task/cost explicit is more valuable than saving those few
setup lines. Thus this is integration-first and contract-preserving, not a
numerical test-count target.

Author's static design checklist (not an independent REVIEW.md approval):
separation of concerns—OK, `ITraderEnv` still owns realized accounting;
programming by intention—OK, the test names the hold/reversal/liquidation
contract; encapsulation—OK, assertions use the public reset/evaluation/evidence
surface rather than private state; high cohesion—OK, the scenario has one
independent episode oracle; low coupling—OK, no production owner or runtime
dependency changed. Independent static Standards and Spec reviews also report
all five principles OK. Runtime execution and base-snapshot comparison remain
pending, so this does not close the full REVIEW.md execution gate.

## Frozen-trace cost-repricing interface proposal

Proposed public function in the existing evaluation owner:

```python
reprice_episode_cost_scenarios(*, steps, evidence, scenario_cost_bps) -> dict
```

It first reconciles the original complete trace against its recorded cost and
terminal evidence once using `attribute_episode`; then it computes each
scenario as `gross_return - turnover_units_including_liquidation *
scenario_cost_bps * 1e-4`. For the fixed lifecycle trace, scenarios `[0, 10,
100]` bp would yield net returns `[0.29090909…, 0.28690909…, 0.25090909…]`;
the algebraic break-even is approximately `727.2727` bp. These values are
derived expectations only; the API/test remains unimplemented. Results retain
requested scenario order, the original declared cost,
gross return, turnover, net return per cost, and break-even bp only when
turnover is nonzero. Zero turnover reports break-even as unavailable. Negative
gross cannot become profitable at any nonnegative cost. Reject missing/mixed
original-cost evidence, malformed/incomplete traces, booleans, negative or
nonfinite scenario costs, and an empty scenario list. Do not replay a policy,
mutate reward, retrain, or imply predictive edge. This contract and its red
test remain proposed; implementation waits for a separate DEV execution lease.

## CUDA profile drafts

Four unexecuted drafts live under ignored
`docs/coordination/_local/continuation22/`: the profile/lock review note, a
PowerShell argv-forwarding launcher, its small Python identity probe, and a
full review-only candidate pyproject. They target the exact Windows x64 CPython
3.11 nightly wheel `2.9.0.dev20250801+cu128`. The packaging choice is accepted:
uv is supported; the default is dev + pinned CPU; CUDA is explicit and mutually
exclusive; plain `pip install .` will not install PyTorch and non-uv users need
an explicit PyTorch setup. Windows/Linux CPU wheels are sourced from the
official CPU index; macOS uses the platform PyPI build. The exact installed CPU
artifact identity and binary parity are not established. The CUDA metadata
SHA-256 is not locally verified; no lock hash, download, install, environment
sync, or root runtime change is claimed. Candidate uv resolution and the
cross-platform lock remain pending.

The official PyTorch previous-versions page documents its 2.3.1 CPU wheel
index; the official index lists the CPython 3.11 Windows x64 CPU wheel. The
candidate reflects this source plan without asserting a payload digest. uv's
documented default-groups, conflicts, and explicit-source mapping are reflected
in the candidate, but its exact cross-group source routing is not qualified.
The official uv guide documents source selection by optional dependency extra;
the draft uses dependency groups. Upstream issue #15730 reports a similar
group-scoped PyTorch source affecting resolution when the group is not selected
on uv 0.8.15, and the issue author reported the behavior still on 0.9.26. This
is a warning signal, not proof of a defect in this candidate or uv 0.11.7. Do
not infer that the source markers fix the CPU-index/CUDA-URL collision until an
isolated exact-version lock-only smoke passes. The uv 0.11.x changelog review
did not establish a fix for this exact case; lack of a changelog entry is not
proof it persists. Sources: [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/), [official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/), [uv dependency and source documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/), [uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/), [uv issue #15730](https://github.com/astral-sh/uv/issues/15730), [uv 0.11.x changelog](https://github.com/astral-sh/uv/blob/main/changelogs/0.11.x.md).

The later authorized bridge should first run uv 0.11.7 against a throwaway
minimal project with tiny local `torch` wheels at distinct file URLs. Resolve
CPU-default and CUDA-only selections separately and inspect lock forks, also
checking default resolution with CUDA absent. Expected: CPU resolves only to
the CPU fixture and CUDA only to the CUDA fixture. Any leakage or ambiguity is
a blocking result; do not redesign the dependency groups silently. This test
avoids acquiring the 3.06 GB CUDA wheel. Only after this routing check passes
and the accepted payload is acquired and hashed should a full candidate lock
be resolved in an isolated copy. Do not sync/create an environment in either
lock-only check.

An independent static Spec follow-up confirmed the candidate is syntactically
group-scoped but rated its source-routing status unresolved. It identified the
concrete failure risk as CPU-index participation or ambiguous resolution in the
CUDA branch; the exact CUDA pin makes silent substitution with the CPU pin less
likely, but does not certify source selection or successful resolution. Its
later smoke oracle is to inspect the exact uv 0.11.7 frozen lock and isolated
group exports: CPU must select only `2.3.1+cpu` from `pytorch-cpu`; CUDA must
select only the exact nightly URL/version. Any ambiguity, CPU lookup in the
CUDA branch, wrong Torch record, or inability to resolve both branches blocks
the candidate. This is static review only, not a full REVIEW.md verdict.

Local draft SHA-256 values:

| File | SHA-256 |
|---|---|
| `cuda-profile-review.md` | `1ED52997ADF38A7E4AD35D4524C6BB88278333F629B8DD876856ABDEB095A020` |
| `pyproject.uv-profile.fragment.toml` (superseded, retained) | `4AEC720B62BB4287730B622642B4D32BA18BFED3F8413C74FDEDE4B2A5A36EF4` |
| `pyproject.uv-profile.candidate.toml` | `F576DBC93B2F3901C76E9678FB290A053674A024DC25F2928763B37A6C4B7314` |
| `invoke-cuda-tested.ps1` | `CDDC3C4DBCF8ACE2750CF63F891AD725E1114C982D871A9D9D6C45285034A112` |
| `python_identity_probe.py` | `81999420C2B7B844D2BEA86D1EE49ECAE2394378BBED0EAAADE754C585C13C3B` |

## Source identities and execution status

| File | SHA-256 / status |
|---|---|
| `itrader/rl_lib/evaluation/economic_diagnostics.py` | `9FDCBAA1E5233EDD6926736D2054645CAB2946226DA27DE4D4E46423E27F33B8` unchanged |
| `itrader/tests/test_economic_diagnostics.py` | `554777842025B15C2CA3F1878EE52E5340D848ADEDD1F46489D50E0E224E7F92` edited |
| `itrader/envs/itrader_env.py` | `CB262C630CBDEFBF19B9A797BCC90E392234AE6CB771004B18DD4902C355A268` read-only identity from prior audit |
| `itrader/rl_lib/evaluation/episode_evidence.py` | `1A73F2EA89F1B15EE5FBD004CA51D35BFA47DD582639028FF498211A96F19E23` read-only identity from prior audit |
| `itrader/rl_lib/tasks/rewards.py` | `DC1B2EB69364A29D4FE781FF6DA994D954CFC511F1890E403F20A67364DAF20F` read-only identity from prior audit |

No tests, imports, healthcheck, pre-commit, Python execution, CUDA command,
download, `uv`, Git status/diff, WSL, or Ray job ran in this scope. The GitNexus
status helper was not run because it invokes Git, which this scope prohibits.
The two independent static reviews both approved with no findings, but did not
execute changed code or compare a base snapshot; the test/review execution gate
remains open. Next: close the TRAINING lease, rebind source identities, obtain
a fresh DEV execution/review scope, run the focused accounting test, and only
then decide whether any overlapping assertion can safely be retired. Keep the
full independent review gate before repricing implementation or R1 experiment
authorization.
