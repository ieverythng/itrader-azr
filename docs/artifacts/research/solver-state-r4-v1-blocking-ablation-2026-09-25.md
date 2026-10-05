# R4-v1 solver-state blocking ablation audit — 2026-09-25

## Status

**Complete; R4-v1 economic promotion failed.** All 12 cells, terminal replays,
state-conditioned diagnostics, controls, and predeclared anchor-block intervals
are frozen. Time-only passed the relative net-return factor gate, but no arm
passed the separate absolute economic gate. No R1, recurrence, PBT, sealed-test
access, 2-bp promotion, or intermediate-result tuning occurred during the round.

## Frozen question and design

Does the accepted fixed-width R4-v1 state connector improve the frozen R0
MLP-FiLM solver under exact 1 bp economics, and is any relative effect strong
and durable enough to pass the separate absolute economic gate?

The design is a sequential, seed-blocked 4 x 3 paired ablation:

| Factor | Frozen levels |
|---|---|
| Solver-state arm | `masked`, `position`, `time`, `both` |
| Training seed | `20260924`, `20260925`, `20260926` |
| Evaluation | seed `30260943`; identical crossed 8-anchor x 4-horizon development panel |
| Runtime | Ray 2.58; local worker only; `workers=0`; `pop=1`; PBT off |
| Exposure | 12 PPO updates, train batch 4,800, approximately 58k sampled steps per cell |
| Economics | exact fixed 1 bp optimization and evidence cost; `net_return` |
| Data | immutable dataset-v2, RTH only, 12-symbol frozen universe |
| Observation | window 60, `semantic_v2`, `market_relative_v1`, `NoFilter` |
| Solver | current frozen PPO/R0 MLP-FiLM recipe; only the state mask varies |

The checked-in loadout is
`itrader/scripts/experiments/configs/phase_a_r4_v1_blocking_4x3_20260925.json`.
It runs seed-major to limit time/resource drift within each paired block. The
runner checks free physical RAM before every cell and hard-stops below 2 GiB.
Checkpoint cadence is terminal update 12. The existing evidence stack also
retains a stable checkpoint at any strictly improved, positive held-out
selection score; those best-of-trajectory checkpoints are secondary only.

## Predeclared decision gates

These gates were frozen before inspecting any result from this round:

1. All 12 cells must complete with identical observation schema, dataset,
   evaluation-panel, and source-tree fingerprints, finite required metrics, and
   no provenance drift.
2. Primary attribution uses terminal update 12, paired by training seed against
   that seed's `masked` arm. Best-of-trajectory evidence is secondary.
3. The final three evaluation reports (updates 10–12) must be reported as a
   durability slice; a single terminal point cannot hide late instability.
4. Every primary terminal checkpoint must restore and replay exactly in a fresh
   process. If the current best-checkpoint pointer is nonterminal, terminal
   replay must use the update-12 checkpoint-index entry explicitly.
5. Each terminal checkpoint requires diagnostics conditioned by prior position
   and remaining-time quarter, including switching, turnover, liquidation,
   inference entropy, explained variance, action probabilities, and value
   outputs.
6. Uncertainty is paired by task/episode and clustered or bootstrapped by the
   eight market anchors. The 32 anchor-horizon tasks are not independent.
   Intervals are reported per seed and as a pooled/replicated interpretation.
7. The relative factor gate requires directionally consistent paired uplift and
   a predeclared 95% anchor-block interval excluding zero for any promoted
   metric. Net-return and Sharpe effects are computed and reported separately.
8. The absolute economic gate is separate: positive exact-1-bp net return and
   Sharpe, superiority to flat and the strongest directional/untrained control,
   reasonable turnover, and exact replay. Relative uplift alone cannot promote.
9. No sealed final-test access, 2-bp promotion claim, PBT, or tuning based on
   intermediate results is allowed.
10. Cells run sequentially. Free RAM is checked between cells and the official
    runner fails closed below 2 GiB.

## Uncertainty method frozen before results

For each seed and candidate arm, compute episode-level candidate-minus-masked
deltas on the identical 32 tasks. Define the cluster as the panel's eight unique
market anchors (`symbol`, `start_idx`); all four horizons belonging to an anchor
travel together. Produce 10,000 seeded nonparametric bootstrap replicates by
sampling eight anchors with replacement and averaging all carried task deltas.
Report the percentile 95% interval for net return and for per-episode Sharpe.

For the pooled interpretation, preserve replication: first sample the three
training seeds with replacement, then independently sample eight anchors with
replacement inside each selected seed, and average the resulting deltas. The
pooled interval is evidence across the three declared training replications, not
32 independent tasks and not an independent final test.

## Controls

Flat, long-only, short-only, seeded-random, and seeded-untrained controls use the
same dataset, evaluation seed, exact 1 bp evidence cost, and frozen panel. The
strongest control is chosen separately for net return and Sharpe from these
predeclared controls; it is not selected from additional post-hoc baselines.

## Protected scope and ownership

`TaskSpec` remains unchanged. `ITraderEnv` continues to own mutable position,
episode progress, and observation assembly. `RewardEngine` remains the sole
reward seam. `SolverMLPFiLM` consumes the already accepted two-value connector
without acquiring broker, risk, or execution semantics. The dirty working tree
containing the accepted dataset-v2 and R4-v1 implementation is the protected
baseline; unrelated modifications will not be staged, committed, deleted, or
rewritten.

## Results

### Completion and provenance

All 12 cells completed sequentially with one trial each at 12 PPO updates and
58,256–58,640 sampled environment steps. The matrix has one unique observation
schema, dataset hash, evaluation-panel hash, and source-tree hash:

```text
observation schema  r4_v1_min / base 332 / state 2 / agent 334
dataset SHA-256     20aac96de6da77e20beeaff214d380ef06184faf301ff0dd53bed0ffd4c04ce9
panel SHA-256       2314a55ceac6aceed575203d91daa0d1468fcd88158ff4d0839751c3b8b36b9f
source SHA-256      351b8d423377feffa22ecd8eb59c7934c92d3ca298f14721ff827bfa8f03c3f1
```

Required economic and diagnostic metrics were finite. The legacy-stack
`clip_fraction` remains unavailable (`NaN`) as already disclosed by the smoke;
it is not a required promotion metric and did not vary by cell. Every inter-cell
RAM check passed the 2 GiB floor. PBT remained off, population remained one, and
no cell was retried.

### Terminal update-12 economics

Values are arithmetic means over the identical 32-task development panel.
Percentages below are return fractions multiplied by 100.

| Seed | Arm | Net | Sharpe / sqrt(bar) | Gross | Turnover | Position changes | Liquidation | Entropy | Explained variance |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260924 | masked | -0.1193% | -0.02381 | -0.0150% | 10.438 | 9.781 | 0.656 | 0.9991 | 0.0130 |
| 20260924 | position | -0.1647% | -0.07004 | +0.0584% | 22.312 | 21.875 | 0.281 | 1.0079 | 0.0149 |
| 20260924 | time | -0.0023% | -0.02516 | +0.0627% | 6.500 | 5.656 | 0.688 | 0.9895 | 0.1539 |
| 20260924 | both | -0.0008% | -0.00855 | +0.0542% | 5.500 | 3.812 | 1.000 | 0.9973 | 0.1170 |
| 20260925 | masked | -0.1582% | -0.03204 | +0.0143% | 17.250 | 16.500 | 0.719 | 0.9656 | 0.0404 |
| 20260925 | position | -0.1045% | -0.02796 | +0.0124% | 11.688 | 7.812 | 0.531 | 0.9128 | 0.0213 |
| 20260925 | time | -0.0816% | -0.02256 | -0.0116% | 7.000 | 6.281 | 0.719 | 0.9223 | 0.0598 |
| 20260925 | both | -0.1289% | -0.02409 | -0.0121% | 11.688 | 10.688 | 0.562 | 0.9313 | 0.0930 |
| 20260926 | masked | -0.0728% | -0.03365 | -0.0359% | 3.688 | 3.625 | 0.062 | 0.7600 | 0.0361 |
| 20260926 | position | -0.0194% | -0.01544 | +0.0068% | 2.625 | 2.625 | 0.000 | 0.7910 | 0.0311 |
| 20260926 | time | +0.0133% | -0.00348 | +0.0439% | 3.062 | 3.062 | 0.000 | 0.7318 | 0.2510 |
| 20260926 | both | +0.0324% | +0.00644 | +0.0481% | 1.562 | 1.562 | 0.000 | 0.7706 | 0.2076 |

Three-seed terminal means were:

| Arm | Net | Sharpe / sqrt(bar) | Gross | Turnover | Position changes | Liquidation | Explained variance |
|---|---:|---:|---:|---:|---:|---:|---:|
| masked | -0.1168% | -0.02983 | -0.0122% | 10.458 | 9.969 | 0.479 | 0.0298 |
| position | -0.0962% | -0.03781 | +0.0259% | 12.208 | 10.771 | 0.271 | 0.0224 |
| time | -0.0235% | -0.01707 | +0.0317% | 5.521 | 5.000 | 0.469 | 0.1549 |
| both | -0.0324% | -0.00873 | +0.0301% | 6.250 | 5.354 | 0.521 | 0.1392 |

Time-only and both improved the average gross path, turnover, value fit, and net
loss relative to masked. Position-only did not: its 20260924 lineage repeated the
smoke's turnover pathology. The combined arm produced the only positive-net and
positive-Sharpe terminal cell, seed 20260926, but the other two combined cells
were negative.

### Final-three durability

The update-10–12 mean net returns were masked `-0.0777%`, position `-0.0894%`,
time `-0.0265%`, and both `-0.0473%` when averaged across seeds. The corresponding
Sharpe means were `-0.02109`, `-0.03432`, `-0.01733`, and `-0.01020`. Only the
20260926 combined lineage had positive final-three mean net and Sharpe
(`+0.0124%`, `+0.00342`); this was not replicated by the other seeds. Full
per-cell update-10–12 receipts are in `analysis.json`.

### Paired anchor-block uncertainty

The predeclared bootstrap resampled the three training seeds, then the eight
market anchors within each sampled seed, carrying all four horizons together.
These are development-panel intervals, not a sealed or selection-adjusted test.

| Arm vs masked | Mean net delta | 95% anchor interval | Mean Sharpe delta | 95% anchor interval | Relative gate |
|---|---:|---:|---:|---:|---|
| position | +0.0206% | [-0.0822%, +0.1154%] | -0.00798 | [-0.05077, +0.02223] | fail |
| time | +0.0932% | **[+0.0265%, +0.1752%]** | +0.01277 | [-0.00933, +0.03355] | **net pass only** |
| both | +0.0843% | [-0.0246%, +0.2019%] | +0.02110 | [-0.00026, +0.04366] | fail |

Time-only had positive terminal net deltas in all three seeds (`+0.1170%`,
`+0.0766%`, `+0.0860%`) and its pooled anchor interval excluded zero. Its Sharpe
delta was negative in seed 20260924, so Sharpe did not pass. Both had positive
net and Sharpe deltas in every seed, but neither pooled interval excluded zero;
the Sharpe lower bound missed by approximately `0.00026` and remains a failure,
not a near-pass promotion.

### Controls and absolute gate

| Control | Net | Sharpe / sqrt(bar) | Turnover |
|---|---:|---:|---:|
| flat | 0.0000% | 0.00000 | 0.000 |
| long-only | -0.0979% | +0.00827 | 2.000 |
| short-only | +0.0579% | -0.01558 | 2.000 |
| untrained seed 20260924 | +0.0579% | -0.01558 | 2.000 |
| untrained seed 20260925 | +0.0618% | -0.00976 | 2.000 |
| untrained seed 20260926 | -0.0716% | -0.01529 | 5.938 |
| random seeds (range) | -1.3505% to -1.2144% | -0.25857 to -0.24465 | 128.750 to 131.312 |

No arm had positive net and Sharpe across all three seeds. No arm's mean beat
flat, the strongest net control (untrained seed 20260925, `+0.0618%`), or the
strongest Sharpe control (long-only, `+0.00827`). Exact relative improvement is
therefore not economic promotion.

### Exact replay and state-conditioned diagnostics

All 12 update-12 checkpoints restored in fresh Ray processes and reproduced the
authoritative terminal net score. Replay deltas were exactly zero or at most
`2.17e-19`.

Each diagnostic contains 4,672 decisions, action probabilities, value outputs,
switching, and slices by prior position and remaining-time quarter. Across
seeds, mean step-level switching was `6.83%` masked, `7.38%` position, `3.42%`
time, and `3.67%` both. Time and both reduced turnover and switching while their
mean explained variance increased to `0.155` and `0.139` from masked `0.030`.
Their mean predicted flat probability rose toward termination: time from
`46.58%` in the early quarter to `48.59%` terminal, and both from `44.18%` to
`46.97%`. The full by-position slices show distinct probabilities and value
outputs when those states were visited, but several near-flat terminal policies
did not visit every prior-position category. These diagnostics establish state
use and value-path influence; they do not rescue the failed absolute economics.

### Evidence/tooling corrections

Three narrow evidence defects were closed with red tests before use:

1. the ablation runner now enforces a top-level free-RAM floor before every
   sequential cell;
2. replay and state diagnostics now select `terminal` explicitly rather than
   silently following a transient `best_checkpoint.json` pointer;
3. control stable evidence is namespaced by experiment so seeded random and
   untrained episode receipts cannot overwrite one another.

The new seed-then-anchor bootstrap also has a regression test proving that the
four crossed horizons move together inside each market-anchor cluster. Only the
six invalidated random/untrained controls were rerun after the control-path fix.

## Verdict

**Fail R4-v1 economic promotion.** Time-only passes the relative net factor gate,
which is useful evidence that remaining-horizon observability changes the learned
solution in a favorable direction. It is not the complete Markov state, its
three-seed mean net and Sharpe remain negative, and it fails the absolute gate.
The complete position-plus-time arm does not pass a pooled relative interval and
also fails the absolute gate. Neither becomes the default solver observation,
enters PBT, or justifies sealed-test or 2-bp promotion work.

This verdict closes the special R4-before-R1 execution block. It does **not**
authorize an automatic R1 launch: the existing user-review and recurrent
lifecycle/placement gates still apply. If R1 proceeds after that review, R4-v1
must remain a named, separate observability factor rather than being silently
folded into the recurrent architecture. The next R4 work, if any, should use a
genuinely new chronological development block and test the time signal as an
explicit factor; rerunning or tuning against these same eight anchors is not
allowed.

### Post-audit architecture decision

The economic verdict above is unchanged. After reviewing control-state
correctness separately from immediate returns, R0 remains the immutable parity
control and the complete `R0 + R4-v1` connector is architecture-qualified as the
forward feed-forward Phase A training baseline. This reflects removal of the
known position/termination alias, not economic promotion or full market
observability. The follow-up uses a fresh December-2025 block, zero-initialized
residual heads, executable turnover/near-flat gates, arm-matched update-zero
controls, counterfactual diagnostics, and a fixed 40-update 4x3 replication.
R1 must retain state-free and explicitly named `R1 + R4-v1` cells.

## Evidence index

- `solver-state-r4-v1-blocking-ablation-2026-09-25-evidence/analysis.json` —
  frozen cell table, final-three durability, diagnostics, controls, intervals,
  and gates;
- `.../training_comparison/` — official 12-cell ablation comparison;
- `.../terminal_replay/` — 12 terminal replay reports and 384 episode receipts;
- `.../terminal_state_diagnostics/` — 12 diagnostic summaries and per-step rows;
- `.../controls/` — matched control summaries and per-seed stable evidence.

## Validation record

- prelaunch healthcheck: passed on Python 3.11.13 / Ray 2.58.0;
- prelaunch full suite: 277 passed;
- RAM-gate red/green tests: passed;
- terminal-selector replay tests: passed;
- control-namespacing red/green tests: passed;
- anchor-bootstrap suite: 10 passed;
- final healthcheck passed on Ray 2.58.0; full suite: 279 passed;
- `git diff --check` and full pre-commit: passed;
- active architecture/masterplan Markdown and HTML companions were synchronized;
- GitNexus status remains stale at commit `1d5ba82` versus current `0894116`;
  it was not refreshed because the documented WSL analyzer crash-dump incident
  still forbids an uncoordinated reindex.
