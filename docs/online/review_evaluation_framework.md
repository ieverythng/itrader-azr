# Review of the evaluation framework on `agent/Watson-dev`

Requested by the CTO ("ask Claude to check the evaluation framework"). Read-only: nothing was run, nothing in the
framework was changed, and no run outputs were regenerated. Branch state reviewed: `agent/Watson-dev` at `6f5f384`.

Sources read: `docs/evaluation/README.md`, `docs/research/trading-rl-evaluation-advisory-2026-10-10.md`,
`itrader/rl_lib/evaluation/{block_uncertainty,data_partition,policy_baselines}.py`, `itrader/envs/itrader_env.py`, the
`replay_jobscope_20261005/cell04` acceptance, declaration and episodes files, and `dataset_manifest.json`. The other
R-numbered solvers' artefacts were not opened.

## 1. What it does, end to end

* **Data:** dataset_v2, Alpaca SIP 1-minute bars, 12 symbols, 2022-01-03 to 2026-09-22, split-adjusted, dividends not
  applied. `DataPartition` splits each symbol chronologically 70/15/15 by row fraction with a 200-row purge.
* **Declared in advance:** task family, horizons 96/128/160/200, the net_return objective, a 1 bp evidence cost and frozen
  episode identities (symbol, start_idx, seed). Each run's `declaration.json` hashes the checkpoint, the evaluator source
  and the launcher source tree.
* **Measured:** per-episode gross, cost and net return, turnover, per-bar Sharpe, drawdown and action fractions.
* **Accept or reject:** `accepted_terminal_replay` means the frozen replay completed with hashes and identities matching.
  It records `"economic_promotion": false`. Acceptance is mechanical, not statistical, and the framework says so.

## 2. Statistical significance

* `block_uncertainty.paired_block_interval` resamples 5,000-row chronological blocks within each symbol, keeping
  policy-versus-control pairs, and gives a 95% percentile interval on the mean net-return difference. The code labels it
  `exploratory_selection_panel_not_selection_adjusted`, and the README lists a qualified calendar-based method as future work.
* **Baselines exist and are a real strength:** flat, long, short, random and untrained architecture-matched controls, with
  paired panels (`policy_baselines.py`). Buy and hold is the long control.
* No deflation, multiplicity correction, probabilistic or deflated Sharpe ratio, or White/SPA-type test was found.
* The README requires a sealed final test and records data exposure. Elsewhere the documents say the validation panels
  have been inspected repeatedly, so there is a real selection-reuse risk on validation.
* The cell04 acceptance reports a mean net return of +0.39% over 32 episodes, 95% of them short, with no interval attached.

## 3. Costs and realism

* Costs are a flat charge per unit of turnover (1 bp in the evidence) plus liquidation at the end of the episode. There is
  no spread, slippage, market-impact or short-borrow model. 1 bp is optimistic for 1-minute equity trading outside the
  most liquid names.
* Bars include pre-market and after-hours (about half the rows), where spreads are wider.
* Lookahead: the action is taken at close t and earns the move to close t+1 (`itrader_env.py`, around line 691), and the
  README documents a causal-window oracle test. No leak was seen, but the full observation path was not traced.

## 4. Does "10.3M rows" give the power it sounds like?

No. 10,370,064 is the whole dataset across all sessions, not one solver's test sample. The advisory states that training
uses about 192,000 sampled steps. Evaluation is on the validation or terminal panel: 128 tasks built from 32 anchors
across 12 symbols by design, and 32 episodes of 96 to 200 bars in the cell04 replay. Minute returns are very noisy and a
day's 12 tech-heavy symbols share one market factor, so the independent evidence is closer to tens of sessions than to
millions of rows. That is why a 4.4 bp gap cannot be resolved at 1 bp costs, as the advisory itself says.

## 5. What is good, and what would add most

Good, specifically:

* Hash-bound provenance, fail-closed gates and an explicit "not promotion" disposition. Description is kept apart from inference.
* Exact accounting identities with literal tests.
* A pre-declared, equal-symbol-weighted panel, a sealed-test discipline and matched controls.
* Candour about limits: the interval is labelled as a selection panel, and the missing qualified method is named.

Improvements, each a concrete change:

1. Emit a per-session (per-day) return series for every run, and bootstrap by session rather than by 5,000 rows.
2. Add a probabilistic Sharpe ratio and a deflated one, using the true count of R-solvers, seeds, checkpoints and
   hyperparameter settings tried.
3. Always report the difference against the long, short and random controls with intervals, and repeat it at 2 to 5 bp costs.
4. Add a spread and slippage model, and report regular hours separately from pre-market and after-hours.
5. Evaluate once on the sealed final test with a pre-registered primary metric, and publish effective sample sizes in
   sessions, not rows.

## 6. Where a significance check plugs in today

Per-episode returns are already in the `*.episodes.jsonl` files (fields `net_return`, `gross_return`, `symbol`,
`start_idx`, `horizon`, `task_seed`). The online stack now ships a tool that reads them directly:

```bash
python -m itrader.online judge --returns "docs/artifacts/research/<run>-evidence/replay_jobscope_20261005/cell04/out/<file>.episodes.jsonl" --column net_return --trials 8
```

It prints the mean and its t-statistic, the probabilistic Sharpe ratio, the deflated one for `--trials` variants, a block
bootstrap of the mean and the record length needed for 95% confidence, and a one-line verdict.

Run on the cell04 replay as recorded (32 episodes, nothing re-run), with `--trials 8` standing in for R1 to R8:

```
mean +0.003928 per episode (t = 2.59, one-sided p = 0.005); hit rate 56%
probability the true Sharpe is above zero: 99.9%
the same after 8 trial(s): 94.7%
block bootstrap of the mean: 5% +0.000393, median +0.003881, 95% +0.007836; chance the mean is not positive 3.3%
VERDICT: significant on its own, NOT after allowing for 8 trials
```

So the direct answer to "is it statistically significant" for that run: yes if it was the only thing tried, borderline
(94.7% against a 95% bar) once eight solvers are counted, and less than that if seeds, checkpoints and hyperparameters
are counted too. Two further cautions pull the same way: the 32 episodes are 95% short positions on co-moving symbols,
so they are far from 32 independent bets, and the 1 bp cost is light. A per-session series and the true trial count
would settle it. Episodes overlap, so set
`--block` at least as long as the overlap and read the result as an upper bound on the evidence. No stored per-day return
file was found; `block_uncertainty.paired_block_interval` is the natural place to add session blocks, and
`analyze_r4_v1_ablation.py` and the acceptance summary are where the result would be reported.

This is a review of method, offered as input for alignment. Nothing here was verified by re-running an evaluation.
