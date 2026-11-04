# Handover to the CTO: the online stack, spec section by section

Branch `roman/itrader-final`, cut from `agent/Watson-dev` (6f5f384, 13 Oct 2026). Every file is new:
`itrader/online/**`, `itrader/tests/test_online_*.py`, `docs/online/**`. Nothing existing was edited,
nothing was pushed. Python 3.11+ (built and tested on 3.14 with pandas 3.1; the repo's pinned 3.11 /
pandas 2.1 should work, the code avoids version-specific behaviour). Deps: pandas, numpy, pyarrow,
requests (certifi), pytest. `bash itrader/online/ops/setup.sh` does the lot.

## Map

```
itrader/online/
  settings.toml, settings.py     one file, every threshold, changelog, hash in every DecisionRecord
  contracts.py                   Proposal -> CandidateTask/ExecutionIntent; wait_candidate(); ONLINE_FAMILIES 10-18, 99
  data/  feeds.py store.py universe.py      Binance mirror + OKX, parquet store, Quality (stale/gap/bad print)
  strategies/  base trend momentum reversion carry seasonality prediction_sleeve experimental, registry
  bench/  engine metrics walkforward profile report run     cost-aware backtester, PSR/DSR, evidence meter, ladder, HTML
  portfolio.py                   vol targeting, ERC budgets, Workbench allocator (cash first-class), book controls
  framer.py gate.py risk.py      MarketFramer, DecisionGate, RiskEngine
  execution/  broker paper binance_plug alpaca_plug executor
  runner.py dashboard.py alerts.py cli.py __main__.py ops/
```

## The spec (docs/iTRADER_Codex_Implementation_Spec.md), what is done and what is left

| section | status | where / note |
|---|---|---|
| 1.1 preserve working paths | done | no existing file touched; repo contracts imported, not redefined |
| 1.4 deterministic online shell | done | no learned component can place an order; `experimental.py` is the only hook and it emits weights that go through the same gate and risk |
| 1.5 WAIT first-class | done | `contracts.wait_candidate()` is in every candidate set; gate selects it when nothing clears |
| 2.2 online architecture | done | data -> framer -> gate -> portfolio -> risk -> executor -> DecisionRecord -> profile |
| 3.3 MarketFramer | done | `framer.py`: 1 to 3 candidates + WAIT, market summary with ref hash, Quality carried to risk |
| 3.4 DecisionGate | done | `gate.py`: validity, confidence floor, utility floor per day, utility over holding period vs 1.5x cost, risk lock; every check logged with value and threshold |
| 3.5 RiskEngine | done | `risk.py`: the 12.1 list plus kill file, gap sizing, fast cancel, keyword calendar; 12.2 modes paper and replay run the same code; live sits behind four locks |
| 3.6 Executor | done | `execution/executor.py` + `PaperBroker`; live adapters exist, guarded, untested against a venue |
| 3.7 WatsonOW | left | nothing built; the hooks are the DecisionRecord stream and `alerts.py` |
| 4 contracts | done | TaskSpec, ExecutionIntent, CandidateTask, CapabilityProfile, DecisionRecord used as is; `validate_*` accepts them. ONLINE_FAMILIES are not in `TASK_FAMILY_REGISTRY` (your file); register ids 10-18 and 99 there if you want `normalize_family_id` to keep them instead of folding to "00" |
| 5 task families | partial | online strategy families 10-18 defined in `contracts.py`; not wired to `families.py` (see above) |
| 6 solver | left | `strategies/experimental.py` has the adapter slot: give it `entry="module:function"` returning a weight frame from a `Market`, and it is benched and gated like the rest |
| 10.2 online loop steps 1-11 | done except 6 | step 6 "solver scores each candidate" is served by the bench's CapabilityProfile (PSR as confidence, mean net per bar as utility). A real solver plugs in via the experimental slot |
| 10.3 candidate count | done | `[gate] max_candidates = 3` |
| 10.4 WAIT semantics | done | all four conditions; test `test_gate_selects_wait_when_nothing_clears` |
| 10.5 outputs | done | `data/online/runs/decisions.jsonl`, one repo `DecisionRecord` per cycle |
| 11 gate standard | done | thresholds in settings, logged, no hidden heuristics |
| 12 risk standard | done | see `operator_guide.md` for every lock and how it clears |
| 13 executor standard | done | fills, acks, errors, audit in the record; never changes a weight except lot rounding |
| 14.1 golden dataset | left | DecisionRecords hold the fields; an export job to the 14.1 row shape is not written |
| 14.2 capability dataset | partial | `profiles.json` from the bench; no regime/horizon breakdown yet |
| 17 tests | done | 26 tests: contracts, no-lookahead for every strategy, cost accounting, funding signs, metrics, ladder, gate/WAIT, every risk block, paper fills, live guard, full replay through the loop |
| 18.2 online metrics | mostly | dashboard shows WAIT rate, selected candidate, confidence, utility, risk adjustments, fees, slippage, equity; calibration by family and regime over time is left |
| 19/20 phases | n/a | this is the Phase B "online shadow stack" (workstream 7) in paper form |

## What the bench found (4 years, 10 USDT pairs, holdout = last 182 days)

See `docs/online/morning_brief.md` for the table. In one line: trend (two forms) and funding carry
have a positive record after costs; cross-sectional momentum, pairs and seasonality lose to costs;
mean reversion looked excellent until a 23-hour lookahead in its daily trend filter was fixed, after
which it loses. No single sleeve passes the paper gate. The Workbench allocator over the survivors,
vol-targeted, made 5.4% a year on development and 5.8% on the holdout at 7 to 10% vol.

## The independent review (risk.py and execution/)

A separate reviewer went through `risk.py` and `execution/` and reported 21 findings. Fixed tonight:
unlock resetting the peak (it relocked at once before), NaN equity and NaN marks bypassing locks,
non-atomic state files and an unreadable state silently clearing a lock (now fails closed), spot legs
reaching the futures API, dust positions that could not be flattened (rounding and minimum notional),
stale-data and spread rules letting sign flips through, spot legs not clamped long-only, spot short
guard only at submit, inconsistent mark lookups, fast-cancel reference prices lost for a missing symbol,
live adapters not recording fills and one failed cancel aborting a cancel-all, keys read at construction,
order ids not deterministic, realised pnl gross of fees, missing marks valued at cost, funding not
idempotent, closed trades never reaching the cooldown rules. Left, by choice or for you:

* paper broker does not restore open orders after a restart (they are cancelled: the safe reading);
* `cooldown_bars_*` are hours (the cycle bar is 1h); rename if you change the cycle;
* `realised_today` is mark-to-market since the day start, which is what the daily lock should look at, but the name is loose;
* `BinanceBroker.book()` is the one unguarded call (public endpoint, no key);
* the Alpaca adapter maps spot legs to `BASE/USD` and refuses perp legs; equities pass through; untested;
* no margin or liquidation model in paper; gross is capped well below where it would matter.

## Known limits, said plainly

* Costs are modelled, not measured. The paper broker uses the live top-of-book spread, a fixed fee and 2 bps
  slippage. The ladder's `slippage_ratio` gate exists to catch the model being wrong.
* Carry ignores basis. Spot and perp are treated as price-neutral; the holdout had no funding above the hurdle.
* The bench's strategies have published, fixed parameters (`n_variants_tried` 1 or 2). If anyone starts tuning,
  bump that counter: the deflated Sharpe depends on it.
* The prediction sleeve (Polymarket) reports through the polyedge kit and never trades here. Polymarket's
  terms restrict the country the operator lives in; this stack does nothing to work around that.
