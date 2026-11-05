# Operator guide: iTRADER online (paper)

Everything runs from the repo root with the venv: `.venv/bin/python -m itrader.online <command>`.

## Daily

| command | what it does |
|---|---|
| `sync` | pulls new 1d and 1h candles and funding for the 10 pairs (incremental, a few seconds) |
| `frame` | one framing pass: which strategies want what, what the gate says, nothing traded |
| `paper --once` | one full cycle: frame, gate, portfolio, risk, paper orders, DecisionRecord |
| `paper` | the loop; a cycle at the close of every hour (`[runner] cycle_bar`) |
| `dashboard` | renders `data/online/runs/dashboard.html` from the loop's own files |
| `status` | risk state, broker state, both live guards, data quality per pair |
| `sleeve` | the Polymarket sleeve's report from the polyedge kit (paper) |

## Files the loop writes (`data/online/runs/`)

* `decisions.jsonl`: one repo `DecisionRecord` per cycle: market summary ref, every candidate (plus WAIT),
  the gate's checks with the thresholds, the risk verdict with each adjustment, orders and fills, equity.
* `risk_state.json`: equity marks, locks, cooldowns, the last 200 risk events. A lock survives a restart.
* `paper_broker.json`: cash, positions, average costs, realised pnl, open orders.
* `state.json`: per-sleeve weights, cycle count, the paper return series.
* `alerts.jsonl`: every alert, delivered or not.

## Locks and how to clear them

* **Daily loss** (`[risk] daily_loss_limit_pct`, 3%): book goes flat, no new risk until the next UTC day. Clears itself.
* **Drawdown** (`drawdown_lock_pct`, 12% from peak): book goes flat and stays flat until `python -m itrader.online unlock`.
  The unlock is logged in `risk_state.json`. Read the dashboard first and decide whether the sizing was wrong.
* **Cooldown**: after a stop-out (24 h) or three losing closed trades (48 h): reductions only.
* **Kill file**: `python -m itrader.online kill` creates `data/online/KILL`; every cycle flattens while it exists.
  `kill --remove` deletes it. Works in every mode and needs no network.
* **Stale data**: a pair whose last bar is more than 2 bars old, has a gap over 24 bars, or an impossible
  print, may only be reduced. Fix the data (`sync`), the block clears on the next cycle.
* **Calendar**: `data/online/calendar.json`, a list of `{"title", "start", "end", "pre_minutes", "post_minutes"}`.
  Any title containing a `[risk] no_trade_keywords` word (FOMC, CPI, NFP, ETF decision, hard fork, exchange outage,
  ceasefire) makes its window reductions-only. Explicit windows go in `no_trade_windows_utc`.
* **Fast cancel**: a move of more than 4% in one pair between two cycles cancels resting orders and blocks new risk
  for that cycle. This is the lesson from the 3 November Polymarket crash: size for the gap, do not try to outrun it.

## Alerts

Set env vars and nothing else: `ITRADER_TG_TOKEN` and `ITRADER_TG_CHAT` for Telegram; `ITRADER_SMTP_URL`
(`smtp://user:pass@host:587`) and `ITRADER_ALERT_EMAIL` for email. Kinds sent: risk_lock, kill, stale_data,
daily_summary, error, fast_cancel (`[alerts] on`). Everything is also appended to `alerts.jsonl`.

## Live mode (off, and how it stays off)

Four locks, all independent, all must be open before an adapter sends an order:

1. `[live] enabled = true` in settings (false, and the changelog records any change);
2. `ITRADER_LIVE=1` in the environment;
3. the venue's key env vars (`ITRADER_BINANCE_KEY` and `_SECRET`, or the Alpaca pair);
4. the file `data/online/I_UNDERSTAND_THIS_IS_REAL_MONEY`.

With any lock closed, `BinanceBroker.submit` and `AlpacaBroker.submit` raise `LiveDisabled`, and `status`
shows which locks are closed. `dry_run()` prints the exact request that would be sent, with no key.
The ladder (`promotion_ladder.md`) says when it would be reasonable to open them; the answer today is not yet.

## Changing a threshold

Edit `itrader/online/settings.toml`, add a line to `[changelog] entries`, restart the loop. Every
DecisionRecord carries the settings hash, so a change in behaviour can be matched to the change in settings.

## Running as a service (Linux)

`itrader/online/ops/itrader-online.service` is a systemd unit for the paper loop (Linux).

On the Mac: `bash itrader/online/ops/install_launchd.sh` installs a LaunchAgent that starts the loop at login,
restarts it if it exits and prevents idle sleep while it runs (`--remove` undoes it). Stop any hand-started loop
first with `pkill -f "itrader.online paper"`. macOS may ask once whether python may read the Documents folder.
A closed laptop on battery still sleeps; the loop then resumes within 30 seconds of waking and carries on at the
next hour, but those hours are missing from the record.

## Second round additions

* **Execution style.** `[execution] style = "maker"`: a change in position is posted as a resting limit order at the
  bid or ask and left for one cycle. If the hour traded through it, it fills at its price with the maker fee; if not,
  it is cancelled and sent as a market order. The dashboard shows the maker share of traded notional and how many
  orders were chased. Anything the risk engine forces (flatten, locks, reductions only) is sent at once as a market
  order. Set `style = "taker"` to go back to immediate market orders. A restart drops resting orders.
* **Bench.** `python -m itrader.online bench` now takes about a minute: both execution styles per strategy, then the
  walk-forward portfolio, BTC benchmark, bootstrap and cost stress. `data/online/runs/bench/portfolio.json` holds
  all of it; `report.html` has a Portfolio section at the top.
* **Plan.** `python -m itrader.online plan --capital 1000 --monthly 200 --horizons 1,3,5,10 --source workbench` projects
  an account from the measured out-of-sample record. Sources are the columns of `portfolio_returns.parquet`:
  `workbench`, `workbench_regime`, `equal_weight`, `btc`, `btc_vol_matched` and so on. Read "paid in" first.
* **Regimes.** The dashboard and each DecisionRecord show the current regime (BTC above or below its 100-day mean,
  30-day volatility above or below its one-year median). `[allocator] regime_aware = true` makes the allocator score
  sleeves on their record in that regime. It is off because it was worse out of sample.
* **Daily summary.** One alert at the first cycle of each UTC day: equity, change, gross, regime, lock state.
* **Funding.** `sync` fills the last weeks of funding from OKX because the Binance archive is monthly. Carry stays flat
  where funding data is missing.
