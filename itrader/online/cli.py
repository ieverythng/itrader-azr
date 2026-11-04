"""Command line: python -m itrader.online <command>

  sync        download or update the data store (4 years, 1d + 1h, funding)
  bench       run every strategy through the bench and write the report
  frame       one framing pass: candidates, gate decision, portfolio and risk verdict, no orders
  paper       the paper loop (--cycles N, --once, --interval S)
  replay      the same loop over past timestamps (--start, --end, --step-hours)
  dashboard   write the HTML dashboard from the paper loop's files
  status      risk state, broker state, live guard, data quality
  sleeve      the prediction sleeve's (Polymarket, paper) report
  unlock      clear a drawdown lock (operator action, logged)
  kill        create the kill file (the loop flattens and stops opening risk); --remove to delete it
  settings    print the effective settings and their hash
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .settings import load_settings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m itrader.online", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sync", "bench", "frame", "paper", "replay", "dashboard", "status", "sleeve", "unlock", "kill", "settings"])
    ap.add_argument("--settings", help="override settings file (or set ITRADER_ONLINE_SETTINGS)")
    ap.add_argument("--years", type=float, default=None)
    ap.add_argument("--cycles", type=int, default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=None, help="seconds between cycles (default: next bar close)")
    ap.add_argument("--no-sync", action="store_true", help="do not refresh data each cycle")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--step-hours", type=int, default=24)
    ap.add_argument("--max-cycles", type=int, default=None)
    ap.add_argument("--out", default=None, help="output directory for paper/replay")
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--strategies", default=None, help="comma list for bench")
    a = ap.parse_args(argv)
    s = load_settings(a.settings)

    if a.command == "settings":
        print(json.dumps(s.to_dict(), indent=1))
        print(f"# hash {s.hash}  sources {s.sources}")
        return 0
    if a.command == "sync":
        from .data import DataStore

        added = DataStore(s.resolve(s.general.data_root)).sync(list(s.data.universe), list(s.data.bars), years=a.years or float(s.data.years))
        print(json.dumps(added, indent=1))
        return 0
    if a.command == "bench":
        from .bench.run import run_bench

        run_bench(s, names=a.strategies.split(",") if a.strategies else None)
        return 0
    if a.command == "sleeve":
        from .strategies import PredictionSleeve

        ps = PredictionSleeve()
        print(json.dumps(ps.status(), indent=1, default=str))
        print(ps.report())
        return 0
    if a.command == "kill":
        p = s.resolve(s.risk.kill_file)
        if a.remove:
            p.unlink(missing_ok=True)
            print(f"removed {p}")
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(datetime.now(timezone.utc).isoformat())
            print(f"kill file written: {p}")
        return 0

    from .runner import PaperRunner

    out = Path(a.out) if a.out else None
    if a.command == "unlock":
        r = PaperRunner(s, out_dir=out, sync=False, alerts_enabled=False)
        r.risk.unlock()
        print("unlocked")
        return 0
    if a.command == "status":
        from .execution.broker import live_guard

        r = PaperRunner(s, out_dir=out, sync=False, alerts_enabled=False)
        q = {k: v.to_dict() for k, v in r.store.quality_report(list(s.data.universe), "1d").items()}
        print(json.dumps({"risk": {k: v for k, v in r.risk.state.to_dict().items() if k != "events"}, "broker": r.broker.snapshot(),
                          "live_guard_binance": live_guard(s, None, "binance"), "live_guard_alpaca": live_guard(s, None, "alpaca"),
                          "data_quality_1d": q, "cycles": r.state.get("cycles"), "profiles_loaded": sorted(r.profiles)}, indent=1, default=str))
        return 0
    if a.command == "dashboard":
        from .dashboard import write_dashboard
        from .strategies import PredictionSleeve

        r = PaperRunner(s, out_dir=out, sync=False, alerts_enabled=False)
        print(write_dashboard(r.out, s, PredictionSleeve().status()))
        return 0
    if a.command == "frame":
        r = PaperRunner(s, out_dir=out, sync=not a.no_sync, alerts_enabled=False)
        f = r.framer.frame()
        gd = r.gate.decide(f, risk_locked=bool(r.risk.state.locked_until))
        print(json.dumps({"candidates": [{"id": p.candidate_id, "strategy": p.strategy, "gross": p.gross, "expected_utility": p.expected_utility,
                                          "confidence": p.confidence, "weights": p.weights} for p in f.proposals],
                          "gate": gd.to_dict(), "quality_not_ok": [k for k, v in f.quality.items() if not v.ok], "notes": f.notes}, indent=1, default=str))
        return 0
    if a.command == "paper":
        r = PaperRunner(s, out_dir=out, sync=not a.no_sync)
        r.run(cycles=1 if a.once else a.cycles, interval_s=a.interval)
        from .dashboard import write_dashboard

        write_dashboard(r.out, s)
        return 0
    if a.command == "replay":
        end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc) if a.end else datetime.now(timezone.utc)
        start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc) if a.start else end - timedelta(days=30)
        r = PaperRunner(s, out_dir=out or s.resolve(s.general.out_root) / "replay", sync=False, alerts_enabled=False)
        recs = r.replay(start, end, step=timedelta(hours=a.step_hours), max_cycles=a.max_cycles)
        from .dashboard import write_dashboard

        print(f"{len(recs)} cycles; final equity {recs[-1]['realized_outcome']['equity_after']:,.2f}" if recs else "no cycles")
        print(write_dashboard(r.out, s))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
