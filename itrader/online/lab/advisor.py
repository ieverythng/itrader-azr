"""The advisor: a plain-language review written from the evidence, every time the lab runs.

It reads the league table, the champion file, the ledger and the paper loop's own records, and writes
`advisor_report.md`: where the system stands, which change earned its place and which did not, what it
got wrong, whether paper trading agrees with the model, and what is worth trying next. It is a set
of explicit rules over measured numbers, not an opinion: the same inputs always give the same report,
and every sentence can be traced to a figure in `league.json`.

It will not tell you the system is good when the numbers do not say so. That is its job.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..settings import Settings, settings as default_settings
from .evaluate import paired_t

# which earlier recipe each recipe should be compared with, and what the difference isolates
LEVERS = [
    ("R1_steady", "R0_baseline", "a slower allocator with no trailing vote"),
    ("R2_stack", "R1_steady", "carry as the core with a directional overlay (capital stacking)"),
    ("R3_stack_carry_core", "R2_stack", "a carry hurdle set by the cost of cash instead of 13%"),
    ("R4_stack_wide", "R3_stack_carry_core", "25 coins instead of 10"),
    ("R5_stack_wide_long", "R4_stack_wide", "adding long-only trend (market exposure)"),
    ("R6_stack_wide_macro", "R4_stack_wide", "adding trend on 20 ETFs across asset classes"),
    ("R7_stack_wide_all", "R4_stack_wide", "adding both long-only trend and multi-asset trend"),
]


def _pct(x: Optional[float], d: int = 1) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.{d}f}%"


def _jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def paper_check(runs: Path, champ: Dict[str, Any]) -> List[str]:
    recs = _jsonl(runs / "decisions.jsonl")
    if not recs:
        return ["No paper cycles recorded yet."]
    t0, t1 = pd.Timestamp(recs[0]["created_at"]), pd.Timestamp(recs[-1]["created_at"])
    days = max((t1 - t0).total_seconds() / 86400.0, 1e-9)
    eq0 = recs[0].get("realized_outcome", {}).get("equity_before") or 0.0
    eq1 = recs[-1].get("realized_outcome", {}).get("equity_after") or 0.0
    mk = [r.get("executor_action", {}).get("maker") or {} for r in recs]
    posted, filled, chased = (sum(m.get(k, 0) for m in mk) for k in ("posted", "filled_since_last_cycle", "chased_as_taker"))
    fees = sum(r.get("executor_action", {}).get("fees", 0.0) for r in recs)
    waits = sum(1 for r in recs if r.get("wait_selected")) / len(recs)
    lines = [f"{len(recs)} cycles over {days:.1f} days. Equity {eq0:,.2f} to {eq1:,.2f} ({_pct(eq1 / eq0 - 1.0, 2) if eq0 else 'n/a'}). WAIT in {_pct(waits, 0)} of cycles. Fees paid {fees:,.2f}."]
    settled = filled + chased
    if settled >= 10:
        lines.append(f"Resting orders: {posted} posted, {filled} filled as maker, {chased} chased as taker: a fill rate of {_pct(filled / settled, 0)} "
                     f"against the 93 to 99% the bench assumes. {'That is in line.' if filled / settled >= 0.85 else 'That is BELOW the model: the bench is flattering the maker style and its numbers should be read down.'}")
    else:
        lines.append(f"Resting orders: {posted} posted, {filled} filled, {chased} chased. Too few to judge the fill model; it needs at least a few dozen.")
    exp = (champ.get("research") or {}).get("cagr")
    if days < 30:
        lines.append(f"The paper record is {days:.0f} days old. Nothing can be read from its return yet; the first honest comparison with the model is at about 60 days.")
    elif exp is not None and eq0:
        got = (eq1 / eq0) ** (365.0 / days) - 1.0
        lines.append(f"Annualised, paper is running at {_pct(got)} against the champion's research figure of {_pct(exp)}. "
                     f"{'Within what noise allows.' if abs(got - exp) < 0.15 else 'A wide gap: check fills, costs and whether the champion changed.'}")
    locks = [r for r in recs if (r.get("risk_decisions") or {}).get("flatten")]
    if locks:
        lines.append(f"The risk engine flattened the book in {len(locks)} cycles. Read those DecisionRecords before changing anything.")
    return lines


def write_report(s: Optional[Settings] = None, lab_root: Optional[Path] = None, runs: Optional[Path] = None) -> Path:
    s = s or default_settings()
    lab_root = Path(lab_root) if lab_root else s.resolve(s.lab.root)
    runs = Path(runs) if runs else s.resolve(s.general.out_root)
    league = json.loads((lab_root / "league.json").read_text()) if (lab_root / "league.json").exists() else {}
    champ = json.loads((lab_root / "champion.json").read_text()) if (lab_root / "champion.json").exists() else {}
    res = {r["name"]: r for r in league.get("results", [])}
    rets = pd.read_parquet(lab_root / "returns.parquet") if (lab_root / "returns.parquet").exists() else pd.DataFrame()
    bench = league.get("benchmarks", {})
    win = league.get("research_window", ["?", "?"])
    L: List[str] = [f"# Advisor report, {datetime.now(timezone.utc):%d %B %Y %H:%M} UTC", ""]

    # ---------------- where it stands
    L += ["## Where the system stands", ""]
    if champ.get("name") and champ["name"] in res:
        c, r = champ, res[champ["name"]]
        rs, v = c["research"], c.get("vault") or {}
        L += [f"**Champion: `{c['name']}`** ({r['policy']} policy, {r['universe']}{', ' + r['label'] if r['label'] else ''}). {c.get('note', '')}.", "",
              f"* Research window ({win[0]} to {win[1]}, every day out of sample): **{_pct(rs['cagr'])} a year at {_pct(rs['vol'])} volatility, Sharpe {rs['sharpe']:.2f}**, "
              f"worst drawdown {_pct(rs['max_dd'])}, worst day {_pct(rs['worst_day'])}.",
              f"* Deflated for the {c['n_trials']} variants tried so far, the probability this is better than nothing is {_pct(c['dsr'], 0)}.",
              f"* Vault ({c.get('vault_start')} onward, never used to choose): {_pct(v.get('total_return'))} in total, Sharpe {v.get('sharpe', 0):.2f}, worst drawdown {_pct(v.get('max_dd'))}. "
              f"**{'Confirmed' if c.get('confirmed') else 'NOT confirmed'}**.",
              f"* At double the assumed costs: Sharpe {r['cost2x']['sharpe']:.2f}. Chance the research record is a losing one: {_pct(r['bootstrap'].get('p_losing_record'), 0)}.",
              f"* Against BTC: beta {r['vs_btc'].get('beta', 0):.2f}, alpha {_pct(r['vs_btc'].get('alpha_ann'))} a year (t {r['vs_btc'].get('alpha_t', 0):.1f}). "
              f"Against the S&P 500: beta {r['vs_spy'].get('beta', 0):.2f}, alpha {_pct(r['vs_spy'].get('alpha_ann'))} a year (t {r['vs_spy'].get('alpha_t', 0):.1f}).", ""]
    else:
        L += [f"No champion. {champ.get('note', 'The lab has not been run.')}", ""]
    if bench:
        L += ["Benchmarks over the same research window: " + "; ".join(
            f"{k.replace('_', ' ')} {_pct(v['cagr'])} a year at {_pct(v['vol'])} vol (Sharpe {v['sharpe']:.2f}, drawdown {_pct(v['max_dd'])})" for k, v in bench.items()) + ".", ""]

    # ---------------- league
    if res:
        L += ["## Every recipe tried", "",
              "| recipe | CAGR | vol | Sharpe | max DD | Sharpe at 2x costs | P(losing) | deflated | BTC beta | alpha t | eligible |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(res.values(), key=lambda r: r.get("dsr", 0), reverse=True):
            rs = r["research"]
            L.append(f"| {'**' + r['name'] + '**' if r.get('champion') else r['name']} | {_pct(rs['cagr'])} | {_pct(rs['vol'])} | {rs['sharpe']:.2f} | {_pct(rs['max_dd'])} | {r['cost2x']['sharpe']:.2f} | "
                     f"{_pct(r['bootstrap'].get('p_losing_record'), 0)} | {_pct(r.get('dsr'), 0)} | {r['vs_btc'].get('beta', 0):.2f} | {r['vs_btc'].get('alpha_t', 0):.1f} | {'yes' if r.get('eligible') else 'no'} |")
        L += ["", f"Research window only. \"Deflated\" is the probability the Sharpe ratio is above what the best of {league.get('n_trials')} no-skill variants would show by luck.", ""]

    # ---------------- what each change did
    if res and len(rets):
        vs = pd.Timestamp(win[1], tz="UTC") if win[1] != "?" else None
        L += ["## What each change was worth", ""]
        for new, old, what in LEVERS:
            if new in res and old in res and new in rets and old in rets:
                a, b = rets[new], rets[old]
                if vs is not None:
                    a, b = a[a.index < vs], b[b.index < vs]
                t = paired_t(a, b)
                dS, dC = res[new]["research"]["sharpe"] - res[old]["research"]["sharpe"], res[new]["research"]["cagr"] - res[old]["research"]["cagr"]
                verdict = "helped, and the difference is unlikely to be chance" if t >= 2 else ("helped, but the difference could be chance" if dS > 0.05 else ("made no real difference" if abs(dS) <= 0.05 else "made it worse"))
                L.append(f"* **{what}** ({new} against {old}): Sharpe {dS:+.2f}, return {100 * dC:+.1f} points a year, t {t:.1f}. It {verdict}.")
        L.append("")

    # ---------------- mistakes and warnings
    W: List[str] = []
    if champ.get("name") in res:
        r, c = res[champ["name"]], champ
        if not c.get("confirmed"):
            W.append("The champion lost money in the vault. Its research record did not travel to the most recent six months. Treat it as unproven.")
        if c.get("dsr", 0) < 0.90:
            W.append(f"After counting {c['n_trials']} trials, the champion's Sharpe is not statistically established (deflated probability {_pct(c['dsr'], 0)}; 90% is the bar).")
        if r["research"]["sharpe"] > 0 and r["cost2x"]["sharpe"] < 0.5 * r["research"]["sharpe"]:
            W.append("Doubling costs removes more than half of the Sharpe ratio. The result depends on the cost model being right; paper fills must confirm it.")
        if r["vs_btc"].get("alpha_t", 0) < 2:
            W.append(f"Alpha against BTC has a t-statistic of {r['vs_btc'].get('alpha_t', 0):.1f}. Below 2, it cannot be told apart from chance.")
        if "market exposure" in r.get("label", ""):
            W.append("The long-only trend sleeve earns by owning crypto while it rises and standing aside when it falls. That is market exposure with a brake, not an edge: in a long decline it earns nothing, and in a choppy one it loses.")
        if abs(r["vs_btc"].get("beta", 0)) > 0.15:
            W.append(f"The book's beta to BTC is {r['vs_btc'].get('beta', 0):.2f}: a meaningful part of the return is simply crypto going up.")
        if "survivor" in r.get("label", ""):
            W.append("The 25-coin universe is a list of coins that survived. Results on it are a little better than a trader could have had at the time.")
        if "carry_core" in r.get("core", []) or "funding_carry" in r.get("core", []):
            W.append("The core is spot-against-perpetual carry. Its real risks, an exchange failing or spot and perpetual prices pulling apart, do not appear in its volatility or its drawdown here.")
        if "macro_trend" in r.get("overlay", []) and r.get("avg_exposure", {}).get("macro_trend", 0) > 0:
            W.append("The multi-asset sleeve assumes margined instruments (futures or perpetuals on the same markets); ETF history is the proxy and the paper loop marks it at daily closes.")
        idle = [k for k, v in (r.get("share_on") or {}).items() if v < 0.2]
        if idle:
            W.append(f"Sleeves that were switched on in under a fifth of months and add little: {', '.join(sorted(idle))}.")
    overfit = [n for n, r in res.items() if r["research"]["sharpe"] - r["cost2x"]["sharpe"] > 0.6]
    if overfit:
        W.append(f"Cost-fragile recipes (Sharpe falls by more than 0.6 at double costs): {', '.join(overfit)}.")
    L += ["## What to be careful about", ""] + ([f"* {w}" for w in W] or ["* Nothing flagged."]) + [""]

    # ---------------- paper
    L += ["## Is paper trading agreeing with the model?", ""] + [f"* {x}" for x in paper_check(runs, champ)] + [""]

    # ---------------- what next
    N: List[str] = []
    if champ.get("name") in res:
        r = res[champ["name"]]
        if champ.get("dsr", 0) < 0.90 or r["vs_btc"].get("alpha_t", 0) < 2:
            N.append("The scarce thing now is evidence, not ideas. Leave the champion running in paper for 60 days without touching a threshold; that record is the only test left that has not been looked at.")
        if "R6_stack_wide_macro" in res and "R4_stack_wide" in res and res["R6_stack_wide_macro"]["research"]["sharpe"] > res["R4_stack_wide"]["research"]["sharpe"] + 0.05:
            N.append("Breadth across asset classes helped. The next step on that road is a venue that offers margined listed markets, and more of them (rates, energy, metals, currencies): each independent market adds to the Sharpe ratio.")
        if r.get("core"):
            N.append("Carry is carrying weight. Before real money: a second venue to split exchange risk, and a model of the spot-perpetual gap in a violent rally.")
        N.append("Give the experimental slot the CTO's solver (`entry=\"module:function\"`). It is judged by this same lab, against this champion.")
    N.append("To try an idea: write the rule and the hypothesis into `itrader/online/lab/registry.py` first, commit it, then run `python -m itrader.online lab`. Every recipe added is one more trial counted against all of them.")
    L += ["## What to do next", ""] + [f"{i + 1}. {x}" for i, x in enumerate(N)] + [""]
    hist = champ.get("history") or []
    if hist:
        L += ["## Champion history", ""] + [f"* {h['at'][:16]}: {h['name']} ({h['note']})" for h in hist] + [""]
    L += ["This report is generated from measured results by fixed rules. It describes software tested on historical data; it is not financial advice.", ""]
    out = lab_root / "advisor_report.md"
    out.write_text("\n".join(L))
    return out
