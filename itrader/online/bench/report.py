"""Self-contained HTML bench report: one table, one equity chart per strategy (inline SVG,
no JavaScript, no external files), the gate results, and the evidence meter. Opens anywhere."""
from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

import pandas as pd

CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;color:#1b1b1b;background:#fff}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 8px} .sub{color:#666;font-size:13px}
table{border-collapse:collapse;font-size:13px;margin:8px 0} th,td{border-bottom:1px solid #e3e3e3;padding:5px 9px;text-align:right}
th{background:#f4f4f4;text-align:right} td:first-child,th:first-child{text-align:left}
.ok{color:#1a7f37;font-weight:600} .bad{color:#b42318;font-weight:600} .muted{color:#888}
.card{border:1px solid #e3e3e3;border-radius:8px;padding:12px 14px;margin:10px 0}
svg{display:block;margin:6px 0} .note{background:#fff8e1;border:1px solid #f1e0a6;padding:8px 10px;border-radius:6px;font-size:13px}
"""


def _pct(x: Optional[float], d: int = 1) -> str:
    if x is None or x != x:
        return "n/a"
    return f"{100 * x:.{d}f}%"


def _num(x: Optional[float], d: int = 2) -> str:
    if x is None or x != x:
        return "n/a"
    if x == float("inf"):
        return "inf"
    return f"{x:.{d}f}"


def _cls(ok: bool) -> str:
    return "ok" if ok else "bad"


def equity_svg(equity: pd.Series, holdout_start: Optional[pd.Timestamp] = None, width: int = 760, height: int = 160) -> str:
    if equity is None or len(equity) < 2:
        return ""
    e = equity.dropna()
    lo, hi = float(e.min()), float(e.max())
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    xs = (e.index.astype("int64") - e.index[0].value) / max(1, (e.index[-1].value - e.index[0].value))
    pts = " ".join(f"{10 + x * (width - 20):.1f},{10 + (1 - (v - lo) / (hi - lo)) * (height - 20):.1f}" for x, v in zip(xs, e.values))
    shade = ""
    if holdout_start is not None and e.index[-1] > holdout_start >= e.index[0]:
        x0 = 10 + (holdout_start.value - e.index[0].value) / max(1, (e.index[-1].value - e.index[0].value)) * (width - 20)
        shade = f'<rect x="{x0:.1f}" y="0" width="{width - x0:.1f}" height="{height}" fill="#eef6ff"/><text x="{x0 + 4:.1f}" y="14" font-size="11" fill="#4a7">holdout</text>'
    one_y = 10 + (1 - (1.0 - lo) / (hi - lo)) * (height - 20)
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">{shade}'
            f'<line x1="10" y1="{one_y:.1f}" x2="{width - 10}" y2="{one_y:.1f}" stroke="#ccc" stroke-dasharray="3,3"/>'
            f'<polyline fill="none" stroke="#1f5fbf" stroke-width="1.5" points="{pts}"/>'
            f'<text x="{width - 60}" y="{height - 4}" font-size="11" fill="#666">{e.index[-1].date()}</text>'
            f'<text x="12" y="{height - 4}" font-size="11" fill="#666">{e.index[0].date()}</text></svg>')


def render(results: List[Dict[str, Any]], equities: Dict[str, pd.Series], title: str, costs: Dict[str, float],
           notes: Optional[List[str]] = None, portfolio: Optional[Dict[str, Any]] = None) -> str:
    rows = []
    for r in results:
        wf, st = r["walk_forward"], r["promotion"]
        d, h = wf["dev"], wf["holdout"]
        rows.append(
            f"<tr><td>{html.escape(r['name'])}<br><span class='muted'>{r['bar']} {r['instrument']}</span></td>"
            f"<td>{_pct(d['cagr'])}</td><td>{_num(d['sharpe'])}</td><td>{_pct(d['dsr'], 0)}</td><td>{_pct(d['max_dd'])}</td>"
            f"<td>{_pct(h['cagr'])}</td><td class='{_cls(h['sharpe'] > 0)}'>{_num(h['sharpe'])}</td><td>{_pct(h['max_dd'])}</td>"
            f"<td>{_pct(wf['folds_positive_share'], 0)}</td><td>{_num(d['turnover_per_year'], 0)}x</td><td>{_pct(min(d['cost_share'], 9.99), 0)}</td>"
            f"<td>{_num(d['evidence'].get('years_needed'), 1)}y</td>"
            f"<td>{html.escape(str(r.get('execution_style', 'taker')))} {_pct((r.get('execution') or {}).get('fill_rate'), 0) if r.get('execution_style') == 'maker' else ''}</td>"
            f"<td class='muted'>{_num((r.get('alt') or {}).get('dev', {}).get('sharpe'))} / {_num((r.get('alt') or {}).get('holdout', {}).get('sharpe'))}</td>"
            f"<td class='{_cls(st['stage'] != 'bench')}'>{st['stage']}</td></tr>")
    table = ("<table><tr><th>strategy</th><th>dev CAGR</th><th>dev Sharpe</th><th>DSR</th><th>dev maxDD</th>"
             "<th>holdout CAGR</th><th>holdout Sharpe</th><th>holdout maxDD</th><th>folds +</th><th>turnover/yr</th><th>cost share</th>"
             "<th>evidence</th><th>execution, fill rate</th><th>other style: dev / holdout Sharpe</th><th>stage</th></tr>" + "".join(rows) + "</table>")
    cards = []
    for r in results:
        wf, st = r["walk_forward"], r["promotion"]
        hs = pd.Timestamp(wf["split"]["holdout_start"]) if wf["split"].get("holdout_start") not in (None, "None") else None
        gates = "".join(
            f"<tr><td>{c['check']}</td><td>{_num(c['value'], 3) if c['value'] is not None else 'n/a'}</td><td>{c['op']} {_num(c['limit'], 3)}</td><td class='{_cls(c['ok'])}'>{'pass' if c['ok'] else 'fail'}</td></tr>"
            for rung in st["rungs"] for c in rung["checks"])
        folds = "".join(f"<tr><td>{f['start']} to {f['end']}</td><td class='{_cls(f['net'] > 0)}'>{_pct(f['net'])}</td><td>{_num(f['sharpe'])}</td><td>{_pct(f['max_dd'])}</td></tr>" for f in wf["folds"])
        ev = wf["dev"]["evidence"]
        cap = wf["dev"].get("capacity", {})
        cards.append(
            f"<div class='card'><h2>{html.escape(r['name'])} <span class='sub'>{html.escape(r.get('description', ''))}</span></h2>"
            f"{equity_svg(equities.get(r['name']), hs)}"
            f"<p class='sub'>Development {wf['dev']['start'][:10]} to {wf['dev']['end'][:10]}; holdout from {str(hs.date()) if hs is not None else 'n/a'}. "
            f"Exposure {_pct(wf['dev']['exposure_share'], 0)} of bars. Worst day {_pct(wf['dev']['worst_day'])}. "
            f"Gross {_pct(wf['dev']['gross_return_total'])}, costs {_pct(wf['dev']['cost_total'])}, funding {_pct(wf['dev']['funding_total'])} over the development period. "
            f"Capacity about ${_num(cap.get('capacity_usd'), 0)} (binding: {cap.get('binding_symbol')}) at {costs.get('participation', 0.01) * 100:.0f}% participation.</p>"
            f"<p class='sub'>Evidence meter: at this Sharpe, {_num(ev.get('years_needed'), 1)} years of record are needed for 95% confidence the edge is above zero; "
            f"this sample covers {_pct(min(ev.get('ratio', 0.0), 9.99), 0)} of that.</p>"
            f"<table><tr><th>fold</th><th>net</th><th>Sharpe</th><th>maxDD</th></tr>{folds}</table>"
            f"<table><tr><th>gate</th><th>value</th><th>rule</th><th></th></tr>{gates}</table></div>")
    port = ""
    if portfolio:
        port = "<h2>Portfolio</h2>" + equity_svg(portfolio.get("equity")) + "<pre class='sub'>" + html.escape(portfolio.get("text", "")) + "</pre>"
    notes_html = "".join(f"<p class='note'>{html.escape(n)}</p>" for n in (notes or []))
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
            f"<h1>{html.escape(title)}</h1><p class='sub'>Costs per side: fee {costs.get('fee_bps')} bps (spot {costs.get('spot_fee_bps')}), "
            f"half spread {costs.get('half_spread_bps')} bps, slippage {costs.get('slippage_bps')} bps. Weights applied one bar after they are decided. "
            f"Holdout is the final {costs.get('holdout_days')} days and was not used for any design choice.</p>{notes_html}{table}{port}{''.join(cards)}</body></html>")
