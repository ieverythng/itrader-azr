"""Self-contained HTML dashboard from the paper loop's own files (decisions.jsonl, state, risk
state, alerts). No server, no JavaScript dependencies; open the file."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .bench.report import CSS, equity_svg
from .settings import Settings, settings as default_settings


def _jsonl(p: Path, limit: Optional[int] = None) -> List[dict]:
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:] if limit else rows


def render_dashboard(out_dir: Path, s: Optional[Settings] = None, sleeve_status: Optional[Dict[str, Any]] = None) -> str:
    s = s or default_settings()
    recs = _jsonl(out_dir / "decisions.jsonl")
    risk = json.loads((out_dir / "risk_state.json").read_text()) if (out_dir / "risk_state.json").exists() else {}
    state = json.loads((out_dir / "state.json").read_text()) if (out_dir / "state.json").exists() else {}
    alerts = _jsonl(out_dir / "alerts.jsonl", 30)
    broker = json.loads((out_dir / "paper_broker.json").read_text()) if (out_dir / "paper_broker.json").exists() else {}
    eq = pd.Series({pd.Timestamp(r["created_at"]): r["realized_outcome"]["equity_after"] for r in recs if r.get("realized_outcome", {}).get("equity_after")}).sort_index()
    wait_rate = sum(1 for r in recs if r.get("wait_selected")) / len(recs) if recs else 0.0
    blocks = sum(1 for r in recs if r.get("risk_decisions", {}).get("adjustments")) / len(recs) if recs else 0.0
    fees = sum(r.get("executor_action", {}).get("fees", 0.0) for r in recs)
    mk = [r.get("executor_action", {}).get("maker") or {} for r in recs]
    mk_n, tk_n = sum(m.get("maker_notional", 0.0) for m in mk), sum(m.get("taker_notional", 0.0) for m in mk)
    maker_line = (f" Maker share of traded notional {100 * mk_n / (mk_n + tk_n):.0f}% ({sum(m.get('posted', 0) for m in mk)} posted, "
                  f"{sum(m.get('filled_since_last_cycle', 0) for m in mk)} filled resting, {sum(m.get('chased_as_taker', 0) for m in mk)} chased).") if (mk_n + tk_n) > 0 else ""
    regime = ((recs[-1].get("metadata") or {}).get("market_summary") or {}).get("regime") if recs else None
    last = recs[-1] if recs else {}
    port = (last.get("metadata") or {}).get("portfolio") or {}
    rows = "".join(
        f"<tr><td>{html.escape(r['created_at'][:16])}</td><td>{r['realized_outcome'].get('equity_after', 0):,.2f}</td>"
        f"<td>{100 * r['realized_outcome'].get('return', 0):+.3f}%</td><td>{'WAIT' if r.get('wait_selected') else html.escape(str(r.get('selected_candidate_id')))}</td>"
        f"<td>{len(r.get('executor_action', {}).get('orders', []))}</td><td>{html.escape('; '.join(r.get('risk_decisions', {}).get('reasons', [])) or 'ok')}</td></tr>"
        for r in reversed(recs[-40:]))
    positions = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:,.6f}</td></tr>" for k, v in (broker.get("pos") or {}).items()) or "<tr><td colspan=2 class='muted'>flat</td></tr>"
    sleeves = "".join(f"<tr><td>{html.escape(k)}</td><td>{100 * v:.1f}%</td><td>{port.get('sleeve_scales', {}).get(k, 1):.2f}x</td></tr>" for k, v in (port.get("sleeve_capital") or {}).items())
    cash = port.get("cash")
    alist = "".join(f"<li><span class='muted'>{html.escape(a['t'][:16])}</span> <b>{html.escape(a['kind'])}</b> {html.escape(a['text'])}</li>" for a in reversed(alerts))
    events = "".join(f"<li><span class='muted'>{html.escape(str(e.get('t', ''))[:16])}</span> {html.escape(str(e.get('kind')))}: {html.escape(str(e.get('reason', '')))}</li>" for e in (risk.get("events") or [])[-15:])
    sl = ""
    if sleeve_status:
        sl = "<h2>Prediction sleeve (Polymarket, paper)</h2><pre class='sub'>" + html.escape(json.dumps(sleeve_status, indent=1, default=str)) + "</pre>"
    lock = risk.get("locked_until")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>iTRADER online</title><style>{CSS}</style></head><body>"
            f"<h1>iTRADER online: paper</h1><p class='sub'>mode {html.escape(str(s.general.mode))}, settings {s.hash}, cycles {state.get('cycles', 0)}, "
            f"last equity {state.get('last_equity') or 0:,.2f}, WAIT rate {100 * wait_rate:.0f}%, cycles with a risk adjustment {100 * blocks:.0f}%, fees paid {fees:,.2f}."
            f"{html.escape(maker_line)} Regime: {html.escape(str(regime))}. "
            f"<b class='{'bad' if lock else 'ok'}'>{'LOCKED: ' + html.escape(str(risk.get('lock_reason'))) if lock else 'risk engine unlocked'}</b>"
            f"{' | cooldown until ' + html.escape(str(risk.get('cooldown_until'))) if risk.get('cooldown_until') else ''}</p>"
            f"{equity_svg(eq) if len(eq) > 1 else '<p class=muted>no equity history yet</p>'}"
            f"<div class='card'><h2>Capital by sleeve (latest cycle)</h2><table><tr><th>sleeve</th><th>capital</th><th>vol scale</th></tr>{sleeves}"
            f"<tr><td>CASH (WAIT)</td><td>{(100 * cash) if cash is not None else 0:.1f}%</td><td></td></tr></table>"
            f"<p class='sub'>{'<br>'.join(html.escape(n) for n in ((port.get('allocator') or {}).get('notes') or []))}</p></div>"
            f"<div class='card'><h2>Positions</h2><table><tr><th>leg</th><th>qty</th></tr>{positions}</table></div>"
            f"<div class='card'><h2>Last 40 cycles</h2><table><tr><th>time</th><th>equity</th><th>return</th><th>selected</th><th>orders</th><th>risk</th></tr>{rows}</table></div>"
            f"<div class='card'><h2>Risk events</h2><ul>{events or '<li class=muted>none</li>'}</ul></div>"
            f"<div class='card'><h2>Alerts</h2><ul>{alist or '<li class=muted>none</li>'}</ul></div>{sl}</body></html>")


def write_dashboard(out_dir: Path, s: Optional[Settings] = None, sleeve_status: Optional[Dict[str, Any]] = None) -> Path:
    s = s or default_settings()
    p = s.resolve(s.runner.dashboard)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_dashboard(Path(out_dir), s, sleeve_status))
    return p
