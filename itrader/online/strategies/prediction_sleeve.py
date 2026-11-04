"""The prediction-market sleeve: Roman's `polyedge` kit (Polymarket), paper only.

This sleeve does not trade the crypto panel, so `targets()` is always flat; the bench skips it.
What it does is report. If the polyedge kit is on this machine (POLYEDGE_ROOT, the `polyedge_root`
param, or the usual places) it reads the reward farm's paper logs and summarises them in the
same shape as every other strategy, so the dashboard and the Workbench allocator can see it.
If the kit is absent it says so and reports nothing invented.

Why paper only: Polymarket's Terms restrict the country the operator lives in and ban VPN use. This module never imports
or calls polyedge's live venue and never asks for keys. The evidence so far (2 and 3 Nov):
makers earn, takers lose, and a 20-second loop cannot dodge a news crash, so even the paper
record must be read with the gap risk in mind.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import Market, Strategy

_DEFAULT_ROOTS = (
    "~/polyedge",
    "~/Downloads/polyedge",
    "~/Documents/polyedge",
    "~/Documents/GitHub/polyedge",
)


def find_polyedge_root(explicit: Optional[str] = None) -> Optional[Path]:
    cands: List[str] = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("POLYEDGE_ROOT"):
        cands.append(os.environ["POLYEDGE_ROOT"])
    cands.extend(_DEFAULT_ROOTS)
    for c in cands:
        p = Path(c).expanduser()
        if (p / "polyedge" / "farm.py").exists():
            return p
    return None


def _read_jsonl(p: Path, limit: Optional[int] = None) -> List[dict]:
    if not p.exists():
        return []
    rows: List[dict] = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:] if limit else rows


class PredictionSleeve(Strategy):
    name = "prediction_sleeve"
    family_id = "17"
    bar = "5m"
    instrument = "polymarket_paper"
    description = "Polymarket liquidity-reward farm via the polyedge kit. Paper only. Reports, never trades here."
    params = {"polyedge_root": None, "mode": "paper"}
    n_variants_tried = 1

    def targets(self, m: Market) -> pd.DataFrame:
        # Flat on the crypto panel by construction: this sleeve's book lives in polyedge.
        return pd.DataFrame(0.0, index=m.close.index, columns=m.close.columns)

    # ------------------------------------------------------------------ reporting
    @property
    def root(self) -> Optional[Path]:
        return find_polyedge_root(self.params.get("polyedge_root"))

    def status(self) -> Dict[str, Any]:
        """A plain summary of the farm's paper run, or a clear 'not available'."""
        root = self.root
        base: Dict[str, Any] = {"strategy": self.name, "mode": "paper", "available": False, "live_allowed": False,
                                "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if root is None:
            base["reason"] = "polyedge kit not found (set POLYEDGE_ROOT)"
            return base
        out_dir = root / "data" / "farm" / "paper"
        cycles = _read_jsonl(out_dir / "cycles.jsonl")
        base.update({"available": True, "root": str(root), "cycles": len(cycles)})
        if not cycles:
            base["reason"] = "no paper cycles logged yet (run: python3 -m polyedge farm paper --capital 1000)"
            return base
        a, b = cycles[0], cycles[-1]
        hours = max((b["t"] - a["t"]) / 3600.0, 1e-9)
        start_eq = float(a.get("equity", 0.0)) or 1.0
        reward = float(b.get("reward", 0.0))
        spread_pnl = float(b.get("spread_pnl", 0.0))
        equity = float(b.get("equity", start_eq))
        marks = equity - start_eq - reward - spread_pnl
        fills = _read_jsonl(out_dir / "fills.jsonl")
        base.update({
            "hours": round(hours, 2),
            "start_equity": start_eq,
            "equity": equity,
            "reward_accrued": reward,
            "spread_pnl": spread_pnl,
            "inventory_marks": marks,
            "return_pct": 100.0 * (equity / start_eq - 1.0),
            "reward_per_day_pct_of_capital": 100.0 * reward / start_eq / hours * 24.0,
            "fills": len(fills),
            "markets": int(b.get("markets", 0)),
            "resting_orders": int(b.get("resting", 0)),
            "stopped": b.get("stopped") or None,
            "last_cycle": datetime.fromtimestamp(b["t"], tz=timezone.utc).isoformat(timespec="seconds"),
        })
        return base

    def report(self) -> str:
        """polyedge's own text report if the kit imports cleanly, else our summary."""
        root = self.root
        if root is None:
            return "prediction_sleeve: polyedge kit not found; nothing to report."
        try:
            import sys

            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from polyedge import farm as fm  # type: ignore

            return str(fm.report(root / "data" / "farm" / "paper"))
        except Exception as exc:  # noqa: BLE001
            s = self.status()
            return f"prediction_sleeve (fallback summary, polyedge import failed: {type(exc).__name__}): {json.dumps(s, indent=1)}"

    def describe(self) -> Dict[str, object]:
        d = super().describe()
        d["polyedge_root"] = str(self.root) if self.root else None
        d["live_allowed"] = False
        return d
