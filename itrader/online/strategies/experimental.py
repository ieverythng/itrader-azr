"""The experimental slot: new ideas and the repo's RL solver, judged on the same bench as the rest.

Two ways to fill the slot, both deterministic at the shell:

1. `entry="package.module:function"`: the function takes a `Market` and returns a weight frame
   (same contract as every strategy). This is how a new idea gets benched without touching
   anything else.
2. `checkpoint="path/to/solver.pt"`: a repo solver checkpoint. The adapter below only confirms
   the file exists and whether torch imports. Wiring a trained solver's observation schema
   (window x features + task embedding) to this panel is the CTO's call, so until an `entry`
   is given the slot stays flat and says so in `describe()`. Nothing is faked.

With neither, the strategy is a no-op: zero weights, zero cost, zero claims.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from .base import Market, Strategy, clean


class SolverAdapter:
    """Loads nothing heavy unless asked. `available` is True only when a usable entry exists."""

    def __init__(self, entry: Optional[str] = None, checkpoint: Optional[str] = None) -> None:
        self.entry = entry
        self.checkpoint = Path(checkpoint).expanduser() if checkpoint else None
        self.fn: Optional[Callable[[Market], pd.DataFrame]] = None
        self.notes: list = []
        if entry:
            try:
                mod, _, name = entry.partition(":")
                self.fn = getattr(importlib.import_module(mod), name)
                self.notes.append(f"entry {entry} loaded")
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"entry {entry} failed: {type(exc).__name__}: {exc}")
        if self.checkpoint is not None:
            if self.checkpoint.exists():
                self.notes.append(f"checkpoint present: {self.checkpoint} ({self.checkpoint.stat().st_size} bytes)")
                try:
                    importlib.import_module("torch")
                    self.notes.append("torch importable")
                except Exception:  # noqa: BLE001
                    self.notes.append("torch not importable in this environment")
                if self.fn is None:
                    self.notes.append("no adapter entry given for the checkpoint; slot stays flat")
            else:
                self.notes.append(f"checkpoint missing: {self.checkpoint}")

    @property
    def available(self) -> bool:
        return self.fn is not None

    def predict(self, m: Market) -> pd.DataFrame:
        assert self.fn is not None
        w = self.fn(m)
        if not isinstance(w, pd.DataFrame):
            raise TypeError("experimental entry must return a DataFrame of weights")
        w = w.reindex(index=m.close.index, columns=m.close.columns).astype(float)
        return clean(w)


class Experimental(Strategy):
    name = "experimental"
    family_id = "18"
    bar = "1d"
    instrument = "perp"
    description = "Slot for new ideas or a repo solver via an adapter entry. Flat until one is wired."
    params = {"entry": None, "checkpoint": None, "max_gross": 1.0, "max_single": 0.35}
    n_variants_tried = 1

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.adapter = SolverAdapter(self.params.get("entry"), self.params.get("checkpoint"))

    def targets(self, m: Market) -> pd.DataFrame:
        if not self.adapter.available:
            return pd.DataFrame(0.0, index=m.close.index, columns=m.close.columns)
        w = self.adapter.predict(m).clip(-self.params["max_single"], self.params["max_single"])
        gross = w.abs().sum(axis=1)
        over = gross > self.params["max_gross"]
        if over.any():
            w.loc[over] = w.loc[over].div(gross[over], axis=0) * self.params["max_gross"]
        return clean(w)

    def describe(self) -> Dict[str, object]:
        d = super().describe()
        d["active"] = self.adapter.available
        d["adapter_notes"] = list(self.adapter.notes)
        return d
