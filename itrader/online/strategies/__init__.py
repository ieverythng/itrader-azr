"""The strategy library and its registry.

Every strategy is a `Strategy` subclass with a unique `name` and an ONLINE_FAMILIES id. The
registry is the single place the bench, the framer and the CLI look strategies up, so adding
one is: write the class, import it here, add it to REGISTRY. Nothing else changes.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Type

from .base import ANN, Market, Strategy, clean, inverse_vol_weights  # noqa: F401
from .carry import FundingCarry
from .experimental import Experimental
from .momentum import CrossSectionalMomentum
from .prediction_sleeve import PredictionSleeve
from .reversion import MeanReversion, Pairs
from .seasonality import Seasonality
from .trend import TrendModern, Turtle

REGISTRY: Dict[str, Type[Strategy]] = {
    cls.name: cls
    for cls in (
        Turtle,
        TrendModern,
        CrossSectionalMomentum,
        MeanReversion,
        FundingCarry,
        Pairs,
        Seasonality,
        PredictionSleeve,
        Experimental,
    )
}

# Strategies that trade the crypto panel and can be benched on candles. The prediction sleeve
# trades Polymarket (paper only) and reports through its own kit; it is excluded from the bench.
BENCHABLE: tuple = tuple(n for n in REGISTRY if n != PredictionSleeve.name)


def make(name: str, **params) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


def all_strategies(params_by_name: Optional[Dict[str, dict]] = None, names: Optional[Iterable[str]] = None) -> List[Strategy]:
    params_by_name = params_by_name or {}
    names = list(names) if names is not None else list(REGISTRY)
    return [make(n, **params_by_name.get(n, {})) for n in names]


def by_bar(strategies: Iterable[Strategy]) -> Dict[str, List[Strategy]]:
    out: Dict[str, List[Strategy]] = {}
    for s in strategies:
        out.setdefault(s.bar, []).append(s)
    return out


__all__ = [
    "ANN", "Market", "Strategy", "clean", "inverse_vol_weights", "REGISTRY", "BENCHABLE", "make", "all_strategies", "by_bar",
    "Turtle", "TrendModern", "CrossSectionalMomentum", "MeanReversion", "FundingCarry", "Pairs", "Seasonality",
    "PredictionSleeve", "Experimental",
]
