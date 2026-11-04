"""The default trading universe: the largest, most liquid USDT crypto pairs.

Liquidity matters more than breadth. Each of these trades hundreds of millions a day on
Binance, so the fee and slippage assumptions in the bench (settings.toml) hold for the sizes
this system will run at.
"""
from __future__ import annotations

DEFAULT_UNIVERSE = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
)

# Pairs with a long record of moving together (for the pairs strategy).
DEFAULT_PAIRS = (
    ("BTCUSDT", "ETHUSDT"),
    ("ETHUSDT", "SOLUSDT"),
    ("LTCUSDT", "BTCUSDT"),
    ("ADAUSDT", "XRPUSDT"),
)


SPOT_SUFFIX = ".SPOT"   # book legs: "BTCUSDT" is the perp, "BTCUSDT.SPOT" the spot holding


def leg_symbol(symbol: str) -> str:
    """The market-data symbol behind a book leg ("BTCUSDT.SPOT" -> "BTCUSDT")."""
    return symbol[: -len(SPOT_SUFFIX)] if symbol.endswith(SPOT_SUFFIX) else symbol


def is_spot(symbol: str) -> bool:
    return symbol.endswith(SPOT_SUFFIX)


def base_asset(symbol: str) -> str:
    symbol = leg_symbol(symbol)
    for quote in ("USDT", "USDC", "USD", "BUSD"):
        if symbol.endswith(quote):
            return symbol[: -len(quote)]
    return symbol
