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


def base_asset(symbol: str) -> str:
    for quote in ("USDT", "USDC", "USD", "BUSD"):
        if symbol.endswith(quote):
            return symbol[: -len(quote)]
    return symbol
