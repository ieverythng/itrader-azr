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


def is_crypto(symbol: str) -> bool:
    """True for the crypto book legs; False for listed instruments such as ETFs."""
    return leg_symbol(symbol).endswith(("USDT", "USDC", "BUSD"))


def base_asset(symbol: str) -> str:
    symbol = leg_symbol(symbol)
    for quote in ("USDT", "USDC", "USD", "BUSD"):
        if symbol.endswith(quote):
            return symbol[: -len(quote)]
    return symbol

# The wider crypto universe: the ten majors plus fifteen more USDT pairs that were already listed, spot and
# perpetual, before November 2022 and still trade. Chosen by age and liquidity, not by performance; it is still
# a list of survivors (coins that died are not in it), which flatters any long-biased result a little.
WIDE_UNIVERSE = DEFAULT_UNIVERSE + (
    "DOTUSDT", "TRXUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT", "XLMUSDT", "BCHUSDT", "NEARUSDT",
    "FILUSDT", "AAVEUSDT", "ALGOUSDT", "ICPUSDT", "SANDUSDT", "APTUSDT", "HBARUSDT",
)

# Multi-asset universe for the macro trend sleeve: liquid US-listed ETFs, one or more per asset class, all with
# history back to 2007 or earlier. Breadth across things that do not move together is what trend following needs.
MACRO_UNIVERSE = (
    "SPY", "QQQ", "IWM", "EFA", "EEM",          # equities: US large, US tech, US small, developed ex-US, emerging
    "TLT", "IEF", "LQD", "HYG", "TIP",          # bonds: long and medium Treasuries, investment grade, high yield, inflation-linked
    "GLD", "SLV", "USO", "DBA", "DBC",          # commodities: gold, silver, oil, agriculture, broad basket
    "UUP", "FXE", "FXY", "FXB",                 # currencies: dollar index, euro, yen, sterling
    "VNQ",                                      # listed real estate
)
