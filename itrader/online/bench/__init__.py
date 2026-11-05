"""The bench: one honest backtester for every strategy, the metrics that judge them, the
capability profiles the online loop consumes, and the promotion ladder.

    targets (weights) -> engine.backtest -> metrics.summarize -> walkforward -> profile -> report
"""

from .engine import BacktestResult, CostModel, backtest, backtest_maker, funding_per_bar, run_style  # noqa: F401
from .metrics import (  # noqa: F401
    binary_evidence,
    capacity_usd,
    deflated_sharpe,
    evidence_meter,
    periods_per_year,
    probabilistic_sharpe,
    summarize,
)
from .profile import capability_profile, promotion_stage  # noqa: F401
from .walkforward import split_points, walk_forward  # noqa: F401
