from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


def _finite(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Episode evidence must be finite, got {value!r}")
    return value


@dataclass(frozen=True)
class EpisodeEvidence:
    """Versioned terminal evidence emitted by ``ITraderEnv``.

    This contract derives economic arithmetic independently from the optimized
    objective. A task may explicitly select the matching ``net_return`` reward,
    but evidence never infers costs from reward components and does not encode
    broker execution semantics.
    """

    schema_version: int
    optimization_reward: float
    gross_return: float
    cost_return: float
    net_return: float
    turnover: float
    liquidation_turnover: float
    max_drawdown: float
    gross_sharpe_per_sqrt_bar: float
    net_sharpe_per_sqrt_bar: float
    annualized_net_sharpe: float | None
    positive_net_bar_fraction: float
    gross_exposure_fraction: float
    position_changes: int
    evidence_cost_bps: float
    success: bool
    family_id: str
    price_id: str
    reward_id: str
    horizon: int
    task_seed: int
    symbol: str
    split: str
    start_idx: int
    end_idx: int
    steps: int

    @classmethod
    def build(
        cls,
        *,
        optimization_reward: float,
        gross_return: float,
        turnover: float,
        liquidation_turnover: float,
        max_drawdown: float,
        evidence_cost_bps: float,
        evidence_periods_per_year: float,
        strategy_returns: list[float],
        turnover_steps: list[float],
        gross_exposure_sum: float,
        position_changes: int,
        task: Any,
        symbol: str,
        split: str,
        start_idx: int,
        end_idx: int,
        steps: int,
    ) -> "EpisodeEvidence":
        gross = _finite(gross_return)
        liquidation = max(0.0, _finite(liquidation_turnover))
        total_turnover = max(0.0, _finite(turnover)) + liquidation
        cost_bps = max(0.0, _finite(evidence_cost_bps))
        cost = total_turnover * cost_bps * 1e-4
        net = gross - cost
        gross_steps = [_finite(value) for value in strategy_returns]
        step_turnover = [max(0.0, _finite(value)) for value in turnover_steps]
        if len(gross_steps) != len(step_turnover):
            raise ValueError("strategy_returns and turnover_steps must have equal length")
        net_steps = [
            value - step_turnover[index] * cost_bps * 1e-4
            for index, value in enumerate(gross_steps)
        ]
        if net_steps:
            net_steps[-1] -= liquidation * cost_bps * 1e-4

        def sharpe(values: list[float]) -> float:
            if len(values) < 2:
                return 0.0
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            std = math.sqrt(max(0.0, variance))
            return 0.0 if std <= 1e-12 else mean / std

        gross_sharpe = sharpe(gross_steps)
        net_sharpe = sharpe(net_steps)
        periods = max(0.0, _finite(evidence_periods_per_year))
        annualized_net_sharpe = net_sharpe * math.sqrt(periods) if periods > 0.0 else None
        positive_net_bar_fraction = (
            sum(value > 0.0 for value in net_steps) / len(net_steps)
            if net_steps
            else 0.0
        )
        step_count = max(1, int(steps))
        return cls(
            schema_version=1,
            optimization_reward=_finite(optimization_reward),
            gross_return=gross,
            cost_return=cost,
            net_return=net,
            turnover=total_turnover,
            liquidation_turnover=liquidation,
            max_drawdown=max(0.0, _finite(max_drawdown)),
            gross_sharpe_per_sqrt_bar=gross_sharpe,
            net_sharpe_per_sqrt_bar=net_sharpe,
            annualized_net_sharpe=annualized_net_sharpe,
            positive_net_bar_fraction=positive_net_bar_fraction,
            gross_exposure_fraction=max(0.0, _finite(gross_exposure_sum)) / step_count,
            position_changes=max(0, int(position_changes)),
            evidence_cost_bps=cost_bps,
            success=bool(net > 0.0),
            family_id=str(task.family_id),
            price_id=str(task.price_id),
            reward_id=str(task.reward_id),
            horizon=int(task.horizon),
            task_seed=int(task.seed),
            symbol=str(symbol),
            split=str(split),
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            steps=int(steps),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
