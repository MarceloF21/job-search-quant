"""Run every strategy plus buy-and-hold benchmarks on one set of prices."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .backtest import BacktestResult, buy_and_hold, run_backtest
from .config import BacktestConfig
from .data import Prices
from .metrics import performance_metrics
from .strategies import MLStrategy, Strategy, default_strategies


@dataclass
class RunOutput:
    results: dict[str, BacktestResult]
    benchmarks: dict[str, pd.Series]
    metrics: dict[str, dict[str, float]]
    ml_diagnostics: dict[str, float] = field(default_factory=dict)


def run_all(
    prices: Prices,
    config: BacktestConfig | None = None,
    strategies: list[Strategy] | None = None,
    benchmark: str | None = "SPY",
) -> RunOutput:
    config = config or BacktestConfig()
    strategies = strategies if strategies is not None else default_strategies()
    results: dict[str, BacktestResult] = {}
    metrics: dict[str, dict[str, float]] = {}
    ml_diag: dict[str, float] = {}

    for strategy in strategies:
        targets, ml_frames = {}, []
        for ticker, frame in prices.items():
            targets[ticker] = strategy.target_positions(frame)
            if isinstance(strategy, MLStrategy):
                ml_frames.append(pd.DataFrame({
                    "prob": strategy.last_probabilities, "label": strategy.last_labels,
                }))
        result = run_backtest(prices, targets, config, strategy.name)
        results[strategy.name] = result
        metrics[strategy.name] = performance_metrics(
            result.equity, result.trades, result.exposure, config.risk_free_rate
        )
        if ml_frames:
            ml_diag = ml_accuracy(pd.concat(ml_frames))

    benchmarks: dict[str, pd.Series] = {}
    if benchmark and benchmark in prices:
        benchmarks[f"{benchmark} buy & hold"] = buy_and_hold({benchmark: prices[benchmark]}, config)
    benchmarks["Equal-weight buy & hold"] = buy_and_hold(prices, config)
    for label, equity in benchmarks.items():
        metrics[label] = performance_metrics(equity, risk_free_rate=config.risk_free_rate)

    return RunOutput(results, benchmarks, metrics, ml_diag)


def ml_accuracy(frame: pd.DataFrame) -> dict[str, float]:
    """Out-of-sample hit rate of the walk-forward model, next to the naive 'always up' baseline."""
    scored = frame.dropna()
    if scored.empty:
        return {}
    predicted_up = scored["prob"] > 0.5
    up_share = float(scored["label"].mean())
    return {
        "predictions": int(len(scored)),
        "accuracy": float((predicted_up == scored["label"].astype(bool)).mean()),
        "always_up_accuracy": up_share,
    }
