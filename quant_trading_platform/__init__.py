"""Quantitative trading platform: rule-based and machine-learning strategies,
tested with a daily portfolio backtester that models costs and avoids look-ahead."""

from .backtest import BacktestResult, buy_and_hold, run_backtest
from .config import DEFAULT_TICKERS, BacktestConfig
from .data import align_prices, fetch_yahoo, generate_synthetic_prices, load_prices, save_prices
from .metrics import performance_metrics
from .pipeline import RunOutput, run_all
from .strategies import (
    MACDStrategy,
    MeanReversionStrategy,
    MLStrategy,
    MomentumStrategy,
    Strategy,
    default_strategies,
)

__version__ = "2.0.0"

__all__ = [
    "DEFAULT_TICKERS", "BacktestConfig", "BacktestResult", "MACDStrategy", "MLStrategy",
    "MeanReversionStrategy", "MomentumStrategy", "RunOutput", "Strategy", "align_prices",
    "buy_and_hold", "default_strategies", "fetch_yahoo", "generate_synthetic_prices",
    "load_prices", "performance_metrics", "run_all", "run_backtest", "save_prices",
]
