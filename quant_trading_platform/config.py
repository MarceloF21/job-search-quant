"""Settings shared by every backtest."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TICKERS: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMZN", "JPM", "GS", "JNJ",
    "PFE", "XOM", "CVX", "WMT", "SPY", "QQQ", "GLD",
)
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestConfig:
    """Portfolio and execution settings.

    One account holds every position, so results are for a single portfolio
    of ``initial_capital``, not one account per ticker.
    """

    initial_capital: float = 100_000.0
    position_size_pct: float = 0.10  # share of current equity put into each new position
    max_positions: int = 10
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    cost_bps: float = 5.0  # commission + slippage per side, in basis points (5 bps = 0.05%)
    risk_free_rate: float = 0.0  # annual rate used in Sharpe and Sortino

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 < self.position_size_pct <= 1:
            raise ValueError("position_size_pct must be in (0, 1]")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")
        if not 0 < self.stop_loss_pct < 1:
            raise ValueError("stop_loss_pct must be in (0, 1)")
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")
