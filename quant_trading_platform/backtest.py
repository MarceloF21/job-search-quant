"""Daily, long-only portfolio backtester.

Rules, all chosen to avoid flattering results:
- Decisions use the close of day t and are executed at the **open of day t+1**.
- Every fill pays ``cost_bps`` (commission + slippage) on top of the price.
- One shared account: each new position gets ``position_size_pct`` of the
  previous close's equity, capped by available cash and ``max_positions``.
- Stop-loss and take-profit are checked against each day's low and high.
  A gap through the level fills at the open; if both are hit the same day,
  the stop is assumed to come first (the conservative choice).
- After a stop or take-profit, a ticker re-enters only after its signal resets to 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .data import Prices

TRADE_COLUMNS = [
    "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
    "shares", "pnl", "return_pct", "holding_days", "exit_reason",
]


@dataclass
class BacktestResult:
    strategy: str
    equity: pd.Series
    exposure: pd.Series
    trades: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=TRADE_COLUMNS))


def shared_dates(prices: Prices) -> pd.DatetimeIndex:
    dates = None
    for frame in prices.values():
        dates = frame.index if dates is None else dates.intersection(frame.index)
    if dates is None or len(dates) < 2:
        raise ValueError("Need at least two dates shared by every ticker.")
    return pd.DatetimeIndex(dates).sort_values()


def run_backtest(
    prices: Prices,
    targets: dict[str, pd.Series],
    config: BacktestConfig | None = None,
    name: str = "strategy",
) -> BacktestResult:
    """Simulate one portfolio trading every ticker's target positions."""
    config = config or BacktestConfig()
    tickers = sorted(prices)
    dates = shared_dates(prices)
    o, h, lo, c = (
        np.column_stack([prices[t][col].reindex(dates).to_numpy(float) for t in tickers])
        for col in ("open", "high", "low", "close")
    )
    if np.isnan(np.stack([o, h, lo, c])).any():
        raise ValueError("Prices contain gaps; run align_prices() first.")
    target = np.column_stack([
        targets.get(t, pd.Series(0, index=dates)).reindex(dates).fillna(0).to_numpy(int)
        for t in tickers
    ])

    cost = config.cost_bps / 10_000
    n_days, n_tickers = c.shape
    cash = config.initial_capital
    shares = np.zeros(n_tickers, dtype=np.int64)
    entry_price = np.zeros(n_tickers)
    entry_day = np.full(n_tickers, -1)
    blocked = np.zeros(n_tickers, dtype=bool)
    equity = np.empty(n_days)
    exposure = np.empty(n_days)
    prev_equity = config.initial_capital
    trades: list[dict] = []

    def record(j: int, day: int, exit_px: float, reason: str) -> None:
        trades.append({
            "ticker": tickers[j],
            "entry_date": dates[entry_day[j]],
            "exit_date": dates[day],
            "entry_price": entry_price[j],
            "exit_price": exit_px,
            "shares": int(shares[j]),
            "pnl": float(shares[j] * (exit_px - entry_price[j])),
            "return_pct": exit_px / entry_price[j] - 1,
            "holding_days": (dates[day] - dates[entry_day[j]]).days,
            "exit_reason": reason,
        })

    def sell(j: int, day: int, raw_price: float, reason: str) -> None:
        nonlocal cash
        fill = raw_price * (1 - cost)
        cash += shares[j] * fill
        record(j, day, fill, reason)
        shares[j] = 0

    for day in range(n_days):
        if day > 0:
            order = target[day - 1]  # decided at yesterday's close, executed at today's open
            for j in np.flatnonzero((shares > 0) & (order == 0)):
                sell(j, day, o[day, j], "Signal")
            held = int((shares > 0).sum())
            for j in np.flatnonzero((shares == 0) & (order == 1) & ~blocked):
                if held >= config.max_positions:
                    break
                fill = o[day, j] * (1 + cost)
                qty = math.floor(min(prev_equity * config.position_size_pct, cash) / fill)
                if qty < 1:
                    continue
                cash -= qty * fill
                shares[j], entry_price[j], entry_day[j] = qty, fill, day
                held += 1

        for j in np.flatnonzero(shares > 0):
            stop = entry_price[j] * (1 - config.stop_loss_pct)
            take = entry_price[j] * (1 + config.take_profit_pct)
            if lo[day, j] <= stop:
                sell(j, day, min(o[day, j], stop), "Stop Loss")
                blocked[j] = True
            elif h[day, j] >= take:
                sell(j, day, max(o[day, j], take), "Take Profit")
                blocked[j] = True

        holdings = float(shares @ c[day])
        equity[day] = cash + holdings
        exposure[day] = holdings / equity[day] if equity[day] > 0 else 0.0
        prev_equity = equity[day]
        blocked &= target[day] != 0

    for j in np.flatnonzero(shares > 0):  # mark positions still open, without selling
        record(j, n_days - 1, c[-1, j], "Open at end")

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return BacktestResult(
        strategy=name,
        equity=pd.Series(equity, index=dates, name=name),
        exposure=pd.Series(exposure, index=dates, name=name),
        trades=trades_df,
    )


def buy_and_hold(prices: Prices, config: BacktestConfig | None = None, name: str = "Buy & hold") -> pd.Series:
    """Equal-weight buy-and-hold of every ticker, bought at the first open with costs."""
    config = config or BacktestConfig()
    tickers = sorted(prices)
    dates = shared_dates(prices)
    per_ticker = config.initial_capital / len(tickers)
    cash = 0.0
    value = pd.Series(0.0, index=dates)
    for t in tickers:
        fill = prices[t]["open"].reindex(dates).iloc[0] * (1 + config.cost_bps / 10_000)
        qty = math.floor(per_ticker / fill)
        cash += per_ticker - qty * fill
        value += qty * prices[t]["close"].reindex(dates)
    return (value + cash).rename(name)
