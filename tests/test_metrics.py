import math

import numpy as np
import pandas as pd
import pytest

from quant_trading_platform.metrics import performance_metrics, trade_statistics


def series(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_max_drawdown():
    assert performance_metrics(series([100, 120, 90, 130]))["max_drawdown"] == pytest.approx(-0.25)


def test_cagr_for_one_year_of_doubling():
    equity = series(np.linspace(100, 200, 253))  # 252 daily returns = one trading year
    assert performance_metrics(equity)["cagr"] == pytest.approx(1.0)


def test_sharpe_uses_daily_returns():
    returns = np.array([0.01, -0.005] * 126)
    equity = series(100 * np.cumprod(np.concatenate(([1.0], 1 + returns))))
    expected = returns.mean() / returns.std(ddof=1) * math.sqrt(252)
    assert performance_metrics(equity)["sharpe"] == pytest.approx(expected)


def test_trade_statistics_ignore_open_positions():
    trades = pd.DataFrame({
        "pnl": [100.0, -50.0, 999.0],
        "return_pct": [0.02, -0.01, 0.5],
        "holding_days": [5, 3, 1],
        "exit_reason": ["Signal", "Stop Loss", "Open at end"],
    })
    stats = trade_statistics(trades)
    assert stats["trades"] == 2
    assert stats["win_rate"] == pytest.approx(0.5)
    assert stats["profit_factor"] == pytest.approx(2.0)
