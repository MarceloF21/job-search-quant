import math

import pandas as pd
import pytest

from quant_trading_platform.backtest import buy_and_hold, run_backtest
from quant_trading_platform.config import BacktestConfig

from .conftest import make_bars

NO_RISK_EXITS = dict(stop_loss_pct=0.99, take_profit_pct=100.0)


def targets_for(bars, values):
    return {"TEST": pd.Series(values, index=bars.index)}


def test_trades_at_next_open_and_books_exact_pnl():
    bars = make_bars(opens=[100, 101, 102, 104, 108, 110], closes=[100, 101, 103, 105, 109, 111])
    config = BacktestConfig(cost_bps=0, **NO_RISK_EXITS)
    result = run_backtest({"TEST": bars}, targets_for(bars, [0, 1, 1, 0, 0, 0]), config)

    trade = result.trades.iloc[0]
    assert trade["entry_date"] == bars.index[2]  # signal at close of day 1 -> buy at open of day 2
    assert trade["entry_price"] == 102
    assert trade["exit_date"] == bars.index[4]  # exit signal at close of day 3 -> sell at day 4 open
    assert trade["exit_price"] == 108
    shares = math.floor(100_000 * 0.10 / 102)
    assert trade["shares"] == shares
    assert trade["pnl"] == pytest.approx(shares * (108 - 102))
    assert result.equity.iloc[-1] == pytest.approx(100_000 + shares * 6)


def test_costs_are_charged_on_both_sides():
    bars = make_bars(opens=[100, 100, 100, 100, 100])
    config = BacktestConfig(cost_bps=10, **NO_RISK_EXITS)
    result = run_backtest({"TEST": bars}, targets_for(bars, [1, 0, 0, 0, 0]), config)
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(100 * 1.001)
    assert trade["exit_price"] == pytest.approx(100 * 0.999)
    assert trade["pnl"] < 0
    assert result.equity.iloc[-1] == pytest.approx(100_000 + trade["pnl"])


def test_stop_loss_fills_at_stop_price_or_at_a_gap_open():
    # Day 2: low touches the stop (95) while the open is above it -> fill at 95.
    bars = make_bars(opens=[100, 100, 99, 99], highs=[100, 100, 99, 99],
                     lows=[100, 100, 94, 99], closes=[100, 100, 96, 99])
    config = BacktestConfig(cost_bps=0, stop_loss_pct=0.05, take_profit_pct=1.0)
    trade = run_backtest({"TEST": bars}, targets_for(bars, [1, 1, 1, 1]), config).trades.iloc[0]
    assert trade["exit_reason"] == "Stop Loss"
    assert trade["exit_price"] == pytest.approx(95)

    # Day 2 opens at 90, already through the stop -> fill at the open.
    gap = make_bars(opens=[100, 100, 90, 90], closes=[100, 100, 91, 90])
    trade = run_backtest({"TEST": gap}, targets_for(gap, [1, 1, 1, 1]), config).trades.iloc[0]
    assert trade["exit_price"] == pytest.approx(90)


def test_take_profit_and_no_reentry_until_signal_resets():
    bars = make_bars(opens=[100, 100, 100, 100, 100, 100, 100],
                     highs=[100, 100, 120, 100, 100, 100, 100])
    config = BacktestConfig(cost_bps=0, stop_loss_pct=0.05, take_profit_pct=0.15)
    # Target stays 1 after the take-profit: no new trade until it drops to 0 and comes back.
    result = run_backtest({"TEST": bars}, targets_for(bars, [1, 1, 1, 1, 0, 1, 1]), config)
    first = result.trades.iloc[0]
    assert first["exit_reason"] == "Take Profit"
    assert first["exit_price"] == pytest.approx(115)
    assert result.trades.iloc[1]["entry_date"] == bars.index[6]


def test_one_shared_account_and_position_limit():
    bars = make_bars(opens=[100] * 6)
    prices = {t: bars for t in ("A", "B", "C")}
    config = BacktestConfig(cost_bps=0, max_positions=2, position_size_pct=0.5, **NO_RISK_EXITS)
    targets = {t: pd.Series(1, index=bars.index) for t in prices}
    result = run_backtest(prices, targets, config)
    assert set(result.trades["ticker"]) == {"A", "B"}  # C never fits under the 2-position cap
    assert result.exposure.iloc[-1] == pytest.approx(1.0)
    assert result.equity.iloc[-1] == pytest.approx(100_000)


def test_buy_and_hold_tracks_prices_after_costs():
    bars = make_bars(opens=[100, 110, 120], closes=[100, 110, 120])
    equity = buy_and_hold({"T": bars}, BacktestConfig(cost_bps=0))
    assert equity.iloc[-1] == pytest.approx(120_000)
