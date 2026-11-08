"""Performance metrics computed from the daily equity curve (not from per-trade returns)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR as DAYS


def performance_metrics(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    exposure: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Return CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar and trade statistics.

    Sharpe and Sortino annualize *daily* portfolio returns with sqrt(252). Annualizing
    per-trade returns that way (as version 1 did) overstates Sharpe several times over.
    """
    equity = equity.dropna()
    returns = equity.pct_change().dropna()
    years = len(returns) / DAYS
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else math.nan

    excess = returns - ((1 + risk_free_rate) ** (1 / DAYS) - 1)
    std = returns.std(ddof=1)
    downside = math.sqrt(float((np.minimum(excess, 0) ** 2).mean())) if len(excess) else 0.0
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())

    metrics = {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "volatility": float(std * math.sqrt(DAYS)) if len(returns) > 1 else math.nan,
        "sharpe": float(excess.mean() / std * math.sqrt(DAYS)) if std > 0 else math.nan,
        "sortino": float(excess.mean() / downside * math.sqrt(DAYS)) if downside > 0 else math.nan,
        "max_drawdown": max_drawdown,
        "calmar": float(cagr / abs(max_drawdown)) if max_drawdown < 0 else math.nan,
        "final_equity": float(equity.iloc[-1]),
    }
    if exposure is not None and len(exposure):
        metrics["avg_exposure"] = float(exposure.mean())
    if trades is not None:
        metrics.update(trade_statistics(trades))
    return metrics


def trade_statistics(trades: pd.DataFrame) -> dict[str, float]:
    closed = trades[trades["exit_reason"] != "Open at end"] if len(trades) else trades
    if closed.empty:
        return {"trades": 0, "win_rate": math.nan, "profit_factor": math.nan,
                "avg_trade_return": math.nan, "avg_holding_days": math.nan}
    gains = closed.loc[closed["pnl"] > 0, "pnl"].sum()
    losses = -closed.loc[closed["pnl"] < 0, "pnl"].sum()
    return {
        "trades": int(len(closed)),
        "win_rate": float((closed["pnl"] > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else math.inf,
        "avg_trade_return": float(closed["return_pct"].mean()),
        "avg_holding_days": float(closed["holding_days"].mean()),
    }
