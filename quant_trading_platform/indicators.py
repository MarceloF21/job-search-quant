"""Technical indicators. Every value at date t uses only prices up to and including t."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_indicators(close: pd.Series) -> pd.DataFrame:
    """Return moving averages, RSI, MACD, Bollinger Bands, volatility and momentum."""
    ind = pd.DataFrame(index=close.index)
    ind["returns"] = close.pct_change()

    for window in (5, 10, 20, 50, 200):
        ind[f"SMA_{window}"] = close.rolling(window).mean()
        ind[f"EMA_{window}"] = close.ewm(span=window, adjust=False).mean()

    # RSI(14) with simple moving averages of gains and losses
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    ind["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ind.loc[(loss == 0) & (gain > 0), "RSI"] = 100.0

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    ind["MACD"] = ema_12 - ema_26
    ind["MACD_signal"] = ind["MACD"].ewm(span=9, adjust=False).mean()
    ind["MACD_histogram"] = ind["MACD"] - ind["MACD_signal"]

    ind["BB_middle"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    ind["BB_upper"] = ind["BB_middle"] + 2 * bb_std
    ind["BB_lower"] = ind["BB_middle"] - 2 * bb_std
    band_width = (ind["BB_upper"] - ind["BB_lower"]).replace(0, np.nan)
    ind["BB_position"] = (close - ind["BB_lower"]) / band_width

    ind["volatility_20"] = ind["returns"].rolling(20).std() * np.sqrt(252)

    for window in (5, 10, 20):
        ind[f"momentum_{window}"] = close.pct_change(window)

    return ind
