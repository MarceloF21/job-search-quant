"""Trading strategies.

Each strategy turns daily bars into a *target position* per day: 1 = hold the stock,
0 = stay out. The target for day t uses only information available at the close of
day t. The backtester trades it at the next day's open, so no strategy can see the future.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .indicators import compute_indicators


class Strategy(ABC):
    """Base class: subclasses emit entry (+1) and exit (-1) events; 0 means no new event."""

    name: str = "strategy"

    @abstractmethod
    def signals(self, data: pd.DataFrame) -> pd.Series:
        """Return +1 (enter), -1 (exit) or 0 (no change) for every date in ``data``."""

    def target_positions(self, data: pd.DataFrame) -> pd.Series:
        """Hold from an entry event until the next exit event."""
        events = self.signals(data).reindex(data.index).fillna(0)
        state = events.replace(0, np.nan).ffill().fillna(0).clip(lower=0)
        return state.astype(int).rename(self.name)


class MomentumStrategy(Strategy):
    """Trend following: buy strong uptrends that aren't overbought; exit when RSI > 75."""

    name = "Momentum_Trend"

    def signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        ind = compute_indicators(close)
        trend_up = (close > ind["SMA_50"]) & (ind["SMA_50"] > ind["SMA_200"])
        entry = trend_up & (ind["momentum_20"] > 0.02) & (ind["RSI"] < 70)
        exit_ = ind["RSI"] > 75
        return _events(entry, exit_)


class MeanReversionStrategy(Strategy):
    """Bollinger Band mean reversion: buy oversold at the lower band, exit overbought at the upper."""

    name = "MeanReversion_BB"

    def signals(self, data: pd.DataFrame) -> pd.Series:
        ind = compute_indicators(data["close"])
        entry = (ind["BB_position"] < 0.1) & (ind["RSI"] < 30)
        exit_ = (ind["BB_position"] > 0.9) & (ind["RSI"] > 70)
        return _events(entry, exit_)


class MACDStrategy(Strategy):
    """MACD crossover: buy when MACD crosses above its signal line, exit on the cross below."""

    name = "MACD_Crossover"

    def signals(self, data: pd.DataFrame) -> pd.Series:
        ind = compute_indicators(data["close"])
        macd, sig = ind["MACD"], ind["MACD_signal"]
        entry = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        exit_ = (macd < sig) & (macd.shift(1) >= sig.shift(1))
        return _events(entry, exit_)


class MLStrategy(Strategy):
    """Random forest predicting tomorrow's direction, retrained walk-forward.

    The model is first trained on ``min_train`` days, then retrained every
    ``retrain_every`` days on all data seen so far. Each prediction uses a model
    that was trained only on earlier days whose outcome was already known.
    """

    name = "ML_RandomForest"
    feature_names = (
        "returns_1d", "returns_5d", "returns_10d", "returns_20d", "rsi", "macd_hist",
        "bb_position", "volatility", "sma_10_20_ratio", "sma_50_200_ratio",
        "price_sma50_ratio", "ema_5_20_ratio",
    )

    def __init__(
        self,
        min_train: int = 252,
        retrain_every: int = 63,
        threshold: float = 0.55,
        n_estimators: int = 50,
        max_depth: int = 5,
        random_state: int = 42,
    ) -> None:
        if not 0.5 < threshold < 1:
            raise ValueError("threshold must be between 0.5 and 1")
        self.min_train = min_train
        self.retrain_every = retrain_every
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.last_probabilities: pd.Series | None = None
        self.last_labels: pd.Series | None = None

    @staticmethod
    def features(close: pd.Series) -> pd.DataFrame:
        ind = compute_indicators(close)
        feats = pd.DataFrame(index=close.index)
        feats["returns_1d"] = ind["returns"]
        feats["returns_5d"] = ind["momentum_5"]
        feats["returns_10d"] = ind["momentum_10"]
        feats["returns_20d"] = ind["momentum_20"]
        feats["rsi"] = ind["RSI"] / 100
        feats["macd_hist"] = ind["MACD_histogram"] / close
        feats["bb_position"] = ind["BB_position"]
        feats["volatility"] = ind["volatility_20"]
        feats["sma_10_20_ratio"] = ind["SMA_10"] / ind["SMA_20"]
        feats["sma_50_200_ratio"] = ind["SMA_50"] / ind["SMA_200"]
        feats["price_sma50_ratio"] = close / ind["SMA_50"]
        feats["ema_5_20_ratio"] = ind["EMA_5"] / ind["EMA_20"]
        return feats.replace([np.inf, -np.inf], np.nan).dropna()

    def probabilities(self, data: pd.DataFrame) -> pd.Series:
        """Out-of-sample probability that the next day closes higher (NaN before the first model)."""
        close = data["close"]
        feats = self.features(close)
        next_return = close.pct_change().shift(-1)
        labels = (next_return > 0).astype(float).where(next_return.notna())
        probs = pd.Series(np.nan, index=close.index, name="prob_up")

        for start in range(self.min_train, len(feats), self.retrain_every):
            train = feats.iloc[:start]
            y = labels.reindex(train.index)
            known = y.notna().to_numpy()
            if len(np.unique(y[known])) < 2:
                continue
            model = RandomForestClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                min_samples_split=20, random_state=self.random_state,
            )
            model.fit(train.to_numpy()[known], y[known].astype(int).to_numpy())
            block = feats.iloc[start:start + self.retrain_every]
            up = list(model.classes_).index(1)
            probs.loc[block.index] = model.predict_proba(block.to_numpy())[:, up]

        self.last_probabilities, self.last_labels = probs, labels
        return probs

    def signals(self, data: pd.DataFrame) -> pd.Series:
        probs = self.probabilities(data)
        events = pd.Series(0, index=data.index)
        events[probs > self.threshold] = 1
        events[probs < 1 - self.threshold] = -1
        return events


def default_strategies(include_ml: bool = True) -> list[Strategy]:
    strategies: list[Strategy] = [MomentumStrategy(), MeanReversionStrategy(), MACDStrategy()]
    if include_ml:
        strategies.append(MLStrategy())
    return strategies


def _events(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    events = pd.Series(0, index=entry.index)
    events[entry.fillna(False).astype(bool)] = 1
    events[exit_.fillna(False).astype(bool)] = -1
    return events
