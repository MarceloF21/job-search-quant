import pandas as pd
import pytest

from quant_trading_platform.data import generate_synthetic_prices


@pytest.fixture(scope="session")
def synthetic_prices():
    return generate_synthetic_prices(tickers=("SIM_A", "SIM_B", "SIM_C"), days=700, seed=11)


def make_bars(opens, highs=None, lows=None, closes=None, start="2024-01-01"):
    """Build a small daily OHLC frame; highs/lows/closes default to the opens."""
    idx = pd.bdate_range(start, periods=len(opens), name="date")
    closes = closes if closes is not None else opens
    highs = highs if highs is not None else [max(o, c) for o, c in zip(opens, closes, strict=True)]
    lows = lows if lows is not None else [min(o, c) for o, c in zip(opens, closes, strict=True)]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1_000.0},
        index=idx, dtype=float,
    )
