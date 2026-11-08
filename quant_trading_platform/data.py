"""Load, save, download and simulate daily OHLCV prices."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
Prices = dict[str, pd.DataFrame]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_data (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def save_prices(db_path: str | Path, prices: Prices, source: str) -> int:
    """Insert or replace daily bars in ``market_data`` and record where they came from."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ticker, frame in prices.items():
        bars = frame[PRICE_COLUMNS].dropna(subset=["open", "high", "low", "close"])
        for date, bar in zip(bars.index, bars.itertuples(index=False), strict=True):
            rows.append((ticker, pd.Timestamp(date).strftime("%Y-%m-%d"), *map(float, bar)))
    with closing(sqlite3.connect(db_path)) as con, con:
        con.executescript(_SCHEMA)
        con.executemany("INSERT OR REPLACE INTO market_data VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        con.executemany(
            "INSERT OR REPLACE INTO meta VALUES (?, ?)",
            [("source", source), ("saved_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))],
        )
    return len(rows)


def load_meta(db_path: str | Path) -> dict[str, str]:
    """Return the ``meta`` table (data source, save time) if the database has one."""
    with closing(sqlite3.connect(Path(db_path))) as con:
        has_meta = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
        ).fetchone()
        return dict(con.execute("SELECT key, value FROM meta").fetchall()) if has_meta else {}


def load_prices(
    db_path: str | Path,
    tickers: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> Prices:
    """Read daily bars from SQLite into one DataFrame per ticker."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"No database at {db_path}. Run `python -m quant_trading_platform fetch` "
            "for real data, or `python -m quant_trading_platform demo` for simulated data."
        )
    query = "SELECT ticker, date, open, high, low, close, volume FROM market_data WHERE 1 = 1"
    params: list[str] = []
    tickers = list(tickers or [])
    if tickers:
        query += f" AND ticker IN ({', '.join('?' * len(tickers))})"
        params += tickers
    if start:
        query += " AND date >= ?"
        params.append(pd.Timestamp(start).strftime("%Y-%m-%d"))
    if end:
        query += " AND date <= ?"
        params.append(pd.Timestamp(end).strftime("%Y-%m-%d"))
    with closing(sqlite3.connect(db_path)) as con:
        frame = pd.read_sql_query(query, con, params=params)
    if frame.empty:
        raise ValueError(f"No price rows in {db_path} match the requested tickers and dates.")
    frame["date"] = pd.to_datetime(frame["date"])
    return {
        str(ticker): group.drop(columns="ticker").set_index("date").sort_index().astype(float)
        for ticker, group in frame.groupby("ticker", sort=True)
    }


def align_prices(prices: Prices, min_rows: int = 300) -> Prices:
    """Keep tickers with enough history and restrict them all to the dates they share."""
    usable = {t: f.dropna() for t, f in prices.items() if len(f.dropna()) >= min_rows}
    if not usable:
        raise ValueError(f"No ticker has at least {min_rows} complete daily bars.")
    common = None
    for frame in usable.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if len(common) < min_rows:
        raise ValueError(f"The tickers share only {len(common)} dates; need {min_rows}.")
    return {t: f.loc[common] for t, f in sorted(usable.items())}


def fetch_yahoo(tickers: Iterable[str], start: str, end: str | None = None) -> Prices:
    """Download split- and dividend-adjusted daily bars from Yahoo Finance (needs yfinance)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("Downloading prices needs yfinance: pip install yfinance") from exc
    tickers = list(tickers)
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True,
        progress=False, threads=False, group_by="ticker",
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data.")
    prices: Prices = {}
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                continue
            sub = raw[ticker]
        else:
            sub = raw
        sub = sub.rename(columns=str.lower)[PRICE_COLUMNS].dropna(subset=["close"])
        sub.index = pd.DatetimeIndex(sub.index).tz_localize(None).rename("date")
        if not sub.empty:
            prices[ticker] = sub.astype(float)
    return prices


def generate_synthetic_prices(
    tickers: Iterable[str] = ("SIM_A", "SIM_B", "SIM_C", "SIM_D", "SIM_E"),
    days: int = 1000,
    seed: int = 7,
    start: str = "2018-01-02",
) -> Prices:
    """Simulate daily bars with geometric Brownian motion, for tests and offline demos only."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=days, name="date")
    prices: Prices = {}
    for ticker in tickers:
        drift = rng.uniform(0.0, 0.12) / 252
        vol = rng.uniform(0.15, 0.40) / np.sqrt(252)
        close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, size=days)))
        prev_close = np.concatenate(([100.0], close[:-1]))
        open_ = prev_close * np.exp(rng.normal(0, vol * 0.3, size=days))
        high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol * 0.5, size=days)))
        low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol * 0.5, size=days)))
        volume = rng.integers(1_000_000, 10_000_000, size=days).astype(float)
        prices[ticker] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
    return prices
