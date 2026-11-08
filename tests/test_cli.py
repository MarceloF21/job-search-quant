import json

from quant_trading_platform.cli import main


def test_demo_runs_end_to_end(tmp_path):
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out), "--days", "600", "--no-ml"]) == 0
    for name in ("results.json", "results.md", "equity.csv", "trades.csv", "dashboard.png"):
        assert (out / name).exists(), name
    results = json.loads((out / "results.json").read_text())
    assert results["data"]["simulated"] is True
    assert "Equal-weight buy & hold" in results["metrics"]
    assert results["metrics"]["Momentum_Trend"]["trades"] >= 0


def test_backtest_reads_saved_prices_and_writes_db(tmp_path):
    from quant_trading_platform.data import generate_synthetic_prices, save_prices

    db = tmp_path / "prices.db"
    save_prices(db, generate_synthetic_prices(days=500, seed=3), source="SIMULATED test")
    assert main(["backtest", "--db", str(db), "--out", str(tmp_path / "out"), "--no-ml", "--write-db"]) == 0

    import sqlite3

    with sqlite3.connect(db) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"market_data", "meta", "strategy_performance"} <= tables
