"""Command line: fetch real prices, run the backtest, or run an offline demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_TICKERS, BacktestConfig
from .data import align_prices, fetch_yahoo, generate_synthetic_prices, load_meta, load_prices, save_prices
from .pipeline import run_all
from .report import metrics_table, save_results_db, write_outputs
from .strategies import default_strategies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m quant_trading_platform", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download adjusted daily prices from Yahoo Finance into SQLite")
    fetch.add_argument("--db", default="data/market_data.db")
    fetch.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    fetch.add_argument("--start", default="2016-01-01")
    fetch.add_argument("--end", default=None, help="exclusive end date (default: today)")

    backtest = sub.add_parser("backtest", help="run all strategies on prices stored in SQLite")
    backtest.add_argument("--db", default="data/market_data.db")
    backtest.add_argument("--tickers", nargs="+", default=None)
    backtest.add_argument("--start", default=None)
    backtest.add_argument("--end", default=None)
    backtest.add_argument("--out", default="output")
    backtest.add_argument("--benchmark", default="SPY")
    backtest.add_argument("--write-db", action="store_true",
                          help="also store trades and metrics in the database")
    _add_common(backtest)

    demo = sub.add_parser("demo", help="run everything on simulated prices (no internet needed)")
    demo.add_argument("--out", default="output/demo")
    demo.add_argument("--days", type=int, default=1000)
    demo.add_argument("--seed", type=int, default=7)
    _add_common(demo)

    args = parser.parse_args(argv)

    if args.command == "fetch":
        prices = fetch_yahoo(args.tickers, args.start, args.end)
        missing = sorted(set(args.tickers) - set(prices))
        source = "Yahoo Finance via yfinance (split- and dividend-adjusted)"
        rows = save_prices(args.db, prices, source=source)
        print(f"Saved {rows:,} daily bars for {len(prices)} tickers to {args.db}")
        if missing:
            print("No data for:", ", ".join(missing))
        return 0

    config = BacktestConfig(
        initial_capital=args.capital, position_size_pct=args.position_size,
        max_positions=args.max_positions, stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit, cost_bps=args.cost_bps, risk_free_rate=args.risk_free,
    )

    if args.command == "demo":
        out = Path(args.out)
        db = out / "demo_prices.db"
        save_prices(db, generate_synthetic_prices(days=args.days, seed=args.seed),
                    source=f"SIMULATED (geometric Brownian motion, seed {args.seed})")
        db_path, tickers, start, end, benchmark = db, None, None, None, None
    else:
        out, db_path, tickers, start, end, benchmark = (
            Path(args.out), Path(args.db), args.tickers, args.start, args.end, args.benchmark)

    prices = align_prices(load_prices(db_path, tickers, start, end))
    source = load_meta(db_path).get("source", "unknown")
    first, last = next(iter(prices.values())).index[[0, -1]]
    run = run_all(prices, config, default_strategies(include_ml=not args.no_ml), benchmark)

    simulated = source.startswith("SIMULATED")
    title = (f"{'SIMULATED DATA – ' if simulated else ''}{len(prices)} tickers, "
             f"{first:%Y-%m-%d} to {last:%Y-%m-%d}, ${config.initial_capital:,.0f} portfolio, "
             f"{config.cost_bps:g} bps costs per trade")
    data_info = {"source": source, "simulated": simulated, "tickers": sorted(prices),
                 "start": f"{first:%Y-%m-%d}", "end": f"{last:%Y-%m-%d}",
                 "trading_days": len(next(iter(prices.values())))}
    paths = write_outputs(run, out, config, data_info, title)
    if getattr(args, "write_db", False):
        save_results_db(run, db_path)

    print(title)
    print(metrics_table(run.metrics))
    if run.ml_diagnostics:
        d = run.ml_diagnostics
        print(f"ML out-of-sample accuracy {d['accuracy']:.1%} vs {d['always_up_accuracy']:.1%} "
              f"for always predicting 'up' ({d['predictions']:,} predictions)")
    print(f"Outputs written to {paths['results.json'].parent}")
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    defaults = BacktestConfig()
    parser.add_argument("--capital", type=float, default=defaults.initial_capital)
    parser.add_argument("--position-size", type=float, default=defaults.position_size_pct)
    parser.add_argument("--max-positions", type=int, default=defaults.max_positions)
    parser.add_argument("--stop-loss", type=float, default=defaults.stop_loss_pct)
    parser.add_argument("--take-profit", type=float, default=defaults.take_profit_pct)
    parser.add_argument("--cost-bps", type=float, default=defaults.cost_bps)
    parser.add_argument("--risk-free", type=float, default=defaults.risk_free_rate)
    parser.add_argument("--no-ml", action="store_true", help="skip the (slower) machine-learning strategy")
