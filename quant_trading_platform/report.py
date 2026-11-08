"""Results table, files and dashboard chart."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import BacktestConfig  # noqa: E402
from .pipeline import RunOutput  # noqa: E402

COLUMNS = [
    ("cagr", "CAGR", "pct"), ("sharpe", "Sharpe", "num"), ("sortino", "Sortino", "num"),
    ("max_drawdown", "Max drawdown", "pct"), ("calmar", "Calmar", "num"),
    ("trades", "Trades", "int"), ("win_rate", "Win rate", "pct"),
    ("avg_exposure", "Avg. invested", "pct"),
]


def _fmt(value: float | None, kind: str) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "–"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "int":
        return f"{int(value):,}"
    return f"{value:.2f}"


def metrics_table(metrics: dict[str, dict[str, float]]) -> str:
    """Markdown table: strategies first, then benchmarks."""
    header = "| Strategy | " + " | ".join(label for _, label, _ in COLUMNS) + " |"
    rule = "|---|" + "---:|" * len(COLUMNS)
    rows = [
        f"| {name} | " + " | ".join(_fmt(m.get(key), kind) for key, _, kind in COLUMNS) + " |"
        for name, m in metrics.items()
    ]
    return "\n".join([header, rule, *rows])


def write_outputs(
    run: RunOutput, out_dir: str | Path, config: BacktestConfig, data_info: dict, title: str
) -> dict[str, Path]:
    """Write results.json, results.md, equity.csv, trades.csv and dashboard.png."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: out / name for name in
             ("results.json", "results.md", "equity.csv", "trades.csv", "dashboard.png")}

    def clean(value):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    payload = {
        "data": data_info,
        "config": asdict(config),
        "metrics": {k: {m: clean(v) for m, v in d.items()} for k, d in run.metrics.items()},
        "ml_out_of_sample": run.ml_diagnostics,
    }
    paths["results.json"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [f"# {title}", "", metrics_table(run.metrics), ""]
    if run.ml_diagnostics:
        d = run.ml_diagnostics
        lines.append(
            f"ML out-of-sample: {d['predictions']:,} predictions, accuracy {d['accuracy']:.1%} "
            f"vs {d['always_up_accuracy']:.1%} for always predicting 'up'."
        )
    paths["results.md"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    curves = {name: r.equity for name, r in run.results.items()} | run.benchmarks
    pd.DataFrame(curves).to_csv(paths["equity.csv"], index_label="date")
    trades = [r.trades.assign(strategy=name) for name, r in run.results.items() if len(r.trades)]
    (pd.concat(trades) if trades else pd.DataFrame()).to_csv(paths["trades.csv"], index=False)
    plot_dashboard(run, paths["dashboard.png"], title)
    return paths


def save_results_db(run: RunOutput, db_path: str | Path) -> None:
    """Store trades and strategy metrics in SQLite (tables ``trades`` and ``strategy_performance``)."""
    trades = [r.trades.assign(strategy=name) for name, r in run.results.items() if len(r.trades)]
    perf = pd.DataFrame.from_dict(run.metrics, orient="index").rename_axis("strategy").reset_index()
    with closing(sqlite3.connect(db_path)) as con:
        if trades:
            pd.concat(trades).astype({"entry_date": str, "exit_date": str}).to_sql(
                "trades", con, if_exists="replace", index=False)
        perf.to_sql("strategy_performance", con, if_exists="replace", index=False)


def _finite(value: float | None) -> float:
    return value if isinstance(value, (int, float)) and math.isfinite(value) else 0.0


def plot_dashboard(run: RunOutput, path: str | Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    (ax_eq, ax_dd), (ax_bar, ax_box) = axes

    for name, result in run.results.items():
        ax_eq.plot(result.equity.index, result.equity, lw=1.4, label=name)
        dd = result.equity / result.equity.cummax() - 1
        ax_dd.plot(dd.index, dd * 100, lw=1.1, label=name)
    for name, equity in run.benchmarks.items():
        ax_eq.plot(equity.index, equity, lw=1.6, ls="--", color="black" if "SPY" in name else "gray",
                   label=name)
        dd = equity / equity.cummax() - 1
        ax_dd.plot(dd.index, dd * 100, lw=1.0, ls="--", color="black" if "SPY" in name else "gray",
                   label=name)
    ax_eq.set_yscale("log")
    ax_eq.set_title("Portfolio value (log scale)")
    ax_eq.legend(fontsize=8)
    ax_dd.set_title("Drawdown (%)")

    names = list(run.metrics)
    sharpe = [_finite(run.metrics[n].get("sharpe")) for n in names]
    cagr = [run.metrics[n].get("cagr", math.nan) for n in names]
    bars = ax_bar.barh(names, sharpe, color=["tab:blue"] * len(run.results) + ["gray"] * len(run.benchmarks))
    for bar, value in zip(bars, cagr, strict=True):
        ax_bar.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f"  CAGR {value:.1%}" if not math.isnan(value) else "", va="center", fontsize=8)
    ax_bar.set_xlim(min(0.0, *sharpe) * 1.3, max(0.1, *sharpe) * 1.3)
    ax_bar.axvline(0, color="black", lw=0.8)
    ax_bar.invert_yaxis()
    ax_bar.set_title("Sharpe ratio (daily returns, annualized)")

    data = [(r.trades["return_pct"].dropna() * 100).tolist() or [0.0] for r in run.results.values()]
    ax_box.boxplot(data, tick_labels=list(run.results), showfliers=False)
    ax_box.axhline(0, color="black", lw=0.8)
    ax_box.set_title("Return per trade (%), after costs")
    ax_box.tick_params(axis="x", labelsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
