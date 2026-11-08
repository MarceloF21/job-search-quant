"""Files and dashboard for the Toronto labour-market analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from .benchmarks import Benchmark  # noqa: E402
from .labour import Bunching, LabourSummary  # noqa: E402

SHIFT_HOURS = [3, 4, 5, 6, 8]
COMMUTE_MINUTES = [15, 30, 45, 60, 75, 90]


def _money(value: float | None) -> str:
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"${value:,.2f}"


def _pct(value: float | None) -> str:
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.0%}"


def write_labour_outputs(latest: LabourSummary, bunching: list[Bunching], grid: list[list[float]],
                         benchmarks: tuple[Benchmark, ...], out_dir: str | Path, title: str,
                         source_note: str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {n: out / n for n in ("labour.md", "labour.json", "labour_dashboard.png")}
    q = latest.quartiles or [math.nan] * 3

    lines = [
        f"# {title}", "",
        f"{latest.postings:,} postings ({latest.vacancies:,} vacancies) in the City of Toronto; "
        f"{latest.with_pay:,} state pay. Median {_money(q[1])}/hour (IQR {_money(q[0])}–{_money(q[2])}).", "",
        "| Benchmark | Hourly | Share of postings at or above |", "|---|---:|---:|",
    ]
    lines += [f"| {b.name} ({b.as_of}) | {_money(b.hourly)} | {_pct(latest.benchmark_shares[b.name])} |"
              for b in benchmarks]
    lines += ["", f"- Part-time median {_money(latest.part_time_median)} (n = {len(latest.part_time_wages):,}) vs "
              f"full-time {_money(latest.full_time_median)} (n = {len(latest.full_time_wages):,}).",
              f"- Entry-level postings (high school or less, under a year of experience; Job Bank's own postings): "
              f"{latest.entry_level_n:,}, median {_money(latest.entry_level_median)}/hour; "
              + ", ".join(f"{_pct(v)} {k.replace('_', '-')}" for k, v in latest.entry_level_schedule.items())
              + " shifts.", "",
              "| Skill level (NOC 2021 TEER) | Median hourly | Postings |", "|---|---:|---:|"]
    lines += [f"| {k} | {_money(m)} | {n:,} |" for k, (m, n) in latest.teer_medians.items()]
    lines += ["", "| Month | Hourly-quoted postings | At old minimum ($17.60) | At new minimum ($17.95) "
              "| Below $17.95 |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {b.label} | {b.hourly_postings:,} | {b.at_old:,} ({b.at_old / b.hourly_postings:.1%}) "
              f"| {b.at_new:,} ({b.at_new / b.hourly_postings:.1%}) | {b.below_new:,} "
              f"({b.below_new / b.hourly_postings:.1%}) |" for b in bunching if b.hourly_postings]
    entry = latest.entry_level_median
    if entry:
        lines += ["", f"Commute tax on the entry-level median ({_money(entry)}): a 6-hour shift with a 45-minute "
                  f"commute each way pays {_money(grid[SHIFT_HOURS.index(6)][COMMUTE_MINUTES.index(45)])} per hour "
                  "of time spent."]
    lines += ["", source_note]
    paths["labour.md"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "title": title, "source": source_note,
        "postings": latest.postings, "vacancies": latest.vacancies, "with_pay": latest.with_pay,
        "hourly_quartiles": q, "benchmark_shares": latest.benchmark_shares,
        "part_time": {"median": latest.part_time_median, "n": len(latest.part_time_wages)},
        "full_time": {"median": latest.full_time_median, "n": len(latest.full_time_wages)},
        "entry_level": {"n": latest.entry_level_n, "median": latest.entry_level_median,
                        "schedule_shares": latest.entry_level_schedule},
        "teer_medians": {k: {"median": m, "n": n} for k, (m, n) in latest.teer_medians.items()},
        "category_medians": {k: {"median": m, "n": n} for k, (m, n) in latest.category_medians.items()},
        "minimum_wage_bunching": [b.__dict__ for b in bunching],
        "commute_tax": {"wage": entry, "shift_hours": SHIFT_HOURS, "commute_minutes_one_way": COMMUTE_MINUTES,
                        "effective_wage": grid},
    }
    paths["labour.json"].write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    plot_labour_dashboard(latest, bunching, grid, benchmarks, paths["labour_dashboard.png"], title)
    return paths


def plot_labour_dashboard(latest: LabourSummary, bunching: list[Bunching], grid: list[list[float]],
                          benchmarks: tuple[Benchmark, ...], path: str | Path, title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    (ax_hist, ax_teer, ax_box), (ax_bunch, ax_sched, ax_heat) = axes
    styles = [("black", "--"), ("tab:red", ":")]

    wages = np.clip(latest.wages, 15, 80)
    ax_hist.hist(wages, bins=np.arange(15, 81, 1), color="tab:blue", alpha=0.8)
    for b, (color, ls) in zip(benchmarks, styles, strict=False):
        ax_hist.axvline(b.hourly, color=color, ls=ls, lw=1.4,
                        label=f"{b.name} ${b.hourly:.2f}: {latest.benchmark_shares[b.name]:.0%} at or above")
    ax_hist.set_xlabel("Posted pay, $/hour (values outside $15–$80 shown at the edges)")
    ax_hist.set_ylabel("Postings")
    ax_hist.set_title(f"Pay distribution, City of Toronto (n = {latest.with_pay:,})")
    ax_hist.legend(fontsize=8)

    labels = list(latest.teer_medians)
    values = [latest.teer_medians[k][0] for k in labels]
    bars = ax_teer.barh(labels, values, color="tab:purple")
    for bar, k in zip(bars, labels, strict=True):
        m, n = latest.teer_medians[k]
        ax_teer.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  ${m:.2f} (n={n:,})", va="center",
                     fontsize=8)
    ax_teer.invert_yaxis()
    ax_teer.set_xlim(0, max(values) * 1.35)
    ax_teer.tick_params(axis="y", labelsize=8)
    ax_teer.set_title("Median pay by skill level: the return to training")

    groups = [("Full-time", latest.full_time_wages), ("Part-time", latest.part_time_wages),
              ("Entry-level", latest.entry_level_wages)]
    ax_box.boxplot([g for _, g in groups], tick_labels=[f"{n}\n(n={len(g):,})" for n, g in groups],
                   showfliers=False)
    for b, (color, ls) in zip(benchmarks, styles, strict=False):
        ax_box.axhline(b.hourly, color=color, ls=ls, lw=1.2)
    ax_box.set_ylabel("$/hour")
    ax_box.set_title("Full-time vs part-time vs entry-level pay")

    x = np.arange(len(bunching))
    width = 0.27
    for i, (name, attr, color) in enumerate([("At old minimum $17.60", "at_old", "tab:gray"),
                                             ("At new minimum $17.95", "at_new", "tab:green"),
                                             ("Below $17.95", "below_new", "tab:red")]):
        shares = [getattr(b, attr) / b.hourly_postings if b.hourly_postings else 0 for b in bunching]
        ax_bunch.bar(x + (i - 1) * width, shares, width, label=name, color=color)
    ax_bunch.set_xticks(x, [b.label for b in bunching])
    ax_bunch.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_bunch.set_ylabel("Share of hourly-quoted postings")
    ax_bunch.set_title("Minimum-wage increase (Oct 1, 2026): pay bunching moves to $17.95")
    ax_bunch.legend(fontsize=8)

    names = list(latest.entry_level_schedule)
    ax_sched.bar([n.replace("_", "-") for n in names], [latest.entry_level_schedule[n] for n in names],
                 color="tab:orange")
    ax_sched.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_sched.set_title(f"Entry-level postings that ask for each shift type (n = {latest.entry_level_n:,})")

    entry = latest.entry_level_median or 0
    data = np.array(grid)
    im = ax_heat.imshow(data, cmap="RdYlGn", aspect="auto", vmin=data.min(), vmax=entry)
    ax_heat.set_xticks(range(len(COMMUTE_MINUTES)), [f"{c}" for c in COMMUTE_MINUTES])
    ax_heat.set_yticks(range(len(SHIFT_HOURS)), [f"{h} h" for h in SHIFT_HOURS])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax_heat.text(j, i, f"${data[i, j]:.2f}\n−{1 - data[i, j] / entry:.0%}", ha="center", va="center",
                         fontsize=7)
    ax_heat.set_xlabel("One-way commute (minutes, unpaid)")
    ax_heat.set_ylabel("Shift length")
    ax_heat.set_title(f"Commute tax on the entry-level median (${entry:.2f}/hour)")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
