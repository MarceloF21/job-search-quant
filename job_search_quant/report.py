"""Tables, files and the two dashboards (market and search performance)."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402

from .benchmarks import Benchmark  # noqa: E402
from .economics import JobEconomics  # noqa: E402
from .journal import STAGES, Funnel  # noqa: E402
from .models import Assumptions, Profile  # noqa: E402
from .odds import Rate  # noqa: E402
from .optimizer import OptimizationResult  # noqa: E402
from .simulate import PipelineSimulation  # noqa: E402

STATUS_COLORS = {"fits": "tab:green", "partial": "tab:olive", "blocked": "tab:red", "clash": "tab:orange",
                 "too_far": "tab:purple", "closed": "gray", "unverified": "lightgray", "excluded": "black",
                 "no_pay": "lightgray"}
BENCH_STYLES = [("black", "--"), ("tab:red", ":")]


def _clean(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _money(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "–"
    return f"−${-value:,.2f}" if value < 0 else f"${value:,.2f}"


def _pct(value: float | None) -> str:
    return "–" if value is None or math.isnan(value) else f"{value:.0%}"


# ---------------------------------------------------------------- market
def ranking_table(jobs: list[JobEconomics]) -> str:
    lines = ["| # | Job | Employer | Status | Posted pay | Expected hrs/wk | Weekly pay | Effective wage "
             "| Weekly surplus | Notes |", "|---:|---|---|---|---:|---:|---:|---:|---:|---|"]
    for i, j in enumerate(jobs, 1):
        notes = [reason.rstrip(".") for reason in j.reasons]
        if j.hours_assumed and not math.isnan(j.expected_hours):
            notes.append("hours assumed")
        if j.to_obtain:
            notes.append("needs " + ", ".join(r.replace("_", " ") for r in j.to_obtain))
        if j.posting.tips:
            notes.append("tips not counted")
        hours = "–" if math.isnan(j.expected_hours) else f"{j.expected_hours:.1f} ± {j.hours_std:.1f}"
        lines.append(
            f"| {i} | {j.posting.title} | {j.posting.employer} | {j.status.replace('_', ' ')} "
            f"| {_money(j.wage)} | {hours} | {_money(j.weekly_pay)} | {_money(j.effective_wage)} "
            f"| {_money(j.weekly_surplus)} | {'; '.join(notes)} |")
    return "\n".join(lines)


def write_market_outputs(jobs: list[JobEconomics], result: OptimizationResult, profile: Profile,
                         assumptions: Assumptions, benchmarks: tuple[Benchmark, ...], summary: dict,
                         out_dir: str | Path, title: str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {n: out / n for n in ("ranking.csv", "market.md", "market.json", "market_dashboard.png")}

    with open(paths["ranking.csv"], "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "id", "title", "employer", "status", "posted_wage", "expected_hours", "hours_std",
                         "weekly_pay", "commute_hours", "setup_cost_weekly", "effective_wage", "weekly_surplus",
                         "source_label", "url"])
        for i, j in enumerate(jobs, 1):
            writer.writerow([i, j.posting.id, j.posting.title, j.posting.employer, j.status, j.wage,
                             j.expected_hours, j.hours_std, j.weekly_pay, j.commute_hours, j.setup_cost_weekly,
                             j.effective_wage, j.weekly_surplus, j.posting.source_label, j.posting.url])

    best = result.best
    q_post, q_eff = summary["posted_wage_quartiles"], summary["effective_wage_quartiles"]
    lines = [f"# {title}", "", ranking_table(jobs), "",
             f"- Median posted pay {_money(q_post[1])} (IQR {_money(q_post[0])}–{_money(q_post[2])}); median "
             f"effective wage {_money(q_eff[1])} (IQR {_money(q_eff[0])}–{_money(q_eff[2])}).",
             f"- Commuting and one-off costs cut pay by {_money(summary['avg_gap_posted_minus_effective'])}"
             f"/hour on average."]
    for name, b in summary["benchmarks"].items():
        lines.append(f"- {name} ({_money(b['hourly'])}, {b['as_of']}): {_pct(b['share_posted_at_or_above'])} of "
                     f"postings pay at least this; {_pct(b['share_effective_at_or_above'])} after commuting.")
    lines += ["", f"**Best combination** (objective: {result.objective}, risk aversion {result.risk_aversion:g}, "
              f"{result.free_hours:g} free hours/week): {best.label}. Expected extra pay "
              f"{_money(best.expected_pay)}/week (± {_money(best.pay_std)}), weekly surplus "
              f"{_money(best.expected_surplus)}, {best.total_time:.1f} hours including commuting."]
    paths["market.md"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "profile": profile.to_dict(), "assumptions": asdict(assumptions),
        "summary": {k: _clean(v) if not isinstance(v, (dict, tuple)) else v for k, v in summary.items()},
        "objective": result.objective, "risk_aversion": result.risk_aversion,
        "best": {"jobs": [j.posting.id for j in best.jobs], "expected_weekly_pay": _clean(best.expected_pay),
                 "expected_weekly_surplus": _clean(best.expected_surplus), "weekly_pay_std": _clean(best.pay_std)},
        "frontier": [{"jobs": [j.posting.id for j in c.jobs], "expected_weekly_pay": _clean(c.expected_pay),
                      "weekly_pay_std": _clean(c.pay_std)} for c in result.frontier],
        "jobs": [{"id": j.posting.id, "status": j.status, "posted_wage": _clean(j.wage),
                  "effective_wage": _clean(j.effective_wage), "weekly_surplus": _clean(j.weekly_surplus),
                  "reasons": j.reasons} for j in jobs],
    }
    paths["market.json"].write_text(json.dumps(payload, indent=2, default=_clean), encoding="utf-8")
    plot_market_dashboard(jobs, result, benchmarks, paths["market_dashboard.png"], title)
    return paths


def plot_market_dashboard(jobs: list[JobEconomics], result: OptimizationResult,
                          benchmarks: tuple[Benchmark, ...], path: str | Path, title: str) -> None:
    fig, ((ax_wage, ax_dist), (ax_frontier, ax_status)) = plt.subplots(2, 2, figsize=(15, 10))
    priced = [j for j in jobs if not math.isnan(j.wage)]

    y = range(len(priced))
    ax_wage.barh([i + 0.2 for i in y], [j.wage for j in priced], height=0.4, color="lightsteelblue",
                 label="Posted pay")
    ax_wage.barh([i - 0.2 for i in y], [0 if math.isnan(j.effective_wage) else j.effective_wage for j in priced],
                 height=0.4, color=[STATUS_COLORS.get(j.status, "gray") for j in priced],
                 label="Effective wage (after commute and costs)")
    for bench, (color, style) in zip(benchmarks, BENCH_STYLES, strict=False):
        ax_wage.axvline(bench.hourly, color=color, ls=style, lw=1.2, label=f"{bench.name} ${bench.hourly:.2f}")
    ax_wage.set_yticks(list(y), [f"#{jobs.index(j) + 1} {j.posting.title[:26]}" for j in priced], fontsize=8)
    ax_wage.invert_yaxis()
    ax_wage.set_title("Posted pay vs effective hourly wage")
    ax_wage.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False)

    posted = [j.wage for j in priced]
    effective = [j.effective_wage for j in priced if not math.isnan(j.effective_wage)]
    ax_dist.boxplot([posted, effective], tick_labels=["Posted pay", "Effective wage"], widths=0.5,
                    showfliers=False)
    rng = np.random.default_rng(0)
    for x, values in ((1, posted), (2, effective)):
        ax_dist.scatter(x + rng.uniform(-0.12, 0.12, len(values)), values, s=22, alpha=0.7, color="tab:blue")
    for bench, (color, style) in zip(benchmarks, BENCH_STYLES, strict=False):
        ax_dist.axhline(bench.hourly, color=color, ls=style, lw=1.2, label=f"{bench.name} ${bench.hourly:.2f}")
    top = max([*posted, *effective, *(b.hourly for b in benchmarks)])
    ax_dist.set_ylim(top=top * 1.06)
    ax_dist.set_ylabel("$ per hour")
    ax_dist.set_title(f"Toronto postings: pay distribution (n = {len(posted)})")
    ax_dist.legend(fontsize=8, loc="upper right")

    def value(c):
        return c.expected_pay if result.objective == "pay" else c.expected_surplus

    ax_frontier.scatter([c.pay_std for c in result.combinations], [value(c) for c in result.combinations],
                        s=18, color="lightgray", label="Feasible combinations")
    ax_frontier.plot([c.pay_std for c in result.frontier], [value(c) for c in result.frontier], "-o",
                     color="tab:blue", label="Efficient frontier")
    ax_frontier.scatter([result.best.pay_std], [value(result.best)], s=160, marker="*", color="tab:red", zorder=5,
                        label=f"Chosen (risk aversion {result.risk_aversion:g})")
    for c in result.frontier[1:]:
        ax_frontier.annotate(" + ".join(j.posting.title[:18] for j in c.jobs), (c.pay_std, value(c)), fontsize=7,
                             xytext=(-6, 8), textcoords="offset points", ha="right")
    ax_frontier.margins(y=0.15)
    ax_frontier.set_xlabel("Std of weekly pay ($)")
    ax_frontier.set_ylabel("Expected extra weekly pay ($)" if result.objective == "pay"
                           else "Expected weekly surplus over reservation wage ($)")
    ax_frontier.set_title("Job combinations: risk vs expected pay")
    ax_frontier.legend(fontsize=8)

    counts: dict[str, int] = {}
    for j in jobs:
        counts[j.status] = counts.get(j.status, 0) + 1
    ax_status.bar([s.replace("_", " ") for s in counts], list(counts.values()),
                  color=[STATUS_COLORS.get(s, "gray") for s in counts])
    ax_status.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_status.set_title("Screening outcome per posting")
    ax_status.set_ylabel("Postings")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- search performance
def write_search_outputs(fun: Funnel, overall: dict[str, Rate], by_channel: list[Rate], by_resume: list[Rate],
                         ab: tuple[str, str, float] | None, sim: PipelineSimulation, curve: dict,
                         needed: int, target: float, out_dir: str | Path, title: str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {n: out / n for n in ("search.md", "search.json", "search_dashboard.png")}

    def rate_line(name: str, r: Rate) -> str:
        return f"| {name} | {r.successes}/{r.trials} | {_pct(r.mean)} | {_pct(r.low)}–{_pct(r.high)} |"

    lines = [f"# {title}", "",
             f"{fun.total} applications, {fun.mature} mature (answered, closed or past the response window), "
             f"{fun.open_applications} still open. Median time to first reply: "
             f"{'–' if fun.median_days_to_response is None else f'{fun.median_days_to_response:g} days'}.", "",
             "| Stage | Count | Conversion |", "|---|---:|---:|"]
    lines.append(f"| applied | {fun.counts['applied']} | |")
    for prev, stage in zip(STAGES, STAGES[1:], strict=False):
        lines.append(f"| {stage} | {fun.counts[stage]} | {_pct(fun.conversion[f'{prev}→{stage}'])} |")
    lines += ["", "| Rate (Bayesian) | Observed | Posterior mean | 90% interval |", "|---|---:|---:|---:|",
              rate_line("Response", overall["response"]), rate_line("Interview", overall["interview"]),
              rate_line("Offer after interview", overall["offer_given_interview"]), "",
              "| Interview rate by channel | Observed | Posterior mean | 90% interval |", "|---|---:|---:|---:|"]
    lines += [rate_line(r.label.replace("_", " "), r) for r in by_channel]
    lines += ["", "| Interview rate by résumé version | Observed | Posterior mean | 90% interval |",
              "|---|---:|---:|---:|"]
    lines += [rate_line(r.label, r) for r in by_resume]
    if ab:
        lines.append(f"\nP({ab[0]} résumé has the higher interview rate than {ab[1]}) = {ab[2]:.0%}.")
    lines += ["", f"- Open pipeline (Monte Carlo, {sim.simulations:,} runs): P(at least one new offer) = "
              f"{sim.prob_at_least_one:.0%}, expected new offers {sim.expected_offers:.2f}; "
              f"offers already in hand: {sim.offers_in_hand}.",
              f"- At the current offer rate per application ({curve['p_offer_mean']:.1%}), about **{needed}** more "
              f"applications give {target:.0%} odds of at least one offer."]
    paths["search.md"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    def rate_dict(r: Rate) -> dict:
        return {k: _clean(v) for k, v in asdict(r).items()}

    payload = {
        "funnel": {"total": fun.total, "mature": fun.mature, "counts": fun.counts,
                   "conversion": {k: _clean(v) for k, v in fun.conversion.items()},
                   "median_days_to_response": fun.median_days_to_response, "open": fun.open_applications},
        "rates": {k: rate_dict(v) for k, v in overall.items()},
        "interview_by_channel": [rate_dict(r) for r in by_channel],
        "interview_by_resume_version": [rate_dict(r) for r in by_resume],
        "ab_test": None if ab is None else {"better": ab[0], "than": ab[1], "probability": ab[2]},
        "pipeline": {"prob_at_least_one_new_offer": sim.prob_at_least_one, "expected_new_offers": sim.expected_offers,
                     "offers_in_hand": sim.offers_in_hand},
        "plan": {"p_offer_per_application": curve["p_offer_mean"], "target": target,
                 "applications_needed": needed},
    }
    paths["search.json"].write_text(json.dumps(payload, indent=2, default=_clean), encoding="utf-8")
    plot_search_dashboard(fun, by_channel, by_resume, ab, sim, curve, needed, target,
                          paths["search_dashboard.png"], title)
    return paths


def plot_search_dashboard(fun: Funnel, by_channel: list[Rate], by_resume: list[Rate],
                          ab: tuple[str, str, float] | None, sim: PipelineSimulation, curve: dict, needed: int,
                          target: float, path: str | Path, title: str) -> None:
    fig, ((ax_funnel, ax_rates), (ax_sim, ax_curve)) = plt.subplots(2, 2, figsize=(15, 10))

    counts = [fun.counts[s] for s in STAGES]
    bars = ax_funnel.bar(STAGES, counts, color=["tab:gray", "tab:blue", "tab:orange", "tab:green"])
    for bar, (i, stage) in zip(bars, enumerate(STAGES), strict=True):
        text = f"{counts[i]}"
        if i:
            text += f"\n({_pct(fun.conversion[f'{STAGES[i - 1]}→{stage}'])} of previous)"
        ax_funnel.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha="center", va="bottom",
                       fontsize=9)
    ax_funnel.margins(y=0.2)
    ax_funnel.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_funnel.set_title(f"Application funnel ({fun.mature} mature of {fun.total} applications)")

    groups = [("Channel", by_channel), ("Résumé", by_resume)]
    labels, means, lows, highs, colors = [], [], [], [], []
    for (group, rates), color in zip(groups, ("tab:blue", "tab:green"), strict=True):
        for r in rates:
            labels.append(f"{group}: {r.label.replace('_', ' ')} ({r.successes}/{r.trials})")
            means.append(r.mean)
            lows.append(r.mean - r.low)
            highs.append(r.high - r.mean)
            colors.append(color)
    ypos = np.arange(len(labels))
    for y, m, lo, hi, color in zip(ypos, means, lows, highs, colors, strict=True):
        ax_rates.errorbar(m, y, xerr=[[lo], [hi]], fmt="o", color=color, capsize=3)
    ax_rates.set_yticks(ypos, labels, fontsize=8)
    ax_rates.invert_yaxis()
    ax_rates.xaxis.set_major_formatter(PercentFormatter(1.0))
    note = f"\nP({ab[0]} > {ab[1]}) = {ab[2]:.0%}" if ab else ""
    ax_rates.set_title("Interview rate: posterior mean and 90% interval" + note)

    values, freq = np.unique(sim.offers, return_counts=True)
    ax_sim.bar(values, freq / sim.simulations, color="tab:green")
    ax_sim.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_sim.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_sim.set_xlabel("New offers from applications still open")
    ax_sim.set_title(f"Monte Carlo of the open pipeline: P(≥1 new offer) = {sim.prob_at_least_one:.0%}")

    ax_curve.fill_between(curve["n"], curve["low"], curve["high"], color="tab:blue", alpha=0.2,
                          label="90% band (rate uncertainty)")
    ax_curve.plot(curve["n"], curve["mean"], color="tab:blue", label="P(at least one offer)")
    ax_curve.axhline(target, color="black", ls="--", lw=1)
    ax_curve.axvline(needed, color="tab:red", ls=":", lw=1.2,
                     label=f"{needed} applications → {target:.0%} (at the mean rate)")
    ax_curve.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_curve.set_xlabel("Additional applications")
    ax_curve.set_title(f"How many more applications? (offer rate {curve['p_offer_mean']:.1%} per application)")
    ax_curve.legend(fontsize=8, loc="lower right")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
