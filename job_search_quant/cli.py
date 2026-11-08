"""Command line for the job-search quant platform."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np

from .benchmarks import DEFAULT_BENCHMARKS
from .db import load_postings, save_postings
from .economics import evaluate_all
from .journal import funnel, load_journal
from .labour import (
    commute_tax,
    fetch_month,
    in_city,
    minimum_wage_bunching,
    month_label,
    summarize,
)
from .labour import (
    load_postings as load_jobbank,
)
from .labour_report import COMMUTE_MINUTES, SHIFT_HOURS, write_labour_outputs
from .market import market_summary
from .match import keyword_match
from .models import Assumptions, Profile
from .odds import applications_needed, probability_better, rates_by, stage_rates
from .optimizer import optimize
from .report import ranking_table, write_market_outputs, write_search_outputs
from .simulate import offer_probability_curve, simulate_pipeline
from .sources import fetch_greenhouse, fetch_lever, fetch_oracle, import_csv, mark_closed

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SAMPLE_DATE = "2026-11-08"  # the bundled sample is a snapshot taken on this day


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m job_search_quant", description=__doc__)
    parser.add_argument("--db", default="data/postings.db")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="collect postings from an employer's public job feed")
    fetch.add_argument("source", choices=["greenhouse", "lever", "oracle"])
    fetch.add_argument("name", help="Greenhouse board token, Lever company, or Oracle host")
    fetch.add_argument("--site", help="Oracle site number (e.g. CX_1)")
    fetch.add_argument("--job-id", help="Oracle job id")
    fetch.add_argument("--employer", help="employer name to display")

    imp = sub.add_parser("import", help="import postings from a CSV file")
    imp.add_argument("csv")

    market = sub.add_parser("market", help="screen and rank postings, pay distribution, best job combination")
    market.add_argument("--profile", default=str(EXAMPLES / "profile.example.json"))
    market.add_argument("--objective", choices=["pay", "surplus"], default="pay")
    market.add_argument("--risk-aversion", type=float, default=0.5)
    market.add_argument("--out", default="output")

    search = sub.add_parser("search", help="funnel, Bayesian success rates and Monte Carlo from your journal")
    search.add_argument("--journal", default="data/applications.csv")
    search.add_argument("--today", default=None, help="evaluate as of this date (YYYY-MM-DD)")
    search.add_argument("--window", type=int, default=21, help="days of silence that count as no response")
    search.add_argument("--target", type=float, default=0.8, help="target probability of at least one offer")
    search.add_argument("--out", default="output")

    labour = sub.add_parser("labour", help="Toronto labour market from Job Bank open data (about 50 MB per month)")
    labour.add_argument("--months", nargs="+", default=["2026-09", "2026-10"],
                        help="months to analyse as YYYY-MM; the last one is the main month")
    labour.add_argument("--data-dir", default="data/jobbank")
    labour.add_argument("--files", nargs="+", help="local monthly CSVs named YYYY-MM.csv instead of downloading")
    labour.add_argument("--out", default="output")

    match = sub.add_parser("match", help="compare a résumé with a posting's keywords")
    match.add_argument("--resume", required=True, help="plain-text résumé")
    match.add_argument("--posting-id", required=True)

    demo = sub.add_parser("demo", help="run everything on the bundled Toronto sample and fictional journal")
    demo.add_argument("--out", default="output/demo")
    demo.add_argument("--objective", choices=["pay", "surplus"], default="pay")
    demo.add_argument("--risk-aversion", type=float, default=0.5)

    args = parser.parse_args(argv)

    if args.command == "fetch":
        if args.source == "greenhouse":
            postings = fetch_greenhouse(args.name)
        elif args.source == "lever":
            postings = fetch_lever(args.name)
        else:
            if not (args.site and args.job_id):
                parser.error("oracle needs --site and --job-id")
            postings = fetch_oracle(args.name, args.site, args.job_id, args.employer)
        print(f"Saved {save_postings(args.db, postings)} postings to {args.db}")
        return 0
    if args.command == "import":
        print(f"Saved {save_postings(args.db, import_csv(args.csv))} postings to {args.db}")
        return 0
    if args.command == "match":
        posting = next((p for p in load_postings(args.db) if p.id == args.posting_id), None)
        if posting is None:
            parser.error(f"No posting with id {args.posting_id}")
        result = keyword_match(Path(args.resume).read_text(encoding="utf-8"), posting.description)
        print(f"Similarity: {result.similarity:.0%}")
        print("Covered keywords:", ", ".join(result.covered) or "none")
        print("Missing keywords:", ", ".join(result.missing) or "none")
        return 0

    if args.command == "demo":
        out = Path(args.out)
        postings = import_csv(EXAMPLES / "sample_postings.csv")
        save_postings(out / "postings.db", postings)
        run_market(postings, EXAMPLES / "profile.example.json", args.objective, args.risk_aversion, out,
                   "Toronto sample, Nov 2026", today=SAMPLE_DATE)
        run_search(EXAMPLES / "sample_applications.csv", date.fromisoformat(SAMPLE_DATE), 21, 0.8, out,
                   "FICTIONAL demo journal")
        return 0
    if args.command == "labour":
        paths = [Path(f) for f in args.files] if args.files else [
            fetch_month(int(m[:4]), int(m[5:7]), args.data_dir) for m in args.months]
        run_labour(paths, Path(args.out))
        return 0
    if args.command == "market":
        run_market(load_postings(args.db), Path(args.profile), args.objective, args.risk_aversion, Path(args.out),
                   f"{args.db}")
        return 0
    today = date.fromisoformat(args.today) if args.today else date.today()
    run_search(Path(args.journal), today, args.window, args.target, Path(args.out), str(args.journal))
    return 0


def run_market(postings, profile_path: Path, objective: str, risk_aversion: float, out: Path, label: str,
               today: str | None = None) -> None:
    profile = Profile.from_json(profile_path)
    assumptions = Assumptions()
    jobs = evaluate_all(mark_closed(postings, today=today), profile, assumptions)
    result = optimize(jobs, profile, risk_aversion, objective)
    summary = market_summary(jobs, DEFAULT_BENCHMARKS)
    title = (f"Toronto job market – {label}: {profile.free_hours:g} free hours/week after a "
             f"{profile.primary_job_hours:g}-hour main job")
    paths = write_market_outputs(jobs, result, profile, assumptions, DEFAULT_BENCHMARKS, summary, out, title)
    print(title)
    print(ranking_table(jobs))
    print(f"\nBest combination: {result.best.label} — expected ${result.best.expected_pay:,.2f}/week "
          f"(± ${result.best.pay_std:,.2f}), {result.best.total_time:.1f} h/week including commuting")
    print(f"Market outputs: {paths['market.md']}")


SOURCE_NOTE = ("Source: Employment and Social Development Canada, \"Job Postings Advertised on Canada's National "
               "Job Bank Website\" (open.canada.ca). Contains information licensed under the Open Government "
               "Licence – Canada.")


def run_labour(paths: list[Path], out: Path) -> None:
    months = [(month_label(p), in_city(load_jobbank(p))) for p in sorted(paths)]
    latest_label, latest_rows = months[-1]
    summary = summarize(latest_rows, DEFAULT_BENCHMARKS, latest_label)
    bunching = [minimum_wage_bunching(rows, label) for label, rows in months]
    base = summary.entry_level_median or summary.median or 0.0
    grid = commute_tax(base, SHIFT_HOURS, COMMUTE_MINUTES)
    title = f"Toronto labour market – Job Bank postings first published in {latest_label}"
    paths_out = write_labour_outputs(summary, bunching, grid, DEFAULT_BENCHMARKS, out, title, SOURCE_NOTE)
    print(paths_out["labour.md"].read_text(encoding="utf-8"))


def run_search(journal_path: Path, today: date, window: int, target: float, out: Path, label: str) -> None:
    apps = load_journal(journal_path)
    rng = np.random.default_rng(2026)
    fun = funnel(apps, today, window)
    overall = stage_rates(apps, today, rng, window_days=window)
    by_channel = rates_by(apps, "channel", "interview", today, rng, window_days=window)
    by_resume = rates_by(apps, "resume_version", "interview", today, rng, window_days=window)
    ab = None
    if len(by_resume) >= 2:
        ab = (by_resume[0].label, by_resume[1].label, probability_better(by_resume[0], by_resume[1], rng))
    sim = simulate_pipeline(apps, overall["interview"], overall["offer_given_interview"], rng)
    curve = offer_probability_curve(overall["interview"], overall["offer_given_interview"], rng)
    needed = applications_needed(curve["p_offer_mean"], target)
    title = f"Job search performance – {label}, as of {today.isoformat()}"
    paths = write_search_outputs(fun, overall, by_channel, by_resume, ab, sim, curve, needed, target, out, title)
    print(paths["search.md"].read_text(encoding="utf-8"))
