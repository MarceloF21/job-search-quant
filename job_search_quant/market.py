"""The Toronto postings as a market: wage distribution, benchmarks and screening mix."""

from __future__ import annotations

import math
import statistics
from collections import Counter

from .benchmarks import Benchmark
from .economics import JobEconomics


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    if len(values) < 2:
        v = values[0] if values else math.nan
        return v, v, v
    q1, q2, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, q2, q3


def market_summary(jobs: list[JobEconomics], benchmarks: tuple[Benchmark, ...]) -> dict:
    priced = [j for j in jobs if not math.isnan(j.wage)]
    posted = [j.wage for j in priced]
    effective = [j.effective_wage for j in priced if not math.isnan(j.effective_wage)]
    summary = {
        "postings": len(jobs),
        "with_pay": len(priced),
        "source_labels": dict(Counter(j.posting.source_label for j in jobs)),
        "screening": dict(Counter(j.status for j in jobs)),
        "posted_wage_quartiles": _quartiles(posted),
        "effective_wage_quartiles": _quartiles(effective),
        "avg_gap_posted_minus_effective": (
            statistics.mean(j.wage - j.effective_wage for j in priced if not math.isnan(j.effective_wage))
            if effective else math.nan),
        "benchmarks": {},
    }
    for bench in benchmarks:
        summary["benchmarks"][bench.name] = {
            "hourly": bench.hourly,
            "source": bench.source,
            "as_of": bench.as_of,
            "share_posted_at_or_above": (sum(w >= bench.hourly for w in posted) / len(posted)) if posted else math.nan,
            "share_effective_at_or_above": (
                sum(w >= bench.hourly for w in effective) / len(effective)) if effective else math.nan,
        }
    return summary
