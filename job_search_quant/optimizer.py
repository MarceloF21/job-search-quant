"""Choose the best combination of jobs, treating them like assets in a portfolio.

Each job has an expected weekly pay and a standard deviation (casual and event work
have uncertain hours; guaranteed hours have almost none). Combinations must fit in the
weekly time budget left after the main job. The efficient frontier lists combinations
that no other feasible combination beats on both expected pay and risk; the chosen
combination maximizes  E[value] - risk_aversion x std(pay), where value is either
weekly pay ("pay": maximize earnings) or weekly surplus over the reservation wage
("surplus": also counts the value of the free time a job uses up).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from .economics import JobEconomics
from .models import Profile


@dataclass
class Combination:
    jobs: tuple[JobEconomics, ...]
    expected_pay: float
    expected_surplus: float
    pay_std: float
    paid_hours: float
    total_time: float
    score: float

    @property
    def label(self) -> str:
        return " + ".join(f"{j.posting.title} ({j.posting.employer})" for j in self.jobs) or "No extra job"


@dataclass
class OptimizationResult:
    combinations: list[Combination]
    frontier: list[Combination]
    best: Combination
    risk_aversion: float
    objective: str
    free_hours: float


def optimize(
    jobs: list[JobEconomics], profile: Profile, risk_aversion: float = 0.5, objective: str = "pay"
) -> OptimizationResult:
    """Enumerate every feasible combination of up to ``profile.max_jobs`` eligible jobs."""
    if risk_aversion < 0:
        raise ValueError("risk_aversion cannot be negative")
    if objective not in ("pay", "surplus"):
        raise ValueError("objective must be 'pay' or 'surplus'")
    candidates = [j for j in jobs if j.eligible and not math.isnan(j.weekly_pay)]
    budget = profile.free_hours
    combos = [Combination((), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    for k in range(1, profile.max_jobs + 1):
        for group in itertools.combinations(candidates, k):
            time = sum(j.weekly_time for j in group)
            if time > budget + 1e-9:
                continue
            pay = sum(j.net_weekly_pay for j in group)
            surplus = sum(j.weekly_surplus for j in group)
            std = math.sqrt(sum(j.pay_std ** 2 for j in group))  # hours assumed independent across jobs
            value = pay if objective == "pay" else surplus
            combos.append(Combination(
                group, pay, surplus, std, sum(j.expected_hours for j in group), time,
                value - risk_aversion * std,
            ))
    frontier = efficient_frontier(combos, objective)
    best = max(combos, key=lambda c: (c.score, -c.total_time))
    return OptimizationResult(combos, frontier, best, risk_aversion, objective, budget)


def efficient_frontier(combos: list[Combination], objective: str = "pay") -> list[Combination]:
    """Combinations not dominated by another with higher-or-equal value and lower-or-equal risk."""
    def value(c: Combination) -> float:
        return c.expected_pay if objective == "pay" else c.expected_surplus

    ordered = sorted(combos, key=lambda c: (c.pay_std, -value(c)))
    frontier, best = [], -math.inf
    for combo in ordered:
        if value(combo) > best + 1e-9:
            frontier.append(combo)
            best = value(combo)
    return frontier
