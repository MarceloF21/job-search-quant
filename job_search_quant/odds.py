"""Bayesian success rates for the job search.

With a few dozen applications, raw percentages swing wildly (1 interview out of 3 is
not a 33% interview rate). Each rate gets a Beta prior, updated with the journal's
counts, which gives a posterior mean and a 90% credible interval that narrows as the
evidence grows. The priors are stated assumptions; change them to your own beliefs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from .journal import Application


@dataclass(frozen=True)
class Prior:
    mean: float
    strength: float  # pseudo-applications of prior belief; small = let the data speak

    @property
    def ab(self) -> tuple[float, float]:
        return self.mean * self.strength, (1 - self.mean) * self.strength


DEFAULT_PRIORS = {
    "response": Prior(0.25, 8),  # an employer replies at all
    "interview": Prior(0.12, 8),  # an application leads to an interview
    "offer_given_interview": Prior(0.35, 6),  # an interview leads to an offer
}


@dataclass
class Rate:
    label: str
    successes: int
    trials: int
    a: float
    b: float
    mean: float
    low: float  # 5th percentile
    high: float  # 95th percentile


def posterior_rate(label: str, successes: int, trials: int, prior: Prior,
                   rng: np.random.Generator, draws: int = 20_000) -> Rate:
    a0, b0 = prior.ab
    a, b = a0 + successes, b0 + trials - successes
    sample = rng.beta(a, b, size=draws)
    return Rate(label, successes, trials, a, b, a / (a + b),
                float(np.percentile(sample, 5)), float(np.percentile(sample, 95)))


def stage_rates(apps: list[Application], today: date, rng: np.random.Generator,
                priors: dict[str, Prior] | None = None, window_days: int = 21, label: str = "all") -> dict[str, Rate]:
    priors = priors or DEFAULT_PRIORS
    mature = [a for a in apps if a.is_mature(today, window_days)]
    interviews = [a for a in apps if a.interview_decided()]
    return {
        "response": posterior_rate(label, sum(a.reached("responded") for a in mature), len(mature),
                                   priors["response"], rng),
        "interview": posterior_rate(label, sum(a.reached("interview") for a in mature), len(mature),
                                    priors["interview"], rng),
        "offer_given_interview": posterior_rate(label, sum(a.reached("offer") for a in interviews),
                                                len(interviews), priors["offer_given_interview"], rng),
    }


def rates_by(apps: list[Application], key: str, stage: str, today: date, rng: np.random.Generator,
             priors: dict[str, Prior] | None = None, window_days: int = 21) -> list[Rate]:
    """Posterior rate of ``stage`` for each value of ``key`` (e.g. channel or resume_version)."""
    groups: dict[str, list[Application]] = {}
    for app in apps:
        groups.setdefault(getattr(app, key) or "unknown", []).append(app)
    rates = [stage_rates(group, today, rng, priors, window_days, label=name)[stage]
             for name, group in sorted(groups.items())]
    return sorted(rates, key=lambda r: -r.mean)


def probability_better(first: Rate, second: Rate, rng: np.random.Generator, draws: int = 50_000) -> float:
    """Posterior probability that ``first`` has the higher true rate (a Bayesian A/B test)."""
    return float(np.mean(rng.beta(first.a, first.b, draws) > rng.beta(second.a, second.b, draws)))


def applications_needed(p_offer: float, target: float = 0.8) -> int:
    """Applications needed so that P(at least one offer) >= target, if each succeeds with p_offer."""
    if not 0 < p_offer < 1:
        raise ValueError("p_offer must be between 0 and 1")
    return math.ceil(math.log(1 - target) / math.log(1 - p_offer))
