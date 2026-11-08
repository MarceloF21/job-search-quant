"""Monte Carlo view of the open pipeline and of future applications.

Each simulation draws the success rates from their posteriors (so parameter
uncertainty is included), then lets every open application succeed or fail.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .journal import Application
from .odds import Rate


@dataclass
class PipelineSimulation:
    simulations: int
    offers: np.ndarray  # new offers per simulation
    prob_at_least_one: float
    expected_offers: float
    offers_in_hand: int
    best_weekly_pay: np.ndarray  # best new offer's weekly pay per simulation (0 if none)


def simulate_pipeline(apps: list[Application], interview: Rate, offer_given_interview: Rate,
                      rng: np.random.Generator, simulations: int = 20_000) -> PipelineSimulation:
    """New offers from applications that are still open (offers already received are counted separately)."""
    waiting = [a for a in apps if a.outcome == "open" and not a.reached("offer")]
    in_hand = sum(a.outcome == "open" and a.reached("offer") for a in apps)
    p_int = rng.beta(interview.a, interview.b, simulations)
    p_off = rng.beta(offer_given_interview.a, offer_given_interview.b, simulations)
    offers = np.zeros(simulations, dtype=int)
    best = np.zeros(simulations)
    for app in waiting:
        p = p_off if app.reached("interview") else p_int * p_off
        hit = rng.random(simulations) < p
        offers += hit
        pay = (app.hourly_wage or 0.0) * (app.expected_hours or 0.0)
        best = np.where(hit, np.maximum(best, pay), best)
    return PipelineSimulation(simulations, offers, float(np.mean(offers > 0)), float(offers.mean()),
                              in_hand, best)


def offer_probability_curve(interview: Rate, offer_given_interview: Rate, rng: np.random.Generator,
                            max_applications: int = 40, simulations: int = 20_000) -> dict[str, np.ndarray]:
    """P(at least one offer) after n more applications, with a 90% band from parameter uncertainty."""
    p = rng.beta(interview.a, interview.b, simulations) * rng.beta(
        offer_given_interview.a, offer_given_interview.b, simulations)
    n = np.arange(0, max_applications + 1)
    curves = 1 - (1 - p[:, None]) ** n[None, :]
    return {"n": n, "mean": curves.mean(axis=0), "low": np.percentile(curves, 5, axis=0),
            "high": np.percentile(curves, 95, axis=0), "p_offer_mean": float(p.mean())}
