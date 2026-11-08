"""Job Search Quant: a quant-style platform for a student's job search.

Postings are the market (screened, priced by effective hourly wage, combined like a
portfolio) and applications are the trade log (funnel, Bayesian success rates,
Monte Carlo odds of landing an offer)."""

from .benchmarks import DEFAULT_BENCHMARKS, GTA_LIVING_WAGE, ONTARIO_MINIMUM_WAGE, Benchmark
from .economics import JobEconomics, evaluate, evaluate_all
from .journal import Application, Funnel, funnel, load_journal
from .labour import LabourSummary, commute_tax, hourly_wage, in_city, minimum_wage_bunching, summarize
from .market import market_summary
from .match import MatchResult, keyword_match
from .models import Assumptions, Posting, Profile
from .odds import DEFAULT_PRIORS, Prior, Rate, applications_needed, probability_better, rates_by, stage_rates
from .optimizer import Combination, OptimizationResult, efficient_frontier, optimize
from .parse import enrich, parse_hours, parse_wage
from .simulate import PipelineSimulation, offer_probability_curve, simulate_pipeline

__version__ = "3.1.0"

__all__ = [
    "Application", "Assumptions", "Benchmark", "Combination", "DEFAULT_BENCHMARKS", "DEFAULT_PRIORS", "Funnel",
    "GTA_LIVING_WAGE", "JobEconomics", "LabourSummary", "MatchResult", "ONTARIO_MINIMUM_WAGE",
    "OptimizationResult", "PipelineSimulation", "Posting", "Prior", "Profile", "Rate", "applications_needed",
    "commute_tax", "efficient_frontier", "enrich", "evaluate", "evaluate_all", "funnel", "hourly_wage",
    "in_city", "keyword_match", "load_journal", "market_summary", "minimum_wage_bunching",
    "offer_probability_curve", "optimize", "parse_hours", "parse_wage", "probability_better", "rates_by",
    "simulate_pipeline", "stage_rates", "summarize",
]
