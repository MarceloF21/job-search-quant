from datetime import date

import numpy as np
import pytest

from job_search_quant.benchmarks import Benchmark
from job_search_quant.economics import evaluate
from job_search_quant.journal import Application, funnel, load_journal
from job_search_quant.market import market_summary
from job_search_quant.models import Posting, Profile
from job_search_quant.odds import Prior, applications_needed, posterior_rate, probability_better, stage_rates
from job_search_quant.simulate import offer_probability_curve, simulate_pipeline

TODAY = date(2026, 11, 8)


def app(id_, applied, **kw):
    return Application(id=id_, employer="E", title="T", category="c", channel=kw.pop("channel", "email"),
                       applied=date.fromisoformat(applied), **kw)


def test_maturity_and_funnel():
    apps = [
        app("old-silent", "2026-09-01"),  # past the window: counts as no response
        app("new-silent", "2026-11-01"),  # too recent: excluded from rates
        app("replied", "2026-11-02", responded=date(2026, 11, 4)),
        app("offer", "2026-10-01", responded=date(2026, 10, 3), interview=date(2026, 10, 6),
            offer=date(2026, 10, 9), outcome="accepted"),
    ]
    fun = funnel(apps, TODAY, window_days=21)
    assert fun.total == 4 and fun.mature == 3
    assert fun.counts == {"applied": 3, "responded": 2, "interview": 1, "offer": 1}
    assert fun.median_days_to_response == pytest.approx(2.0)


def test_dates_must_not_go_backwards():
    with pytest.raises(ValueError):
        app("bad", "2026-10-10", responded=date(2026, 10, 1))


def test_beta_posterior_matches_closed_form():
    rate = posterior_rate("x", successes=3, trials=10, prior=Prior(0.2, 10), rng=np.random.default_rng(0))
    assert (rate.a, rate.b) == pytest.approx((5.0, 15.0))
    assert rate.mean == pytest.approx(0.25)
    assert rate.low < rate.mean < rate.high


def test_more_evidence_narrows_the_interval():
    rng = np.random.default_rng(1)
    small = posterior_rate("s", 2, 10, Prior(0.2, 2), rng)
    large = posterior_rate("l", 20, 100, Prior(0.2, 2), rng)
    assert (large.high - large.low) < (small.high - small.low)


def test_probability_better_and_applications_needed():
    rng = np.random.default_rng(2)
    strong = posterior_rate("strong", 30, 60, Prior(0.5, 2), rng)
    weak = posterior_rate("weak", 6, 60, Prior(0.5, 2), rng)
    assert probability_better(strong, weak, rng) > 0.99
    assert applications_needed(0.2, 0.8) == 8  # 1 - 0.8**8 = 0.832 >= 0.8; 7 gives 0.79


def test_pipeline_simulation_counts_open_applications_only():
    apps = load_journal("examples/sample_applications.csv")
    rng = np.random.default_rng(3)
    rates = stage_rates(apps, TODAY, rng)
    sim = simulate_pipeline(apps, rates["interview"], rates["offer_given_interview"], rng, simulations=5000)
    assert sim.offers_in_hand == 1                       # A15 holds an offer
    assert sim.offers.max() <= 4                         # A12 + A16 + A17 + A18 are still open
    assert 0 < sim.prob_at_least_one < 1
    curve = offer_probability_curve(rates["interview"], rates["offer_given_interview"], rng, max_applications=10)
    assert curve["mean"][0] == 0 and np.all(np.diff(curve["mean"]) > 0)


def test_market_summary_against_benchmarks():
    profile = Profile(available=("evening",))
    jobs = [evaluate(Posting(id=str(w), employer="E", title="T", source_label="verified", wage_min=w,
                             hours_min=10, hours_max=10, schedule=("evening",), commute_minutes=0), profile)
            for w in (16.0, 18.0, 20.0, 30.0)]
    summary = market_summary(jobs, (Benchmark("min", 17.95, "s", "d"), Benchmark("living", 27.2, "s", "d")))
    assert summary["benchmarks"]["min"]["share_posted_at_or_above"] == pytest.approx(0.75)
    assert summary["benchmarks"]["living"]["share_posted_at_or_above"] == pytest.approx(0.25)
    assert summary["posted_wage_quartiles"][1] == pytest.approx(19.0)
