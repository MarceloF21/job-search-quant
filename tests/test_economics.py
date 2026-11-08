import math

import pytest

from job_search_quant.economics import evaluate
from job_search_quant.models import Assumptions, Posting, Profile


def posting(**kw):
    base = dict(id="x", employer="E", title="T", source_label="verified", wage_min=20.0,
                hours_min=12.0, hours_max=12.0, schedule=("evening",), commute_minutes=30)
    base.update(kw)
    return Posting(**base)


PROFILE = Profile(available=("evening", "weekend"), reservation_wage=15.0)


def test_effective_wage_counts_unpaid_commute():
    job = evaluate(posting(), PROFILE, Assumptions(shift_hours=6))
    # 12 h -> 2 shifts -> 2 round trips of 30 min each way = 2 h unpaid
    assert job.commute_hours == pytest.approx(2.0)
    assert job.weekly_pay == pytest.approx(240.0)
    assert job.effective_wage == pytest.approx(240 / 14)
    assert job.weekly_surplus == pytest.approx(240 - 15 * 14)
    assert job.status == "fits"


def test_certificate_cost_is_spread_over_the_horizon():
    job = evaluate(posting(requirements=("smart_serve",)), PROFILE,
                   Assumptions(shift_hours=6, obtainable={"smart_serve": (40.0, 4.0)}))
    assert job.to_obtain == ["smart_serve"]
    assert job.setup_cost_weekly == pytest.approx((40 + 4 * 15) / PROFILE.horizon_weeks)


@pytest.mark.parametrize(("kw", "status"), [
    (dict(requirements=("drivers_licence",)), "blocked"),
    (dict(requirements=("french",)), "blocked"),
    (dict(requirements=("experience_6m",)), "blocked"),
    (dict(schedule=("night",)), "clash"),
    (dict(schedule=("night", "evening")), "partial"),
    (dict(commute_minutes=120), "too_far"),
    (dict(source_label="closed"), "closed"),
    (dict(source_label="lead"), "unverified"),
    (dict(wage_min=None), "no_pay"),
])
def test_screening(kw, status):
    assert evaluate(posting(**kw), PROFILE).status == status


def test_missing_hours_use_reported_assumptions():
    job = evaluate(posting(hours_min=None, hours_max=None, schedule=("casual", "evening")), PROFILE,
                   Assumptions(hours_if_casual=(8.0, 6.0)))
    assert job.hours_assumed
    assert (job.expected_hours, job.hours_std) == (8.0, 6.0)
    assert job.pay_std == pytest.approx(120.0)


def test_hours_are_capped_by_free_time():
    job = evaluate(posting(hours_min=40, hours_max=40), Profile(weekly_hour_budget=30, primary_job_hours=20,
                                                                available=("evening",)))
    assert job.expected_hours == 10
    assert not math.isnan(job.effective_wage)
