import pytest

from job_search_quant.economics import evaluate
from job_search_quant.models import Assumptions, Posting, Profile
from job_search_quant.optimizer import optimize

PROFILE = Profile(weekly_hour_budget=40, primary_job_hours=20, available=("evening", "weekend"),
                  reservation_wage=10.0, max_jobs=2)
A = Assumptions(shift_hours=10)


def job(id_, wage, hours, std=0.0):
    lo, hi = hours - 2 * std, hours + 2 * std
    return evaluate(Posting(id=id_, employer="E", title=id_, source_label="verified", wage_min=wage,
                            hours_min=lo, hours_max=hi, schedule=("evening",), commute_minutes=0), PROFILE, A)


def test_respects_time_budget_and_picks_highest_pay_when_risk_neutral():
    jobs = [job("a", 20, 10), job("b", 25, 10), job("c", 29, 15)]
    result = optimize(jobs, PROFILE, risk_aversion=0)
    assert all(c.total_time <= 20 + 1e-9 for c in result.combinations)  # a + c or b + c would need 25 h
    assert {j.posting.id for j in result.best.jobs} == {"a", "b"}  # 450 beats c alone (435)
    assert result.best.expected_pay == pytest.approx(450)


def test_ties_go_to_the_option_that_uses_less_time():
    jobs = [job("a", 20, 10), job("b", 25, 10), job("c", 30, 15)]  # a + b and c both pay 450
    assert [j.posting.id for j in optimize(jobs, PROFILE, risk_aversion=0).best.jobs] == ["c"]


def test_risk_aversion_prefers_steady_hours():
    steady, volatile = job("steady", 20, 10, std=0.0), job("volatile", 24, 10, std=4.0)
    one_job = Profile(**{**PROFILE.to_dict(), "max_jobs": 1})
    assert optimize([steady, volatile], one_job, risk_aversion=0).best.jobs[0].posting.id == "volatile"
    assert optimize([steady, volatile], one_job, risk_aversion=1).best.jobs[0].posting.id == "steady"


def test_frontier_is_not_dominated():
    jobs = [job("a", 20, 10, 1), job("b", 25, 10, 3), job("c", 18, 5, 0)]
    result = optimize(jobs, PROFILE, risk_aversion=0.5)
    for f in result.frontier:
        assert not any(c.expected_pay > f.expected_pay + 1e-9 and c.pay_std <= f.pay_std - 1e-9
                       for c in result.combinations)


def test_surplus_objective_can_choose_no_job():
    low = job("low", 9, 10)  # pays below the 10/h reservation wage
    assert optimize([low], PROFILE, objective="surplus").best.jobs == ()
    assert optimize([low], PROFILE, objective="pay").best.jobs[0].posting.id == "low"
