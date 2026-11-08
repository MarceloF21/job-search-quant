"""Screen each job and price it the way an economist would.

Effective hourly wage = (expected weekly pay - weekly share of one-off costs)
                        / (paid hours + unpaid commute hours)

Weekly surplus = what the job pays above the job seeker's reservation wage
                 for all the time it takes (work + commute).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .models import SCHEDULE_WINDOWS, Assumptions, Posting, Profile

HARD_REQUIREMENTS = ("drivers_licence", "forklift", "security_licence", "french")


@dataclass
class JobEconomics:
    posting: Posting
    status: str  # fits | partial | blocked | clash | too_far | closed | unverified | excluded | no_pay
    reasons: list[str] = field(default_factory=list)
    wage: float = math.nan
    expected_hours: float = math.nan
    hours_std: float = math.nan
    hours_assumed: bool = False
    weekly_pay: float = math.nan
    pay_std: float = math.nan
    commute_hours: float = math.nan
    setup_cost_weekly: float = 0.0
    to_obtain: list[str] = field(default_factory=list)
    effective_wage: float = math.nan
    weekly_surplus: float = math.nan
    schedule_share: float = math.nan

    @property
    def eligible(self) -> bool:
        return self.status in ("fits", "partial")

    @property
    def net_weekly_pay(self) -> float:
        return self.weekly_pay - self.setup_cost_weekly

    @property
    def weekly_time(self) -> float:
        return self.expected_hours + self.commute_hours


def evaluate(posting: Posting, profile: Profile, assumptions: Assumptions | None = None) -> JobEconomics:
    a = assumptions or Assumptions()
    job = JobEconomics(posting=posting, status="fits")

    # --- screens, most decisive first ---
    if posting.source_label == "closed":
        job.status, job.reasons = "closed", ["The employer has closed this posting."]
    elif posting.source_label == "lead":
        job.status, job.reasons = "unverified", ["Only a search snippet; find the full posting first."]
    elif posting.employer in profile.excluded_employers:
        job.status, job.reasons = "excluded", ["Employer is on the exclusion list."]

    blockers = _blockers(posting, profile)
    if blockers and job.status == "fits":
        job.status, job.reasons = "blocked", blockers

    windows = [w for w in posting.schedule if w in SCHEDULE_WINDOWS]
    if windows:
        job.schedule_share = sum(w in profile.available for w in windows) / len(windows)
        if job.schedule_share == 0 and job.status == "fits":
            job.status, job.reasons = "clash", [f"Works only {', '.join(windows)}; not available then."]
        elif job.schedule_share < 1 and job.status == "fits":
            missing = [w for w in windows if w not in profile.available]
            job.status, job.reasons = "partial", [f"Some shifts fall outside availability: {', '.join(missing)}."]

    commute = posting.commute_minutes if posting.commute_minutes is not None else profile.default_commute_minutes
    if commute > profile.max_commute_minutes and job.status in ("fits", "partial"):
        limit = profile.max_commute_minutes
        job.status, job.reasons = "too_far", [f"{commute:.0f}-minute commute exceeds {limit:.0f}."]

    # --- economics (computed for every job with pay, so screened-out jobs can still be compared) ---
    if posting.wage is None:
        if job.status in ("fits", "partial"):
            job.status = "no_pay"
        job.reasons.append("Pay not stated.")
        return job

    job.wage = posting.wage
    job.expected_hours, job.hours_std, job.hours_assumed = _hours(posting, a)
    job.expected_hours = min(job.expected_hours, profile.free_hours)
    job.weekly_pay = job.wage * job.expected_hours
    job.pay_std = job.wage * job.hours_std
    shifts = math.ceil(job.expected_hours / a.shift_hours) if job.expected_hours > 0 else 0
    job.commute_hours = shifts * 2 * commute / 60

    for req in posting.requirements:
        if req in a.obtainable and req not in profile.certifications:
            cost, hours = a.obtainable[req]
            job.setup_cost_weekly += (cost + hours * profile.reservation_wage) / profile.horizon_weeks
            job.to_obtain.append(req)

    time = job.weekly_time
    if time > 0:
        job.effective_wage = job.net_weekly_pay / time
        job.weekly_surplus = job.net_weekly_pay - profile.reservation_wage * time
    return job


def evaluate_all(
    postings: list[Posting], profile: Profile, assumptions: Assumptions | None = None
) -> list[JobEconomics]:
    jobs = [evaluate(p, profile, assumptions) for p in postings]
    order = {"fits": 0, "partial": 1}
    return sorted(jobs, key=lambda j: (order.get(j.status, 2), -_nan_to(j.effective_wage, -1e9)))


def _blockers(posting: Posting, profile: Profile) -> list[str]:
    reasons = []
    for req in posting.requirements:
        if req == "drivers_licence" and not profile.can_drive:
            reasons.append("Requires a driver's licence or vehicle.")
        elif req == "french" and "french" not in profile.languages:
            reasons.append("Requires French.")
        elif req in ("forklift", "security_licence") and req not in profile.certifications:
            reasons.append(f"Requires {req.replace('_', ' ')}.")
        elif req.startswith("experience_"):
            months = float(req.removeprefix("experience_").removesuffix("m"))
            if profile.experience_months < months:
                reasons.append(f"Requires {months:g} months of directly related experience.")
    return reasons


def _hours(posting: Posting, a: Assumptions) -> tuple[float, float, bool]:
    if posting.hours_min is not None:
        hi = posting.hours_max if posting.hours_max is not None else posting.hours_min
        return (posting.hours_min + hi) / 2, (hi - posting.hours_min) / 4, False
    flags = set(posting.schedule)
    if flags & {"casual", "24_7"}:
        mean, std = a.hours_if_casual
    elif "full_time" in flags:
        mean, std = a.hours_if_full_time
    elif "part_time" in flags:
        mean, std = a.hours_if_part_time
    else:
        mean, std = a.hours_if_unknown
    return mean, std, True


def _nan_to(value: float, default: float) -> float:
    return default if value is None or math.isnan(value) else value
