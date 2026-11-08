"""The applications journal: the job-search equivalent of a trade log.

Each row is one application with the dates it reached each stage. An application
counts toward a rate only once it is *mature*: it got a reply, it was closed, or the
response window has passed (silence after that counts as no response).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

STAGES = ("applied", "responded", "interview", "offer")
OUTCOMES = ("open", "rejected", "no_response", "withdrawn", "accepted", "declined")


@dataclass
class Application:
    id: str
    employer: str
    title: str
    category: str
    channel: str  # employer_portal | email | job_board | walk_in | referral ...
    applied: date
    resume_version: str = ""
    responded: date | None = None
    interview: date | None = None
    offer: date | None = None
    outcome: str = "open"
    hourly_wage: float | None = None
    expected_hours: float | None = None
    posting_id: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"{self.id}: outcome must be one of {OUTCOMES}, got {self.outcome!r}")
        dates = [self.applied, self.responded, self.interview, self.offer]
        known = [d for d in dates if d is not None]
        if known != sorted(known):
            raise ValueError(f"{self.id}: stage dates must not go backwards")

    def reached(self, stage: str) -> bool:
        return {"applied": True, "responded": self.responded is not None or self.interview is not None
                or self.offer is not None, "interview": self.interview is not None or self.offer is not None,
                "offer": self.offer is not None}[stage]

    @property
    def furthest_stage(self) -> str:
        return next(s for s in reversed(STAGES) if self.reached(s))

    def is_mature(self, today: date, window_days: int = 21) -> bool:
        return (self.outcome != "open" or self.reached("responded")
                or (today - self.applied).days >= window_days)

    def interview_decided(self) -> bool:
        """An interview whose result is known (offer made, or the application closed)."""
        return self.reached("interview") and (self.offer is not None or self.outcome != "open")

    def days_to_response(self) -> int | None:
        first = self.responded or self.interview or self.offer
        return None if first is None else (first - self.applied).days


def _date(value: str) -> date | None:
    value = (value or "").strip()
    return date.fromisoformat(value) if value else None


def _number(value: str) -> float | None:
    value = (value or "").strip()
    return float(value) if value else None


def load_journal(path: str | Path) -> list[Application]:
    with open(path, newline="", encoding="utf-8") as handle:
        return [
            Application(
                id=row["id"], employer=row["employer"], title=row["title"], category=row["category"],
                channel=row["channel"], applied=_date(row["applied"]),
                resume_version=row.get("resume_version", ""),
                responded=_date(row.get("responded", "")), interview=_date(row.get("interview", "")),
                offer=_date(row.get("offer", "")), outcome=(row.get("outcome") or "open").strip(),
                hourly_wage=_number(row.get("hourly_wage", "")),
                expected_hours=_number(row.get("expected_hours", "")),
                posting_id=row.get("posting_id", ""),
            )
            for row in csv.DictReader(handle)
        ]


@dataclass
class Funnel:
    total: int
    mature: int
    counts: dict[str, int]  # among mature applications
    conversion: dict[str, float]  # stage-to-stage, among mature applications
    median_days_to_response: float | None
    open_applications: int


def funnel(apps: list[Application], today: date, window_days: int = 21) -> Funnel:
    mature = [a for a in apps if a.is_mature(today, window_days)]
    counts = {stage: sum(a.reached(stage) for a in mature) for stage in STAGES}
    conversion = {}
    for prev, stage in zip(STAGES, STAGES[1:], strict=False):
        conversion[f"{prev}→{stage}"] = counts[stage] / counts[prev] if counts[prev] else float("nan")
    delays = sorted(d for a in apps if (d := a.days_to_response()) is not None)
    median = None
    if delays:
        mid = len(delays) // 2
        median = float(delays[mid]) if len(delays) % 2 else (delays[mid - 1] + delays[mid]) / 2
    open_apps = sum(a.outcome == "open" and not a.reached("offer") for a in apps)
    return Funnel(len(apps), len(mature), counts, conversion, median, open_apps)
