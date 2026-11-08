"""Data models: a job posting and the job seeker's constraints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SOURCE_LABELS = ("verified", "mirror", "lead", "closed")
SCHEDULE_WINDOWS = ("day", "evening", "night", "weekend", "holiday")


@dataclass
class Posting:
    """One job posting. Pay and hours can be given directly or parsed from the description."""

    id: str
    employer: str
    title: str
    location: str = ""
    url: str = ""
    source: str = "manual"
    source_label: str = "lead"  # verified = employer's own page or feed, mirror = job board, lead = snippet
    checked_at: str = ""
    closing_date: str = ""
    description: str = ""
    wage_min: float | None = None
    wage_max: float | None = None
    hours_min: float | None = None
    hours_max: float | None = None
    tips: bool = False
    schedule: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    commute_minutes: float | None = None  # one way, estimated by the user

    def __post_init__(self) -> None:
        if self.source_label not in SOURCE_LABELS:
            raise ValueError(f"source_label must be one of {SOURCE_LABELS}, got {self.source_label!r}")
        self.schedule = tuple(self.schedule)
        self.requirements = tuple(self.requirements)

    @property
    def wage(self) -> float | None:
        """Midpoint of the posted hourly range."""
        if self.wage_min is None:
            return None
        return self.wage_min if self.wage_max is None else (self.wage_min + self.wage_max) / 2


@dataclass
class Profile:
    """The job seeker's time budget, availability and qualifications."""

    name: str = "Sample student"
    weekly_hour_budget: float = 45.0  # all paid work plus commuting, per week
    primary_job_hours: float = 20.0  # hours already committed to a main job
    available: tuple[str, ...] = ("day", "evening", "weekend")
    certifications: tuple[str, ...] = ()
    languages: tuple[str, ...] = ("english",)
    can_drive: bool = False
    experience_months: float = 0.0  # in the kind of role a posting asks experience for
    reservation_wage: float = 17.95  # value of an hour of free time (default: Ontario minimum wage, Oct 2026)
    default_commute_minutes: float = 40.0
    max_commute_minutes: float = 75.0
    max_jobs: int = 2  # extra jobs to combine on top of the main job
    horizon_weeks: int = 20  # weeks over which one-off costs (certificates) are spread
    excluded_employers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = set(self.available) - set(SCHEDULE_WINDOWS)
        if unknown:
            raise ValueError(f"Unknown availability windows: {sorted(unknown)}")
        for name in ("available", "certifications", "languages", "excluded_employers"):
            setattr(self, name, tuple(getattr(self, name)))

    @property
    def free_hours(self) -> float:
        return max(0.0, self.weekly_hour_budget - self.primary_job_hours)

    @classmethod
    def from_json(cls, path: str | Path) -> Profile:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown profile fields: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Assumptions:
    """Values used when a posting doesn't say. All are editable and reported with the results."""

    shift_hours: float = 6.0  # used to count commutes per week
    hours_if_casual: tuple[float, float] = (8.0, 6.0)  # (expected weekly hours, std) for casual/on-call/event work
    hours_if_part_time: tuple[float, float] = (16.0, 4.0)
    hours_if_full_time: tuple[float, float] = (37.5, 2.0)
    hours_if_unknown: tuple[float, float] = (12.0, 6.0)
    # requirement -> (cost in dollars, hours to obtain) when it can be earned quickly
    obtainable: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"smart_serve": (40.0, 4.0), "food_handler": (60.0, 8.0)}
    )
