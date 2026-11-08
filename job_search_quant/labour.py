"""Toronto's labour market from official open data.

Source: "Job Postings Advertised on Canada's National Job Bank Website", Employment
and Social Development Canada, one CSV per month on open.canada.ca. Contains
information licensed under the Open Government Licence – Canada.

Pay is converted to an hourly rate. Hourly postings use the midpoint of their range.
Annual, monthly, weekly and bi-weekly pay is divided by the posting's weekly hours
when stated, otherwise by 40. Daily pay is divided by 8. Values outside $10–$200/hour
are treated as data errors and dropped.
"""

from __future__ import annotations

import csv
import io
import json
import statistics
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .benchmarks import Benchmark

DATASET_ID = "ea639e28-c0fc-48bf-b5dd-b8899bd43072"
PACKAGE_URL = f"https://open.canada.ca/data/api/action/package_show?id={DATASET_ID}"
CITY_OF_TORONTO = frozenset({"Toronto", "North York", "Scarborough", "Etobicoke", "East York", "York"})
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
TEER_LABELS = {
    "0": "TEER 0 Management",
    "1": "TEER 1 University degree",
    "2": "TEER 2 College / apprenticeship (2+ yrs)",
    "3": "TEER 3 College (<2 yrs) / some training",
    "4": "TEER 4 High school",
    "5": "TEER 5 No formal education",
}
ENTRY_EDUCATION = {"No degree, certificate or diploma", "Secondary (high) school graduation certificate"}
ENTRY_EXPERIENCE = {"No experience", "1 to less than 7 months", "7 months to less than 1 year"}
SCHEDULE_COLUMNS = {"evening": "Employment Term Evening", "weekend": "Employment Term Weekend",
                    "night": "Employment Term Night", "on_call": "Employment Term Oncall"}


# ------------------------------------------------------------------ loading
def resource_url(package: dict, year: int, month: int, language: str = "en") -> str:
    """Pick the CSV for one month from the dataset's CKAN metadata."""
    wanted = f"{MONTHS[month - 1]} {year} Job Postings"
    for res in package["result"]["resources"]:
        names = res.get("name_translated") or {}
        name = names.get(language) or res.get("name", "")
        if name.startswith(wanted) and language in (res.get("language") or [language]):
            return res["url"]
    raise LookupError(f"No {language} file for {MONTHS[month - 1]} {year} in the Job Bank dataset.")


def fetch_month(year: int, month: int, data_dir: str | Path = "data/jobbank") -> Path:
    """Download one month's file (about 50 MB) unless it is already on disk."""
    path = Path(data_dir) / f"{year}-{month:02d}.csv"
    if path.exists():
        return path
    with urllib.request.urlopen(PACKAGE_URL, timeout=60) as response:  # noqa: S310 - fixed https host
        package = json.loads(response.read().decode("utf-8"))
    url = resource_url(package, year, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=300) as response, open(tmp, "wb") as out:  # noqa: S310
        while chunk := response.read(1 << 20):
            out.write(chunk)
    tmp.rename(path)
    return path


def load_postings(path: str | Path) -> list[dict[str, str]]:
    """Read a monthly file (UTF-16, tab-separated in recent months; UTF-8 handled too)."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    first = text.split("\n", 1)[0]
    delimiter = "\t" if first.count("\t") > first.count(",") else ","
    return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))


def in_city(rows: list[dict[str, str]], cities: frozenset[str] = CITY_OF_TORONTO,
            province: str = "Ontario") -> list[dict[str, str]]:
    return [r for r in rows if r.get("Province/Territory") == province and r.get("City") in cities]


# ------------------------------------------------------------------ measures
def _num(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "", "NA") else None
    except ValueError:
        return None


def weekly_hours(row: dict[str, str]) -> float | None:
    lo, hi = _num(row.get("Hours Minimum")), _num(row.get("Hours Maximum"))
    if not lo:
        return None
    hours = (lo + (hi or lo)) / 2
    per = row.get("Hours Per")
    return hours / 2 if per == "Bi-weekly" else hours * 12 / 52 if per == "Month" else hours


def hourly_wage(row: dict[str, str]) -> float | None:
    lo, hi = _num(row.get("Salary Minimum")), _num(row.get("Salary Maximum"))
    if lo is None:
        return None
    mid = (lo + (hi or lo)) / 2
    hours = weekly_hours(row) or 40.0
    per = row.get("Salary Per")
    value = {"Hour": mid, "Year": mid / (52 * hours), "Month": mid * 12 / (52 * hours), "Week": mid / hours,
             "Bi-weekly": mid / (2 * hours), "Day": mid / 8}.get(per)
    return value if value is not None and 10 <= value <= 200 else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _quartiles(values: list[float]) -> list[float] | None:
    return statistics.quantiles(values, n=4, method="inclusive") if len(values) >= 2 else None


def is_entry_level(row: dict[str, str]) -> bool:
    """Job Bank's own postings that ask for high school or less and under a year of experience."""
    return (row.get("External Indicator") == "0" and row.get("Education LOS") in ENTRY_EDUCATION
            and row.get("Experience Level") in ENTRY_EXPERIENCE)


@dataclass
class LabourSummary:
    label: str
    postings: int
    vacancies: int
    with_pay: int
    wages: list[float]
    quartiles: list[float] | None
    benchmark_shares: dict[str, float]
    part_time_wages: list[float]
    full_time_wages: list[float]
    teer_medians: dict[str, tuple[float, int]]
    category_medians: dict[str, tuple[float, int]]
    entry_level_n: int
    entry_level_wages: list[float]
    entry_level_schedule: dict[str, float]

    @property
    def median(self) -> float | None:
        return _median(self.wages)

    @property
    def part_time_median(self) -> float | None:
        return _median(self.part_time_wages)

    @property
    def full_time_median(self) -> float | None:
        return _median(self.full_time_wages)

    @property
    def entry_level_median(self) -> float | None:
        return _median(self.entry_level_wages)


NOC_CATEGORIES = {
    "0": "Legislative & senior management", "1": "Business, finance & administration",
    "2": "Natural & applied sciences", "3": "Health", "4": "Education, law, social & government",
    "5": "Art, culture, recreation & sport", "6": "Sales & service", "7": "Trades & transport",
    "8": "Natural resources & agriculture", "9": "Manufacturing & utilities",
}


def summarize(rows: list[dict[str, str]], benchmarks: tuple[Benchmark, ...], label: str) -> LabourSummary:
    priced = [(r, w) for r in rows if (w := hourly_wage(r)) is not None]
    wages = [w for _, w in priced]

    def medians_by(key) -> dict[str, tuple[float, int]]:
        groups: dict[str, list[float]] = {}
        for r, w in priced:
            k = key(r)
            if k is not None:
                groups.setdefault(k, []).append(w)
        return {k: (statistics.median(v), len(v)) for k, v in sorted(groups.items())}

    part = [w for r, w in priced if (r.get("Employment Type") or "").startswith("Part time")]
    full = [w for r, w in priced if r.get("Employment Type") == "Full time"]
    entry = [r for r in rows if is_entry_level(r)]
    entry_wages = [w for r in entry if (w := hourly_wage(r)) is not None]
    return LabourSummary(
        label=label,
        postings=len(rows),
        vacancies=sum(int(_num(r.get("Vacancy Count")) or 1) for r in rows),
        with_pay=len(wages),
        wages=wages,
        quartiles=_quartiles(wages),
        benchmark_shares={b.name: (sum(w >= b.hourly for w in wages) / len(wages) if wages else float("nan"))
                          for b in benchmarks},
        part_time_wages=part,
        full_time_wages=full,
        teer_medians=medians_by(lambda r: TEER_LABELS.get((r.get("NOC21 Code") or "  ")[1:2])),
        category_medians=medians_by(lambda r: NOC_CATEGORIES.get((r.get("NOC21 Code") or " ")[:1])),
        entry_level_n=len(entry),
        entry_level_wages=entry_wages,
        entry_level_schedule={name: (sum(r.get(col) == "Yes" for r in entry) / len(entry) if entry else float("nan"))
                              for name, col in SCHEDULE_COLUMNS.items()},
    )


@dataclass
class Bunching:
    label: str
    hourly_postings: int
    at_old: int
    at_new: int
    below_new: int


def minimum_wage_bunching(rows: list[dict[str, str]], label: str, old: float = 17.60,
                          new: float = 17.95) -> Bunching:
    """Count hourly postings whose minimum pay sits exactly at the old or new minimum wage."""
    pays = [p for r in rows if r.get("Salary Per") == "Hour" and (p := _num(r.get("Salary Minimum"))) is not None]
    return Bunching(label, len(pays), sum(abs(p - old) < 0.005 for p in pays),
                    sum(abs(p - new) < 0.005 for p in pays), sum(p < new - 0.005 for p in pays))


def commute_tax(wage: float, shift_hours: list[float], commute_minutes: list[float]) -> list[list[float]]:
    """Effective hourly wage when every shift needs a round trip of 2 x commute minutes, unpaid."""
    return [[wage * h / (h + 2 * c / 60) for c in commute_minutes] for h in shift_hours]


def month_label(path: str | Path) -> str:
    stem = Path(path).stem  # YYYY-MM
    try:
        year, month = stem.split("-")
        return f"{MONTHS[int(month) - 1][:3]} {year}"
    except ValueError:
        return stem


def occupations(rows: list[dict[str, str]], top: int = 10) -> list[tuple[str, int]]:
    return Counter(r.get("NOC21 Code Name", "") for r in rows).most_common(top)
