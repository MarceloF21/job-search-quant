"""Collect postings from employers' documented public job feeds, or from a CSV file.

Only documented, public endpoints are used: postings read this way are labelled
"verified" because they come straight from the employer's own hiring system.
Job boards that block automated access are deliberately not scraped.
"""

from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

from .models import Posting
from .parse import enrich, strip_html

USER_AGENT = "job-search-quant/2.0 (+https://github.com/MarceloF21/quant-trading-platform)"


def _get_json(url: str, timeout: float = 30.0):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https hosts
        return json.loads(response.read().decode("utf-8"))


def _today() -> str:
    return date.today().isoformat()


def fetch_greenhouse(board: str) -> list[Posting]:
    """Greenhouse Job Board API (documented, public): boards-api.greenhouse.io."""
    url = f"https://boards-api.greenhouse.io/v1/boards/{urllib.parse.quote(board)}/jobs?content=true"
    data = _get_json(url)
    postings = []
    for job in data.get("jobs", []):
        postings.append(enrich(Posting(
            id=f"greenhouse:{board}:{job['id']}",
            employer=board,
            title=job.get("title", ""),
            location=(job.get("location") or {}).get("name", ""),
            url=job.get("absolute_url", ""),
            source="greenhouse",
            source_label="verified",
            checked_at=_today(),
            description=strip_html(job.get("content", "")),
        )))
    return postings


def fetch_lever(company: str) -> list[Posting]:
    """Lever Postings API (documented, public): api.lever.co/v0/postings."""
    url = f"https://api.lever.co/v0/postings/{urllib.parse.quote(company)}?mode=json"
    postings = []
    for job in _get_json(url):
        lists = " ".join(f"{item.get('text', '')}: {strip_html(item.get('content', ''))}"
                         for item in job.get("lists", []))
        salary = job.get("salaryRange") or {}
        per_hour = str(salary.get("interval", "")).lower().startswith("per-hour")
        wage_min = wage_max = None
        if salary.get("min"):
            divisor = 1 if per_hour else 2080
            wage_min = round(float(salary["min"]) / divisor, 2)
            wage_max = round(float(salary.get("max", salary["min"])) / divisor, 2)
        categories = job.get("categories") or {}
        postings.append(enrich(Posting(
            id=f"lever:{company}:{job['id']}",
            employer=company,
            title=job.get("text", ""),
            location=categories.get("location", ""),
            url=job.get("hostedUrl", ""),
            source="lever",
            source_label="verified",
            checked_at=_today(),
            description=" ".join([job.get("descriptionPlain", ""), lists, categories.get("commitment", "")]),
            wage_min=wage_min,
            wage_max=wage_max,
        )))
    return postings


def fetch_oracle(host: str, site: str, job_id: str, employer: str | None = None) -> list[Posting]:
    """Oracle Recruiting Cloud's documented REST resource ``recruitingCEJobRequisitionDetails``."""
    finder = f'ById;Id="{job_id}",siteNumber={site}'
    url = (f"https://{host}/hcmRestApi/resources/latest/recruitingCEJobRequisitionDetails"
           f"?expand=all&onlyData=true&finder={urllib.parse.quote(finder, safe=';=,')}")
    items = _get_json(url).get("items", [])
    postings = []
    for item in items:
        end = str(item.get("ExternalPostedEndDate") or "")[:10]
        closed = bool(end) and end < _today()
        postings.append(enrich(Posting(
            id=f"oracle:{host}:{job_id}",
            employer=employer or host.split(".")[0],
            title=item.get("Title", ""),
            location=item.get("PrimaryLocation", ""),
            url=f"https://{host}/hcmUI/CandidateExperience/en/sites/{site}/job/{job_id}",
            source="oracle",
            source_label="closed" if closed else "verified",
            checked_at=_today(),
            closing_date=end,
            description=" ".join(filter(None, [
                strip_html(item.get("ExternalDescriptionStr", "")), item.get("JobSchedule", "")])),
        )))
    return postings


CSV_FIELDS = [
    "id", "employer", "title", "location", "url", "source", "source_label", "checked_at", "closing_date",
    "description", "wage_min", "wage_max", "hours_min", "hours_max", "tips", "schedule", "requirements",
    "commute_minutes",
]


def import_csv(path: str | Path) -> list[Posting]:
    """Read postings from CSV. ``schedule`` and ``requirements`` are ;-separated flag lists."""
    postings = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            def number(key: str, _row=row) -> float | None:
                value = (_row.get(key) or "").strip()
                return float(value) if value else None

            postings.append(enrich(Posting(
                id=row["id"], employer=row["employer"], title=row["title"],
                location=row.get("location", ""), url=row.get("url", ""),
                source=row.get("source") or "csv", source_label=row.get("source_label") or "lead",
                checked_at=row.get("checked_at", ""), closing_date=row.get("closing_date", ""),
                description=row.get("description", ""),
                wage_min=number("wage_min"), wage_max=number("wage_max"),
                hours_min=number("hours_min"), hours_max=number("hours_max"),
                tips=(row.get("tips") or "").strip().lower() in ("1", "true", "yes"),
                schedule=tuple(f for f in (row.get("schedule") or "").split(";") if f),
                requirements=tuple(f for f in (row.get("requirements") or "").split(";") if f),
                commute_minutes=number("commute_minutes"),
            )))
    return postings


def mark_closed(postings: list[Posting], today: str | None = None) -> list[Posting]:
    """Label postings whose closing date has passed as closed."""
    today = today or datetime.now(timezone.utc).date().isoformat()
    for posting in postings:
        if posting.closing_date and posting.closing_date < today and posting.source_label != "closed":
            posting.source_label = "closed"
    return postings
