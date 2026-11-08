import json

import pytest

from job_search_quant import sources
from job_search_quant.cli import main
from job_search_quant.db import load_postings, save_postings
from job_search_quant.match import keyword_match


def test_keyword_match_lists_missing_terms():
    posting = "Cash handling and customer service at banquet events. Smart Serve certification required."
    resume = "Customer service and cash handling at a bank."
    result = keyword_match(resume, posting)
    assert 0 < result.similarity < 1
    assert "cash handling" in result.covered
    assert any("banquet" in term for term in result.missing)


def test_greenhouse_feed_is_parsed(monkeypatch):
    payload = {"jobs": [{"id": 7, "title": "Barista", "location": {"name": "Toronto"},
                         "absolute_url": "https://example.com/7",
                         "content": "&lt;p&gt;Part-time, weekends. Pay $19.50 per hour.&lt;/p&gt;"}]}
    monkeypatch.setattr(sources, "_get_json", lambda url: payload)
    [p] = sources.fetch_greenhouse("cafe")
    assert (p.title, p.source_label, p.wage_min) == ("Barista", "verified", 19.5)
    assert "weekend" in p.schedule and "part_time" in p.schedule


def test_lever_salary_range_converts_to_hourly(monkeypatch):
    payload = [{"id": "abc", "text": "Analyst", "hostedUrl": "https://example.com/abc",
                "categories": {"location": "Toronto", "commitment": "Full-time"},
                "descriptionPlain": "Evenings sometimes.", "lists": [],
                "salaryRange": {"min": 52000, "max": 62400, "interval": "per-year-salary"}}]
    monkeypatch.setattr(sources, "_get_json", lambda url: payload)
    [p] = sources.fetch_lever("acme")
    assert (p.wage_min, p.wage_max) == (25.0, 30.0)


def test_oracle_marks_past_closing_dates_closed(monkeypatch):
    payload = {"items": [{"Title": "Operator", "PrimaryLocation": "Toronto", "JobSchedule": "Part time",
                          "ExternalPostedEndDate": "2001-01-01T00:00:00+00:00",
                          "ExternalDescriptionStr": "<p>Starting pay is $19/hr.</p>"}]}
    monkeypatch.setattr(sources, "_get_json", lambda url: payload)
    [p] = sources.fetch_oracle("host.example.com", "CX_1", "1", "Arcadis")
    assert p.source_label == "closed" and p.wage_min == 19.0


def test_db_roundtrip(tmp_path):
    postings = sources.import_csv("examples/sample_postings.csv")
    db = tmp_path / "p.db"
    assert save_postings(db, postings) == len(postings)
    loaded = load_postings(db)
    assert {p.id for p in loaded} == {p.id for p in postings}


@pytest.mark.parametrize("objective", ["pay", "surplus"])
def test_demo_end_to_end(tmp_path, objective):
    out = tmp_path / objective
    assert main(["demo", "--out", str(out), "--objective", objective]) == 0
    for name in ("ranking.csv", "market.md", "market.json", "market_dashboard.png",
                 "search.md", "search.json", "search_dashboard.png"):
        assert (out / name).exists(), name
    market = json.loads((out / "market.json").read_text())
    statuses = {j["id"]: j["status"] for j in market["jobs"]}
    assert statuses["ovg-34372"] == "blocked"            # bartending experience required
    assert statuses["jobbank-10317067158"] == "blocked"  # French required
    assert statuses["jobbank-3712298"] == "too_far"
    assert statuses["ovg-33848"] == "closed"             # closed Nov 6, sample judged as of Nov 8
    assert statuses["lead-seoul-dishwasher"] == "unverified"
    search = json.loads((out / "search.json").read_text())
    assert search["funnel"]["total"] == 18
    assert search["plan"]["applications_needed"] > 0


def test_search_command_reads_a_journal(tmp_path):
    out = tmp_path / "search"
    assert main(["search", "--journal", "examples/sample_applications.csv", "--today", "2026-11-08",
                 "--out", str(out)]) == 0
    assert (out / "search_dashboard.png").exists()
