import math

import pytest

from job_search_quant.benchmarks import Benchmark
from job_search_quant.cli import main
from job_search_quant.labour import (
    commute_tax,
    hourly_wage,
    in_city,
    load_postings,
    minimum_wage_bunching,
    resource_url,
    summarize,
)

COLUMNS = ["Province/Territory", "City", "NOC21 Code", "NOC21 Code Name", "Vacancy Count", "Employment Type",
           "External Indicator", "Education LOS", "Experience Level", "Employment Term Evening",
           "Employment Term Weekend", "Employment Term Night", "Employment Term Oncall", "Salary Per",
           "Salary Minimum", "Salary Maximum", "Hours Per", "Hours Minimum", "Hours Maximum"]


def row(**kw):
    base = {c: "NA" for c in COLUMNS}
    base.update({"Province/Territory": "Ontario", "City": "Toronto", "NOC21 Code": "65200",
                 "NOC21 Code Name": "Food counter attendants", "Vacancy Count": "1", "Employment Type": "Part time",
                 "External Indicator": "0", "Education LOS": "No degree, certificate or diploma",
                 "Experience Level": "No experience", "Employment Term Evening": "Yes",
                 "Employment Term Weekend": "No", "Employment Term Night": "No", "Employment Term Oncall": "No",
                 "Salary Per": "Hour", "Salary Minimum": "17.95", "Salary Maximum": "17.95"})
    base.update(kw)
    return base


def write_month(path, rows):
    lines = ["\t".join(COLUMNS)] + ["\t".join(r[c] for c in COLUMNS) for r in rows]
    path.write_bytes("\n".join(lines).encode("utf-16"))  # the open data files are UTF-16 TSV


@pytest.mark.parametrize(("kw", "expected"), [
    (dict(Salary_Per="Hour", lo="20", hi="24"), 22.0),
    (dict(Salary_Per="Year", lo="52000", hi="52000"), 25.0),  # 52,000 / (52 x 40)
    (dict(Salary_Per="Year", lo="52000", hi="52000", hours="20"), 50.0),  # stated 20 h/week
    (dict(Salary_Per="Day", lo="160", hi="160"), 20.0),
    (dict(Salary_Per="Hour", lo="5", hi="5"), None),  # outside $10–$200: data error
])
def test_hourly_wage(kw, expected):
    r = row(**{"Salary Per": kw["Salary_Per"], "Salary Minimum": kw["lo"], "Salary Maximum": kw["hi"]})
    if "hours" in kw:
        r.update({"Hours Per": "Week", "Hours Minimum": kw["hours"], "Hours Maximum": kw["hours"]})
    assert hourly_wage(r) == (pytest.approx(expected) if expected else None)


def test_city_filter_and_summary():
    rows = [row(), row(**{"Salary Minimum": "30", "Salary Maximum": "30", "Employment Type": "Full time",
                          "NOC21 Code": "11100", "External Indicator": "1"}),
            row(City="Mississauga"), row(**{"Province/Territory": "Alberta"})]
    toronto = in_city(rows)
    assert len(toronto) == 2
    s = summarize(toronto, (Benchmark("min", 17.95, "s", "d"), Benchmark("living", 27.2, "s", "d")), "Oct 2026")
    assert s.benchmark_shares == {"min": 1.0, "living": 0.5}
    assert s.part_time_median == pytest.approx(17.95) and s.full_time_median == pytest.approx(30)
    assert s.entry_level_n == 1 and s.entry_level_schedule["evening"] == 1.0
    assert set(s.teer_medians) == {"TEER 5 No formal education", "TEER 1 University degree"}


def test_minimum_wage_bunching():
    rows = [row(**{"Salary Minimum": v}) for v in ("17.60", "17.60", "17.95", "16.90", "21")]
    b = minimum_wage_bunching(rows, "Sep 2026")
    assert (b.hourly_postings, b.at_old, b.at_new, b.below_new) == (5, 2, 1, 3)


def test_commute_tax():
    grid = commute_tax(20.0, [6], [45])
    assert grid[0][0] == pytest.approx(20 * 6 / 7.5)  # 1.5 h unpaid round trip


def test_resource_url_picks_the_english_month():
    package = {"result": {"resources": [
        {"name_translated": {"en": "October 2026 Job Postings Advertised on Canada's National Job Bank Website"},
         "language": ["fr"], "url": "fr.csv"},
        {"name_translated": {"en": "October 2026 Job Postings Advertised on Canada's National Job Bank Website"},
         "language": ["en"], "url": "en.csv"}]}}
    assert resource_url(package, 2026, 10) == "en.csv"
    with pytest.raises(LookupError):
        resource_url(package, 2026, 9)


def test_utf16_file_round_trip_and_cli(tmp_path):
    write_month(tmp_path / "2026-09.csv", [row(**{"Salary Minimum": "17.60", "Salary Maximum": "17.60"})] * 3)
    write_month(tmp_path / "2026-10.csv", [row(), row(**{"Salary Minimum": "25", "Salary Maximum": "25"})])
    assert len(load_postings(tmp_path / "2026-10.csv")) == 2
    out = tmp_path / "out"
    assert main(["labour", "--files", str(tmp_path / "2026-09.csv"), str(tmp_path / "2026-10.csv"),
                 "--out", str(out)]) == 0
    text = (out / "labour.md").read_text()
    assert "Sep 2026" in text and "Oct 2026" in text
    assert (out / "labour_dashboard.png").exists()
    assert not math.isnan(len(text))
