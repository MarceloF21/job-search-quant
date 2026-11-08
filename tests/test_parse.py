import pytest

from job_search_quant.parse import parse_hours, parse_wage, requirement_flags, schedule_flags, strip_html


@pytest.mark.parametrize(("text", "expected"), [
    ("This role will pay an hourly rate of $23.00 CAD.", (23.0, None)),
    ("This role pays an hourly rate of $18.00-$20.00 CAD.", (18.0, 20.0)),
    ("Salary$17.95HOUR hourly / 40 hours per week", (17.95, None)),
    ("Starting pay for this role is $19/hr.", (19.0, None)),
    ("Salary $50,000 to $60,000 annually", (24.04, 28.85)),
    ("Classes start at $299. Pay is $21 per hour.", (21.0, None)),
    ("No pay information here.", (None, None)),
])
def test_parse_wage(text, expected):
    assert parse_wage(text) == expected


@pytest.mark.parametrize(("text", "expected"), [
    ("$17.95 hourly / 40 hours per week", (40.0, 40.0)),
    ("12 to 40 hours bi-weekly", (6.0, 20.0)),
    ("30-44 hours per week", (30.0, 44.0)),
    ("Hours depend on events.", (None, None)),
])
def test_parse_hours(text, expected):
    assert parse_hours(text) == expected


def test_required_vs_asset_requirements():
    text = ("Valid SmartServe certification is required for alcohol service. "
            "Food Handler Certification is considered an asset. Six months or more experience bartending.")
    assert requirement_flags(text) == ("smart_serve", "experience_6m")


def test_schedule_flags():
    text = "Regular Part-Time. Shifts include early mornings, evenings, weekends, and holidays. 24/7 centre."
    assert set(schedule_flags(text)) == {"day", "evening", "weekend", "holiday", "part_time", "24_7"}


def test_strip_html_handles_escaped_markup():
    assert strip_html("&lt;p&gt;Pay: $20/hr&lt;/p&gt;") == "Pay: $20/hr"
