"""Turn posting text into structured fields: pay, hours, schedule, requirements, tips.

Heuristic by design: anything the text doesn't state stays None, and the economics
module fills gaps with explicit, reported assumptions instead of guesses.
"""

from __future__ import annotations

import html
import re
from dataclasses import replace

from .models import Posting

_MONEY = r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)"
_RANGE = re.compile(_MONEY + r"\s*(?:CAD\s*)?(?:-|–|to)\s*" + _MONEY, re.I)
_SINGLE = re.compile(_MONEY, re.I)
_PAY_CONTEXT = re.compile(r"pay|rate|wage|salary|hour|hourly|/\s?hr|compensation|earn|annual|year", re.I)
_ANNUAL = re.compile(r"year|annual|annum|salary", re.I)

_HOURS_RANGE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*(?:-|–|to)\s*(\d{1,2}(?:\.\d)?)\s*hours?\s*(?:per|a|/|each)?\s*(bi-?weekly|week)", re.I)
_HOURS_SINGLE = re.compile(r"(\d{1,2}(?:\.\d)?)\s*hours?\s*(?:per|a|/|each)\s*(bi-?weekly|week)", re.I)

SCHEDULE_PATTERNS = {
    "day": r"\b(?:mornings?|daytime|day shifts?|weekdays?)\b",
    "evening": r"\bevenings?\b",
    "night": r"\b(?:nights?|overnight|graveyard)\b",
    "weekend": r"\b(?:weekends?|saturdays?|sundays?)\b",
    "holiday": r"\bholidays?\b",
    "casual": r"\b(?:casual|as[- ]needed|on[- ]call)\b",
    "part_time": r"\bpart[- ]time\b",
    "full_time": r"\bfull[- ]time\b",
    "seasonal": r"\bseasonal\b",
    "24_7": r"24\s?/\s?7",
}
REQUIREMENT_PATTERNS = {
    "drivers_licence": r"driver'?s?\s+licen[cs]e|\bG\s?licen[cs]e|own vehicle|access to a vehicle",
    "smart_serve": r"smart\s?serve",
    "food_handler": r"food handler",
    "forklift": r"\bforklift\b",
    "security_licence": r"security (?:guard )?licen[cs]e",
    "french": r"\bfrench\b",
}
_ASSET = re.compile(r"\b(?:asset|preferred|nice to have|bonus|an advantage)\b", re.I)
_EXPERIENCE = re.compile(
    r"(\d+|one|two|three|four|five|six)\+?\s*(months?|years?)\s*(?:or more\s*)?(?:of\s+)?(?:\w+\s+){0,3}experience",
    re.I)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def strip_html(text: str) -> str:
    text = html.unescape(html.unescape(text or ""))
    text = re.sub(r"<\s*(?:br|/p|/li|/div|/h\d)\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _to_float(value: str) -> float:
    return float(value.replace(",", ""))


def _hourly(amount: float, context: str) -> float | None:
    """Convert to hourly pay; amounts that can't be pay (e.g. a $299 course fee) return None."""
    if amount >= 10_000 or (_ANNUAL.search(context) and amount >= 1_000):
        return round(amount / 2080, 2)  # 52 weeks x 40 hours
    if 10 <= amount <= 200:
        return amount
    return None


def parse_wage(text: str) -> tuple[float | None, float | None]:
    """Return the posted hourly pay range (min, max). Prefers amounts near pay words."""
    candidates = []
    taken = []
    for match in _RANGE.finditer(text):
        context = text[max(0, match.start() - 60): match.end() + 40]
        lo, hi = (_hourly(_to_float(g), context) for g in match.groups())
        if lo is not None and hi is not None:
            candidates.append((bool(_PAY_CONTEXT.search(context)), match.start(), lo, hi))
            taken.append((match.start(), match.end()))
    for match in _SINGLE.finditer(text):
        if any(start <= match.start() < end for start, end in taken):
            continue
        context = text[max(0, match.start() - 60): match.end() + 40]
        value = _hourly(_to_float(match.group(1)), context)
        if value is not None:
            candidates.append((bool(_PAY_CONTEXT.search(context)), match.start(), value, None))
    if not candidates:
        return None, None
    best = sorted(candidates, key=lambda c: (not c[0], c[1]))[0]
    return best[2], best[3]


def parse_hours(text: str) -> tuple[float | None, float | None]:
    """Return weekly hours (min, max); bi-weekly figures are halved."""
    match = _HOURS_RANGE.search(text)
    if match:
        lo, hi = float(match.group(1)), float(match.group(2))
        factor = 0.5 if match.group(3).lower().startswith("bi") else 1.0
        return lo * factor, hi * factor
    match = _HOURS_SINGLE.search(text)
    if match:
        value = float(match.group(1)) * (0.5 if match.group(2).lower().startswith("bi") else 1.0)
        return value, value
    return None, None


def schedule_flags(text: str) -> tuple[str, ...]:
    return tuple(name for name, pattern in SCHEDULE_PATTERNS.items() if re.search(pattern, text, re.I))


def requirement_flags(text: str) -> tuple[str, ...]:
    """Requirements stated as required. Sentences that call them an asset or preferred are skipped."""
    found = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+|•", text):
        if _ASSET.search(sentence):
            continue
        for name, pattern in REQUIREMENT_PATTERNS.items():
            if name not in found and re.search(pattern, sentence, re.I):
                found.append(name)
        months = experience_months(sentence)
        if months and "experience" not in " ".join(found):
            found.append(f"experience_{months:g}m")
    return tuple(found)


def experience_months(text: str) -> float | None:
    match = _EXPERIENCE.search(text)
    if not match:
        return None
    raw = match.group(1).lower()
    number = float(_NUMBER_WORDS.get(raw, raw))
    return number if match.group(2).lower().startswith("month") else number * 12


def tips_mentioned(text: str) -> bool:
    return bool(re.search(r"\btips?\b|tip[- ]eligible|gratuit", text, re.I))


def enrich(posting: Posting) -> Posting:
    """Fill fields the posting didn't provide from its description; given values always win."""
    text = strip_html(posting.description)
    wage_min, wage_max = parse_wage(text)
    hours_min, hours_max = parse_hours(text)
    return replace(
        posting,
        description=text,
        wage_min=posting.wage_min if posting.wage_min is not None else wage_min,
        wage_max=posting.wage_max if posting.wage_min is not None else wage_max,
        hours_min=posting.hours_min if posting.hours_min is not None else hours_min,
        hours_max=posting.hours_max if posting.hours_min is not None else hours_max,
        tips=posting.tips or tips_mentioned(text),
        schedule=posting.schedule or schedule_flags(text),
        requirements=posting.requirements or requirement_flags(text),
    )
