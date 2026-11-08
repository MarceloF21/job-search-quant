"""SQLite storage for postings."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from .models import Posting

_COLUMNS = [
    "id", "employer", "title", "location", "url", "source", "source_label", "checked_at", "closing_date",
    "description", "wage_min", "wage_max", "hours_min", "hours_max", "tips", "schedule", "requirements",
    "commute_minutes",
]
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS postings (
    {", ".join(c + (" TEXT PRIMARY KEY" if c == "id" else "") for c in _COLUMNS)}
);
"""


def save_postings(db_path: str | Path, postings: list[Posting]) -> int:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in postings:
        values = {c: getattr(p, c) for c in _COLUMNS}
        values["schedule"] = ";".join(p.schedule)
        values["requirements"] = ";".join(p.requirements)
        values["tips"] = int(p.tips)
        rows.append(tuple(values[c] for c in _COLUMNS))
    with closing(sqlite3.connect(db_path)) as con, con:
        con.executescript(_SCHEMA)
        con.executemany(f"INSERT OR REPLACE INTO postings VALUES ({', '.join('?' * len(_COLUMNS))})", rows)
    return len(rows)


def load_postings(db_path: str | Path) -> list[Posting]:
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"No database at {db_path}. Import or fetch postings first.")
    with closing(sqlite3.connect(db_path)) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(f"SELECT {', '.join(_COLUMNS)} FROM postings ORDER BY employer, title").fetchall()
    postings = []
    for row in rows:
        data = dict(row)
        data["schedule"] = tuple(f for f in (data["schedule"] or "").split(";") if f)
        data["requirements"] = tuple(f for f in (data["requirements"] or "").split(";") if f)
        data["tips"] = bool(data["tips"])
        postings.append(Posting(**{k: ("" if v is None and k not in _NUMERIC else v) for k, v in data.items()}))
    return postings


_NUMERIC = {"wage_min", "wage_max", "hours_min", "hours_max", "commute_minutes"}
