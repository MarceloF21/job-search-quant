"""How well does a résumé speak the posting's language?

Terms are words and two-word phrases taken *within* phrases (stop words and punctuation
break phrases), so "cash handling and customer service" yields "cash handling" and
"customer service" but never the junk pair "handling customer".
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_GENERIC = {
    "ability", "able", "including", "work", "working", "role", "position", "team", "strong", "will", "must",
    "required", "requirement", "requirements", "preferred", "asset", "experience", "candidate", "candidates",
    "job", "responsibilities", "duties", "qualifications", "based", "per", "hour", "hourly", "pay", "rate",
    "cad", "year", "years", "skills", "excellent", "good", "great", "new", "time", "involving", "positions",
    "valid", "varying", "various", "within", "regular",
}
_STOP = ENGLISH_STOP_WORDS | _GENERIC
_TOKEN = re.compile(r"[a-z][a-z+&'-]*|\d+|[^\w\s]")


def terms(text: str) -> list[str]:
    """Words and adjacent two-word phrases, never crossing stop words or punctuation."""
    found, phrase = [], []
    for token in _TOKEN.findall(text.lower()) + ["."]:
        if not token[0].isalpha() or len(token) < 3 or token in _STOP:
            found += phrase + [" ".join(pair) for pair in zip(phrase, phrase[1:], strict=False)]
            phrase = []
        else:
            phrase.append(token.strip("'"))
    return found


@dataclass
class MatchResult:
    similarity: float  # cosine similarity of TF-IDF vectors, 0 to 1
    covered: list[str]  # the posting's main keywords that the résumé already uses
    missing: list[str]  # the posting's main keywords the résumé never mentions


def keyword_match(resume_text: str, posting_text: str, top_n: int = 15) -> MatchResult:
    vectorizer = TfidfVectorizer(analyzer=terms, sublinear_tf=True)
    matrix = vectorizer.fit_transform([posting_text, resume_text])
    similarity = float(cosine_similarity(matrix[0], matrix[1])[0, 0])

    counts = Counter(terms(posting_text))
    order = {term: i for i, term in enumerate(dict.fromkeys(terms(posting_text)))}
    ranked = sorted(counts, key=lambda t: (-counts[t], -(" " in t), order[t]))
    chosen: list[str] = []
    for term in ranked:
        inside_phrase = any(term in phrase.split() and counts[phrase] >= counts[term]
                            for phrase in chosen if " " in phrase)
        if not inside_phrase:
            chosen.append(term)
        if len(chosen) == top_n:
            break
    resume_terms = set(terms(resume_text))
    return MatchResult(
        similarity=similarity,
        covered=[t for t in chosen if t in resume_terms],
        missing=[t for t in chosen if t not in resume_terms],
    )
