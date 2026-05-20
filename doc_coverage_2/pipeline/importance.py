from __future__ import annotations

import re


CRITICAL_TERMS = {
    "must",
    "required",
    "shall",
    "critical",
    "warning",
    "caution",
    "error",
    "security",
}

HTTP_CODE_RE = re.compile(r"\b[1-5][0-9]{2}\b")
CONFIG_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
ENDPOINT_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\s+/\S+")


def score_importance(
    text: str,
    entities: list[str],
    symbols: list[str],
    in_procedure_list: bool,
) -> float:
    score = 1.0
    lower = text.lower()
    if any(term in lower for term in CRITICAL_TERMS):
        score += 2.0
    if ENDPOINT_RE.search(text) or CONFIG_RE.search(text) or HTTP_CODE_RE.search(text):
        score += 1.0
    if in_procedure_list:
        score += 1.0
    score += 0.5 * len(set(entities))
    return max(1.0, min(5.0, round(score, 2)))
