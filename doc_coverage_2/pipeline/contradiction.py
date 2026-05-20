from __future__ import annotations

import re

from .extractor import CoverageUnit
from .matcher import MatchResult, QAPair


STATUS_RE = re.compile(r"\b([1-5][0-9]{2})\b")
NUMBER_UNIT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)?\b")
VERSION_RE = re.compile(r"\b(v?\d+(?:\.\d+)+(?:-[a-zA-Z0-9]+)?)\b")
CONFIG_VALUE_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\s*[=:]\s*([^,;\n]+)")
NEGATION_TERMS = {"not", "never", "no", "without"}


def _contains_negation(text: str, term: str) -> bool:
    words = text.lower().split()
    target = term.lower()
    for index, word in enumerate(words):
        if target in word:
            window = words[max(0, index - 3): index + 1]
            if any(token in NEGATION_TERMS for token in window):
                return True
    return False


def detect_contradictions(units: list[CoverageUnit], match_result: MatchResult) -> list[dict]:
    qa_by_id = {qa.id: qa for qa in match_result.qa_pairs}
    unit_by_id = {unit.unit_id: unit for unit in units}
    contradictions: list[dict] = []
    for score in match_result.scores:
        if score.score < 0.4:
            continue
        qa = qa_by_id[score.qa_id]
        unit = unit_by_id[score.unit_id]
        contradictions.extend(_compare_status_codes(unit, qa))
        contradictions.extend(_compare_numbers(unit, qa))
        contradictions.extend(_compare_versions(unit, qa))
        contradictions.extend(_compare_config_values(unit, qa))
        contradictions.extend(_compare_negation(unit, qa))
    return contradictions


def _compare_status_codes(unit: CoverageUnit, qa: QAPair) -> list[dict]:
    source = STATUS_RE.findall(unit.text)
    answer = STATUS_RE.findall(qa.answer)
    if source and answer and set(source) != set(answer):
        return [{"type": "http_status_code_mismatch", "unit_id": unit.unit_id, "source_value": source, "qa_value": answer, "qa_id": qa.id}]
    return []


def _compare_numbers(unit: CoverageUnit, qa: QAPair) -> list[dict]:
    source = NUMBER_UNIT_RE.findall(unit.text)
    answer = NUMBER_UNIT_RE.findall(qa.answer)
    if not source or not answer:
        return []
    source_values = {f"{value}{unit_name or ''}" for value, unit_name in source}
    answer_values = {f"{value}{unit_name or ''}" for value, unit_name in answer}
    if source_values != answer_values:
        return [{"type": "numeric_value_mismatch", "unit_id": unit.unit_id, "source_value": sorted(source_values), "qa_value": sorted(answer_values), "qa_id": qa.id}]
    return []


def _compare_versions(unit: CoverageUnit, qa: QAPair) -> list[dict]:
    source = VERSION_RE.findall(unit.text)
    answer = VERSION_RE.findall(qa.answer)
    if source and answer and set(source) != set(answer):
        return [{"type": "version_mismatch", "unit_id": unit.unit_id, "source_value": source, "qa_value": answer, "qa_id": qa.id}]
    return []


def _compare_config_values(unit: CoverageUnit, qa: QAPair) -> list[dict]:
    source = dict(CONFIG_VALUE_RE.findall(unit.text))
    answer = dict(CONFIG_VALUE_RE.findall(qa.answer))
    mismatches = []
    for key, source_value in source.items():
        if key in answer and source_value.strip() != answer[key].strip():
            mismatches.append({"type": "config_key_value_mismatch", "unit_id": unit.unit_id, "source_value": {key: source_value.strip()}, "qa_value": {key: answer[key].strip()}, "qa_id": qa.id})
    return mismatches


def _compare_negation(unit: CoverageUnit, qa: QAPair) -> list[dict]:
    key_terms = [symbol for symbol in unit.symbols if symbol.isalpha()] or unit.keywords[:3]
    mismatches = []
    for term in key_terms:
        if _contains_negation(unit.text, term) != _contains_negation(qa.answer, term):
            mismatches.append({"type": "boolean_negation_mismatch", "unit_id": unit.unit_id, "source_value": unit.text, "qa_value": qa.answer, "qa_id": qa.id})
            break
    return mismatches
