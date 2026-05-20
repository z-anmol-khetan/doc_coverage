from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict

from .extractor import CoverageUnit
from .matcher import MatchResult
from .parser import DocSection
from .scorer import Scorecard


def build_results(
    sections: list[DocSection],
    units: list[CoverageUnit],
    match_result: MatchResult,
    scorecard: Scorecard,
    contradictions: list[dict],
) -> dict:
    uncovered_units = []
    partially_covered_units = []
    entity_symbol_rows: dict[str, dict] = {}
    section_unit_rows: dict[str, list[dict]] = defaultdict(list)
    section_units_covered_rows: dict[str, list[dict]] = defaultdict(list)
    section_units_not_covered_rows: dict[str, list[dict]] = defaultdict(list)
    for unit in units:
        label = match_result.best_unit_labels[unit.unit_id]
        row = {
            "unit_id": unit.unit_id,
            "section_id": unit.section_id,
            "section_title": unit.section_title,
            "text": unit.text,
            "importance": unit.importance,
            "member_units": unit.member_units,
            "coverage_label": label,
            "best_match_score": match_result.best_unit_scores[unit.unit_id],
            "best_match_bm25": match_result.best_unit_bm25_scores[unit.unit_id],
        }
        section_unit_rows[unit.section_id].append(row)
        if label != "not_covered":
            section_units_covered_rows[unit.section_id].append(row)
        else:
            section_units_not_covered_rows[unit.section_id].append(row)
        if label == "not_covered":
            uncovered_units.append(row)
        elif label in {"partially_covered", "mentioned_only"}:
            partially_covered_units.append(row)
        for item in unit.entities + unit.symbols:
            key = item.strip()
            if not key:
                continue
            current = entity_symbol_rows.setdefault(
                key,
                {"name": key, "kind": "symbol" if item in unit.symbols else "entity", "covered": False, "unit_ids": []},
            )
            current["covered"] = current["covered"] or label != "not_covered"
            current["unit_ids"].append(unit.unit_id)

    over_covered_topics = []
    for qa_id, unit_ids in match_result.qa_coverage_map.items():
        if len(unit_ids) > 5:
            over_covered_topics.append({"qa_id": qa_id, "unit_count": len(unit_ids), "unit_ids": unit_ids})

    sections_json = []
    for section in scorecard.sections:
        section_rows = section_unit_rows[section.section_id]
        sections_json.append(
            {
                "section_id": section.section_id,
                "title": section.title,
                "coverage": section.coverage,
                "units_total": section.units_total,
                "units_covered": section.units_covered,
                "bucket_counts": section.bucket_counts,
                "units": section_rows,
                "covered_units": section_units_covered_rows[section.section_id],
                "not_covered_units": section_units_not_covered_rows[section.section_id],
            }
        )

    return {
        "final_score": scorecard.final_score,
        "atomic_coverage": scorecard.atomic_coverage,
        "critical_coverage": scorecard.critical_coverage,
        "section_balance": scorecard.section_balance,
        "redundancy_ratio": scorecard.redundancy_ratio,
        "diversity_score": scorecard.diversity_score,
        "entity_coverage": scorecard.entity_coverage,
        "sections": sections_json,
        "coverage_distribution": scorecard.coverage_distribution,
        "uncovered_units": uncovered_units,
        "partially_covered_units": partially_covered_units,
        "over_covered_topics": over_covered_topics,
        "contradictions": contradictions,
        "qa_coverage_map": match_result.qa_coverage_map,
        "qa_overlap_map": match_result.overlap_map,
        "entity_symbol_coverage": sorted(entity_symbol_rows.values(), key=lambda item: item["name"].lower()),
        "document": [asdict(section) for section in sections],
        "units": [asdict(unit) for unit in units],
        "qa_matches": [asdict(score) for score in match_result.scores],
    }
