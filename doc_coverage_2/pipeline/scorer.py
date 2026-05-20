from __future__ import annotations

from dataclasses import dataclass, field
from statistics import pstdev

import numpy as np

from .extractor import CoverageUnit
from .matcher import MatchResult


@dataclass
class SectionScore:
    section_id: str
    title: str
    coverage: float
    units_total: int
    units_covered: int
    bucket_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class Scorecard:
    final_score: float
    atomic_coverage: float
    critical_coverage: float
    section_balance: float
    redundancy_ratio: float
    diversity_score: float
    entity_coverage: float
    sections: list[SectionScore]
    coverage_distribution: dict[str, int]


def score_coverage(units: list[CoverageUnit], match_result: MatchResult) -> Scorecard:
    covered_labels = {"fully_covered", "probably_covered", "partially_covered"}
    total_weight = sum(unit.importance for unit in units) or 1.0
    covered_weight = sum(
        unit.importance
        for unit in units
        if match_result.best_unit_labels[unit.unit_id] in covered_labels
    )
    atomic_coverage = covered_weight / total_weight

    critical_units = [unit for unit in units if unit.importance >= 4]
    critical_total = sum(unit.importance for unit in critical_units) or 1.0
    critical_covered = sum(
        unit.importance
        for unit in critical_units
        if match_result.best_unit_labels[unit.unit_id] in covered_labels
    )
    critical_coverage = critical_covered / critical_total

    section_map: dict[tuple[str, str], list[CoverageUnit]] = {}
    for unit in units:
        section_map.setdefault((unit.section_id, unit.section_title), []).append(unit)
    sections: list[SectionScore] = []
    section_scores: list[float] = []
    distribution = {
        "fully_covered": 0,
        "probably_covered": 0,
        "partially_covered": 0,
        "mentioned_only": 0,
        "not_covered": 0,
    }
    for (section_id, title), section_units in section_map.items():
        covered = 0
        bucket_counts = {key: 0 for key in distribution}
        for unit in section_units:
            label = match_result.best_unit_labels[unit.unit_id]
            bucket_counts[label] += 1
            distribution[label] += 1
            if label in {"fully_covered", "probably_covered", "partially_covered"}:
                covered += 1
        coverage = covered / len(section_units) if section_units else 0.0
        section_scores.append(coverage)
        sections.append(
            SectionScore(
                section_id=section_id,
                title=title,
                coverage=round(coverage, 4),
                units_total=len(section_units),
                units_covered=covered,
                bucket_counts=bucket_counts,
            )
        )
    sections.sort(key=lambda item: item.coverage, reverse=True)

    stddev = pstdev(section_scores) if len(section_scores) > 1 else 0.0
    section_balance = max(0.0, 1.0 - min(stddev / 0.5, 1.0))

    unique_covered_units = {unit.unit_id for unit in units if match_result.best_unit_labels[unit.unit_id] != "not_covered"}
    total_matches = sum(len(unit_ids) for unit_ids in match_result.qa_coverage_map.values())
    redundancy_ratio = total_matches / max(len(unique_covered_units), 1)

    if len(match_result.qa_embeddings) > 1:
        similarity_matrix = np.clip(match_result.qa_embeddings @ match_result.qa_embeddings.T, -1.0, 1.0)
        upper = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        diversity_score = 1.0 - float(np.mean(upper)) if upper.size else 1.0
    else:
        diversity_score = 1.0
    diversity_score = max(0.0, min(diversity_score, 1.0))

    all_entities = {entity.lower() for unit in units for entity in unit.entities}
    covered_entities = {
        entity.lower()
        for unit in units
        if match_result.best_unit_labels[unit.unit_id] != "not_covered"
        for entity in unit.entities
    }
    entity_coverage = len(covered_entities) / len(all_entities) if all_entities else 1.0

    final_score = (
        0.45 * atomic_coverage
        + 0.20 * section_balance
        + 0.15 * critical_coverage
        + 0.10 * entity_coverage
        + 0.10 * diversity_score
    )

    return Scorecard(
        final_score=round(final_score, 4),
        atomic_coverage=round(atomic_coverage, 4),
        critical_coverage=round(critical_coverage, 4),
        section_balance=round(section_balance, 4),
        redundancy_ratio=round(redundancy_ratio, 4),
        diversity_score=round(diversity_score, 4),
        entity_coverage=round(entity_coverage, 4),
        sections=sections,
        coverage_distribution=distribution,
    )
