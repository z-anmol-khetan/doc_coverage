from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path

import numpy as np

from .embedder import embed_texts
from .extractor import CoverageUnit
from .indexer import CoverageIndex, build_index


@dataclass
class QAPair:
    id: str
    question: str
    answer: str


@dataclass
class MatchScore:
    qa_id: str
    unit_id: str
    score: float
    label: str
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class MatchResult:
    qa_pairs: list[QAPair]
    qa_embeddings: np.ndarray
    title_embeddings: np.ndarray
    unit_embeddings: np.ndarray
    scores: list[MatchScore]
    best_unit_labels: dict[str, str]
    best_unit_scores: dict[str, float]
    best_unit_bm25_scores: dict[str, float]
    qa_coverage_map: dict[str, list[str]]
    overlap_map: dict[str, list[str]]
    index: CoverageIndex


def load_qa_pairs(qa_path: str) -> list[QAPair]:
    path = Path(qa_path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for index, row in enumerate(reader, start=1):
                question = (row.get("question") or row.get("Question") or "").strip()
                answer = (row.get("answer") or row.get("Answer") or row.get("Verified Answer") or "").strip()
                qa_id = (row.get("id") or row.get("ID") or f"qa_{index:03d}").strip()
                if question:
                    rows.append(QAPair(id=qa_id, question=question, answer=answer))
            return rows
    if path.suffix.lower() in (".xlsx", ".xls"):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows_iter = iter(ws.rows)
        header = [str(cell.value).strip() if cell.value is not None else "" for cell in next(rows_iter)]
        def _col(row_cells, *names):
            for name in names:
                for i, h in enumerate(header):
                    if h.lower() == name.lower() and i < len(row_cells):
                        v = row_cells[i].value
                        return str(v).strip() if v is not None else ""
            return ""
        pairs = []
        for index, row_cells in enumerate(rows_iter, start=1):
            question = _col(row_cells, "question", "Question")
            answer = _col(row_cells, "answer", "Answer", "Verified Answer")
            qa_id = _col(row_cells, "id", "ID") or f"qa_{index:03d}"
            if question:
                pairs.append(QAPair(id=qa_id, question=question, answer=answer))
        wb.close()
        return pairs
    data = json.loads(path.read_text(encoding="utf-8"))
    return [QAPair(id=item.get("id", f"qa_{index:03d}"), question=item["question"], answer=item.get("answer", "")) for index, item in enumerate(data, start=1)]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _score_label(score: float) -> str:
    if score >= 0.80:
        return "fully_covered"
    if score >= 0.60:
        return "probably_covered"
    if score >= 0.30:
        return "partially_covered"
    if score >= 0.25:
        return "mentioned_only"
    return "not_covered"


def match_qa_pairs(units: list[CoverageUnit], qa_pairs: list[QAPair], preferred_model: str | None = None) -> MatchResult:
    unit_texts = [unit.text for unit in units]
    title_texts = [unit.section_title for unit in units]
    qa_texts = [f"{qa.question}\n{qa.answer}".strip() for qa in qa_pairs]

    unit_embeddings, model_name = embed_texts(unit_texts, cache_prefix="units", preferred_model=preferred_model)
    title_embeddings, _ = embed_texts(title_texts, cache_prefix="titles", preferred_model=model_name)
    qa_embeddings, _ = embed_texts(qa_texts, cache_prefix="qas", preferred_model=model_name)
    index = build_index(units, unit_embeddings)

    # Pre-embed individual member texts for cluster parents
    member_texts_flat: list[str] = []
    member_coords: list[tuple[int, int]] = []  # (unit_idx, member_idx)
    for u_idx, unit in enumerate(units):
        for m_idx, member in enumerate(unit.member_units):
            member_texts_flat.append(member["text"])
            member_coords.append((u_idx, m_idx))

    member_embeddings: np.ndarray | None = None
    unit_member_slices: dict[int, tuple[int, int]] = {}  # unit_idx -> (start, end) into member_embeddings
    if member_texts_flat:
        member_embeddings, _ = embed_texts(member_texts_flat, cache_prefix="members", preferred_model=model_name)
        offset = 0
        for u_idx, unit in enumerate(units):
            n = len(unit.member_units)
            if n:
                unit_member_slices[u_idx] = (offset, offset + n)
                offset += n

    bm25_matrix = []
    for qa in qa_pairs:
        bm25_matrix.append(index.bm25.get_scores(qa.question.lower().split() + qa.answer.lower().split()))
    bm25_matrix = np.asarray(bm25_matrix)
    bm25_max = float(bm25_matrix.max()) if bm25_matrix.size else 0.0

    scores: list[MatchScore] = []
    best_unit_scores = {unit.unit_id: 0.0 for unit in units}
    best_unit_bm25_scores = {unit.unit_id: 0.0 for unit in units}
    best_unit_labels = {unit.unit_id: "not_covered" for unit in units}
    qa_coverage_map: dict[str, list[str]] = {qa.id: [] for qa in qa_pairs}

    for qa_index, qa in enumerate(qa_pairs):
        qa_entities = {token.lower() for token in qa.question.split() + qa.answer.split() if token[:1].isupper()}
        qa_symbols = {token.lower().strip(',.') for token in qa.answer.split() + qa.question.split() if any(char in token for char in ['/', '-', '_']) or token.isupper()}
        qa_keywords = {token.lower().strip(',.?') for token in qa.question.split() + qa.answer.split() if len(token) > 3}
        qa_vector = qa_embeddings[qa_index]
        for unit_index, unit in enumerate(units):
            bm25_score = float(bm25_matrix[qa_index][unit_index]) / bm25_max if bm25_max else 0.0
            entity_symbol_overlap = _jaccard(qa_entities | qa_symbols, index.entity_sets[unit_index] | index.symbol_sets[unit_index])
            keyphrase_overlap = _jaccard(qa_keywords, index.keyword_sets[unit_index])
            heading_similarity = float(np.dot(qa_vector, title_embeddings[unit_index]))
            table_schema_overlap = _jaccard(qa_keywords, index.table_schema_sets[unit_index]) if unit.table_schema else 0.0
            non_cosine = (
                0.20 * max(bm25_score, 0.0)
                + 0.20 * entity_symbol_overlap
                + 0.15 * keyphrase_overlap
                + 0.10 * max(heading_similarity, 0.0)
                + 0.05 * table_schema_overlap
            )

            if unit_index in unit_member_slices:
                # Score each member individually; parent score = max across members
                start, end = unit_member_slices[unit_index]
                mem_cosines = np.clip(member_embeddings[start:end] @ qa_vector, 0.0, 1.0)
                best_cosine = float(mem_cosines.max())
            else:
                best_cosine = max(float(np.dot(qa_vector, unit_embeddings[unit_index])), 0.0)

            score = round(min(0.30 * best_cosine + non_cosine, 1.0), 4)
            label = _score_label(score)
            scores.append(
                MatchScore(
                    qa_id=qa.id,
                    unit_id=unit.unit_id,
                    score=score,
                    label=label,
                    components={
                        "cosine_similarity": round(best_cosine, 4),
                        "bm25": round(max(bm25_score, 0.0), 4),
                        "entity_symbol_overlap": round(entity_symbol_overlap, 4),
                        "keyphrase_overlap": round(keyphrase_overlap, 4),
                        "heading_similarity": round(max(heading_similarity, 0.0), 4),
                        "table_schema_overlap": round(table_schema_overlap, 4),
                    },
                )
            )
            if score >= 0.25:
                qa_coverage_map[qa.id].append(unit.unit_id)
            if score > best_unit_scores[unit.unit_id]:
                best_unit_scores[unit.unit_id] = score
                best_unit_bm25_scores[unit.unit_id] = round(max(bm25_score, 0.0), 4)
                best_unit_labels[unit.unit_id] = label

    # Propagate parent's final score (= max across all QAs and all members) to every member
    for unit in units:
        if unit.member_units:
            final_score = best_unit_scores[unit.unit_id]
            final_label = best_unit_labels[unit.unit_id]
            for member in unit.member_units:
                member["score"] = final_score
                member["label"] = final_label

    normalized_map = {qa_id: sorted(set(unit_ids)) for qa_id, unit_ids in qa_coverage_map.items()}
    overlap_map: dict[str, list[str]] = {}
    qa_ids = [qa.id for qa in qa_pairs]
    for qa_id in qa_ids:
        overlaps = []
        current = set(normalized_map[qa_id])
        for other_id in qa_ids:
            if qa_id == other_id:
                continue
            if current & set(normalized_map[other_id]):
                overlaps.append(other_id)
        overlap_map[qa_id] = overlaps

    return MatchResult(
        qa_pairs=qa_pairs,
        qa_embeddings=qa_embeddings,
        title_embeddings=title_embeddings,
        unit_embeddings=unit_embeddings,
        scores=scores,
        best_unit_labels=best_unit_labels,
        best_unit_scores=best_unit_scores,
        best_unit_bm25_scores=best_unit_bm25_scores,
        qa_coverage_map=normalized_map,
        overlap_map=overlap_map,
        index=index,
    )
