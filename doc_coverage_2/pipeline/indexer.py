from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .extractor import CoverageUnit



@dataclass
class CoverageIndex:
    bm25: BM25Okapi
    faiss_index: faiss.IndexFlatL2
    embeddings: np.ndarray
    tokenized_texts: list[list[str]]
    entity_sets: list[set[str]]
    symbol_sets: list[set[str]]
    keyword_sets: list[set[str]]
    table_schema_sets: list[set[str]]


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def build_index(units: list[CoverageUnit], embeddings: np.ndarray) -> CoverageIndex:
    tokenized_texts = [_tokenize(unit.text) for unit in units]
    bm25 = BM25Okapi(tokenized_texts)
    vectors = np.asarray(embeddings, dtype="float32")
    faiss_index = faiss.IndexFlatL2(vectors.shape[1])
    faiss_index.add(vectors)
    return CoverageIndex(
        bm25=bm25,
        faiss_index=faiss_index,
        embeddings=vectors,
        tokenized_texts=tokenized_texts,
        entity_sets=[{item.lower() for item in unit.entities} for unit in units],
        symbol_sets=[{item.lower() for item in unit.symbols} for unit in units],
        keyword_sets=[{item.lower() for item in unit.keywords} for unit in units],
        table_schema_sets=[{item.lower() for item in unit.table_schema} for unit in units],
    )
