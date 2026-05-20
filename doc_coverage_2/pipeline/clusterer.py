from __future__ import annotations

from collections import defaultdict

import numpy as np

from .embedder import embed_texts
from .extractor import CoverageUnit

SIMILARITY_THRESHOLD = 0.85


class _DSU:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px != py:
            self._parent[px] = py


def _merge(cluster: list[CoverageUnit], cluster_id: int) -> CoverageUnit:
    rep = cluster[0]
    seen: set[str] = set()
    keywords: list[str] = []
    for u in cluster:
        for kw in u.keywords:
            if kw not in seen:
                seen.add(kw)
                keywords.append(kw)
    members = [
        {"unit_id": u.unit_id, "text": u.text, "importance": u.importance}
        for u in cluster
    ] if len(cluster) > 1 else []
    return CoverageUnit(
        unit_id=f"cluster_{cluster_id:04d}",
        section_id=rep.section_id,
        section_title=rep.section_title,
        text=" ".join(u.text for u in cluster),
        keywords=keywords[:8],
        entities=list({e for u in cluster for e in u.entities}),
        symbols=list({s for u in cluster for s in u.symbols}),
        importance=max(u.importance for u in cluster),
        section_level=rep.section_level,
        block_type=rep.block_type,
        table_schema=rep.table_schema,
        member_units=members,
    )


def cluster_units(
    units: list[CoverageUnit],
    threshold: float = SIMILARITY_THRESHOLD,
    preferred_model: str | None = None,
) -> list[CoverageUnit]:
    if not units:
        return units

    text_units = [u for u in units if u.block_type == "paragraph"]
    other_units = [u for u in units if u.block_type != "paragraph"]

    if not text_units:
        return other_units

    embeddings, _ = embed_texts(
        [u.text for u in text_units], cache_prefix="units", preferred_model=preferred_model
    )

    # embeddings are L2-normalized → dot product == cosine similarity
    sim: np.ndarray = embeddings @ embeddings.T

    dsu = _DSU(len(text_units))
    rows, cols = np.where(np.triu(sim >= threshold, k=1))
    for i, j in zip(rows.tolist(), cols.tolist()):
        dsu.union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(text_units)):
        groups[dsu.find(i)].append(i)

    clustered = [
        _merge([text_units[i] for i in indices], cluster_id)
        for cluster_id, indices in enumerate(groups.values(), start=1)
    ]
    return clustered + other_units
