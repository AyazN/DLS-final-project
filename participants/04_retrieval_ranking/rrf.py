from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from aise.contracts import SearchResult


class ReciprocalRankFusion:
    def __init__(self, k: int = 60) -> None:
        if k < 0:
            raise ValueError("RRF k must be non-negative.")
        self.k = k

    def fuse(
        self,
        bm25_results: Sequence[SearchResult],
        dense_results: Sequence[SearchResult],
    ) -> Sequence[SearchResult]:
        scores: dict[str, float] = {}
        results_map: dict[str, SearchResult] = {}

        for results in (bm25_results, dense_results):
            seen: set[str] = set()
            for result in results:
                if result.doc_id in seen:
                    continue
                seen.add(result.doc_id)
                scores[result.doc_id] = (
                    scores.get(result.doc_id, 0.0)
                    + 1.0 / (self.k + result.rank)
                )
                results_map.setdefault(result.doc_id, result)

        ranked = sorted(
            scores,
            key=lambda doc_id: (-scores[doc_id], doc_id),
        )
        return [
            replace(results_map[doc_id], score=scores[doc_id], rank=rank)
            for rank, doc_id in enumerate(ranked, start=1)
        ]
