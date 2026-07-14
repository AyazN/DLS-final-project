from __future__ import annotations

from collections.abc import Sequence

from aise.contracts import SearchResult


class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        bm25_results: Sequence[SearchResult],
        dense_results: Sequence[SearchResult],
    ) -> Sequence[SearchResult]:

        scores: dict[str, float] = {}
        results_map: dict[str, SearchResult] = {}

        for results in (bm25_results, dense_results):
            for result in results:
                scores[result.doc_id] = (
                    scores.get(result.doc_id, 0.0)
                    + 1.0 / (self.k + result.rank)
                )
                results_map[result.doc_id] = result

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        fused_results: list[SearchResult] = []

        for rank, (doc_id, score) in enumerate(ranked, start=1):
            result = results_map[doc_id]

            fused_results.append(
                SearchResult(
                    doc_id=result.doc_id,
                    model_id=result.model_id,
                    score=score,
                    rank=rank,
                    title=result.title,
                    snippet=result.snippet,
                    metadata=result.metadata,
                )
            )

        return fused_results