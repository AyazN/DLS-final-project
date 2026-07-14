from __future__ import annotations

from collections.abc import Sequence

from aise.contracts import SearchResult


class ReciprocalRankFusion:
    def __init__(self, k: int = 60) -> None:
        if k < 0:
            raise ValueError("k must be non-negative")
        self.k = k

    def fuse(
        self,
        *rankings: Sequence[SearchResult],
        top_k: int | None = None,
    ) -> Sequence[SearchResult]:
        scores: dict[str, float] = {}
        representative: dict[str, SearchResult] = {}
        first_seen: dict[str, int] = {}

        seen_counter = 0
        for results in rankings:
            seen_in_ranking: set[str] = set()
            for fallback_rank, result in enumerate(results, start=1):
                if result.doc_id in seen_in_ranking:
                    continue
                seen_in_ranking.add(result.doc_id)
                rank = result.rank if result.rank > 0 else fallback_rank
                scores[result.doc_id] = scores.get(result.doc_id, 0.0) + 1.0 / (
                    self.k + rank
                )
                if result.doc_id not in representative:
                    representative[result.doc_id] = result
                    first_seen[result.doc_id] = seen_counter
                    seen_counter += 1

        ordered_ids = sorted(
            scores,
            key=lambda doc_id: (-scores[doc_id], first_seen[doc_id]),
        )
        if top_k is not None:
            ordered_ids = ordered_ids[: max(0, int(top_k))]

        fused: list[SearchResult] = []
        for rank, doc_id in enumerate(ordered_ids, start=1):
            result = representative[doc_id]
            fused.append(
                SearchResult(
                    doc_id=result.doc_id,
                    model_id=result.model_id,
                    score=float(scores[doc_id]),
                    rank=rank,
                    title=result.title,
                    snippet=result.snippet,
                    metadata=result.metadata,
                )
            )
        return fused
