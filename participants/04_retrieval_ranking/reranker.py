from __future__ import annotations

from collections.abc import Sequence

from aise.contracts import (
    Query,
    SearchResult,
    Reranker,
)


class CrossEncoderReranker(Reranker):
    def __init__(self, model):
        # TODO:
        # Select and integrate the final CrossEncoder model.
        self.model = model

    def rank(
        self,
        query: Query,
        candidates: Sequence[SearchResult],
    ) -> Sequence[SearchResult]:

        pairs = [
            (query.text, candidate.snippet)
            for candidate in candidates
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results: list[SearchResult] = []

        for rank, (candidate, score) in enumerate(ranked, start=1):
            results.append(
                SearchResult(
                    doc_id=candidate.doc_id,
                    model_id=candidate.model_id,
                    score=float(score),
                    rank=rank,
                    title=candidate.title,
                    snippet=candidate.snippet,
                    metadata=candidate.metadata,
                )
            )

        return results