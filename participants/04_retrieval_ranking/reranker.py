from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from aise.contracts import Query, Reranker, SearchResult


class CrossEncoderReranker(Reranker):
    def __init__(self, model: Any) -> None:
        self.model = model

    def rank(
        self,
        query: Query,
        candidates: Sequence[SearchResult],
    ) -> Sequence[SearchResult]:
        if not candidates:
            return []

        pairs = [
            (query.text, candidate.snippet or candidate.title)
            for candidate in candidates
        ]
        scores = np.asarray(self.model.predict(pairs)).reshape(-1)
        if len(scores) != len(candidates):
            raise ValueError("Reranker returned an invalid number of scores.")

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            replace(candidate, score=float(score), rank=rank)
            for rank, (candidate, score) in enumerate(
                ranked[: query.top_k],
                start=1,
            )
        ]
