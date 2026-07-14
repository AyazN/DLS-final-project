from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from aise.contracts import Query, SearchResult


class CrossEncoderReranker:
    def __init__(
        self,
        model: Any,
        *,
        top_k: int | None = None,
        document_text_key: str = "body",
    ) -> None:
        self.model = model
        self.top_k = top_k
        self.document_text_key = document_text_key

    def _document_text(self, candidate: SearchResult) -> str:
        full_text = candidate.metadata.get(self.document_text_key, "")
        useful_text = str(full_text).strip() or candidate.snippet.strip() or candidate.title
        return f"{candidate.title}\n{useful_text}" if candidate.title else useful_text

    def rank(
        self,
        query: Query,
        candidates: Sequence[SearchResult],
    ) -> Sequence[SearchResult]:
        if not candidates:
            return []

        pairs = [(query.text, self._document_text(candidate)) for candidate in candidates]
        scores = np.asarray(self.model.predict(pairs), dtype=np.float64).reshape(-1)
        if len(scores) != len(candidates):
            raise ValueError("Cross-encoder returned a different number of scores than candidates")

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        limit = query.top_k if self.top_k is None else self.top_k
        ranked = ranked[: max(0, int(limit))]

        return [
            SearchResult(
                doc_id=candidate.doc_id,
                model_id=candidate.model_id,
                score=float(score),
                rank=rank,
                title=candidate.title,
                snippet=candidate.snippet,
                metadata=candidate.metadata,
            )
            for rank, (candidate, score) in enumerate(ranked, start=1)
        ]
