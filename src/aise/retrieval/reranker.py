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
        """Build deterministic reranking text.

        Mirrors the structured fields (model id, task, tags) that the dense
        retriever's embedding text already includes, plus the body/snippet.
        Without these fields the cross-encoder only sees free text and drifts
        from candidates that were retrieved via pipeline_tag/tag matches.
        """
        metadata = candidate.metadata
        body = str(metadata.get(self.document_text_key, "")).strip() or candidate.snippet.strip()

        pipeline_tag = str(metadata.get("pipeline_tag") or metadata.get("task") or "").strip()

        tags = metadata.get("tags") or ()
        if isinstance(tags, str):
            tags = [tags]
        tags_text = ", ".join(str(tag).strip() for tag in list(tags)[:10] if str(tag).strip())

        parts = [
            f"Title: {candidate.title}" if candidate.title else "",
            f"Model ID: {candidate.model_id}" if candidate.model_id else "",
            f"Task: {pipeline_tag}" if pipeline_tag else "",
            f"Tags: {tags_text}" if tags_text else "",
            body,
        ]
        text = "\n".join(part for part in parts if part)
        return text or candidate.title or candidate.model_id

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