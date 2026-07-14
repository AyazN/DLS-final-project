from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from aise.contracts import Query, SearchDocument, SearchResult, VectorIndex

from .encoding import format_query_for_encoder


class DenseRetriever:
    def __init__(
        self,
        index: VectorIndex,
        documents: Sequence[SearchDocument],
        model: Any,
        *,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        ids: Sequence[Any] | None = None,
        metadata: Any | None = None,
        normalize_embeddings: bool = True,
        query_formatter: Callable[[str, str], str] = format_query_for_encoder,
    ) -> None:
        self.index = index
        self.documents = documents
        self.model = model
        self.encoder_name = encoder_name
        self.ids = ids
        self.metadata = metadata
        self.normalize_embeddings = normalize_embeddings
        self.query_formatter = query_formatter

        if ids is not None and len(ids) != len(documents):
            raise ValueError("ids and documents must have the same number of rows")
        if metadata is not None and len(metadata) != len(documents):
            raise ValueError("metadata and documents must have the same number of rows")
        if index.metric not in {"inner_product", "l2"}:
            raise ValueError(f"Unsupported index metric: {index.metric!r}")

    def _metadata_at(self, position: int) -> dict[str, Any]:
        doc = self.documents[position]
        result = dict(doc.metadata)
        if self.metadata is not None:
            row = (
                self.metadata.iloc[position]
                if hasattr(self.metadata, "iloc")
                else self.metadata[position]
            )
            result.update(row.to_dict() if hasattr(row, "to_dict") else dict(row))
        result.setdefault("body", doc.body)
        result["dense_metric"] = self.index.metric
        result["score_direction"] = (
            "higher_is_better" if self.index.higher_is_better else "lower_is_better"
        )
        return result

    def search(self, query: Query) -> Sequence[SearchResult]:
        top_k = max(0, int(query.top_k))
        if top_k == 0 or not self.documents:
            return []

        formatted_query = self.query_formatter(query.text, self.encoder_name)
        query_vector = self.model.encode(
            [formatted_query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        values, positions = self.index.search(query_vector, top_k)
        results: list[SearchResult] = []
        for value, raw_position in zip(values[0], positions[0]):
            position = int(raw_position)
            if position < 0 or position >= len(self.documents):
                continue
            doc = self.documents[position]
            doc_id = str(self.ids[position]) if self.ids is not None else doc.doc_id
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    model_id=doc.model_id,
                    score=float(value),
                    rank=len(results) + 1,
                    title=doc.title,
                    snippet=doc.body[:500],
                    metadata=self._metadata_at(position),
                )
            )
        return results
