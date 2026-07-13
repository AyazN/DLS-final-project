from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from aise.contracts import Query, Retriever, SearchDocument, SearchResult, VectorIndex


class DenseRetriever(Retriever):
    def __init__(
        self,
        index: VectorIndex,
        documents: Sequence[SearchDocument],
        model: Any,
    ) -> None:
        if not documents:
            raise ValueError("DenseRetriever requires at least one document.")
        self.index = index
        self.documents = list(documents)
        self.model = model

    def search(self, query: Query) -> Sequence[SearchResult]:
        query_vector = np.asarray(
            self.model.encode([query.text], convert_to_numpy=True),
            dtype=np.float32,
        )
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        scores, indices = self.index.search(query_vector, query.top_k)
        results: list[SearchResult] = []
        for score, raw_index in zip(scores[0], indices[0]):
            index = int(raw_index)
            if index < 0:
                continue
            if index >= len(self.documents):
                raise IndexError(
                    f"Vector index returned row {index}, but only "
                    f"{len(self.documents)} documents are loaded."
                )
            document = self.documents[index]
            results.append(
                SearchResult(
                    doc_id=document.doc_id,
                    model_id=document.model_id,
                    score=float(score),
                    rank=len(results) + 1,
                    title=document.title,
                    snippet=document.body[:500],
                    metadata=document.metadata,
                )
            )
        return results
