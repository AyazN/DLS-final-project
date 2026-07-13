from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from aise.contracts import Query, Retriever, SearchDocument, SearchResult

from .tokenizer import tokenize

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


class BM25Retriever(Retriever):
    def __init__(
        self,
        documents: Sequence[SearchDocument],
        engine_factory: Callable[[list[list[str]]], Any] | None = None,
    ) -> None:
        if not documents:
            raise ValueError("BM25Retriever requires at least one document.")
        self.documents = list(documents)
        corpus = [tokenize(document.body) for document in self.documents]

        factory = engine_factory or BM25Okapi
        if factory is None:
            raise ImportError(
                "BM25 retrieval requires rank-bm25. Install project requirements first."
            )
        self.bm25 = factory(corpus)

    def search(self, query: Query) -> Sequence[SearchResult]:
        scores = self.bm25.get_scores(tokenize(query.text))
        best_indices = np.argsort(scores)[::-1][: query.top_k]

        results: list[SearchResult] = []
        for rank, index in enumerate(best_indices, start=1):
            document = self.documents[int(index)]
            results.append(
                SearchResult(
                    doc_id=document.doc_id,
                    model_id=document.model_id,
                    score=float(scores[index]),
                    rank=rank,
                    title=document.title,
                    snippet=document.body[:500],
                    metadata=document.metadata,
                )
            )
        return results
