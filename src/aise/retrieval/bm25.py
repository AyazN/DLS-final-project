from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from aise.contracts import Query, SearchDocument, SearchResult

from .tokenizer import tokenize


class BM25Retriever:
    def __init__(self, documents: Sequence[SearchDocument]) -> None:
        self.documents = list(documents)
        corpus = [tokenize(f"{doc.title} {doc.body}") for doc in self.documents]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: Query) -> Sequence[SearchResult]:
        top_k = max(0, int(query.top_k))
        if self.bm25 is None or top_k == 0:
            return []

        scores = np.asarray(self.bm25.get_scores(tokenize(query.text)), dtype=np.float64)
        best_positions = np.argsort(-scores, kind="stable")[:top_k]
        results: list[SearchResult] = []
        for rank, position in enumerate(best_positions, start=1):
            doc = self.documents[int(position)]
            metadata = dict(doc.metadata)
            metadata.setdefault("body", doc.body)
            metadata.setdefault("tags", doc.tags)
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    model_id=doc.model_id,
                    score=float(scores[position]),
                    rank=rank,
                    title=doc.title,
                    snippet=doc.body[:500],
                    metadata=metadata,
                )
            )
        return results
