from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from aise.contracts import Query, SearchDocument, SearchResult, Retriever

from .tokenizer import tokenize


class BM25Retriever(Retriever):
    def __init__(self, documents: Sequence[SearchDocument]) -> None:
        self.documents = list(documents)

        # corpus for BM25
        corpus = [
            tokenize(doc.text)
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(corpus)

    def search(
        self,
        query: Query,
        top_k: int = 100,
    ) -> Sequence[SearchResult]:

        tokens = tokenize(query.text)

        scores = self.bm25.get_scores(tokens)

        best_idx = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in best_idx:

            results.append(
                SearchResult(
                    document=self.documents[idx],
                    score=float(scores[idx]),
                )
            )

        return results