from __future__ import annotations

from collections.abc import Sequence

from aise.contracts import (
    Query,
    SearchResult,
    Retriever,
)

from .rrf import ReciprocalRankFusion


class HybridRetriever(Retriever):
    def __init__(
        self,
        bm25: Retriever,
        dense: Retriever,
        fusion: ReciprocalRankFusion,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.fusion = fusion

    def search(
        self,
        query: Query,
    ) -> Sequence[SearchResult]:
        # TODO:
        # Replace the temporary DenseRetriever with the final
        # implementation after the embeddings module is integrated.

        bm25_results = self.bm25.search(query)
        dense_results = self.dense.search(query)

        return self.fusion.fuse(
            bm25_results,
            dense_results,
        )