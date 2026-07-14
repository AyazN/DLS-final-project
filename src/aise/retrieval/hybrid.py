from __future__ import annotations

from collections.abc import Sequence

from aise.contracts import Query, Retriever, SearchResult

from .rrf import ReciprocalRankFusion


class HybridRetriever:
    def __init__(
        self,
        bm25: Retriever,
        dense: Retriever,
        fusion: ReciprocalRankFusion | None = None,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.fusion = fusion or ReciprocalRankFusion()

    def search(self, query: Query) -> Sequence[SearchResult]:
        bm25_results = self.bm25.search(query)
        dense_results = self.dense.search(query)
        return self.fusion.fuse(bm25_results, dense_results, top_k=query.top_k)
