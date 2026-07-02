from __future__ import annotations

from collections.abc import Sequence

from aise.contracts import Query, SearchResult


class HybridRetriever:
    def search(self, query: Query) -> Sequence[SearchResult]:
        raise NotImplementedError


class Reranker:
    def rank(self, query: Query, candidates: Sequence[SearchResult]) -> Sequence[SearchResult]:
        raise NotImplementedError
