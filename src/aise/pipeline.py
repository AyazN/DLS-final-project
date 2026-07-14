from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import Query, Reranker, Retriever, SearchResult


@dataclass
class SearchPipeline:
    """Thin orchestrator for retrieval and optional ranking."""

    retriever: Retriever
    ranker: Reranker | None = None

    def search(self, query: Query) -> Sequence[SearchResult]:
        candidates = self.retriever.search(query)
        if self.ranker is None:
            return candidates
        return self.ranker.rank(query, candidates)


class EmptyRetriever:
    """Placeholder retriever used until participant modules are connected."""

    def search(self, query: Query) -> Sequence[SearchResult]:
        return []
