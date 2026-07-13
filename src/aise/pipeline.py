from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import Query, Reranker, Retriever, SearchResult


@dataclass
class SearchPipeline:
    """Thin orchestrator for retrieval and optional reranking."""

    retriever: Retriever
    reranker: Reranker | None = None

    def search(self, query: Query) -> Sequence[SearchResult]:
        candidates = self.retriever.search(query)
        if self.reranker is None:
            return candidates
        return self.reranker.rank(query, candidates)


class EmptyRetriever:
    """Placeholder retriever used by contract tests."""

    def search(self, query: Query) -> Sequence[SearchResult]:
        return []
