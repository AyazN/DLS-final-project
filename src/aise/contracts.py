from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence


Metadata = Mapping[str, Any]


@dataclass(frozen=True)
class ModelCard:
    """Raw or lightly normalized model card from the source dataset."""

    model_id: str
    name: str
    text: str
    tags: tuple[str, ...] = ()
    task: str | None = None
    source_url: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class SearchDocument:
    """Document that is ready for lexical or dense indexing."""

    doc_id: str
    title: str
    body: str
    model_id: str
    tags: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Query:
    """User search query after optional normalization."""

    text: str
    top_k: int = 10
    filters: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """Ranked result returned by a retriever, ranker, or full pipeline."""

    doc_id: str
    model_id: str
    score: float
    rank: int
    title: str
    snippet: str = ""
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationExample:
    """Single labeled query for offline evaluation."""

    query: Query
    relevant_model_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregated retrieval metrics."""

    metrics: Mapping[str, float]
    details: Metadata = field(default_factory=dict)


class DatasetLoader(Protocol):
    def load(self) -> Iterable[ModelCard]:
        """Load model cards from a dataset source."""


class DocumentBuilder(Protocol):
    def build(self, cards: Iterable[ModelCard]) -> Iterable[SearchDocument]:
        """Convert model cards into indexable search documents."""


class VectorIndex(Protocol):
    def build(self, documents: Sequence[SearchDocument]) -> None:
        """Build or update the index from documents."""

    def save(self, path: str) -> None:
        """Persist index artifacts."""

    def load(self, path: str) -> None:
        """Load index artifacts."""


class Retriever(Protocol):
    def search(self, query: Query) -> Sequence[SearchResult]:
        """Return initial candidates for a query."""


class Ranker(Protocol):
    def rank(self, query: Query, candidates: Sequence[SearchResult]) -> Sequence[SearchResult]:
        """Re-rank candidate results."""


class Evaluator(Protocol):
    def evaluate(
        self,
        examples: Sequence[EvaluationExample],
        retriever: Retriever,
    ) -> EvaluationReport:
        """Evaluate a retriever on labeled examples."""
