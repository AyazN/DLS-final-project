from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


Metadata = Mapping[str, Any]
FloatMatrix = NDArray[np.float32]
IndexArray = NDArray[np.int64]


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
    """Document ready for lexical or dense indexing."""

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

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Query text must not be empty.")
        if self.top_k <= 0:
            raise ValueError("Query top_k must be positive.")


@dataclass(frozen=True)
class SearchResult:
    """Ranked result returned by a retriever, reranker, or full pipeline."""

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
    """Aggregated retrieval metrics and serializable experiment details."""

    metrics: Mapping[str, float]
    details: Metadata = field(default_factory=dict)


class DatasetLoader(Protocol):
    def load(self) -> Iterable[ModelCard]:
        """Load model cards from a dataset source."""


class DocumentBuilder(Protocol):
    def build(self, cards: Iterable[ModelCard]) -> Iterable[SearchDocument]:
        """Convert model cards into indexable search documents."""


class VectorIndex(Protocol):
    """Index over dense vectors; larger returned scores are always better."""

    def build(self, vectors: FloatMatrix) -> None:
        """Build or replace the index from a two-dimensional vector matrix."""

    def search(self, query_vectors: FloatMatrix, k: int) -> tuple[FloatMatrix, IndexArray]:
        """Return similarity scores and row indices for each query vector."""

    def save(self, path: str | Path) -> None:
        """Persist index artifacts."""

    def load(self, path: str | Path) -> None:
        """Load index artifacts."""


class Retriever(Protocol):
    def search(self, query: Query) -> Sequence[SearchResult]:
        """Return ranked candidates for a query."""


class Reranker(Protocol):
    def rank(self, query: Query, candidates: Sequence[SearchResult]) -> Sequence[SearchResult]:
        """Re-rank candidate results."""


class Evaluator(Protocol):
    def evaluate(
        self,
        examples: Sequence[EvaluationExample],
        retriever: Retriever,
    ) -> EvaluationReport:
        """Evaluate a retriever on labeled examples."""
