from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from aise.contracts import EvaluationExample, Query, SearchDocument, SearchResult
from aise.evaluation import RetrievalEvaluator
from aise.pipeline import SearchPipeline
from aise.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    ReciprocalRankFusion,
    format_query_for_encoder,
)
from aise.search_index import EmbeddingArtifacts, load_search_documents


def document(position: int, body: str) -> SearchDocument:
    return SearchDocument(
        doc_id=f"doc-{position}",
        title=f"Title {position}",
        body=body,
        model_id=f"model-{position}",
        metadata={"position": position},
    )


def result(doc_id: str, rank: int, *, model_id: str | None = None) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        model_id=model_id or doc_id,
        score=1.0,
        rank=rank,
        title=doc_id,
        snippet=f"snippet {doc_id}",
        metadata={"source": doc_id},
    )


def test_bm25_returns_structured_ranked_results() -> None:
    retriever = BM25Retriever(
        [
            document(0, "football video analysis"),
            document(1, "text generation"),
            document(2, "image segmentation"),
        ]
    )

    results = retriever.search(Query("football", top_k=2))

    assert [item.rank for item in results] == [1, 2]
    assert results[0].doc_id == "doc-0"
    assert results[0].model_id == "model-0"
    assert results[0].metadata["body"] == "football video analysis"


def test_low_memory_documents_keep_reranking_metadata() -> None:
    metadata = pd.DataFrame(
        [
            {
                "model_id": "model-0",
                "name": "Video model",
                "body": "classifies videos",
                "tags": ["vision", "video-classification"],
                "pipeline_tag": "video-classification",
            }
        ]
    )
    artifacts = EmbeddingArtifacts(
        embeddings=np.zeros((1, 2), dtype=np.float32),
        ids=np.array(["doc-0"]),
        metadata=metadata,
        run_config={"model_name": "fake"},
        directory=Path("."),
    )

    loaded = load_search_documents(artifacts, include_metadata=False)

    assert loaded[0].metadata == {"pipeline_tag": "video-classification"}
    assert loaded[0].tags == ("vision", "video-classification")


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append((texts, kwargs))
        return np.array([[1.0, 0.0]], dtype=np.float64)[:, ::-1]


class FakeIndex:
    dim = 2
    metric = "inner_product"
    higher_is_better = True

    def __init__(self, positions: Sequence[int] = (2, 0, -1)) -> None:
        self.positions = positions
        self.seen_query: np.ndarray | None = None

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert query.dtype == np.float32
        assert query.flags.c_contiguous
        self.seen_query = query
        values = np.array([[0.9, 0.8, -np.inf]], dtype=np.float32)
        positions = np.array([self.positions], dtype=np.int64)
        return values[:, :k], positions[:, :k]


class L2FakeIndex(FakeIndex):
    metric = "l2"
    higher_is_better = False


def test_dense_maps_faiss_positions_and_ignores_invalid_values() -> None:
    documents = [document(0, "zero"), document(1, "one"), document(2, "two")]
    encoder = FakeEncoder()
    retriever = DenseRetriever(
        FakeIndex(),
        documents,
        encoder,
        ids=np.array(["artifact-0", "artifact-1", "artifact-2"]),
        metadata=[{"row": 0}, {"row": 1}, {"row": 2}],
    )

    results = retriever.search(Query("query", top_k=3))

    assert [item.doc_id for item in results] == ["artifact-2", "artifact-0"]
    assert [item.model_id for item in results] == ["model-2", "model-0"]
    assert [item.rank for item in results] == [1, 2]
    assert results[0].metadata["row"] == 2
    assert encoder.calls[0][0] == ["query"]
    assert encoder.calls[0][1]["normalize_embeddings"] is True


def test_bge_query_formatting_uses_existing_instruction() -> None:
    formatted = format_query_for_encoder(" image classification ", "BAAI/bge-small-en-v1.5")

    assert formatted == (
        "Represent this sentence for searching relevant passages: image classification"
    )


def test_dense_labels_l2_values_as_lower_is_better() -> None:
    retriever = DenseRetriever(
        L2FakeIndex(positions=(0,)),
        [document(0, "zero")],
        FakeEncoder(),
    )

    result_item = retriever.search(Query("query", top_k=1))[0]

    assert result_item.metadata["dense_metric"] == "l2"
    assert result_item.metadata["score_direction"] == "lower_is_better"


def test_retrievers_propagate_task_tags_and_body() -> None:
    doc = SearchDocument(
        doc_id="doc-0",
        title="Video model",
        body="classifies videos",
        model_id="model-0",
        tags=("vision", "video-classification"),
        metadata={"pipeline_tag": "video-classification"},
    )

    bm25_result = BM25Retriever([doc]).search(Query("videos", top_k=1))[0]
    dense_result = DenseRetriever(
        FakeIndex(positions=(0,)),
        [doc],
        FakeEncoder(),
    ).search(Query("videos", top_k=1))[0]

    for item in (bm25_result, dense_result):
        assert item.metadata["pipeline_tag"] == "video-classification"
        assert item.metadata["tags"] == ("vision", "video-classification")
        assert item.metadata["body"] == "classifies videos"


def test_rrf_removes_duplicates_and_regenerates_ranks() -> None:
    fusion = ReciprocalRankFusion(k=60)

    fused = fusion.fuse(
        [result("a", 1), result("b", 2), result("b", 3)],
        [result("b", 1), result("c", 2)],
        top_k=3,
    )

    assert [item.doc_id for item in fused] == ["b", "a", "c"]
    assert [item.rank for item in fused] == [1, 2, 3]
    assert fused[0].metadata == {"source": "b"}


class SpyRetriever:
    def __init__(self, results: Sequence[SearchResult]) -> None:
        self.results = results
        self.calls = 0

    def search(self, query: Query) -> Sequence[SearchResult]:
        self.calls += 1
        return self.results


def test_hybrid_invokes_both_retrievers_and_limits_output() -> None:
    lexical = SpyRetriever([result("a", 1), result("b", 2)])
    dense = SpyRetriever([result("b", 1), result("c", 2)])
    hybrid = HybridRetriever(lexical, dense)

    results = hybrid.search(Query("query", top_k=2))

    assert lexical.calls == dense.calls == 1
    assert len(results) == 2
    assert len({item.doc_id for item in results}) == 2


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        self.pairs = pairs
        return np.array([0.1, 0.9], dtype=np.float32)


def test_reranker_preserves_identifiers_and_changes_ranking() -> None:
    candidates = [
        SearchResult("a", "model-a", 10.0, 1, "A", "", {"body": "first full body"}),
        SearchResult("b", "model-b", 9.0, 2, "B", "", {"body": "second full body"}),
    ]
    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker(model, top_k=2)

    reranked = reranker.rank(Query("query", top_k=10), candidates)

    assert [item.doc_id for item in reranked] == ["b", "a"]
    assert [item.model_id for item in reranked] == ["model-b", "model-a"]
    assert [item.rank for item in reranked] == [1, 2]
    assert [item.score for item in reranked] == pytest.approx([0.9, 0.1])
    assert "first full body" in model.pairs[0][1]


def test_evaluator_accepts_integrated_retriever_and_computes_metrics() -> None:
    retriever = SpyRetriever(
        [result("doc-2", 1, model_id="model-2"), result("doc-1", 2, model_id="model-1")]
    )
    evaluator = RetrievalEvaluator(metric_k_values=(1, 2))
    examples = [
        EvaluationExample(Query("query", top_k=2), relevant_model_ids=("model-1",))
    ]

    report = evaluator.evaluate(examples, retriever)

    assert report.metrics["mrr"] == 0.5
    assert report.metrics["recall@1"] == 0.0
    assert report.metrics["recall@2"] == 1.0
    assert report.metrics["ndcg@2"] == pytest.approx(1.0 / np.log2(3.0))
    assert report.metrics["mean_latency_ms"] >= 0.0


def test_evaluator_handles_query_with_no_relevant_documents() -> None:
    evaluator = RetrievalEvaluator(metric_k_values=(1,))
    examples = [EvaluationExample(Query("query", top_k=1), relevant_model_ids=())]

    report = evaluator.evaluate(examples, SpyRetriever([result("doc-1", 1)]))

    assert report.metrics["mrr"] == 0.0
    assert report.metrics["recall@1"] == 0.0
    assert report.metrics["ndcg@1"] == 0.0


def test_evaluator_uses_final_reranked_order() -> None:
    candidates = [
        SearchResult("a", "relevant", 1.0, 1, "A", "", {"body": "short"}),
        SearchResult(
            "b",
            "irrelevant",
            0.5,
            2,
            "B",
            "",
            {"body": "much longer body"},
        ),
    ]

    class FixedRetriever:
        def search(self, query):
            return candidates

    pipeline = SearchPipeline(
        retriever=FixedRetriever(),
        ranker=CrossEncoderReranker(FakeCrossEncoder(), top_k=2),
    )
    report = RetrievalEvaluator(metric_k_values=(1, 2)).evaluate(
        [EvaluationExample(Query("query", top_k=2), relevant_model_ids=("relevant",))],
        pipeline,
    )

    assert report.details["per_query"][0]["retrieved_model_ids"] == [
        "irrelevant",
        "relevant",
    ]
    assert report.metrics["mrr"] == 0.5


def test_empty_inputs_do_not_crash() -> None:
    assert BM25Retriever([]).search(Query("query")) == []
    assert DenseRetriever(FakeIndex(), [], FakeEncoder()).search(Query("query")) == []
    assert ReciprocalRankFusion().fuse([], []) == []
    assert CrossEncoderReranker(FakeCrossEncoder()).rank(Query("query"), []) == []
    report = RetrievalEvaluator().evaluate([], SpyRetriever([]))
    assert report.metrics == {}


@pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss is optional")
def test_faiss_flat_validates_search_and_round_trips(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    from aise.search_index import FaissFlatIndex

    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = FaissFlatIndex(2, metric="inner_product")
    index.build(vectors)
    values, positions = index.search(vectors[:1], 2)
    assert positions.dtype == np.int64
    assert positions.tolist() == [[0, 1]]
    assert values[0, 0] == pytest.approx(1.0)

    path = tmp_path / "flat.faiss"
    index.save(path)
    loaded = FaissFlatIndex(2)
    loaded.load(path)
    assert loaded.search(vectors[:1], 1)[1].tolist() == [[0]]

    with pytest.raises(TypeError):
        loaded.search(vectors.astype(np.float64), 1)
    with pytest.raises(ValueError):
        loaded.search(np.ones((1, 3), dtype=np.float32), 1)
