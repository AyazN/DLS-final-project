from __future__ import annotations

import importlib
import unittest

import numpy as np

from aise.cli import build_parser, build_pipeline
from aise.contracts import (
    EvaluationExample,
    ModelCard,
    Query,
    SearchDocument,
    SearchResult,
)
from aise.pipeline import EmptyRetriever, SearchPipeline


BM25_MODULE = importlib.import_module(
    "participants.04_retrieval_ranking.bm25"
)
DENSE_MODULE = importlib.import_module(
    "participants.04_retrieval_ranking.dense"
)
HYBRID_MODULE = importlib.import_module(
    "participants.04_retrieval_ranking.hybrid"
)
RERANKER_MODULE = importlib.import_module(
    "participants.04_retrieval_ranking.reranker"
)
RRF_MODULE = importlib.import_module(
    "participants.04_retrieval_ranking.rrf"
)
EVALUATION_MODULE = importlib.import_module(
    "participants.05_evaluation.evaluation"
)


def documents() -> list[SearchDocument]:
    return [
        SearchDocument(
            doc_id="football",
            model_id="hf/football",
            title="Football Vision",
            body="football video player tracking computer vision",
            tags=("sports",),
        ),
        SearchDocument(
            doc_id="medical",
            model_id="hf/medical",
            title="Medical Model",
            body="medicine diagnosis radiology classification",
            tags=("medicine",),
        ),
    ]


class FakeBM25Okapi:
    def __init__(self, corpus):
        self.corpus = corpus

    def get_scores(self, query_tokens):
        return np.asarray(
            [
                sum(tokens.count(token) for token in query_tokens)
                for tokens in self.corpus
            ],
            dtype=np.float32,
        )


class FakeIndex:
    def search(self, query, k):
        scores = np.asarray([[0.9, 0.2]], dtype=np.float32)
        indices = np.asarray([[0, 1]], dtype=np.int64)
        return scores[:, :k], indices[:, :k]


class FakeEncoder:
    def encode(self, texts, convert_to_numpy=True):
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class FakeCrossEncoder:
    def predict(self, pairs):
        return np.asarray(
            [
                2.0 if "medicine" in text.casefold() else 1.0
                for _, text in pairs
            ],
            dtype=np.float32,
        )


class ContractAndIntegrationTests(unittest.TestCase):
    def test_contract_shapes(self) -> None:
        card = ModelCard(
            model_id="hf/demo",
            name="Demo",
            text="football model",
        )
        result = SearchResult(
            doc_id=card.model_id,
            model_id=card.model_id,
            score=0.9,
            rank=1,
            title=card.name,
        )
        self.assertEqual(result.model_id, "hf/demo")
        with self.assertRaises(ValueError):
            Query(text="", top_k=1)

    def test_empty_pipeline(self) -> None:
        pipeline = SearchPipeline(EmptyRetriever())
        self.assertEqual(
            pipeline.search(Query("football")),
            [],
        )

    def test_participant_bm25_uses_shared_contract(self) -> None:
        retriever = BM25_MODULE.BM25Retriever(
            documents(),
            engine_factory=FakeBM25Okapi,
        )
        results = retriever.search(
            Query("football tracking", top_k=1)
        )
        self.assertEqual(results[0].model_id, "hf/football")
        self.assertEqual(results[0].rank, 1)

    def test_participant_dense_uses_index_and_documents(self) -> None:
        retriever = DENSE_MODULE.DenseRetriever(
            FakeIndex(),
            documents(),
            FakeEncoder(),
        )
        results = retriever.search(Query("football", top_k=2))
        self.assertEqual(
            [result.model_id for result in results],
            ["hf/football", "hf/medical"],
        )

    def test_participant_hybrid_rrf(self) -> None:
        bm25 = BM25_MODULE.BM25Retriever(
            documents(),
            engine_factory=FakeBM25Okapi,
        )
        dense = DENSE_MODULE.DenseRetriever(
            FakeIndex(),
            documents(),
            FakeEncoder(),
        )
        hybrid = HYBRID_MODULE.HybridRetriever(
            bm25,
            dense,
            RRF_MODULE.ReciprocalRankFusion(),
        )
        results = hybrid.search(Query("football", top_k=2))
        self.assertEqual(results[0].model_id, "hf/football")
        self.assertEqual([result.rank for result in results], [1, 2])

    def test_participant_reranker_and_pipeline(self) -> None:
        dense = DENSE_MODULE.DenseRetriever(
            FakeIndex(),
            documents(),
            FakeEncoder(),
        )
        reranker = RERANKER_MODULE.CrossEncoderReranker(
            FakeCrossEncoder()
        )
        pipeline = SearchPipeline(dense, reranker)
        results = pipeline.search(Query("medicine", top_k=2))
        self.assertEqual(results[0].model_id, "hf/medical")

    def test_participant_evaluation(self) -> None:
        retriever = BM25_MODULE.BM25Retriever(
            documents(),
            engine_factory=FakeBM25Okapi,
        )
        evaluator = EVALUATION_MODULE.RetrievalEvaluator()
        report = evaluator.evaluate(
            [
                EvaluationExample(
                    query=Query("football", top_k=2),
                    relevant_model_ids=("hf/football",),
                )
            ],
            retriever,
            top_k=2,
            metric_k_values=(1, 2),
            save_report=False,
        )
        self.assertEqual(report.metrics["mrr"], 1.0)
        self.assertEqual(report.metrics["recall@1"], 1.0)

    def test_cli_builds_participant_retriever(self) -> None:
        original = BM25_MODULE.BM25Okapi
        BM25_MODULE.BM25Okapi = FakeBM25Okapi
        try:
            pipeline = build_pipeline(
                documents(),
                mode="bm25",
            )
        finally:
            BM25_MODULE.BM25Okapi = original
        self.assertEqual(
            pipeline.retriever.__class__.__module__,
            "participants.04_retrieval_ranking.bm25",
        )

    def test_cli_parser(self) -> None:
        args = build_parser().parse_args(
            ["search", "football", "--mode", "hybrid"]
        )
        self.assertEqual(args.mode, "hybrid")


if __name__ == "__main__":
    unittest.main()
