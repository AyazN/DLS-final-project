from aise.contracts import Query, SearchResult
from aise.retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """Deterministic fake: score = length of the document text."""

    def predict(self, pairs):
        return [float(len(doc_text)) for _query, doc_text in pairs]


def _result(doc_id: str, model_id: str, *, title="T", snippet="s", metadata=None) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        model_id=model_id,
        score=0.5,
        rank=1,
        title=title,
        snippet=snippet,
        metadata=metadata or {},
    )


def test_higher_fake_score_moves_candidate_up() -> None:
    short = _result("a", "m-a", metadata={"body": "x"})
    long = _result("b", "m-b", metadata={"body": "x" * 200})
    reranker = CrossEncoderReranker(FakeCrossEncoder())

    ranked = reranker.rank(Query(text="q", top_k=10), [short, long])

    assert [r.doc_id for r in ranked] == ["b", "a"]


def test_score_to_document_mapping_preserved() -> None:
    candidates = [
        _result("a", "m-a", metadata={"body": "aaaaa"}),
        _result("b", "m-b", metadata={"body": "bb"}),
        _result("c", "m-c", metadata={"body": "ccc"}),
    ]
    reranker = CrossEncoderReranker(FakeCrossEncoder())

    ranked = reranker.rank(Query(text="q", top_k=10), candidates)

    by_id = {r.doc_id: r.score for r in ranked}
    assert by_id["a"] > by_id["c"] > by_id["b"]


def test_ranks_restart_from_one() -> None:
    candidates = [_result(str(i), f"m-{i}", metadata={"body": "x" * i}) for i in range(1, 4)]
    reranker = CrossEncoderReranker(FakeCrossEncoder())

    ranked = reranker.rank(Query(text="q", top_k=10), candidates)

    assert [r.rank for r in ranked] == [1, 2, 3]


def test_ids_and_metadata_survive_reranking() -> None:
    meta = {"body": "hello", "pipeline_tag": "text-classification"}
    candidate = _result("doc-1", "hf/model-1", title="Model One", snippet="snip", metadata=meta)
    reranker = CrossEncoderReranker(FakeCrossEncoder())

    ranked = reranker.rank(Query(text="q", top_k=10), [candidate])

    assert ranked[0].doc_id == "doc-1"
    assert ranked[0].model_id == "hf/model-1"
    assert ranked[0].title == "Model One"
    assert ranked[0].snippet == "snip"
    assert ranked[0].metadata == meta


def test_empty_input_returns_empty_list() -> None:
    reranker = CrossEncoderReranker(FakeCrossEncoder())

    assert reranker.rank(Query(text="q", top_k=10), []) == []


def test_all_candidates_scored_before_truncation() -> None:
    seen_pairs: list[tuple[str, str]] = []

    class RecordingEncoder:
        def predict(self, pairs):
            seen_pairs.extend(pairs)
            return [float(i) for i in range(len(pairs))]

    candidates = [_result(str(i), f"m-{i}", metadata={"body": "x"}) for i in range(50)]
    reranker = CrossEncoderReranker(RecordingEncoder(), top_k=10)

    ranked = reranker.rank(Query(text="q", top_k=50), candidates)

    assert len(seen_pairs) == 50
    assert len(ranked) == 10


def test_pipeline_returns_reranked_order() -> None:
    from aise.pipeline import SearchPipeline

    class StubRetriever:
        def search(self, query):
            return [
                _result("a", "m-a", metadata={"body": "x"}),
                _result("b", "m-b", metadata={"body": "x" * 50}),
            ]

    pipeline = SearchPipeline(retriever=StubRetriever(), ranker=CrossEncoderReranker(FakeCrossEncoder()))

    results = pipeline.search(Query(text="q", top_k=10))

    assert [r.doc_id for r in results] == ["b", "a"]
    assert [r.rank for r in results] == [1, 2]


def test_reranking_text_contains_intended_fields() -> None:
    candidate = _result(
        "a",
        "hf/model-a",
        title="Model A",
        metadata={"body": "a useful description", "pipeline_tag": "text-classification", "tags": ["nlp", "bert"]},
    )
    reranker = CrossEncoderReranker(FakeCrossEncoder())

    text = reranker._document_text(candidate)

    assert "Model A" in text
    assert "hf/model-a" in text
    assert "text-classification" in text
    assert "nlp" in text and "bert" in text
    assert "a useful description" in text