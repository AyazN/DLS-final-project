from aise.contracts import Query, SearchResult
from aise.retrieval.reranker import CrossEncoderReranker, RankFusionReranker


class FakeCrossEncoder:
    """Deterministic fake: score = length of the document text."""

    def predict(self, pairs):
        return [float(len(doc_text)) for _query, doc_text in pairs]


def _result(doc_id: str, model_id: str, rank: int = 1, *, title="T", snippet="s", metadata=None) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        model_id=model_id,
        score=0.5,
        rank=rank,
        title=title,
        snippet=snippet,
        metadata=metadata or {},
    )


# --- CrossEncoderReranker: ordering, identity, truncation ---------------------------------


def test_higher_fake_score_moves_candidate_up() -> None:
    short = _result("a", "m-a", 1, metadata={"body": "x"})
    long = _result("b", "m-b", 2, metadata={"body": "x" * 200})
    reranker = CrossEncoderReranker(FakeCrossEncoder())

    ranked = reranker.rank(Query(text="q", top_k=10), [short, long])

    assert [r.doc_id for r in ranked] == ["b", "a"]


def test_score_to_document_mapping_preserved() -> None:
    candidates = [
        _result("a", "m-a", 1, metadata={"body": "aaaaa"}),
        _result("b", "m-b", 2, metadata={"body": "bb"}),
        _result("c", "m-c", 3, metadata={"body": "ccc"}),
    ]
    reranker = CrossEncoderReranker(FakeCrossEncoder())

    ranked = reranker.rank(Query(text="q", top_k=10), candidates)

    by_id = {r.doc_id: r.score for r in ranked}
    assert by_id["a"] > by_id["c"] > by_id["b"]


def test_ranks_restart_from_one() -> None:
    candidates = [_result(str(i), f"m-{i}", i, metadata={"body": "x" * i}) for i in range(1, 4)]
    reranker = CrossEncoderReranker(FakeCrossEncoder())

    ranked = reranker.rank(Query(text="q", top_k=10), candidates)

    assert [r.rank for r in ranked] == [1, 2, 3]


def test_ids_and_metadata_survive_reranking() -> None:
    meta = {"body": "hello", "pipeline_tag": "text-classification"}
    candidate = _result("doc-1", "hf/model-1", 1, title="Model One", snippet="snip", metadata=meta)
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

    candidates = [_result(str(i), f"m-{i}", i + 1, metadata={"body": "x"}) for i in range(50)]
    reranker = CrossEncoderReranker(RecordingEncoder(), top_k=10)

    ranked = reranker.rank(Query(text="q", top_k=50), candidates)

    assert len(seen_pairs) == 50
    assert len(ranked) == 10


def test_pipeline_returns_reranked_order() -> None:
    from aise.pipeline import SearchPipeline

    class StubRetriever:
        def search(self, query):
            return [
                _result("a", "m-a", 1, metadata={"body": "x"}),
                _result("b", "m-b", 2, metadata={"body": "x" * 50}),
            ]

    pipeline = SearchPipeline(retriever=StubRetriever(), ranker=CrossEncoderReranker(FakeCrossEncoder()))

    results = pipeline.search(Query(text="q", top_k=10))

    assert [r.doc_id for r in results] == ["b", "a"]
    assert [r.rank for r in results] == [1, 2]


# --- text strategies: natural language, no raw labels --------------------------------------


def test_text_strategies_produce_clean_natural_language_text() -> None:
    candidate = _result(
        "a",
        "hf/model-a",
        1,
        title="Model A",
        metadata={
            "body": "a useful description",
            "pipeline_tag": "video-classification",
            "tags": ["computer-vision", "transformers"],
        },
    )

    for strategy in ("title_body", "title_task_body", "title_task_tags_body", "body_only"):
        reranker = CrossEncoderReranker(FakeCrossEncoder(), text_strategy=strategy)
        text = reranker._document_text(candidate)

        # no raw labeled/structured fields or raw model ids leak into the text
        for forbidden in ("Model ID:", "Task:", "Tags:", "['", "hf/model-a"):
            assert forbidden not in text

        assert "a useful description" in text

    task_text = CrossEncoderReranker(FakeCrossEncoder(), text_strategy="title_task_body")._document_text(candidate)
    assert "video classification" in task_text

    tags_text = CrossEncoderReranker(FakeCrossEncoder(), text_strategy="title_task_tags_body")._document_text(
        candidate
    )
    assert "computer vision" in tags_text and "transformers" in tags_text


def test_text_strategy_length_is_bounded() -> None:
    candidate = _result("a", "m-a", 1, title="T", metadata={"body": "x" * 5000})
    reranker = CrossEncoderReranker(FakeCrossEncoder(), max_text_chars=200)

    assert len(reranker._document_text(candidate)) <= 200


def test_invalid_text_strategy_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        CrossEncoderReranker(FakeCrossEncoder(), text_strategy="not-a-strategy")


# --- RankFusionReranker: fusion, identity, weight=0 baseline --------------------------------


def test_fusion_weight_zero_reproduces_hybrid_order() -> None:
    candidates = [
        _result("a", "m-a", 1, metadata={"body": "z"}),
        _result("b", "m-b", 2, metadata={"body": "z" * 100}),
        _result("c", "m-c", 3, metadata={"body": "z" * 50}),
    ]
    fusion = RankFusionReranker(CrossEncoderReranker(FakeCrossEncoder()), cross_encoder_weight=0.0)

    fused = fusion.rank(Query(text="q", top_k=10), candidates)

    assert [r.doc_id for r in fused] == ["a", "b", "c"]


def test_fusion_cross_encoder_can_promote_a_candidate() -> None:
    # hybrid order: a, b, c (by .rank) but CE strongly prefers c (longest body)
    candidates = [
        _result("a", "m-a", 1, metadata={"body": "z"}),
        _result("b", "m-b", 2, metadata={"body": "zz"}),
        _result("c", "m-c", 3, metadata={"body": "z" * 500}),
    ]
    fusion = RankFusionReranker(CrossEncoderReranker(FakeCrossEncoder()), cross_encoder_weight=5.0, k=1)

    fused = fusion.rank(Query(text="q", top_k=10), candidates)

    assert fused[0].doc_id == "c"


def test_fusion_preserves_candidate_identity() -> None:
    candidates = [
        _result("a", "m-a", 1, title="A", snippet="sa", metadata={"body": "z", "x": 1}),
        _result("b", "m-b", 2, title="B", snippet="sb", metadata={"body": "zz", "x": 2}),
    ]
    fusion = RankFusionReranker(CrossEncoderReranker(FakeCrossEncoder()))

    fused = fusion.rank(Query(text="q", top_k=10), candidates)

    by_id = {r.doc_id: r for r in fused}
    assert by_id["a"].title == "A" and by_id["a"].snippet == "sa" and by_id["a"].metadata["x"] == 1
    assert by_id["b"].title == "B" and by_id["b"].snippet == "sb" and by_id["b"].metadata["x"] == 2


def test_fusion_ranks_restart_from_one_and_no_duplicates() -> None:
    candidates = [_result(str(i), f"m-{i}", i, metadata={"body": "x" * i}) for i in range(1, 6)]
    fusion = RankFusionReranker(CrossEncoderReranker(FakeCrossEncoder()))

    fused = fusion.rank(Query(text="q", top_k=10), candidates)

    assert [r.rank for r in fused] == [1, 2, 3, 4, 5]
    assert len(fused) == len({r.doc_id for r in fused})


def test_fusion_scores_all_candidates_before_truncation() -> None:
    seen_pairs: list[tuple[str, str]] = []

    class RecordingEncoder:
        def predict(self, pairs):
            seen_pairs.extend(pairs)
            return [float(i) for i in range(len(pairs))]

    candidates = [_result(str(i), f"m-{i}", i + 1, metadata={"body": "x"}) for i in range(50)]
    fusion = RankFusionReranker(CrossEncoderReranker(RecordingEncoder()), final_k=10)

    fused = fusion.rank(Query(text="q", top_k=50), candidates)

    assert len(seen_pairs) == 50
    assert len(fused) == 10


def test_fusion_empty_input_returns_empty_list() -> None:
    fusion = RankFusionReranker(CrossEncoderReranker(FakeCrossEncoder()))

    assert fusion.rank(Query(text="q", top_k=10), []) == []
