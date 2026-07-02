from aise.contracts import ModelCard, Query, SearchDocument, SearchResult
from aise.pipeline import EmptyRetriever, SearchPipeline


def test_model_card_to_search_document_contract_shape() -> None:
    card = ModelCard(
        model_id="hf/demo-model",
        name="Demo Model",
        text="A model for football video analysis.",
        tags=("sports", "vision"),
    )

    document = SearchDocument(
        doc_id=card.model_id,
        title=card.name,
        body=card.text,
        model_id=card.model_id,
        tags=card.tags,
    )

    assert document.model_id == "hf/demo-model"
    assert "football" in document.body


def test_empty_pipeline_returns_no_results() -> None:
    pipeline = SearchPipeline(retriever=EmptyRetriever())

    results = pipeline.search(Query(text="AI for football"))

    assert results == []


def test_search_result_contract_shape() -> None:
    result = SearchResult(
        doc_id="doc-1",
        model_id="hf/demo-model",
        score=0.9,
        rank=1,
        title="Demo Model",
    )

    assert result.rank == 1
    assert result.score == 0.9
