from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from aise.evaluation import load_relevance_csv, parse_relevant_model_ids


@pytest.fixture
def qrels_csv_path() -> Iterator[Path]:
    path = Path.cwd() / f".test-relevance-{uuid4().hex}.csv"
    yield path
    path.unlink(missing_ok=True)


def write_csv(path: Path, rows: str) -> Path:
    path.write_text(
        "query,relevant_model_ids,label_source\n" + rows,
        encoding="utf-8",
    )
    return path


def test_parse_relevant_model_ids_supports_common_formats() -> None:
    assert parse_relevant_model_ids("model-a|model-b|model-a") == (
        "model-a",
        "model-b",
    )
    assert parse_relevant_model_ids('["model-a", "model-b"]') == (
        "model-a",
        "model-b",
    )
    assert parse_relevant_model_ids("model-a;model-b") == (
        "model-a",
        "model-b",
    )


def test_load_relevance_csv_builds_evaluation_examples(
    qrels_csv_path: Path,
) -> None:
    path = write_csv(
        qrels_csv_path,
        "image classification,model-a|model-b,metadata\n"
        "text generation,model-c,metadata\n",
    )

    examples = load_relevance_csv(
        path,
        top_k=20,
        available_model_ids={"model-a", "model-b", "model-c"},
    )

    assert [example.query.text for example in examples] == [
        "image classification",
        "text generation",
    ]
    assert all(example.query.top_k == 20 for example in examples)
    assert examples[0].relevant_model_ids == ("model-a", "model-b")


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("query one,,metadata\n", "no relevant model IDs"),
        (
            "query one,model-a,metadata\nQuery One,model-b,metadata\n",
            "Duplicate evaluation query",
        ),
    ],
)
def test_load_relevance_csv_rejects_invalid_rows(
    qrels_csv_path: Path,
    rows: str,
    message: str,
) -> None:
    path = write_csv(qrels_csv_path, rows)

    with pytest.raises(ValueError, match=message):
        load_relevance_csv(path)


def test_load_relevance_csv_rejects_unknown_model_ids(
    qrels_csv_path: Path,
) -> None:
    path = write_csv(
        qrels_csv_path,
        "query one,missing-model,metadata\n",
    )

    with pytest.raises(ValueError, match="Unknown relevant model IDs"):
        load_relevance_csv(path, available_model_ids={"known-model"})
