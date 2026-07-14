"""Load and validate retrieval relevance judgments (qrels)."""

from __future__ import annotations

import ast
import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from aise.contracts import EvaluationExample, Query


def parse_relevant_model_ids(value: str | Sequence[object]) -> tuple[str, ...]:
    """Parse model IDs from a JSON-like list or a pipe/semicolon-separated value."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_values: Iterable[object] = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "[]"}:
            return ()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(text)
                except (SyntaxError, ValueError) as error:
                    raise ValueError("Invalid relevant_model_ids list") from error
            if not isinstance(parsed, (list, tuple, set)):
                raise ValueError("relevant_model_ids must contain a list of IDs")
            raw_values = parsed
        else:
            separator = "|" if "|" in text else ";"
            raw_values = text.split(separator)

    deduplicated: list[str] = []
    seen: set[str] = set()
    for value_item in raw_values:
        model_id = str(value_item).strip()
        if model_id and model_id not in seen:
            seen.add(model_id)
            deduplicated.append(model_id)
    return tuple(deduplicated)


def load_relevance_csv(
    path: str | Path,
    *,
    top_k: int = 100,
    available_model_ids: Iterable[str] | None = None,
) -> list[EvaluationExample]:
    """Load qrels and reject malformed, duplicate, or unknown relevance entries."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    csv_path = Path(path)
    available_ids = (
        {str(model_id) for model_id in available_model_ids}
        if available_model_ids is not None
        else None
    )

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        required_columns = {"query", "relevant_model_ids"}
        missing_columns = required_columns - fieldnames
        if missing_columns:
            raise ValueError(
                f"Missing relevance columns: {sorted(missing_columns)}"
            )

        examples: list[EvaluationExample] = []
        seen_queries: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            query_text = str(row.get("query") or "").strip()
            if not query_text:
                raise ValueError(f"Empty query at CSV row {row_number}")
            normalized_query = query_text.casefold()
            if normalized_query in seen_queries:
                raise ValueError(
                    f"Duplicate evaluation query at CSV row {row_number}: {query_text}"
                )

            relevant_ids = parse_relevant_model_ids(
                str(row.get("relevant_model_ids") or "")
            )
            if not relevant_ids:
                raise ValueError(
                    f"Query at CSV row {row_number} has no relevant model IDs"
                )
            if available_ids is not None:
                unknown_ids = [
                    model_id for model_id in relevant_ids if model_id not in available_ids
                ]
                if unknown_ids:
                    preview = ", ".join(unknown_ids[:5])
                    raise ValueError(
                        f"Unknown relevant model IDs at CSV row {row_number}: {preview}"
                    )

            seen_queries.add(normalized_query)
            examples.append(
                EvaluationExample(
                    query=Query(query_text, top_k=top_k),
                    relevant_model_ids=relevant_ids,
                )
            )

    if not examples:
        raise ValueError("The relevance file contains no evaluation queries")
    return examples
