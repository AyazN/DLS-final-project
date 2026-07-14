from __future__ import annotations

import re


QUERY_PREFIX_BY_MODEL = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def format_query_for_encoder(query: str, model_name: str) -> str:
    """Apply the same query-side formatting used by the embeddings stage."""
    query = re.sub(r"\s+", " ", str(query)).strip()
    prefix = QUERY_PREFIX_BY_MODEL.get(model_name, "")
    return f"{prefix}{query}" if prefix else query
