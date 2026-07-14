from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from aise.contracts import Query, SearchResult
from .reranker import CrossEncoderReranker


def rerank_diagnostics(
    query: Query,
    hybrid_results: Sequence[SearchResult],
    reranker: CrossEncoderReranker,
    qrels: Mapping[str, set[str]] | None = None,
    *,
    preview_chars: int = 120,
) -> pd.DataFrame:
    """Row-per-candidate view of what changed during cross-encoder reranking.

    Notebook-compatible: pass a single query's hybrid_results and the same
    reranker instance used in the pipeline. `qrels` is optional
    {query_text: {relevant_model_id, ...}}.
    """
    reranked = list(reranker.rank_all(query, hybrid_results))
    hybrid_rank_by_id = {r.doc_id: (r.rank, r.score) for r in hybrid_results}
    relevant = (qrels or {}).get(query.text, set())

    rows = []
    for result in reranked:
        hybrid_rank, rrf_score = hybrid_rank_by_id.get(result.doc_id, (None, None))
        candidate = next(c for c in hybrid_results if c.doc_id == result.doc_id)
        rows.append(
            {
                "query": query.text,
                "model_id": result.model_id,
                "pipeline_tag": candidate.metadata.get("pipeline_tag", ""),
                "hybrid_rank": hybrid_rank,
                "reranked_rank": result.rank,
                "rank_change": (hybrid_rank - result.rank) if hybrid_rank else None,
                "rrf_score": rrf_score,
                "cross_encoder_score": result.score,
                "qrels_relevant": result.model_id in relevant,
                "reranker_text_preview": reranker._document_text(candidate)[:preview_chars],
            }
        )
    return pd.DataFrame(rows)
