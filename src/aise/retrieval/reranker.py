from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aise.contracts import Query, SearchResult

TEXT_STRATEGIES = ("title_body", "title_task_body", "title_task_tags_body", "body_only")

_MAX_TEXT_CHARS = 500  # keeps well within a MiniLM cross-encoder's token budget


def _clean_phrase(value: str) -> str:
    return value.replace("-", " ").replace("_", " ").strip()


def _natural_task_sentence(pipeline_tag: str) -> str:
    tag = _clean_phrase(str(pipeline_tag or ""))
    return f"This model is designed for {tag}." if tag else ""


def _natural_tags_sentence(tags: Any) -> str:
    if isinstance(tags, str):
        tags = [tags]
    cleaned = [_clean_phrase(str(tag)) for tag in (tags or []) if str(tag).strip()]
    cleaned = [tag for tag in cleaned if tag][:5]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return f"Relevant tags include {cleaned[0]}."
    return "Relevant tags include " + ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}."


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    # only snap back to the last word boundary if that doesn't throw away
    # most of the text (e.g. a body with few/no spaces near the cutoff)
    if last_space > max_chars * 0.5:
        truncated = truncated[:last_space]
    return truncated.strip()


class CrossEncoderReranker:
    def __init__(
        self,
        model: Any,
        *,
        top_k: int | None = None,
        document_text_key: str = "body",
        text_strategy: str = "title_task_body",
        max_text_chars: int = _MAX_TEXT_CHARS,
    ) -> None:
        if text_strategy not in TEXT_STRATEGIES:
            raise ValueError(f"Unknown text_strategy: {text_strategy!r}, expected one of {TEXT_STRATEGIES}")
        if max_text_chars <= 0:
            raise ValueError("max_text_chars must be positive")
        self.model = model
        self.top_k = top_k
        self.document_text_key = document_text_key
        self.text_strategy = text_strategy
        self.max_text_chars = max_text_chars

    def _document_text(self, candidate: SearchResult) -> str:
        """Build deterministic, natural-language reranking text.

        Deliberately avoids raw labeled fields (e.g. "Model ID:", "Tags:")
        since those are out-of-distribution for an MS MARCO-trained model.
        Text strategies keep task/tag inclusion explicit and testable.
        """
        metadata = candidate.metadata
        body = str(metadata.get(self.document_text_key, "")).strip() or candidate.snippet.strip()
        title = candidate.title.strip()

        if self.text_strategy == "body_only":
            text = body or title
        else:
            sentences = [f"{title}." if title else ""]
            if self.text_strategy in ("title_task_body", "title_task_tags_body"):
                sentences.append(_natural_task_sentence(metadata.get("pipeline_tag") or metadata.get("task") or ""))
            if self.text_strategy == "title_task_tags_body":
                sentences.append(_natural_tags_sentence(metadata.get("tags")))
            sentences.append(body)
            text = " ".join(part for part in sentences if part)

        text = text.strip() or title or candidate.model_id
        return _truncate(text, self.max_text_chars)

    def rank_all(self, query: Query, candidates: Sequence[SearchResult]) -> Sequence[SearchResult]:
        """Score and sort every candidate, without truncating to top_k."""
        if not candidates:
            return []

        pairs = [(query.text, self._document_text(candidate)) for candidate in candidates]
        scores = np.asarray(self.model.predict(pairs), dtype=np.float64).reshape(-1)
        if len(scores) != len(candidates):
            raise ValueError("Cross-encoder returned a different number of scores than candidates")

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            SearchResult(
                doc_id=candidate.doc_id,
                model_id=candidate.model_id,
                score=float(score),
                rank=rank,
                title=candidate.title,
                snippet=candidate.snippet,
                metadata=candidate.metadata,
            )
            for rank, (candidate, score) in enumerate(ranked, start=1)
        ]

    def rank(
        self,
        query: Query,
        candidates: Sequence[SearchResult],
    ) -> Sequence[SearchResult]:
        if not candidates:
            return []
        ranked = self.rank_all(query, candidates)
        limit = query.top_k if self.top_k is None else self.top_k
        limit = max(0, int(limit))
        return [
            SearchResult(
                doc_id=result.doc_id,
                model_id=result.model_id,
                score=result.score,
                rank=new_rank,
                title=result.title,
                snippet=result.snippet,
                metadata=result.metadata,
            )
            for new_rank, result in enumerate(ranked[:limit], start=1)
        ]


@dataclass
class RankFusionReranker:
    """Combines the original Hybrid ranking with Cross-Encoder ranking via RRF.

    final_score = 1 / (k + hybrid_rank) + cross_encoder_weight / (k + ce_rank)

    cross_encoder_weight=0 reproduces the Hybrid order exactly (no ties, since
    hybrid ranks are unique). Candidates are always fully scored by the
    Cross-Encoder before any truncation.
    """

    cross_encoder: CrossEncoderReranker
    cross_encoder_weight: float = 1.0
    k: int = 60
    final_k: int | None = None
    _last_ce_ranked: list[SearchResult] = field(default_factory=list, init=False, repr=False)

    def rank(self, query: Query, candidates: Sequence[SearchResult]) -> Sequence[SearchResult]:
        if not candidates:
            self._last_ce_ranked = []
            return []

        hybrid_rank_by_id = {c.doc_id: c.rank for c in candidates}
        ce_ranked = self.cross_encoder.rank_all(query, candidates)
        self._last_ce_ranked = list(ce_ranked)
        ce_rank_by_id = {r.doc_id: r.rank for r in ce_ranked}
        ce_score_by_id = {r.doc_id: r.score for r in ce_ranked}

        fused = []
        for candidate in candidates:
            hybrid_rank = hybrid_rank_by_id[candidate.doc_id]
            ce_rank = ce_rank_by_id[candidate.doc_id]
            final_score = 1.0 / (self.k + hybrid_rank) + self.cross_encoder_weight / (self.k + ce_rank)
            fused.append((candidate, final_score, ce_score_by_id[candidate.doc_id]))

        fused.sort(key=lambda item: item[1], reverse=True)
        limit = query.top_k if self.final_k is None else self.final_k
        limit = max(0, int(limit))
        fused = fused[:limit]

        return [
            SearchResult(
                doc_id=candidate.doc_id,
                model_id=candidate.model_id,
                score=float(final_score),
                rank=rank,
                title=candidate.title,
                snippet=candidate.snippet,
                metadata=candidate.metadata,
            )
            for rank, (candidate, final_score, _ce_score) in enumerate(fused, start=1)
        ]
