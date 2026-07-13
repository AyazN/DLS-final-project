from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aise.contracts import EvaluationExample, EvaluationReport, Retriever


class RetrievalEvaluator:
    """Evaluate a Retriever with standard ranking metrics."""

    def evaluate(
        self,
        examples: Sequence[EvaluationExample],
        retriever: Retriever,
        top_k: int = 20,
        metric_k_values: Sequence[int] = (1, 5, 10, 20),
        save_report: bool = True,
        output_dir: str = "data/results",
        experiment_name: str | None = None,
    ) -> EvaluationReport:
        if not examples:
            raise ValueError("At least one evaluation example is required.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        cutoffs = tuple(sorted(set(metric_k_values)))
        if not cutoffs or any(k <= 0 for k in cutoffs):
            raise ValueError("metric_k_values must contain positive integers.")

        retrieval_k = max(top_k, max(cutoffs))
        per_query_metrics: list[dict[str, float]] = []
        per_query_details: list[dict[str, Any]] = []

        for example in examples:
            results = retriever.search(replace(example.query, top_k=retrieval_k))
            retrieved_ids = [result.model_id for result in results[:retrieval_k]]
            relevant_ids = set(example.relevant_model_ids)
            ranks = [
                position
                for position, model_id in enumerate(retrieved_ids, start=1)
                if model_id in relevant_ids
            ]

            metrics: dict[str, float] = {
                "mrr": 1.0 / ranks[0] if ranks else 0.0,
            }
            for k in cutoffs:
                relevant_at_k = sum(rank <= k for rank in ranks)
                metrics[f"precision@{k}"] = relevant_at_k / k
                metrics[f"recall@{k}"] = (
                    relevant_at_k / len(relevant_ids)
                    if relevant_ids
                    else 0.0
                )

                dcg = sum(
                    1.0 / math.log2(position + 1)
                    for position, model_id in enumerate(
                        retrieved_ids[:k],
                        start=1,
                    )
                    if model_id in relevant_ids
                )
                ideal_count = min(k, len(relevant_ids))
                idcg = sum(
                    1.0 / math.log2(position + 1)
                    for position in range(1, ideal_count + 1)
                )
                metrics[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0

            per_query_metrics.append(metrics)
            per_query_details.append(
                {
                    "query": example.query.text,
                    "relevant_model_ids": sorted(relevant_ids),
                    "retrieved_model_ids": retrieved_ids,
                    "ranks": ranks,
                    "metrics": metrics,
                }
            )

        metric_names = per_query_metrics[0]
        aggregated = {
            name: sum(metrics[name] for metrics in per_query_metrics)
            / len(per_query_metrics)
            for name in metric_names
        }
        details = {
            "per_query": per_query_details,
            "experiment_name": experiment_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "top_k": top_k,
            "metric_k_values": list(cutoffs),
        }
        report = EvaluationReport(metrics=aggregated, details=details)

        if save_report:
            self._save_report(report, output_dir, experiment_name)
        return report

    def _save_report(
        self,
        report: EvaluationReport,
        output_dir: str,
        experiment_name: str | None,
    ) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        name = experiment_name or datetime.now(timezone.utc).strftime(
            "eval_%Y%m%d_%H%M%S"
        )
        path = output_path / f"{name}.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "metrics": dict(report.metrics),
                    "details": dict(report.details),
                },
                stream,
                indent=2,
                ensure_ascii=False,
            )
        return path
