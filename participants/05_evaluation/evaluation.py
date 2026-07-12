from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from aise.contracts import EvaluationExample, EvaluationReport, Retriever


class RetrievalEvaluator:
    """
    Evaluates a retrieval system on a set of labeled queries.

    Computes standard ranking metrics: Precision@K, Recall@K, MRR, and nDCG@K.
    Aggregates results across queries and saves a detailed report to disk.
    """

    def evaluate(
        self,
        examples: Sequence[EvaluationExample],
        retriever: Retriever,
        top_k: int = 20,
        metric_k_values: Sequence[int] = (1, 5, 10, 20),
        save_report: bool = True,
        output_dir: str = "data/results",
        experiment_name: Optional[str] = None,
    ) -> EvaluationReport:
        """
        Runs the evaluation.

        Args:
            examples: List of labeled queries with their relevant document IDs.
            retriever: A retriever implementing `retrieve(query, top_k) -> Sequence[Document]`.
            top_k: How many documents to retrieve per query for the evaluation.
            metric_k_values: Cut-off positions for which metrics are computed.
            save_report: Whether to persist the evaluation report to disk.
            output_dir: Directory where the report JSON will be saved.
            experiment_name: Optional name for the experiment; used in the filename.
                              If not provided, a timestamp is used.

        Returns:
            EvaluationReport: Aggregated metrics and per‑query details.
        """
        if not examples:
            raise ValueError("At least one evaluation example is required.")

        per_query_metrics: list[dict[str, float]] = []
        per_query_details: list[dict[str, Any]] = []

        for ex in examples:
            retrieved = retriever.retrieve(ex.query, top_k=top_k)
            retrieved_ids = [doc.id for doc in retrieved]
            relevant_set = set(ex.relevant_docs)

            # Positions (1‑based) of relevant documents in the ranked list
            ranks = []
            for pos, doc_id in enumerate(retrieved_ids, start=1):
                if doc_id in relevant_set:
                    ranks.append(pos)

            total_relevant = len(ex.relevant_docs)
            query_metrics: dict[str, float] = {}
            query_metrics["mrr"] = 1.0 / ranks[0] if ranks else 0.0

            # Pre‑compute DCG and IDCG for all k values more efficiently
            # but for simplicity we calculate inside the loop
            for k in metric_k_values:
                # Precision@K
                rel_in_top_k = sum(1 for r in ranks if r <= k)
                precision = rel_in_top_k / k
                query_metrics[f"precision@{k}"] = precision

                # Recall@K
                recall = rel_in_top_k / total_relevant if total_relevant > 0 else 0.0
                query_metrics[f"recall@{k}"] = recall

                # nDCG@K (binary relevance)
                dcg = 0.0
                for pos, doc_id in enumerate(retrieved_ids[:k], start=1):
                    if doc_id in relevant_set:
                        dcg += 1.0 / math.log2(pos + 1)

                ideal_rel = min(k, total_relevant)
                idcg = 0.0
                for pos in range(1, ideal_rel + 1):
                    idcg += 1.0 / math.log2(pos + 1)

                ndcg = dcg / idcg if idcg > 0 else 0.0
                query_metrics[f"ndcg@{k}"] = ndcg

            per_query_metrics.append(query_metrics)
            per_query_details.append(
                {
                    "query": ex.query,
                    "relevant_ids": list(relevant_set),
                    "retrieved_ids": retrieved_ids[:k],
                    "ranks": ranks,
                    **query_metrics,
                }
            )

        # Aggregate metrics across all queries (mean)
        aggregated_metrics: dict[str, float] = {}
        if per_query_metrics:
            for key in per_query_metrics[0]:
                values = [qm[key] for qm in per_query_metrics]
                aggregated_metrics[key] = sum(values) / len(values)

        # Build the report object
        report = EvaluationReport(
            metrics=aggregated_metrics,
            per_query=per_query_details,
            experiment_name=experiment_name,
            timestamp=datetime.now().isoformat(),
            top_k=top_k,
            metric_k_values=list(metric_k_values),
        )

        if save_report:
            self._save_report(report, output_dir)

        return report

    def _save_report(self, report: EvaluationReport, output_dir: str) -> None:
        """Persists the evaluation report as a JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build a sensible filename
        name = report.experiment_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filename = output_path / f"{name}.json"

        # Convert to a serializable dict
        data = {
            "metrics": report.metrics,
            "per_query": report.per_query,
            "experiment_name": report.experiment_name,
            "timestamp": report.timestamp,
            "top_k": report.top_k,
            "metric_k_values": report.metric_k_values,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
