"""Contract-compatible integration of the participant 05 retrieval metrics."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from aise.contracts import EvaluationExample, EvaluationReport, Retriever


class RetrievalEvaluator:
    def __init__(
        self,
        *,
        metric_k_values: Sequence[int] = (1, 5, 10, 20),
        top_k: int | None = None,
        save_report: bool = False,
        output_dir: str | Path = "data/results",
        experiment_name: str | None = None,
    ) -> None:
        if any(k <= 0 for k in metric_k_values):
            raise ValueError("metric_k_values must contain positive integers")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative")
        self.metric_k_values = tuple(int(k) for k in metric_k_values)
        self.top_k = top_k
        self.save_report = save_report
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name

    def evaluate(
        self,
        examples: Sequence[EvaluationExample],
        retriever: Retriever,
    ) -> EvaluationReport:
        if not examples:
            return EvaluationReport(
                metrics={},
                details={"query_count": 0, "per_query": [], "mean_latency_ms": 0.0},
            )

        per_query_metrics: list[dict[str, float]] = []
        per_query_details: list[dict[str, Any]] = []
        latencies_ms: list[float] = []

        for example in examples:
            query = (
                example.query
                if self.top_k is None
                else replace(example.query, top_k=self.top_k)
            )
            started = time.perf_counter()
            retrieved = list(retriever.search(query))
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(latency_ms)

            retrieved_ids = list(dict.fromkeys(result.model_id for result in retrieved))
            relevant_set = set(example.relevant_model_ids)
            ranks = [
                position
                for position, model_id in enumerate(retrieved_ids, start=1)
                if model_id in relevant_set
            ]
            total_relevant = len(relevant_set)
            metrics: dict[str, float] = {"mrr": 1.0 / ranks[0] if ranks else 0.0}

            for k in self.metric_k_values:
                relevant_in_top_k = sum(rank <= k for rank in ranks)
                metrics[f"precision@{k}"] = relevant_in_top_k / k
                metrics[f"recall@{k}"] = (
                    relevant_in_top_k / total_relevant if total_relevant else 0.0
                )

                dcg = sum(
                    1.0 / math.log2(position + 1)
                    for position, model_id in enumerate(retrieved_ids[:k], start=1)
                    if model_id in relevant_set
                )
                ideal_count = min(k, total_relevant)
                idcg = sum(
                    1.0 / math.log2(position + 1)
                    for position in range(1, ideal_count + 1)
                )
                metrics[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0

            per_query_metrics.append(metrics)
            per_query_details.append(
                {
                    "query": query.text,
                    "relevant_model_ids": list(example.relevant_model_ids),
                    "retrieved_model_ids": retrieved_ids,
                    "ranks": ranks,
                    "latency_ms": latency_ms,
                    "metrics": metrics,
                }
            )

        aggregated = {
            name: sum(metrics[name] for metrics in per_query_metrics) / len(per_query_metrics)
            for name in per_query_metrics[0]
        }
        aggregated["mean_latency_ms"] = sum(latencies_ms) / len(latencies_ms)
        details = {
            "query_count": len(examples),
            "per_query": per_query_details,
            "metric_k_values": list(self.metric_k_values),
            "top_k": self.top_k,
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "mean_latency_ms": aggregated["mean_latency_ms"],
        }
        report = EvaluationReport(metrics=aggregated, details=details)
        if self.save_report:
            self._save_report(report)
        return report

    def _save_report(self, report: EvaluationReport) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        name = self.experiment_name or datetime.now().strftime("eval_%Y%m%d_%H%M%S")
        path = self.output_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(
                {"metrics": dict(report.metrics), "details": dict(report.details)},
                stream,
                indent=2,
                ensure_ascii=False,
            )
        return path
