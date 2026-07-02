from __future__ import annotations

from collections.abc import Sequence

from aise.contracts import EvaluationExample, EvaluationReport, Retriever


class RetrievalEvaluator:
    def evaluate(
        self,
        examples: Sequence[EvaluationExample],
        retriever: Retriever,
    ) -> EvaluationReport:
        raise NotImplementedError
