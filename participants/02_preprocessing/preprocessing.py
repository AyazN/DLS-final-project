from __future__ import annotations

from collections.abc import Iterable

from aise.contracts import ModelCard, SearchDocument


class ModelCardDocumentBuilder:
    def build(self, cards: Iterable[ModelCard]) -> Iterable[SearchDocument]:
        raise NotImplementedError
