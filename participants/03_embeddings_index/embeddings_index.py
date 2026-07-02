from __future__ import annotations

from aise.contracts import SearchDocument


class EmbeddingVectorIndex:
    def build(self, documents: list[SearchDocument]) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
