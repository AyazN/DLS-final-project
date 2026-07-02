from __future__ import annotations

from collections.abc import Iterable

from aise.contracts import ModelCard


class ModelCardLoader:
    def load(self) -> Iterable[ModelCard]:
        raise NotImplementedError
