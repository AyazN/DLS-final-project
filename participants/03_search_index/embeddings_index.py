from __future__ import annotations


import numpy as np
from abc import ABC, abstractmethod


class EmbeddingVectorIndex(ABC):
    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass
