from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from .embeddings_index import EmbeddingVectorIndex
except ImportError:
    from embeddings_index import EmbeddingVectorIndex


def _faiss() -> Any:
    if faiss is None:
        raise ImportError(
            "FAISS is not installed. Install faiss-cpu on a supported platform."
        )
    return faiss


def _matrix(values: np.ndarray, dim: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dim:
        raise ValueError(f"Expected matrix with shape (n, {dim}); got {matrix.shape}.")
    return np.ascontiguousarray(matrix)


class FaissFlatIndex(EmbeddingVectorIndex):
    def __init__(self, dim: int):
        self.dim = dim
        self.index = None

    def build(self, vectors: np.ndarray) -> None:
        module = _faiss()
        matrix = _matrix(vectors, self.dim)
        self.index = module.IndexFlatL2(self.dim)
        self.index.add(matrix)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index has not been built or loaded.")
        distances, indices = self.index.search(_matrix(query, self.dim), k)
        return -distances, indices

    def save(self, path: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Index has not been built.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _faiss().write_index(self.index, str(path))

    def load(self, path: str | Path) -> None:
        self.index = _faiss().read_index(str(path))
        self.dim = self.index.d


class FaissHNSWIndex(EmbeddingVectorIndex):
    def __init__(self, dim: int, M: int = 32):
        self.dim = dim
        self.M = M
        self.index = None

    def build(self, vectors: np.ndarray) -> None:
        module = _faiss()
        matrix = _matrix(vectors, self.dim)
        self.index = module.IndexHNSWFlat(self.dim, self.M)
        self.index.add(matrix)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index has not been built or loaded.")
        distances, indices = self.index.search(_matrix(query, self.dim), k)
        return -distances, indices

    def save(self, path: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Index has not been built.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _faiss().write_index(self.index, str(path))

    def load(self, path: str | Path) -> None:
        self.index = _faiss().read_index(str(path))
        self.dim = self.index.d


class FaissIVFPQIndex(EmbeddingVectorIndex):
    def __init__(
        self,
        dim: int,
        nprobe: int = 10,
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8,
    ):
        self.dim = dim
        self.nprobe = nprobe
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.index = None

    def build(self, vectors: np.ndarray) -> None:
        module = _faiss()
        matrix = _matrix(vectors, self.dim)
        quantizer = module.IndexFlatL2(self.dim)
        self.index = module.IndexIVFPQ(
            quantizer,
            self.dim,
            self.nlist,
            self.m,
            self.nbits,
        )
        self.index.train(matrix)
        self.index.add(matrix)
        self.index.nprobe = self.nprobe

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index has not been built or loaded.")
        self.index.nprobe = self.nprobe
        distances, indices = self.index.search(_matrix(query, self.dim), k)
        return -distances, indices

    def save(self, path: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Index has not been built.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _faiss().write_index(self.index, str(path))

    def load(self, path: str | Path) -> None:
        self.index = _faiss().read_index(str(path))
        self.dim = self.index.d
        self.nlist = self.index.nlist
        self.m = self.index.pq.M
        self.index.nprobe = self.nprobe
