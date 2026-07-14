from __future__ import annotations

from os import PathLike
from typing import Any

import numpy as np


SUPPORTED_METRICS = {"inner_product", "l2"}


def _require_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "FAISS is required for vector indexing. Install the faiss-cpu dependency."
        ) from exc
    return faiss


def _validate_matrix(vectors: np.ndarray, dim: int, name: str) -> np.ndarray:
    if not isinstance(vectors, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if vectors.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32, got {vectors.dtype}")
    if vectors.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {vectors.shape}")
    if vectors.shape[1] != dim:
        raise ValueError(f"{name} dimension {vectors.shape[1]} does not match index dimension {dim}")
    return np.ascontiguousarray(vectors)


def _metric_type(metric: str) -> int:
    faiss = _require_faiss()
    if metric == "inner_product":
        return faiss.METRIC_INNER_PRODUCT
    if metric == "l2":
        return faiss.METRIC_L2
    raise ValueError(f"Unsupported metric {metric!r}; expected one of {sorted(SUPPORTED_METRICS)}")


def _metric_name(metric_type: int) -> str:
    faiss = _require_faiss()
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return "inner_product"
    if metric_type == faiss.METRIC_L2:
        return "l2"
    raise ValueError(f"Unsupported FAISS metric type: {metric_type}")


class _FaissIndex:
    """Common validated FAISS interface.

    ``inner_product`` returns similarities (higher is better). ``l2`` returns
    squared L2 distances (lower is better).
    """

    def __init__(self, dim: int, metric: str = "inner_product") -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric {metric!r}")
        self.dim = dim
        self.metric = metric
        self.index: Any | None = None

    @property
    def higher_is_better(self) -> bool:
        return self.metric == "inner_product"

    @property
    def ntotal(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    def _require_built(self) -> Any:
        if self.index is None:
            raise RuntimeError("The FAISS index has not been built or loaded")
        return self.index

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        index = self._require_built()
        if k < 0:
            raise ValueError("k must be non-negative")
        query = _validate_matrix(query, self.dim, "query")
        if k == 0:
            shape = (query.shape[0], 0)
            return np.empty(shape, dtype=np.float32), np.empty(shape, dtype=np.int64)
        values, positions = index.search(query, int(k))
        return np.asarray(values, dtype=np.float32), np.asarray(positions, dtype=np.int64)

    def save(self, path: str | PathLike[str]) -> None:
        _require_faiss().write_index(self._require_built(), str(path))

    def load(self, path: str | PathLike[str]) -> None:
        index = _require_faiss().read_index(str(path))
        if int(index.d) != self.dim:
            raise ValueError(f"Loaded index dimension {index.d} does not match expected {self.dim}")
        self.index = index
        self.metric = _metric_name(int(index.metric_type))


class FaissFlatIndex(_FaissIndex):
    """Exact flat index using inner product/cosine by default, or squared L2."""

    def build(self, vectors: np.ndarray) -> None:
        vectors = _validate_matrix(vectors, self.dim, "vectors")
        faiss = _require_faiss()
        self.index = (
            faiss.IndexFlatIP(self.dim)
            if self.metric == "inner_product"
            else faiss.IndexFlatL2(self.dim)
        )
        self.index.add(vectors)


class FaissHNSWIndex(_FaissIndex):
    """Approximate HNSW index using inner product/cosine by default, or squared L2."""

    def __init__(self, dim: int, M: int = 32, metric: str = "inner_product") -> None:
        super().__init__(dim, metric)
        if M <= 0:
            raise ValueError("M must be positive")
        self.M = M

    def build(self, vectors: np.ndarray) -> None:
        vectors = _validate_matrix(vectors, self.dim, "vectors")
        faiss = _require_faiss()
        self.index = faiss.IndexHNSWFlat(self.dim, self.M, _metric_type(self.metric))
        self.index.add(vectors)


class FaissIVFPQIndex(_FaissIndex):
    """Approximate IVF-PQ index using inner product/cosine by default, or squared L2."""

    def __init__(
        self,
        dim: int,
        nprobe: int = 10,
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8,
        metric: str = "inner_product",
    ) -> None:
        super().__init__(dim, metric)
        if min(nprobe, nlist, m, nbits) <= 0:
            raise ValueError("nprobe, nlist, m, and nbits must be positive")
        if dim % m != 0:
            raise ValueError(f"dim ({dim}) must be divisible by m ({m})")
        self.nprobe = nprobe
        self.nlist = nlist
        self.m = m
        self.nbits = nbits

    def build(self, vectors: np.ndarray) -> None:
        vectors = _validate_matrix(vectors, self.dim, "vectors")
        if vectors.shape[0] == 0:
            raise ValueError("IVFPQ requires at least one training vector")
        faiss = _require_faiss()
        quantizer = (
            faiss.IndexFlatIP(self.dim)
            if self.metric == "inner_product"
            else faiss.IndexFlatL2(self.dim)
        )
        self.index = faiss.IndexIVFPQ(
            quantizer,
            self.dim,
            self.nlist,
            self.m,
            self.nbits,
            _metric_type(self.metric),
        )
        self.index.train(vectors)
        self.index.add(vectors)
        self.index.nprobe = self.nprobe

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        index = self._require_built()
        index.nprobe = self.nprobe
        return super().search(query, k)

    def load(self, path: str | PathLike[str]) -> None:
        super().load(path)
        index = self._require_built()
        self.nlist = int(index.nlist)
        self.m = int(index.pq.M)
        index.nprobe = self.nprobe
