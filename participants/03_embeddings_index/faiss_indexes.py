import faiss
import numpy as np
from embeddings_index import EmbeddingVectorIndex


class FaissFlatIndex(EmbeddingVectorIndex):
    def __init__(self, dim: int):
        self.dim = dim
        self.index = None

    def build(self, vectors: np.ndarray):
        vectors = vectors.astype(np.float32)
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(vectors)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        query = query.astype(np.float32)
        return self.index.search(query, k)

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)
        self.dim = self.index.d
    
class FaissHNSWIndex(EmbeddingVectorIndex):
    def __init__(self, dim: int, M: int = 32):
        self.dim = dim
        self.M = M
        self.index = None

    def build(self, vectors: np.ndarray):
        vectors = vectors.astype(np.float32)
        self.index = faiss.IndexHNSWFlat(self.dim, self.M)
        self.index.add(vectors)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        query = query.astype(np.float32)
        return self.index.search(query, k)

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)
        self.dim = self.index.d

class FaissIVFPQIndex(EmbeddingVectorIndex):
    def __init__(self, dim: int, nprobe: int = 10, nlist: int = 100, m: int = 8, nbits: int = 8):
        self.dim = dim
        self.nprobe = nprobe
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.index = None

    def build(self, vectors: np.ndarray):
        vectors = vectors.astype(np.float32)
        quantizer = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.m, self.nbits)
        self.index.train(vectors)
        self.index.add(vectors)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        query = query.astype(np.float32)
        self.index.nprobe = self.nprobe
        return self.index.search(query, k)

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)
        self.dim = self.index.d
        self.nlist = self.index.nlist
        self.m = self.index.pq.M
