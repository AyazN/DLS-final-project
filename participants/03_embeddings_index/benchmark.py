import time
import numpy as np
import psutil
import os
import gc

from faiss_indexes import *
from config import *

import json
import pandas as pd
from pathlib import Path


def measure_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def time_build(index, vectors):
    start = time.time()
    index.build(vectors)
    end = time.time()
    return end - start


def time_search(index, queries, k=10, n_runs=10):
    total = 0.0

    for _ in range(n_runs):
        start = time.time()
        index.search(queries, k)
        total += (time.time() - start)

    return total / (n_runs * len(queries))


def run_benchmark():
    np.random.seed(42)

    dim = DIM
    n_vectors = N_VECTORS
    n_queries = N_QUERIES
    k = DEFAULT_TOP_K

    print("Generating data...")

    vectors = np.random.random((n_vectors, dim)).astype("float32")
    queries = np.random.random((n_queries, dim)).astype("float32")

    indices = {
        "Flat": lambda: FaissFlatIndex(dim),
        "HNSW": lambda: FaissHNSWIndex(dim, M=HNSW_M),
        "IVFPQ": lambda: FaissIVFPQIndex(dim, nprobe=IVF_NPROBE, nlist=IVF_NLIST, m=IVF_M, nbits=IVF_NBITS)
    }

    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

    results = []

    for name, index_factory in indices.items():
        print(f"\n=== {name} ===")

        gc.collect()

        mem_before = measure_memory_mb()

        index = index_factory()

        build_time = time_build(index, vectors)

        mem_after = measure_memory_mb()

        search_time = time_search(index, queries, k=k)

        # save index
        index_path = f"{INDEX_DIR}/{name.lower()}.index"
        index.save(index_path)

        results.append({
            "index": name,
            "build_time_sec": round(build_time, 4),
            "search_time_sec_per_query": round(search_time, 6),
            "memory_mb": round(mem_after - mem_before, 2),
            "n_vectors": n_vectors,
            "n_queries": n_queries,
            "dim": dim
        })

        del index
        gc.collect()

    print("\n\n===== FINAL RESULTS =====\n")

    for r in results:
        print(
            f"{r['index']:6} | "
            f"build: {r['build_time_sec']}s | "
            f"search: {r['search_time_sec_per_query']}s | "
            f"mem: {r['memory_mb']} MB"
        )

    # save results
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{RESULTS_DIR}/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/benchmark_results.csv", index=False)

if __name__ == "__main__":
    run_benchmark()