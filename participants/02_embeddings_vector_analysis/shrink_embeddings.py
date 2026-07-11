from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from embedding_utils import (  # noqa: E402
    batched_range,
    ensure_dir,
    estimate_array_memory_mb,
    load_json,
    normalize_l2,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shrink embeddings with PCA dimensionality reduction."
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        required=True,
        help="Directory containing original embeddings.npy, ids.npy, metadata.parquet.",
    )
    parser.add_argument("--target-dim", type=int, default=128)
    parser.add_argument(
        "--fit-sample-size",
        type=int,
        default=100_000,
        help="Number of vectors sampled to fit PCA. Use all if dataset is smaller.",
    )
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <embedding-dir>/pca<TARGET_DIM>.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not L2-normalize PCA-reduced embeddings. Normalization is recommended.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output embeddings.npy.",
    )
    return parser.parse_args()


def load_original(embedding_dir: Path) -> tuple[np.ndarray, Path, Path]:
    embeddings_path = embedding_dir / "embeddings.npy"
    ids_path = embedding_dir / "ids.npy"
    metadata_path = embedding_dir / "metadata.parquet"
    for path in [embeddings_path, ids_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(embeddings_path, mmap_mode="r"), ids_path, metadata_path


def sample_for_pca(embeddings: np.ndarray, sample_size: int, random_state: int) -> np.ndarray:
    n = embeddings.shape[0]
    size = min(sample_size, n)
    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.choice(n, size=size, replace=False))
    return np.asarray(embeddings[idx], dtype="float32")


def main() -> int:
    args = parse_args()
    embeddings, ids_path, metadata_path = load_original(args.embedding_dir)
    n_vectors, original_dim = embeddings.shape

    if args.target_dim <= 0 or args.target_dim >= original_dim:
        raise ValueError(
            f"target_dim must be between 1 and original_dim-1. Got {args.target_dim}, original_dim={original_dim}."
        )

    output_dir = ensure_dir(args.output_dir or args.embedding_dir / f"pca{args.target_dim}")
    output_embeddings_path = output_dir / "embeddings.npy"
    if output_embeddings_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_embeddings_path} already exists. Pass --overwrite to regenerate."
        )

    print("Original embeddings:", embeddings.shape, embeddings.dtype)
    print("Fitting PCA...")
    fit_sample = sample_for_pca(embeddings, args.fit_sample_size, args.random_state)

    pca = PCA(
        n_components=args.target_dim,
        svd_solver="randomized",
        random_state=args.random_state,
    )
    pca.fit(fit_sample)

    normalize = not args.no_normalize
    reduced = np.lib.format.open_memmap(
        output_embeddings_path,
        mode="w+",
        dtype="float32",
        shape=(n_vectors, args.target_dim),
    )

    for start, end in tqdm(
        batched_range(n_vectors, args.batch_size),
        total=(n_vectors + args.batch_size - 1) // args.batch_size,
        desc="Applying PCA",
    ):
        batch = np.asarray(embeddings[start:end], dtype="float32")
        transformed = pca.transform(batch).astype("float32", copy=False)
        if normalize:
            transformed = normalize_l2(transformed).astype("float32", copy=False)
        reduced[start:end] = transformed

    reduced.flush()

    shutil.copy2(ids_path, output_dir / "ids.npy")
    shutil.copy2(metadata_path, output_dir / "metadata.parquet")
    joblib.dump(pca, output_dir / "pca_model.joblib")

    source_config_path = args.embedding_dir / "run_config.json"
    source_config: dict[str, Any] = {}
    if source_config_path.exists():
        source_config = load_json(source_config_path)

    original_memory_mb = estimate_array_memory_mb((n_vectors, original_dim), dtype="float32")
    reduced_memory_mb = estimate_array_memory_mb((n_vectors, args.target_dim), dtype="float32")
    explained = np.cumsum(pca.explained_variance_ratio_)

    stats = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_embedding_dir": str(args.embedding_dir),
        "source_model_name": source_config.get("model_name"),
        "output_dir": str(output_dir),
        "num_vectors": int(n_vectors),
        "original_dim": int(original_dim),
        "target_dim": int(args.target_dim),
        "original_memory_mb": round(original_memory_mb, 3),
        "reduced_memory_mb": round(reduced_memory_mb, 3),
        "memory_compression_ratio": round(original_memory_mb / reduced_memory_mb, 3),
        "fit_sample_size": int(min(args.fit_sample_size, n_vectors)),
        "normalized_after_pca": bool(normalize),
        "cumulative_explained_variance": float(explained[-1]),
        "embedding_file": str(output_embeddings_path),
        "ids_file": str(output_dir / "ids.npy"),
        "metadata_file": str(output_dir / "metadata.parquet"),
        "pca_model_file": str(output_dir / "pca_model.joblib"),
    }
    save_json(stats, output_dir / "pca_stats.json")

    run_config = {
        **source_config,
        "representation_type": "pca_reduced_embeddings",
        "source_embedding_dir": str(args.embedding_dir),
        "embedding_shape": [int(n_vectors), int(args.target_dim)],
        "embedding_dtype": "float32",
        "normalized": bool(normalize),
        "pca_stats_file": str(output_dir / "pca_stats.json"),
    }
    save_json(run_config, output_dir / "run_config.json")

    print("Saved reduced embeddings to:", output_embeddings_path)
    print("Saved PCA stats to:", output_dir / "pca_stats.json")
    print(f"Memory: {original_memory_mb:.2f} MB -> {reduced_memory_mb:.2f} MB")
    print(f"Cumulative explained variance: {explained[-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
