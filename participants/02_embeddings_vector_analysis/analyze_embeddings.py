from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from embedding_utils import ensure_dir, normalize_l2, parse_list_value, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze generated embedding vectors.")
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        required=True,
        help="Directory containing embeddings.npy, ids.npy, metadata.parquet.",
    )
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-clusters", type=int, default=12)
    parser.add_argument("--n-random-pairs", type=int, default=20_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <embedding-dir>/analysis.",
    )
    return parser.parse_args()


def load_artifacts(embedding_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    embeddings_path = embedding_dir / "embeddings.npy"
    ids_path = embedding_dir / "ids.npy"
    metadata_path = embedding_dir / "metadata.parquet"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing {embeddings_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing {ids_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")

    embeddings = np.load(embeddings_path, mmap_mode="r")
    ids = np.load(ids_path, allow_pickle=True)
    metadata = pd.read_parquet(metadata_path)
    return embeddings, ids, metadata


def sample_indices(n: int, sample_size: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    size = min(sample_size, n)
    return np.sort(rng.choice(n, size=size, replace=False))


def compute_basic_stats(embeddings: np.ndarray) -> dict[str, Any]:
    # Norms require reading the whole matrix once; this is fine and useful for diagnostics.
    norms = np.linalg.norm(np.asarray(embeddings), axis=1)
    return {
        "num_vectors": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "matrix_memory_mb": round(embeddings.size * embeddings.dtype.itemsize / 1024 / 1024, 3),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "zero_norm_count": int((norms == 0).sum()),
    }


def cosine_distribution(sample: np.ndarray, n_pairs: int, random_state: int) -> dict[str, Any]:
    rng = np.random.default_rng(random_state)
    sample = normalize_l2(sample.astype("float32", copy=False))
    n = len(sample)
    if n < 2:
        return {}

    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i = i[mask]
    j = j[mask]
    sims = np.sum(sample[i] * sample[j], axis=1)

    return {
        "num_pairs": int(len(sims)),
        "mean": float(sims.mean()),
        "std": float(sims.std()),
        "min": float(sims.min()),
        "p01": float(np.quantile(sims, 0.01)),
        "p05": float(np.quantile(sims, 0.05)),
        "p50": float(np.quantile(sims, 0.50)),
        "p95": float(np.quantile(sims, 0.95)),
        "p99": float(np.quantile(sims, 0.99)),
        "max": float(sims.max()),
    }


def summarize_cluster_rows(cluster_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, group in cluster_df.groupby("cluster"):
        tags_counter: Counter[str] = Counter()
        for value in group.get("pipeline_tag", pd.Series(dtype=object)).fillna(""):
            if str(value).strip():
                tags_counter[str(value)] += 1

        library_counter: Counter[str] = Counter()
        for value in group.get("library_name", pd.Series(dtype=object)).fillna(""):
            if str(value).strip():
                library_counter[str(value)] += 1

        extra_tag_counter: Counter[str] = Counter()
        for value in group.get("tags", pd.Series(dtype=object)):
            extra_tag_counter.update(parse_list_value(value)[:20])

        examples = group["model_id"].astype(str).head(8).tolist()
        rows.append(
            {
                "cluster": int(cluster_id),
                "size": int(len(group)),
                "top_pipeline_tags": "; ".join(f"{k}:{v}" for k, v in tags_counter.most_common(8)),
                "top_libraries": "; ".join(f"{k}:{v}" for k, v in library_counter.most_common(8)),
                "top_tags": "; ".join(f"{k}:{v}" for k, v in extra_tag_counter.most_common(12)),
                "example_model_ids": "; ".join(examples),
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False)


def maybe_save_plots(output_dir: Path, pca_df: pd.DataFrame, explained_df: pd.DataFrame) -> list[str]:
    saved: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping PNG plots.")
        return saved

    # PCA scatter. Plot top 12 pipeline tags separately and group the rest as Other.
    plot_df = pca_df.copy()
    top_tags = plot_df["pipeline_tag"].fillna("unknown").value_counts().head(12).index
    plot_df["plot_tag"] = plot_df["pipeline_tag"].where(plot_df["pipeline_tag"].isin(top_tags), "Other")

    fig = plt.figure(figsize=(10, 7))
    for tag, group in plot_df.groupby("plot_tag"):
        plt.scatter(group["pc1"], group["pc2"], s=8, alpha=0.55, label=str(tag))
    plt.title("Embedding PCA projection by pipeline_tag")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.tight_layout()
    path = output_dir / "pca_2d_by_pipeline_tag.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(str(path))

    fig = plt.figure(figsize=(9, 5))
    plt.plot(explained_df["component"], explained_df["cumulative_explained_variance"])
    plt.title("PCA cumulative explained variance")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.tight_layout()
    path = output_dir / "pca_explained_variance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(str(path))

    return saved


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir or args.embedding_dir / "analysis")

    embeddings, ids, metadata = load_artifacts(args.embedding_dir)
    if len(ids) != embeddings.shape[0]:
        raise ValueError("ids.npy length does not match embeddings.npy rows")

    print("Embedding matrix:", embeddings.shape, embeddings.dtype)
    print("Metadata rows:", len(metadata))

    stats = compute_basic_stats(embeddings)
    indices = sample_indices(embeddings.shape[0], args.sample_size, args.random_state)
    sample = np.asarray(embeddings[indices], dtype="float32")
    sample_meta = metadata.iloc[indices].reset_index(drop=True).copy()

    cos_stats = cosine_distribution(sample, args.n_random_pairs, args.random_state)

    pca_components = min(50, sample.shape[1], sample.shape[0])
    pca = PCA(n_components=pca_components, random_state=args.random_state)
    pca_values = pca.fit_transform(sample)

    pca_df = pd.DataFrame(
        {
            "embedding_row": indices,
            "model_id": ids[indices].astype(str),
            "pc1": pca_values[:, 0],
            "pc2": pca_values[:, 1] if pca_values.shape[1] > 1 else 0.0,
        }
    )
    for col in ["pipeline_tag", "library_name", "likes", "downloads"]:
        if col in sample_meta.columns:
            pca_df[col] = sample_meta[col].to_numpy()
    pca_df.to_csv(output_dir / "pca_2d.csv", index=False)

    explained_df = pd.DataFrame(
        {
            "component": np.arange(1, pca_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained_df.to_csv(output_dir / "pca_explained_variance.csv", index=False)

    n_clusters = min(args.n_clusters, len(sample))
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(sample)
    cluster_df = sample_meta.copy()
    cluster_df["embedding_row"] = indices
    cluster_df["model_id"] = ids[indices].astype(str)
    cluster_df["cluster"] = cluster_labels
    cluster_df.to_csv(output_dir / "cluster_assignments_sample.csv", index=False)

    cluster_summary = summarize_cluster_rows(cluster_df)
    cluster_summary.to_csv(output_dir / "cluster_summary.csv", index=False)

    plot_paths = maybe_save_plots(output_dir, pca_df, explained_df)

    summary = {
        "embedding_dir": str(args.embedding_dir),
        "output_dir": str(output_dir),
        "basic_stats": stats,
        "sample_size": int(len(indices)),
        "random_pair_cosine_similarity": cos_stats,
        "pca": {
            "components_fitted": int(pca_components),
            "explained_variance_pc1": float(explained_df.loc[0, "explained_variance_ratio"]),
            "explained_variance_pc2": float(explained_df.loc[1, "explained_variance_ratio"])
            if pca_components > 1
            else None,
            "cumulative_explained_variance_10": float(explained_df.loc[min(9, pca_components - 1), "cumulative_explained_variance"]),
            "cumulative_explained_variance_50": float(explained_df.loc[pca_components - 1, "cumulative_explained_variance"]),
        },
        "clustering": {
            "n_clusters": int(n_clusters),
            "cluster_summary_file": str(output_dir / "cluster_summary.csv"),
        },
        "plots": plot_paths,
    }
    save_json(summary, output_dir / "analysis_summary.json")

    print("Saved analysis to:", output_dir)
    print("Key files:")
    for name in [
        "analysis_summary.json",
        "pca_2d.csv",
        "pca_explained_variance.csv",
        "cluster_summary.csv",
    ]:
        print("-", output_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
