from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from aise.contracts import SearchDocument


@dataclass(frozen=True)
class EmbeddingArtifacts:
    embeddings: np.ndarray
    ids: np.ndarray
    metadata: Any
    run_config: Mapping[str, Any]
    directory: Path

    @property
    def model_name(self) -> str:
        return str(self.run_config["model_name"])

    @property
    def normalized(self) -> bool:
        return bool(self.run_config.get("normalized", False))


def load_embedding_artifacts(
    embedding_dir: str | Path,
    *,
    mmap_embeddings: bool = True,
    expected_dim: int | None = None,
) -> EmbeddingArtifacts:
    """Load and validate aligned embeddings, ids, and parquet metadata."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas and pyarrow are required to load embedding artifacts") from exc

    directory = Path(embedding_dir)
    embeddings_path = directory / "embeddings.npy"
    ids_path = directory / "ids.npy"
    metadata_path = directory / "metadata.parquet"
    run_config_path = directory / "run_config.json"
    for path in (embeddings_path, ids_path, metadata_path, run_config_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding artifact: {path}")

    embeddings = np.load(embeddings_path, mmap_mode="r" if mmap_embeddings else None)
    ids = np.load(ids_path, allow_pickle=True)
    metadata = pd.read_parquet(metadata_path)
    with run_config_path.open("r", encoding="utf-8") as stream:
        run_config = json.load(stream)

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be two-dimensional, got {embeddings.shape}")
    if embeddings.dtype != np.float32:
        raise TypeError(f"embeddings must have dtype float32, got {embeddings.dtype}")
    if ids.ndim != 1:
        raise ValueError(f"ids must be one-dimensional, got {ids.shape}")
    if embeddings.shape[0] != ids.shape[0] or embeddings.shape[0] != len(metadata):
        raise ValueError(
            "Artifact row count mismatch: "
            f"embeddings={embeddings.shape[0]}, ids={ids.shape[0]}, metadata={len(metadata)}"
        )
    configured_shape = tuple(run_config.get("embedding_shape", ()))
    if configured_shape and configured_shape != embeddings.shape:
        raise ValueError(
            f"run_config embedding_shape {configured_shape} does not match {embeddings.shape}"
        )
    if run_config.get("embedding_dtype") not in (None, "float32"):
        raise ValueError("run_config embedding_dtype does not match float32 embeddings")
    if not run_config.get("model_name"):
        raise ValueError("run_config must contain model_name")
    if expected_dim is not None and embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding dimension {embeddings.shape[1]} does not match expected {expected_dim}"
        )

    return EmbeddingArtifacts(embeddings, ids, metadata, run_config, directory)


def _parse_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if str(item))
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return ()
    return tuple(part.strip() for part in text.strip("[]").split(",") if part.strip())


def _first_value(row: Any, columns: Sequence[str], default: str = "") -> str:
    for column in columns:
        if column in row and row[column] is not None:
            value = str(row[column]).strip()
            if value and value.lower() != "nan":
                return value
    return default


def load_search_documents(
    artifacts: EmbeddingArtifacts,
    *,
    processed_metadata_path: str | Path | None = None,
    max_body_chars: int | None = None,
    include_metadata: bool = True,
) -> list[SearchDocument]:
    """Create row-aligned search documents for lexical retrieval and reranking.

    The embedding metadata intentionally omits the long model-card text. Pass the
    processed dataset parquet to recover it; if metadata already contains a body,
    the extra parquet is unnecessary.
    """
    if max_body_chars is not None and max_body_chars <= 0:
        raise ValueError("max_body_chars must be positive when provided")

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas and pyarrow are required to load search documents") from exc

    metadata = artifacts.metadata.reset_index(drop=True)
    source = metadata
    body_columns = ("body", "modelCard", "text")
    if not any(column in source.columns for column in body_columns):
        if processed_metadata_path is None:
            raise ValueError(
                "Artifact metadata has no full document text; pass processed_metadata_path"
            )
        source = pd.read_parquet(processed_metadata_path).reset_index(drop=True)
        if len(source) != len(artifacts.ids):
            raise ValueError(
                f"Processed metadata rows ({len(source)}) do not match ids ({len(artifacts.ids)})"
            )
        if "model_id" not in source.columns:
            raise ValueError("Processed metadata must contain model_id")
        source_ids = source["model_id"].astype(str).to_numpy()
        artifact_ids = artifacts.ids.astype(str)
        if not np.array_equal(source_ids, artifact_ids):
            raise ValueError("Processed metadata model_id rows are not aligned with ids.npy")

    documents: list[SearchDocument] = []
    artifact_ids = artifacts.ids.astype(str)
    for position, doc_id in enumerate(artifact_ids):
        meta_row = metadata.iloc[position].to_dict() if include_metadata else {}
        source_row = source.iloc[position]
        model_id = _first_value(source_row, ("model_id",), default=doc_id)
        title = _first_value(source_row, ("title", "name"), default=model_id)
        body = _first_value(source_row, body_columns)
        if max_body_chars is not None:
            body = body[:max_body_chars]
        tags = _parse_tags(source_row.get("tags"))
        documents.append(
            SearchDocument(
                doc_id=doc_id,
                title=title,
                body=body,
                model_id=model_id,
                tags=tags,
                metadata=meta_row,
            )
        )
    return documents
