from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_MARKERS = ("config.py", "README.md")


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root by walking up until config.py / README.md is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return Path.cwd().resolve()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_model_dir_name(model_name: str) -> str:
    """Convert HF model name to a filesystem-friendly directory name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name.strip())


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict, np.ndarray)):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def parse_list_value(value: Any) -> list[str]:
    """Normalize tags/languages values that may arrive as lists, tuples, arrays, strings, or NaNs."""
    if _is_missing_scalar(value):
        return []

    if isinstance(value, np.ndarray):
        return [str(x).strip() for x in value.tolist() if str(x).strip()]

    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "[]"}:
            return []

        # Some parquet/csv exports can store lists as strings like "['a', 'b']".
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                return parse_list_value(parsed)
            except (SyntaxError, ValueError):
                pass

        # Fallback: split comma-separated strings.
        return [part.strip() for part in text.split(",") if part.strip()]

    return [str(value).strip()] if str(value).strip() else []


def compact_text(value: Any, max_chars: int | None = None) -> str:
    """Convert a field to clean one-line-ish text and optionally truncate it."""
    if _is_missing_scalar(value):
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def build_model_text(row: pd.Series, max_model_card_chars: int = 2500) -> str:
    """Build a searchable text representation for a Hugging Face model card row.

    The representation intentionally combines short structured metadata with the
    longer model card. This helps short user queries match task/library/tags even
    when the model card wording is noisy.
    """
    model_id = compact_text(row.get("model_id"))
    pipeline_tag = compact_text(row.get("pipeline_tag"))
    library_name = compact_text(row.get("library_name"))
    tags = parse_list_value(row.get("tags"))
    languages = parse_list_value(row.get("languages"))
    likes = compact_text(row.get("likes"))
    downloads = compact_text(row.get("downloads"))
    model_card = compact_text(row.get("modelCard"), max_chars=max_model_card_chars)

    parts = [
        f"Model ID: {model_id}" if model_id else "",
        f"Task: {pipeline_tag}" if pipeline_tag else "",
        f"Library: {library_name}" if library_name else "",
        f"Tags: {', '.join(tags[:40])}" if tags else "",
        f"Languages: {', '.join(languages[:20])}" if languages else "",
        f"Popularity: {likes} likes, {downloads} downloads" if likes or downloads else "",
        f"Model card: {model_card}" if model_card else "",
    ]
    return "\n".join(part for part in parts if part)


def normalize_l2(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize vectors row-wise."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def estimate_array_memory_mb(shape: Iterable[int], dtype: str | np.dtype = "float32") -> float:
    n_items = int(np.prod(tuple(shape)))
    return n_items * np.dtype(dtype).itemsize / 1024 / 1024


def read_dataset(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    elif path.suffix.lower() in {".jsonl", ".json"}:
        df = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    required = {"model_id", "modelCard"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if limit is not None and limit > 0:
        df = df.head(limit).copy()

    df = df.reset_index(drop=True)
    return df


def metadata_without_long_text(df: pd.DataFrame) -> pd.DataFrame:
    """Keep metadata useful for search results, but drop the long modelCard text."""
    keep = [
        "model_id",
        "likes",
        "downloads",
        "tags",
        "pipeline_tag",
        "library_name",
        "createdAt",
        "languages",
    ]
    cols = [col for col in keep if col in df.columns]
    meta = df[cols].copy()
    meta.insert(0, "embedding_row", np.arange(len(meta), dtype=np.int64))
    return meta


def batched_range(n_items: int, batch_size: int) -> Iterable[tuple[int, int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)

# ---------------------------------------------------------------------------
# Integration helpers for downstream index / retrieval modules
# ---------------------------------------------------------------------------

QUERY_PREFIX_BY_MODEL = {
    # BGE models are usually queried with an instruction prefix for retrieval.
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def get_embedding_dir(
    model_name: str,
    embeddings_root: str | Path = "data/processed/embeddings",
) -> Path:
    """Return the standard embedding output directory for a model name."""
    return Path(embeddings_root) / safe_model_dir_name(model_name)


def load_embedding_artifacts(
    embedding_dir: str | Path,
    *,
    mmap_embeddings: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load embeddings, ids, and metadata from an embedding directory.

    `ids.npy` often stores Python strings and therefore cannot be memory-mapped.
    `embeddings.npy` can be memory-mapped for large 500k/600k matrices.
    """
    embedding_dir = Path(embedding_dir)
    embeddings_path = embedding_dir / "embeddings.npy"
    ids_path = embedding_dir / "ids.npy"
    metadata_path = embedding_dir / "metadata.parquet"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing ids file: {ids_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    embeddings = np.load(embeddings_path, mmap_mode="r" if mmap_embeddings else None)
    ids = np.load(ids_path, allow_pickle=True)
    metadata = pd.read_parquet(metadata_path)

    if embeddings.shape[0] != ids.shape[0] or embeddings.shape[0] != len(metadata):
        raise ValueError(
            "Artifact row count mismatch: "
            f"embeddings={embeddings.shape[0]}, ids={ids.shape[0]}, metadata={len(metadata)}"
        )

    return embeddings, ids, metadata


def format_query_for_encoder(query: str, model_name: str) -> str:
    """Apply encoder-specific query formatting.

    MiniLM does not need a prefix. BGE retrieval models benefit from an instruction
    prefix on the query side. Document embeddings are still generated without this prefix.
    """
    query = compact_text(query)
    prefix = QUERY_PREFIX_BY_MODEL.get(model_name, "")
    return f"{prefix}{query}" if prefix else query
