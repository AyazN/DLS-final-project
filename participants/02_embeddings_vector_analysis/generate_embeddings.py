from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# Make the script runnable both from project root and from this participant folder.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from config import DEFAULT_EMBEDDING_MODEL, PROCESSED_DATA_DIR
except Exception:  # pragma: no cover - fallback for standalone use
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    PROCESSED_DATA_DIR = _PROJECT_ROOT / "data" / "processed"

from embedding_utils import (  # noqa: E402
    batched_range,
    build_model_text,
    ensure_dir,
    estimate_array_memory_mb,
    load_json,
    metadata_without_long_text,
    read_dataset,
    safe_model_dir_name,
    save_json,
)


DEFAULT_SECOND_MODEL = "BAAI/bge-small-en-v1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dense embeddings for the AI model-card dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(PROCESSED_DATA_DIR) / "clean_dataset.parquet",
        help="Path to clean_dataset.parquet / csv / jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(PROCESSED_DATA_DIR) / "embeddings",
        help="Directory where model-specific embedding folders will be created.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_EMBEDDING_MODEL, DEFAULT_SECOND_MODEL],
        help="SentenceTransformer model names to run.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick experiments, e.g. --limit 5000.",
    )
    parser.add_argument(
        "--max-model-card-chars",
        type=int,
        default=2500,
        help="Truncate long model cards before encoding to keep inference cheaper.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for SentenceTransformer, e.g. cpu, cuda. Default: auto.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not L2-normalize embeddings. Normalization is recommended for cosine/L2 search.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing artifacts and regenerate from row 0.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run from progress.json. This is safer for large 500k/600k runs. "
            "Do not combine with --overwrite."
        ),
    )
    parser.add_argument(
        "--save-every-batches",
        type=int,
        default=10,
        help="Flush embeddings and progress after this many batches.",
    )
    return parser.parse_args()


def _atomic_save_json(data: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_json(data, tmp)
    tmp.replace(path)


def _write_metadata_artifacts(df, model_dir: Path, max_model_card_chars: int) -> None:
    """Write row mapping before heavy encoding so interrupted runs still have aligned metadata."""
    ids = df["model_id"].astype(str).to_numpy(dtype=object)
    np.save(model_dir / "ids.npy", ids)

    metadata = metadata_without_long_text(df)
    metadata["text_length_chars"] = [
        len(build_model_text(row, max_model_card_chars=max_model_card_chars))
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing text lengths")
    ]
    metadata.to_parquet(model_dir / "metadata.parquet", index=False)

    sample_size = min(20, len(df))
    samples = []
    for i in range(sample_size):
        row = df.iloc[i]
        samples.append(
            {
                "embedding_row": int(i),
                "model_id": str(row.get("model_id", "")),
                "text": build_model_text(row, max_model_card_chars=max_model_card_chars),
            }
        )
    save_json({"samples": samples}, model_dir / "text_representation_samples.json")


def _load_completed_until(progress_path: Path) -> int:
    if not progress_path.exists():
        return 0
    try:
        progress = load_json(progress_path)
        return int(progress.get("completed_until", 0))
    except Exception:
        return 0


def _validate_existing_shape(embeddings_path: Path, expected_shape: tuple[int, int]) -> None:
    if not embeddings_path.exists():
        return
    arr = np.load(embeddings_path, mmap_mode="r")
    if tuple(arr.shape) != expected_shape:
        raise ValueError(
            f"Existing {embeddings_path} has shape {arr.shape}, expected {expected_shape}. "
            "Use --overwrite to regenerate from scratch."
        )


def generate_for_model(
    *,
    df,
    model_name: str,
    output_root: Path,
    batch_size: int,
    max_model_card_chars: int,
    device: str | None,
    normalize: bool,
    overwrite: bool,
    resume: bool,
    save_every_batches: int,
) -> Path:
    if overwrite and resume:
        raise ValueError("Use either --overwrite or --resume, not both.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Install project dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    model_dir = ensure_dir(output_root / safe_model_dir_name(model_name))
    embeddings_path = model_dir / "embeddings.npy"
    progress_path = model_dir / "progress.json"

    print(f"\n=== Loading encoder: {model_name} ===")
    encoder = SentenceTransformer(model_name, device=device)
    dim = int(encoder.get_sentence_embedding_dimension())
    n_docs = len(df)
    expected_shape = (n_docs, dim)
    print(f"Documents: {n_docs:,}; embedding dimension: {dim}")
    print(f"Estimated embedding matrix size: {estimate_array_memory_mb(expected_shape):.2f} MB")

    if embeddings_path.exists() and not overwrite and not resume:
        raise FileExistsError(
            f"{embeddings_path} already exists. Pass --overwrite to regenerate or --resume to continue."
        )

    if overwrite:
        for file_name in [
            "embeddings.npy",
            "ids.npy",
            "metadata.parquet",
            "run_config.json",
            "progress.json",
            "text_representation_samples.json",
        ]:
            path = model_dir / file_name
            if path.exists():
                path.unlink()

    # Metadata is cheap compared with encoding and must be aligned with embeddings rows.
    # We write it before the heavy loop, so an interrupted run still has valid row mapping.
    if overwrite or not (model_dir / "ids.npy").exists() or not (model_dir / "metadata.parquet").exists():
        _write_metadata_artifacts(df, model_dir, max_model_card_chars=max_model_card_chars)

    if resume and embeddings_path.exists():
        _validate_existing_shape(embeddings_path, expected_shape)
        embeddings = np.load(embeddings_path, mmap_mode="r+")
        completed_until = min(_load_completed_until(progress_path), n_docs)
        print(f"Resuming from row: {completed_until:,}")
    else:
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype="float32",
            shape=expected_shape,
        )
        completed_until = 0

    run_start = time.perf_counter()
    batch_counter = 0

    try:
        ranges = list(batched_range(n_docs, batch_size))
        start_batch_idx = completed_until // batch_size
        for start, end in tqdm(
            ranges[start_batch_idx:],
            total=len(ranges) - start_batch_idx,
            desc=f"Encoding {safe_model_dir_name(model_name)}",
        ):
            # If completed_until is inside the current batch, skip the already completed rows.
            if end <= completed_until:
                continue
            if start < completed_until < end:
                start = completed_until

            batch_df = df.iloc[start:end]
            texts = [
                build_model_text(row, max_model_card_chars=max_model_card_chars)
                for _, row in batch_df.iterrows()
            ]
            batch_embeddings = encoder.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            ).astype("float32", copy=False)
            embeddings[start:end] = batch_embeddings

            completed_until = end
            batch_counter += 1

            if batch_counter % max(1, save_every_batches) == 0 or completed_until == n_docs:
                embeddings.flush()
                _atomic_save_json(
                    {
                        "status": "running" if completed_until < n_docs else "complete",
                        "model_name": model_name,
                        "completed_until": int(completed_until),
                        "total_rows": int(n_docs),
                        "embedding_shape": [int(n_docs), int(dim)],
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    progress_path,
                )
                gc.collect()
    finally:
        embeddings.flush()

    elapsed = time.perf_counter() - run_start
    status = "complete" if completed_until == n_docs else "interrupted"

    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input_rows": int(n_docs),
        "input_columns": list(df.columns),
        "model_name": model_name,
        "model_dir": str(model_dir),
        "embedding_file": str(embeddings_path),
        "ids_file": str(model_dir / "ids.npy"),
        "metadata_file": str(model_dir / "metadata.parquet"),
        "progress_file": str(progress_path),
        "completed_until": int(completed_until),
        "embedding_shape": [int(n_docs), int(dim)],
        "embedding_dtype": "float32",
        "normalized": bool(normalize),
        "batch_size": int(batch_size),
        "max_model_card_chars": int(max_model_card_chars),
        "elapsed_seconds_this_run": round(elapsed, 3),
        "estimated_embedding_memory_mb": round(estimate_array_memory_mb(expected_shape), 3),
    }
    save_json(run_config, model_dir / "run_config.json")
    _atomic_save_json(
        {
            "status": status,
            "model_name": model_name,
            "completed_until": int(completed_until),
            "total_rows": int(n_docs),
            "embedding_shape": [int(n_docs), int(dim)],
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        progress_path,
    )

    print(f"Saved embeddings to: {embeddings_path}")
    print(f"Saved ids to: {model_dir / 'ids.npy'}")
    print(f"Saved metadata to: {model_dir / 'metadata.parquet'}")
    print(f"Progress: {completed_until:,}/{n_docs:,} rows ({status})")
    print(f"Elapsed in this run: {elapsed:.2f}s")
    return model_dir


def main() -> int:
    args = parse_args()
    df = read_dataset(args.input, limit=args.limit)
    ensure_dir(args.output_dir)

    print("Input dataset:", args.input)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    normalize = not args.no_normalize
    for model_name in args.models:
        generate_for_model(
            df=df,
            model_name=model_name,
            output_root=args.output_dir,
            batch_size=args.batch_size,
            max_model_card_chars=args.max_model_card_chars,
            device=args.device,
            normalize=normalize,
            overwrite=args.overwrite,
            resume=args.resume,
            save_every_batches=args.save_every_batches,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
