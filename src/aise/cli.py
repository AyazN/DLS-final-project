from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Query, SearchDocument
from .pipeline import SearchPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "clean_dataset.parquet"


def _module(name: str):
    return import_module(name)


def load_documents(
    dataset_path: str | Path,
    limit: int | None = None,
) -> list[SearchDocument]:
    """Adapt participant 01/02 output to the shared SearchDocument contract."""
    utils = _module(
        "participants.02_embeddings_vector_analysis.embedding_utils"
    )
    dataframe = utils.read_dataset(dataset_path, limit=limit)

    documents: list[SearchDocument] = []
    for _, row in dataframe.iterrows():
        model_id = str(row["model_id"])
        metadata = {
            key: row[key]
            for key in (
                "likes",
                "downloads",
                "pipeline_tag",
                "library_name",
                "createdAt",
                "languages",
            )
            if key in dataframe.columns
        }
        documents.append(
            SearchDocument(
                doc_id=model_id,
                model_id=model_id,
                title=model_id.rsplit("/", 1)[-1],
                body=utils.build_model_text(row),
                tags=tuple(utils.parse_list_value(row.get("tags"))),
                metadata=metadata,
            )
        )
    return documents


class _QueryEncoder:
    """Apply participant 02 query formatting and optional PCA transform."""

    def __init__(
        self,
        model: Any,
        model_name: str,
        formatter: Any,
        *,
        normalize: bool,
        pca: Any = None,
        normalize_after_pca: bool = False,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.formatter = formatter
        self.normalize = normalize
        self.pca = pca
        self.normalize_after_pca = normalize_after_pca

    def encode(self, texts, convert_to_numpy: bool = True):
        formatted = [
            self.formatter(text, self.model_name)
            for text in texts
        ]
        vectors = self.model.encode(
            formatted,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.pca is not None:
            vectors = self.pca.transform(vectors).astype(
                np.float32,
                copy=False,
            )
            if self.normalize_after_pca:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / np.maximum(norms, 1e-12)
        return vectors


def _dense_retriever(
    documents: list[SearchDocument],
    embedding_dir: Path,
):
    utils = _module(
        "participants.02_embeddings_vector_analysis.embedding_utils"
    )
    embeddings, ids, _ = utils.load_embedding_artifacts(embedding_dir)
    document_by_id = {
        document.model_id: document
        for document in documents
    }
    missing_ids = [
        str(model_id)
        for model_id in ids
        if str(model_id) not in document_by_id
    ]
    if missing_ids:
        raise ValueError(
            f"Dataset is missing {len(missing_ids)} embedding IDs."
        )
    aligned_documents = [
        document_by_id[str(model_id)]
        for model_id in ids
    ]

    config_path = embedding_dir / "run_config.json"
    run_config: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as stream:
            run_config = json.load(stream)
    model_name = str(
        run_config.get("model_name")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Dense retrieval requires sentence-transformers."
        ) from exc

    pca = None
    source_normalized = bool(run_config.get("normalized", False))
    normalize_after_pca = False
    if run_config.get("representation_type") == "pca_reduced_embeddings":
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("PCA retrieval requires joblib.") from exc
        pca_path = embedding_dir / "pca_model.joblib"
        if not pca_path.exists():
            raise FileNotFoundError(f"Missing PCA model: {pca_path}")
        pca = joblib.load(pca_path)
        source_normalized = bool(
            run_config.get("source_normalized", True)
        )
        normalize_after_pca = bool(run_config.get("normalized", False))

    encoder = _QueryEncoder(
        SentenceTransformer(model_name),
        model_name,
        utils.format_query_for_encoder,
        normalize=source_normalized,
        pca=pca,
        normalize_after_pca=normalize_after_pca,
    )

    index_module = _module(
        "participants.03_search_index.faiss_indexes"
    )
    index = index_module.FaissFlatIndex(int(embeddings.shape[1]))
    index.build(np.asarray(embeddings, dtype=np.float32))

    dense_module = _module(
        "participants.04_retrieval_ranking.dense"
    )
    return dense_module.DenseRetriever(
        index,
        aligned_documents,
        encoder,
    )


def build_pipeline(
    documents: list[SearchDocument],
    *,
    mode: str = "bm25",
    embedding_dir: Path | None = None,
    reranker_model: str | None = None,
) -> SearchPipeline:
    bm25_module = _module(
        "participants.04_retrieval_ranking.bm25"
    )

    if mode == "bm25":
        retriever = bm25_module.BM25Retriever(documents)
    else:
        if embedding_dir is None:
            raise ValueError(
                f"--embedding-dir is required for {mode} search."
            )
        dense = _dense_retriever(documents, embedding_dir)
        if mode == "dense":
            retriever = dense
        elif mode == "hybrid":
            hybrid_module = _module(
                "participants.04_retrieval_ranking.hybrid"
            )
            rrf_module = _module(
                "participants.04_retrieval_ranking.rrf"
            )
            retriever = hybrid_module.HybridRetriever(
                bm25_module.BM25Retriever(documents),
                dense,
                rrf_module.ReciprocalRankFusion(),
            )
        else:
            raise ValueError(f"Unknown search mode: {mode}")

    reranker = None
    if reranker_model:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "Reranking requires sentence-transformers."
            ) from exc
        reranker_module = _module(
            "participants.04_retrieval_ranking.reranker"
        )
        reranker = reranker_module.CrossEncoderReranker(
            CrossEncoder(reranker_model)
        )

    return SearchPipeline(retriever=retriever, reranker=reranker)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aise",
        description="AI model-card search CLI",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    search_parser = subparsers.add_parser(
        "search",
        help="Search relevant AI model cards",
    )
    search_parser.add_argument("query")
    search_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )
    search_parser.add_argument("--limit", type=int, default=None)
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument(
        "--mode",
        choices=("bm25", "dense", "hybrid"),
        default="bm25",
    )
    search_parser.add_argument("--embedding-dir", type=Path)
    search_parser.add_argument("--reranker-model")
    return parser


def run_search(
    query_text: str,
    top_k: int,
    *,
    dataset_path: Path = DEFAULT_DATASET,
    limit: int | None = None,
    mode: str = "bm25",
    embedding_dir: Path | None = None,
    reranker_model: str | None = None,
) -> int:
    documents = load_documents(dataset_path, limit=limit)
    if not documents:
        raise ValueError("Dataset contains no searchable documents.")
    pipeline = build_pipeline(
        documents,
        mode=mode,
        embedding_dir=embedding_dir,
        reranker_model=reranker_model,
    )
    results = pipeline.search(
        Query(text=query_text, top_k=top_k)
    )

    if not results:
        print("No matching models found.")
        return 0
    for result in results:
        print(
            f"{result.rank}. {result.title} "
            f"[{result.score:.4f}]"
        )
        print(f"   model_id={result.model_id}")
        if result.snippet:
            print(f"   {result.snippet}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "search":
            return run_search(
                args.query,
                args.top_k,
                dataset_path=args.dataset,
                limit=args.limit,
                mode=args.mode,
                embedding_dir=args.embedding_dir,
                reranker_model=args.reranker_model,
            )
    except (
        FileNotFoundError,
        ImportError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
