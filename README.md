# AISE — AI Model Search Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AyazN/DLS-final-project/blob/develop/notebooks/full_search_pipeline_colab.ipynb)

AISE is an educational search engine for discovering relevant AI models from model-card metadata. The project was developed for the **Deep Learning for Search** course and integrates the original team submissions into one contract-compatible retrieval pipeline.

The primary demonstration entry point is [`notebooks/full_search_pipeline_colab.ipynb`](notebooks/full_search_pipeline_colab.ipynb).

## Integrated pipeline

```text
processed model metadata
        ↓
precomputed normalized embeddings
        ↓
FAISS vector index ──────────────┐
                                ├─→ Reciprocal Rank Fusion
BM25 title + body index ─────────┘              ↓
                                      cross-encoder reranking
                                                 ↓
                              Precision / Recall / MRR / nDCG / latency
```

The integration keeps [`src/aise/contracts.py`](src/aise/contracts.py) as the shared source of truth. Production imports live under `src/aise`; the numbered `participants/*` directories remain intact as the original team submissions.

## What is implemented

### Shared contracts

- `SearchDocument`: `doc_id`, `title`, `body`, `model_id`, `tags`, and `metadata`.
- `Query`: `text`, `top_k`, and optional `filters`.
- `SearchResult`: document/model identity, score, rank, title, snippet, and metadata.
- A common `Retriever.search(Query)` interface.
- A common `Reranker.rank(Query, candidates)` interface.
- Evaluation examples and reports shared by retrieval and evaluation modules.

### Artifact loading

`aise.search_index.load_embedding_artifacts` validates:

- aligned `embeddings.npy`, `ids.npy`, and `metadata.parquet` row counts;
- a two-dimensional `float32` embedding matrix;
- the configured embedding shape and model name;
- the expected vector dimension when supplied.

`load_search_documents` restores full model-card text from the processed dataset, verifies row alignment with `ids.npy`, and creates contract-compatible `SearchDocument` objects. A configurable body limit supports low-memory environments.

### FAISS indexes

The importable index implementations expose one validated interface:

- `FaissFlatIndex` — exact search;
- `FaissHNSWIndex` — approximate graph search;
- `FaissIVFPQIndex` — trained approximate compressed search.

All implementations validate dimensions and `float32` input, return `float32` values plus integer positions, and support save/load. Inner product returns similarity values where higher is better; L2 returns squared distances where lower is better.

The main notebook uses an inner-product flat index because the supplied document embeddings are already L2-normalized, making inner product equivalent to cosine similarity.

### Retrieval and ranking

- `BM25Retriever` indexes document title and body with the existing tokenizer.
- `DenseRetriever` encodes queries with the same encoder used for document embeddings and preserves FAISS row-to-document alignment.
- MiniLM queries are used as written.
- BGE queries use `format_query_for_encoder` and its retrieval instruction.
- Invalid FAISS positions such as `-1` are ignored safely.
- `ReciprocalRankFusion` combines ranks by `doc_id`; raw BM25 and dense scores are never mixed directly.
- `HybridRetriever` calls both retrievers, removes duplicate documents through RRF, and respects `Query.top_k`.
- `CrossEncoderReranker` uses query text plus useful document text, replaces retrieval scores with cross-encoder scores, preserves identity/metadata, and regenerates ranks.
- `SearchPipeline` connects hybrid retrieval and optional reranking through the shared contracts.

### Evaluation

`aise.evaluation.RetrievalEvaluator` evaluates any compatible retriever with:

- Precision@K;
- Recall@K;
- MRR;
- nDCG@K;
- mean query latency.

The repository includes [`data/evaluation/relevance.csv`](data/evaluation/relevance.csv), generated independently from `data/processed/clean_dataset.parquet`:

- 10 natural-language task queries;
- 830 unique relevant model IDs;
- 35–191 relevant models per query;
- relevance defined by an exact normalized `pipeline_tag` match.

These are reproducible **metadata-derived silver labels**, not human expert judgments. Retrieval outputs were not used to create the labels. `aise.evaluation.load_relevance_csv` validates required columns, non-empty relevance sets, duplicate queries, ID formats, and optionally whether every relevant ID exists in the searchable collection.

The notebook chooses evaluation labels in this order:

1. an optional manually reviewed `evaluation/relevance.csv` on Google Drive;
2. the bundled repository `data/evaluation/relevance.csv`;
3. a smaller runtime metadata fallback if neither CSV exists.

## Supplied artifact assumptions

The preprocessing and embedding stages are already completed. The expected directories are:

```text
data/processed/embeddings/
├── sentence-transformers__all-MiniLM-L6-v2/
└── BAAI__bge-small-en-v1.5/
```

Each encoder directory contains:

```text
embeddings.npy
ids.npy
metadata.parquet
run_config.json
progress.json
text_representation_samples.json
```

The integrated demo expects:

- shape `(600000, 384)`;
- dtype `float32`;
- L2-normalized vectors;
- aligned rows across embeddings, IDs, and metadata;
- FAISS position `i` mapping to `ids[i]` and `metadata.iloc[i]`.

The large processed data, embeddings, and generated indexes are intentionally not stored in Git.

## Colab demonstration

Open the notebook using the badge at the top of this README. It checks out the `develop` branch and adds `src/` to `sys.path` before importing AISE.

Place the source artifacts in Google Drive under one of the paths documented in the notebook configuration cell. The default layout is:

```text
MyDrive/Inno/DLS/AISE/
├── embeddings/
│   └── sentence-transformers__all-MiniLM-L6-v2/
├── data/
│   └── processed/
│       └── clean_dataset.parquet
├── indices/
├── models/
└── results/
```

The notebook then:

1. mounts Google Drive and updates the repository;
2. installs project dependencies;
3. copies or extracts the embedding artifacts to local Colab storage;
4. validates all artifacts and row alignment;
5. loads or builds a cached BM25 retriever;
6. loads or builds and saves the FAISS index without regenerating embeddings;
7. runs BM25, dense, and hybrid RRF retrieval;
8. reranks hybrid candidates with a cross-encoder;
9. compares rankings and overlap between systems;
10. evaluates all four retrieval configurations and saves the outputs.

### Colab memory mode

The default notebook configuration is intended for a free Colab runtime with approximately 12.7 GB of system RAM:

```python
LOW_MEMORY_MODE = True
BODY_MAX_CHARS_OVERRIDE = 300
```

The 300-character limit affects only the text retained for BM25 and reranking; it does not reduce the number of indexed documents and does not modify dense embeddings. Increase it only when more system RAM is available.

GPU is not required for BM25 or FAISS index construction. It is useful for loading the sentence encoder, encoding queries, and cross-encoder reranking. System RAM, rather than GPU RAM, is the main constraint during the full 600,000-document BM25 stage.

### Cached and saved outputs

The notebook persists expensive artifacts and presentation outputs to Drive:

- compressed BM25 cache;
- FAISS index;
- Hugging Face model cache;
- BM25, dense, hybrid, and reranked result CSV files;
- rank-change and retrieval-overlap comparison tables;
- evaluation query summary;
- complete metrics in CSV and JSON;
- final run summary written after the pipeline completes.

On later runs, the notebook reuses compatible caches rather than rebuilding them.

## Local installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Minimal Python integration

```python
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from aise.contracts import Query
from aise.evaluation import RetrievalEvaluator, load_relevance_csv
from aise.pipeline import SearchPipeline
from aise.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    ReciprocalRankFusion,
)
from aise.search_index import (
    FaissFlatIndex,
    load_embedding_artifacts,
    load_search_documents,
)

root = Path(".")
embedding_dir = (
    root
    / "data/processed/embeddings/sentence-transformers__all-MiniLM-L6-v2"
)
artifacts = load_embedding_artifacts(
    embedding_dir,
    mmap_embeddings=True,
    expected_dim=384,
)
documents = load_search_documents(
    artifacts,
    processed_metadata_path=root / "data/processed/clean_dataset.parquet",
    max_body_chars=300,
    include_metadata=False,
)

index_path = root / "data/indexes/minilm_flat_ip.faiss"
index_path.parent.mkdir(parents=True, exist_ok=True)
index = FaissFlatIndex(dim=384, metric="inner_product")
if index_path.exists():
    index.load(index_path)
else:
    index.build(artifacts.embeddings)
    index.save(index_path)

encoder = SentenceTransformer(artifacts.model_name)
bm25 = BM25Retriever(documents)
dense = DenseRetriever(
    index=index,
    documents=documents,
    model=encoder,
    encoder_name=artifacts.model_name,
    ids=artifacts.ids,
    normalize_embeddings=artifacts.normalized,
)
hybrid = HybridRetriever(
    bm25=bm25,
    dense=dense,
    fusion=ReciprocalRankFusion(k=60),
)
reranker = CrossEncoderReranker(
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2"),
    top_k=20,
)
pipeline = SearchPipeline(retriever=hybrid, ranker=reranker)

results = pipeline.search(
    Query("small multilingual text classification model", top_k=100)
)
for result in results[:5]:
    print(result.rank, result.model_id, result.score)

examples = load_relevance_csv(
    root / "data/evaluation/relevance.csv",
    top_k=100,
)
report = RetrievalEvaluator(
    metric_k_values=(1, 5, 10, 20),
    top_k=100,
).evaluate(examples, pipeline)
print(report.metrics)
```

For the full 600,000-document dataset, prefer the notebook because it explicitly releases memory between stages and caches BM25 and FAISS artifacts.

## Tests

Run all fast synthetic tests without loading the real 600,000-vector artifacts:

```bash
pytest -q
```

The tests cover:

- shared contracts;
- BM25 result structure and ranking;
- dense FAISS-position mapping and invalid positions;
- RRF duplicate removal and final rank generation;
- hybrid invocation of both retrievers;
- reranker identity preservation and rank changes;
- evaluator compatibility and empty inputs;
- FAISS validation and save/load when FAISS is installed;
- relevance CSV parsing and validation.

## Project structure

```text
AISE/
├── data/
│   ├── evaluation/relevance.csv
│   ├── processed/                 # local/Drive artifacts, ignored by Git
│   ├── indexes/                   # generated indexes, ignored by Git
│   └── results/                   # generated reports, ignored by Git
├── notebooks/
│   └── full_search_pipeline_colab.ipynb
├── participants/
│   ├── 01_preprocess/
│   ├── 02_embeddings_vector_analysis/
│   ├── 03_search_index/
│   ├── 04_retrieval_ranking/
│   └── 05_evaluation/
├── src/aise/
│   ├── contracts.py
│   ├── pipeline.py
│   ├── evaluation/
│   ├── retrieval/
│   └── search_index/
├── tests/
├── requirements.txt
└── README.md
```

## Team

| Participant | Area | Original submission |
| --- | --- | --- |
| Ayaz | Data ingestion and preprocessing | `participants/01_preprocess/` |
| Damir | Embeddings and vector analysis | `participants/02_embeddings_vector_analysis/` |
| Magomedgadzhi | Vector search and indexing | `participants/03_search_index/` |
| Malik | Retrieval and ranking | `participants/04_retrieval_ranking/` |
| Denis | Evaluation and metrics | `participants/05_evaluation/` |

## Current limitations

- The bundled relevance judgments are metadata-derived silver labels. A manually reviewed benchmark remains the preferred next evaluation improvement.
- The complete run depends on large artifacts that are not committed to Git.
- Free Colab RAM can be tight during BM25 construction; keep low-memory mode enabled or use a machine with more system RAM.
- The current command-line module is still a scaffold with an empty retriever. The notebook and importable `src/aise` modules are the supported integrated demonstration path.
- MiniLM is the primary verified notebook configuration. BGE query formatting is supported in code, but switching encoders also requires selecting the matching artifact directory and index cache.

## Git workflow

Development uses `feature/*` branches with pull requests into `develop`; `main` is reserved for stable releases. Shared interface changes belong in `src/aise/contracts.py` and should remain compatible with the integrated production modules.
