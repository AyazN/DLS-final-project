# 02 — Embeddings + Vector Analysis

This module is responsible for the representation stage of the AISE search pipeline.
It takes the cleaned Hugging Face model-card dataset, builds searchable text representations,
generates dense embeddings, analyzes the vector space, and provides a PCA-based shrink step.

Pipeline position:

```text
clean_dataset.parquet
        ↓
text representation = model_id + task + library + tags + languages + model card
        ↓
sentence-transformer encoder
        ↓
embeddings.npy + ids.npy + metadata.parquet
        ↓
FAISS / retrieval / evaluation
```

## Why this module exists

Project requirements ask us to select a representation model, generate vectors, analyze generated
vectors, and use this analysis to shrink the representation. This folder covers exactly that part:

- dense representation model selection;
- embedding generation for model cards;
- comparison of two encoders: MiniLM and BGE-small;
- vector statistics and PCA/cluster analysis;
- PCA dimensionality reduction as a cheap representation shrink optimization.

## Files

```text
embedding_utils.py          shared helper functions and loading helpers
generate_embeddings.py      generate dense embeddings and metadata artifacts
analyze_embeddings.py       analyze vector norms, cosine distribution, PCA, clusters
shrink_embeddings.py        reduce embedding dimensionality with PCA
run_embedding_experiment.py convenience script for two-encoder experiments
README.md                   this file
```

## Expected input

Default input:

```text
data/processed/clean_dataset.parquet
```

Expected columns:

```text
model_id, likes, downloads, tags, pipeline_tag, library_name, createdAt, languages, modelCard
```

## Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

If your environment cannot read parquet or cannot save plots, add:

```bash
pip install pyarrow matplotlib joblib sentence-transformers scikit-learn
```

## Recommended smoke test: two encoders on 5k rows

This is the safest way to test the whole module without freezing a laptop:

```bash
python participants/02_embeddings_vector_analysis/run_embedding_experiment.py \
  --input data/processed/clean_dataset.parquet \
  --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 \
  --limit 5000 \
  --batch-size 64 \
  --analyze \
  --shrink-baseline \
  --overwrite
```

This generates two embedding folders:

```text
data/processed/embeddings/sentence-transformers__all-MiniLM-L6-v2/
data/processed/embeddings/BAAI__bge-small-en-v1.5/
```

## Larger experiment on 50k rows

```bash
python participants/02_embeddings_vector_analysis/run_embedding_experiment.py \
  --input data/processed/clean_dataset.parquet \
  --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 \
  --limit 50000 \
  --batch-size 64 \
  --analysis-sample-size 10000 \
  --fit-sample-size 10000 \
  --analyze \
  --shrink-baseline \
  --overwrite
```

## Full dataset generation with resume support

For 500k/600k rows, use `--resume` if a previous run was interrupted.
The script writes `ids.npy`, `metadata.parquet`, and `progress.json` early, so downstream row mapping stays aligned.

Fresh full MiniLM run:

```bash
python participants/02_embeddings_vector_analysis/generate_embeddings.py \
  --input data/processed/clean_dataset.parquet \
  --models sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 64 \
  --max-model-card-chars 2500 \
  --overwrite
```

Resume interrupted MiniLM run:

```bash
python participants/02_embeddings_vector_analysis/generate_embeddings.py \
  --input data/processed/clean_dataset.parquet \
  --models sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 64 \
  --max-model-card-chars 2500 \
  --resume
```

For CPU-only machines, start from `--limit 5000` or `--limit 50000`. The full dataset can take a long time.

## Generated files per model

```text
embeddings.npy                     float32 matrix, shape = (n_docs, dim)
ids.npy                            model_id for every embedding row
metadata.parquet                   lightweight metadata without long modelCard text
run_config.json                    run settings and embedding shape
progress.json                      completed rows for resume/debugging
text_representation_samples.json   examples of the exact text sent to the encoder
```

Embeddings are L2-normalized by default. This is useful because FAISS `IndexFlatL2` over normalized
vectors gives the same ranking as cosine similarity.

## Analyze embeddings

Example:

```bash
python participants/02_embeddings_vector_analysis/analyze_embeddings.py \
  --embedding-dir data/processed/embeddings/sentence-transformers__all-MiniLM-L6-v2 \
  --sample-size 10000 \
  --n-clusters 12
```

Output:

```text
analysis/analysis_summary.json
analysis/pca_2d.csv
analysis/pca_explained_variance.csv
analysis/cluster_assignments_sample.csv
analysis/cluster_summary.csv
analysis/pca_2d_by_pipeline_tag.png
analysis/pca_explained_variance.png
```

What to discuss in the presentation:

- vector norms show whether embeddings are normalized;
- random-pair cosine distribution shows anisotropy / concentration;
- PCA projection shows whether similar model tasks cluster together;
- cluster summary shows top pipeline tags and libraries per embedding cluster.

## Shrink embeddings with PCA

Example: reduce 384 dimensions to 128 dimensions.

```bash
python participants/02_embeddings_vector_analysis/shrink_embeddings.py \
  --embedding-dir data/processed/embeddings/sentence-transformers__all-MiniLM-L6-v2 \
  --target-dim 128 \
  --fit-sample-size 100000 \
  --batch-size 50000 \
  --overwrite
```

Output folder:

```text
data/processed/embeddings/sentence-transformers__all-MiniLM-L6-v2/pca128/
```

Example memory reduction:

```text
600000 × 384 float32 ≈ 879 MB
600000 × 128 float32 ≈ 293 MB
```

So PCA-128 reduces embedding storage by about 3x. The indexing/evaluation participants can compare
whether this smaller representation keeps enough search quality while improving memory and latency.

## Notes for downstream retrieval

The index stage can load produced artifacts with the helper:

```python
from participants.02_embeddings_vector_analysis.embedding_utils import load_embedding_artifacts

embeddings, ids, metadata = load_embedding_artifacts(
    "data/processed/embeddings/sentence-transformers__all-MiniLM-L6-v2"
)
```

The row index returned by FAISS corresponds to the same row in `ids.npy` and `metadata.parquet`.

BGE query-side formatting: document embeddings are generated normally, but query embeddings for BGE should use an instruction prefix. Use:

```python
from embedding_utils import format_query_for_encoder

query_text = format_query_for_encoder("image classification model", "BAAI/bge-small-en-v1.5")
```

MiniLM does not need a query prefix.
