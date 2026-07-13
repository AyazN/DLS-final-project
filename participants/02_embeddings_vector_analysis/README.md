# 02 - Embeddings and Vector Analysis

This module builds searchable text representations, generates dense embeddings, analyzes the vector space, and optionally reduces embedding dimensions with PCA.

## Files

~~~text
embedding_utils.py          Shared loading and formatting helpers
generate_embeddings.py      Generates embeddings and metadata artifacts
analyze_embeddings.py       Computes vector statistics, PCA, and clusters
shrink_embeddings.py        Reduces dimensions with PCA
run_embedding_experiment.py Runs generation, analysis, and PCA as one workflow
~~~

## Input

The default input file is:

~~~text
data/processed/clean_dataset.parquet
~~~

Expected columns:

~~~text
model_id, likes, downloads, tags, pipeline_tag, library_name, createdAt, languages, modelCard
~~~

Run all commands below from the project root.

## Install Dependencies

Windows PowerShell:

~~~powershell
python -m venv .venv; ./.venv/Scripts/python.exe -m pip install -r requirements.txt
~~~

Linux or macOS:

~~~bash
python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
~~~

## Fast Smoke Test

Generate MiniLM embeddings for 100 rows:

~~~powershell
./.venv/Scripts/python.exe participants/02_embeddings_vector_analysis/generate_embeddings.py --input data/processed/clean_dataset.parquet --models sentence-transformers/all-MiniLM-L6-v2 --limit 100 --batch-size 32 --overwrite
~~~

Expected output directory:

~~~text
data/processed/embeddings/sentence-transformers__all-MiniLM-L6-v2
~~~

## Complete 5k Experiment

This single command generates embeddings for MiniLM and BGE-small, analyzes both models, and creates PCA-128 embeddings for MiniLM:

~~~powershell
./.venv/Scripts/python.exe participants/02_embeddings_vector_analysis/run_embedding_experiment.py --input data/processed/clean_dataset.parquet --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 --limit 5000 --batch-size 64 --analysis-sample-size 5000 --n-clusters 12 --target-dim 128 --fit-sample-size 5000 --analyze --shrink-baseline --overwrite
~~~

Expected model directories:

~~~text
data/processed/embeddings/sentence-transformers__all-MiniLM-L6-v2
data/processed/embeddings/BAAI__bge-small-en-v1.5
~~~

## Complete 50k Experiment

~~~powershell
./.venv/Scripts/python.exe participants/02_embeddings_vector_analysis/run_embedding_experiment.py --input data/processed/clean_dataset.parquet --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 --limit 50000 --batch-size 64 --analysis-sample-size 10000 --n-clusters 12 --target-dim 128 --fit-sample-size 10000 --analyze --shrink-baseline --overwrite
~~~

## Generate Full Embeddings

Start a fresh MiniLM run:

~~~powershell
./.venv/Scripts/python.exe participants/02_embeddings_vector_analysis/generate_embeddings.py --input data/processed/clean_dataset.parquet --models sentence-transformers/all-MiniLM-L6-v2 --batch-size 64 --max-model-card-chars 2500 --overwrite
~~~

Resume an interrupted MiniLM run:

~~~powershell
./.venv/Scripts/python.exe participants/02_embeddings_vector_analysis/generate_embeddings.py --input data/processed/clean_dataset.parquet --models sentence-transformers/all-MiniLM-L6-v2 --batch-size 64 --max-model-card-chars 2500 --resume
~~~

Do not combine --overwrite and --resume.

## Analyze Existing Embeddings

~~~powershell
./.venv/Scripts/python.exe participants/02_embeddings_vector_analysis/analyze_embeddings.py --embedding-dir data/processed/embeddings/sentence-transformers__all-MiniLM-L6-v2 --sample-size 10000 --n-clusters 12
~~~

Analysis output is written under the model directory:

~~~text
analysis/analysis_summary.json
analysis/pca_2d.csv
analysis/pca_explained_variance.csv
analysis/cluster_assignments_sample.csv
analysis/cluster_summary.csv
analysis/pca_2d_by_pipeline_tag.png
analysis/pca_explained_variance.png
~~~

## Reduce Embeddings with PCA

Reduce 384 dimensions to 128:

~~~powershell
./.venv/Scripts/python.exe participants/02_embeddings_vector_analysis/shrink_embeddings.py --embedding-dir data/processed/embeddings/sentence-transformers__all-MiniLM-L6-v2 --target-dim 128 --fit-sample-size 100000 --batch-size 50000 --overwrite
~~~

Expected output directory:

~~~text
data/processed/embeddings/sentence-transformers__all-MiniLM-L6-v2/pca128
~~~

Approximate storage:

~~~text
600000 x 384 float32 = 879 MB
600000 x 128 float32 = 293 MB
~~~

## Generated Artifacts

Each model directory contains:

~~~text
embeddings.npy
ids.npy
metadata.parquet
run_config.json
progress.json
text_representation_samples.json
~~~

The row returned by FAISS maps to the same row in ids.npy and metadata.parquet.

## Load Artifacts in Python

~~~python
from importlib import import_module

embedding_utils = import_module("participants.02_embeddings_vector_analysis.embedding_utils")
embeddings, ids, metadata = embedding_utils.load_embedding_artifacts("data/processed/embeddings/sentence-transformers__all-MiniLM-L6-v2")
~~~

## Linux or macOS Commands

The arguments are identical. Replace ./.venv/Scripts/python.exe with ./.venv/bin/python. Example:

~~~bash
./.venv/bin/python participants/02_embeddings_vector_analysis/run_embedding_experiment.py --input data/processed/clean_dataset.parquet --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 --limit 5000 --batch-size 64 --analyze --shrink-baseline --overwrite
~~~
