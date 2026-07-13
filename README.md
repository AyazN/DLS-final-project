# AISE - AI Search Engine

A project for the Deep Learning for Search course. The system searches for suitable AI models in a Hugging Face model-card dataset.

## Architecture

The team implementations are located under participants:

| Role | Directory |
|---|---|
| Data ingestion / preprocessing | participants/01_preprocess |
| Embeddings / vector analysis | participants/02_embeddings_vector_analysis |
| FAISS indexes | participants/03_search_index |
| BM25, Dense, RRF, Hybrid, Reranker | participants/04_retrieval_ranking |
| Evaluation and metrics | participants/05_evaluation |

The shared src/aise layer does not duplicate participant algorithms:

- contracts.py defines shared data structures and Protocol contracts;
- pipeline.py orchestrates Retriever and Reranker calls;
- cli.py loads artifacts and assembles participant modules into a pipeline.

## Pipeline

~~~text
participants/01: clean_dataset.parquet
    -> participants/02: texts + embeddings
    -> participants/03: FAISS index
    -> participants/04: BM25 / Dense / RRF / Reranker
    -> participants/05: EvaluationReport
~~~

## Quick Start on Windows PowerShell

Run every command from the project root.

Create the virtual environment and install all dependencies:

~~~powershell
python -m venv .venv; ./.venv/Scripts/python.exe -m pip install -r requirements.txt
~~~

Download and clean the dataset:

~~~powershell
./.venv/Scripts/python.exe participants/01_preprocess/data_load.py
~~~

Generate a small embedding sample for a smoke test:

~~~powershell
./.venv/Scripts/python.exe participants/02_embeddings_vector_analysis/generate_embeddings.py --input data/processed/clean_dataset.parquet --models sentence-transformers/all-MiniLM-L6-v2 --limit 100 --batch-size 32 --overwrite
~~~

Run BM25 search:

~~~powershell
$env:PYTHONPATH="src"; ./.venv/Scripts/python.exe -m aise.cli search "AI for football" --mode bm25
~~~

Run dense search with the generated embeddings:

~~~powershell
$env:PYTHONPATH="src"; ./.venv/Scripts/python.exe -m aise.cli search "model for medicine" --mode dense --embedding-dir data/processed/embeddings/sentence-transformers__all-MiniLM-L6-v2
~~~

Run hybrid search:

~~~powershell
$env:PYTHONPATH="src"; ./.venv/Scripts/python.exe -m aise.cli search "model for medicine" --mode hybrid --embedding-dir data/processed/embeddings/sentence-transformers__all-MiniLM-L6-v2
~~~

Run all tests:

~~~powershell
$env:PYTHONPATH="src"; ./.venv/Scripts/python.exe -m unittest discover -s tests -v
~~~

## Quick Start on Linux or macOS

Create the virtual environment and install dependencies:

~~~bash
python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
~~~

Run the same scripts with ./.venv/bin/python. For example:

~~~bash
PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -v
~~~

## Embedding Experiments

All embedding commands are documented as copy-ready one-line commands in participants/02_embeddings_vector_analysis/README.md.

## Integration Notes

The CLI loads implementations directly from participants through importlib. Numeric participant directory names cannot be referenced with regular static imports, so the integration layer uses import_module.
